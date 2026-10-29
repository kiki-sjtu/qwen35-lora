"""Load a frozen Qwen VLM and adapt only its language-model linear layers."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
# The native Windows path uses eager PyTorch kernels, not torch.compile/Triton.
if os.name == 'nt':
    os.environ.setdefault('TORCHDYNAMO_DISABLE','1')

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


def load_qwen_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(str(config.path('model_path')),local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer with offsets is required for precise loss masking')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def select_precision(config, require_cuda=True):
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use this project .venv with the CUDA PyTorch installation.')
    precision = config.precision
    if precision == 'auto':
        precision = 'bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ('fp16' if torch.cuda.is_available() else 'fp32')
    if precision=='bf16' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('This GPU does not support BF16; select fp16')
    return precision, {'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[precision]


def load_qwen_model(config, adapter_path=None, training=True):
    precision,dtype = select_precision(config,require_cuda=training)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu': dtype = torch.float32
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(config.path('model_path')),local_files_only=True,dtype=dtype,
        attn_implementation=config.attention_implementation)
    base.config.use_cache = not training
    base.config.text_config.use_cache = not training
    if adapter_path:
        model = PeftModel.from_pretrained(base,str(adapter_path),is_trainable=training)
    elif training:
        targets = [name for name,module in base.named_modules()
                   if name.startswith('model.language_model.layers.') and isinstance(module,torch.nn.Linear)]
        if not targets:
            raise RuntimeError('Qwen language modules not found; verify Transformers version')
        model = get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=config.lora_rank,
            lora_alpha=config.lora_alpha,lora_dropout=config.lora_dropout,bias='none',target_modules=targets))
    else:
        model = base
    if training:
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.enable_input_require_grads()
        model.train()
        model.print_trainable_parameters()
        if any(p.requires_grad and 'lora_' not in name for name,p in model.named_parameters()):
            raise RuntimeError('A non-LoRA base parameter was unexpectedly unfrozen')
    else:
        model.eval()
    return model.to(device),precision,device


def qwen_answer_loss_sum(model,batch,chunk_size=32):
    """Causal CE sum and token count, without allocating full [B,T,vocab] logits.

    The backbone includes injected LoRA layers. Chunked checkpointed LM projection
    is algebraically equivalent to normal causal CE with ignore_index=-100.
    """
    base = model.get_base_model() if isinstance(model,PeftModel) else model
    result = base.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                        use_cache=False,return_dict=True)
    labels = batch['labels'][:,1:]
    mask = labels.ne(-100)
    hidden = result.last_hidden_state[:,:-1,:][mask]
    targets = labels[mask]
    if not targets.numel(): raise ValueError('No supervised answer tokens')
    def chunk_loss(h,y):
        return F.cross_entropy(base.lm_head(h).float(),y,reduction='sum')
    total = hidden.new_zeros((),dtype=torch.float32)
    for start in range(0,targets.numel(),chunk_size):
        h,y = hidden[start:start+chunk_size],targets[start:start+chunk_size]
        if model.training and torch.is_grad_enabled():
            total = total+checkpoint(chunk_loss,h,y,use_reentrant=False)
        else:
            total = total+chunk_loss(h,y)
    return total,int(targets.numel())
