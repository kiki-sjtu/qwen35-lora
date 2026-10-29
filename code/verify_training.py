"""Verify masking, real-model CE, and one longest-sample LoRA optimizer step."""

from runtime import ensure_project_python
ensure_project_python(__file__)

import argparse
import json
import math
from pathlib import Path
import torch
from qwen_config import QwenConfig, ROOT
from data_handle.data_loader import QwenDataset,QwenCollator
from utils.modeling import qwen_answer_loss_sum,load_qwen_model,load_qwen_tokenizer
from utils.common_utils import save_checkpoint,seed_everything,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backward',action='store_true',help='Perform one test update on the longest training sample')
    args=parser.parse_args()
    config=QwenConfig.load()
    seed_everything(config.seed)
    tokenizer=load_qwen_tokenizer(config)
    data=QwenDataset(config.path('train_path'),tokenizer,config.max_seq_length)
    collator=QwenCollator(tokenizer.pad_token_id)
    shortest=min(data.examples,key=lambda r:len(r['input_ids']))
    longest=max(data.examples,key=lambda r:len(r['input_ids']))
    padded=collator([shortest,longest])
    start=len(shortest['input_ids'])
    assert bool(padded['labels'][0,start:].eq(-100).all())
    assert bool(padded['attention_mask'][0,start:].eq(0).all())
    # The supervised suffix must decode to the original answer followed by EOS,
    # never the user/system prompt or the empty thinking prefix.
    for row in data.examples:
        supervised=tokenizer.decode([x for x in row['labels'] if x!=-100],skip_special_tokens=False)
        assert supervised.endswith(tokenizer.eos_token)
        json.loads(supervised[:-len(tokenizer.eos_token)])
    model,precision,device=load_qwen_model(config,training=True)
    model.eval()
    batch={k:v.to(device) for k,v in collator([shortest]).items()}
    with torch.no_grad():
        reference=float(model(**batch).loss)
        summed,tokens=qwen_answer_loss_sum(model,batch,config.loss_chunk_size)
        actual=float(summed)/tokens
    assert math.isfinite(actual) and abs(reference-actual)<0.003,(reference,actual)
    report={'cuda_gpu':torch.cuda.get_device_name(),'precision':precision,
            'all_5000_answer_masks_valid':True,'padding_masks_valid':True,
            'reference_causal_loss':reference,'chunked_causal_loss':actual,
            'loss_difference':abs(reference-actual),'backward_step_completed':False}
    if args.backward:
        model.train();torch.cuda.reset_peak_memory_stats()
        parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
        before={n:p.detach().cpu().clone() for n,p in parameters.items()}
        optimizer=torch.optim.AdamW(parameters.values(),lr=config.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        batch={k:v.to(device) for k,v in collator([longest]).items()}
        with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision!='fp32'):
            loss_sum,count=qwen_answer_loss_sum(model,batch,config.loss_chunk_size)
            loss=loss_sum/count
        # This workstation supports BF16. No FP16 smoke claim without a scaler.
        if precision=='fp16': raise RuntimeError('Use train.py for the FP16 scaled training path')
        loss.backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in parameters.values())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        torch.nn.utils.clip_grad_norm_(parameters.values(),config.max_grad_norm,error_if_nonfinite=True)
        optimizer.step()
        changed=sum(not torch.equal(before[n],p.detach().cpu()) for n,p in parameters.items())
        assert changed>0
        adapter=ROOT/'artifacts'/'checks'/'smoke_adapter'
        save_checkpoint(adapter,model,tokenizer,config)
        report.update(backward_step_completed=True,checked_sequence_tokens=len(longest['input_ids']),
                      test_loss=float(loss.detach()),updated_lora_tensors=changed,
                      trainable_parameters=sum(p.numel() for p in parameters.values()),
                      peak_allocated_gb=round(torch.cuda.max_memory_allocated()/1024**3,3),
                      peak_reserved_gb=round(torch.cuda.max_memory_reserved()/1024**3,3),
                      smoke_adapter=str(adapter),full_dataset_training_started=False)
    write_json(ROOT/'artifacts'/'checks'/'verification_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
