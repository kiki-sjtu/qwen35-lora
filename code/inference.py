"""Load the local Qwen base and an optional trained LoRA adapter."""

from runtime import ensure_project_python
ensure_project_python(__file__)

import argparse
import json
from pathlib import Path
import torch
from qwen_config import QwenConfig, ROOT
from utils.modeling import load_qwen_model, load_qwen_tokenizer


def qwen_inference(model,tokenizer,instruction,sentence,max_new_tokens=512):
    messages=[{'role':'system','content':instruction},{'role':'user','content':sentence}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    batch=tokenizer(text,add_special_tokens=False,return_tensors='pt').to(next(model.parameters()).device)
    with torch.inference_mode():
        output=model.generate(**batch,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
                              pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--base-only',action='store_true')
    parser.add_argument('--prompt',default='判断这段评论的主题类别，只输出JSON数组。\n\n这台笔记本电脑开机很快，运行编程软件也很流畅。')
    parser.add_argument('--max-new-tokens',type=int,default=512)
    args=parser.parse_args()
    config=QwenConfig.load(args.config)
    if args.base_only and args.adapter: parser.error('--base-only cannot be combined with --adapter')
    adapter=None if args.base_only else (args.adapter.resolve() if args.adapter else config.path('output_dir')/'best')
    if adapter and not (adapter/'adapter_config.json').exists():
        raise FileNotFoundError(f'No trained adapter: {adapter}. Train first or use --base-only.')
    tokenizer=load_qwen_tokenizer(config)
    model,_,_=load_qwen_model(config,adapter,training=False)
    print(qwen_inference(model,tokenizer,'你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。',args.prompt,args.max_new_tokens))


if __name__=='__main__': main()
