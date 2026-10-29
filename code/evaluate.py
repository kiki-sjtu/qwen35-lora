"""Answer-token-weighted dev loss; this is not generation accuracy."""

from runtime import ensure_project_python
ensure_project_python(__file__)

import argparse
import json
from pathlib import Path
from qwen_config import QwenConfig
from data_handle.data_loader import QwenDataset,get_qwen_data_loader
from utils.modeling import load_qwen_model,load_qwen_tokenizer
from train import evaluate_qwen_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--base-only',action='store_true')
    parser.add_argument('--original-dev',action='store_true')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    config=QwenConfig.load(args.config)
    if args.base_only and args.adapter: parser.error('--base-only cannot be combined with --adapter')
    if args.limit<0: parser.error('--limit must be nonnegative')
    adapter=None if args.base_only else (args.adapter.resolve() if args.adapter else config.path('output_dir')/'best')
    if adapter and not (adapter/'adapter_config.json').exists(): raise FileNotFoundError(adapter)
    tokenizer=load_qwen_tokenizer(config)
    data_path=config.path('original_dev_path' if args.original_dev else 'dev_path')
    data=QwenDataset(data_path,tokenizer,config.max_seq_length,args.limit)
    model,precision,device=load_qwen_model(config,adapter,training=False)
    metrics=evaluate_qwen_model(model,get_qwen_data_loader(data,tokenizer,config),device,precision,config.loss_chunk_size)
    metrics.update(dataset=str(data_path),adapter=str(adapter) if adapter else None)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
