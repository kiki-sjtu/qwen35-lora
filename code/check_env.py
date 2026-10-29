
from runtime import ensure_project_python
ensure_project_python(__file__)

import importlib.metadata
import json
import sys
from pathlib import Path
import torch
from transformers import Qwen3_5ForConditionalGeneration
from peft import LoraConfig


def main():
    report={'python':sys.executable,'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','peft','accelerate')},
            'cuda_available':torch.cuda.is_available()}
    if not report['cuda_available']: raise RuntimeError('CUDA is unavailable; run setup_env.ps1 with CUDA PyTorch installed')
    report.update(gpu=torch.cuda.get_device_name(),vram_gb=round(torch.cuda.get_device_properties(0).total_memory/1024**3,2),bf16=torch.cuda.is_bf16_supported())
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':main()
