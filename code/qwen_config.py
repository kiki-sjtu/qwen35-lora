"""Project-relative configuration for Qwen3.5 text-only LoRA."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent


@dataclass
class QwenConfig:
    model_path: str = 'models/Qwen3.5-0.8B'
    train_path: str = 'data/train.jsonl'
    dev_path: str = 'data/dev.jsonl'
    original_dev_path: str = 'data/dev_original.jsonl'
    output_dir: str = 'checkpoints/qwen_lora'
    max_seq_length: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 2
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    precision: str = 'auto'
    attention_implementation: str = 'sdpa'
    loss_chunk_size: int = 32
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 20261029
    max_steps: int = 0
    train_limit: int = 0
    eval_limit: int = 0

    @classmethod
    def load(cls, path=None):
        file = Path(path).resolve() if path else CODE_ROOT/'train_config.json'
        values = json.loads(file.read_text(encoding='utf-8')) if file.exists() else {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        for name in ('max_seq_length','batch_size','gradient_accumulation_steps','epochs','lora_rank','lora_alpha','loss_chunk_size','logging_steps','save_steps'):
            if getattr(self,name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.learning_rate <= 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError('Invalid learning rate/warmup ratio')
        if self.precision not in ('auto','bf16','fp16','fp32'):
            raise ValueError('precision must be auto/bf16/fp16/fp32')
        if not 0 <= self.lora_dropout < 1 or min(self.train_limit,self.eval_limit,self.max_steps)<0:
            raise ValueError('Invalid dropout or limits')

    def path(self, name):
        value = Path(getattr(self,name))
        return value.resolve() if value.is_absolute() else (ROOT/value).resolve()

    def dictionary(self):
        return asdict(self)
