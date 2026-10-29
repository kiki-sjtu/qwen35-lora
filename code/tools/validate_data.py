"""Validate converted SFT data against the downloaded Qwen chat template."""

from pathlib import Path as _RuntimePath
import sys as _runtime_sys
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
from runtime import ensure_project_python
ensure_project_python(__file__)

import json
import math
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'data' / 'reports'
DATA = ROOT / 'data'


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    model = ROOT / 'models' / 'Qwen3.5-0.8B'
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def raise_exception(message):
        raise ValueError(message)
    environment.globals['raise_exception'] = raise_exception
    template = environment.from_string((model / 'chat_template.jinja').read_text(encoding='utf-8'))
    def render(messages, add_generation_prompt=False):
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    report = {'model': 'Qwen/Qwen3.5-0.8B', 'enable_thinking': False,
              'uses_local_official_chat_template': True,
              'validator': 'Jinja2 sandbox + native tokenizers; no model weights loaded',
              'truncated_rows': 0, 'splits': {}}
    for name in ('train.jsonl', 'dev.jsonl'):
        rows = [json.loads(line) for line in (DATA / name).read_text(encoding='utf-8').splitlines()]
        lengths, prompts = [], set()
        for i, row in enumerate(rows, 1):
            assert set(row) == {'messages'}, (name, i)
            messages = row['messages']
            assert [m['role'] for m in messages] == ['system', 'user', 'assistant'], (name, i)
            assert all(isinstance(m['content'], str) and m['content'].strip() for m in messages), (name, i)
            json.loads(messages[-1]['content'])
            text = render(messages)
            prompt = render(messages[:-1], add_generation_prompt=True)
            assert text == prompt + messages[-1]['content'].strip() + '<|im_end|>\n', (name, i)
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            assert ids, (name, i)
            lengths.append(len(ids))
            prompts.add(json.dumps(messages[:-1], ensure_ascii=False, sort_keys=True))
            if name == 'train.jsonl' and i == 1:
                (REPORTS / 'formatted_example.txt').write_text(text, encoding='utf-8')
        ordered = sorted(lengths)
        report['splits'][name] = {'rows': len(rows), 'unique_prompts': len(prompts),
            'min_tokens': min(lengths), 'max_tokens': max(lengths),
            'mean_tokens': round(sum(lengths) / len(lengths), 2),
            'p95_tokens': ordered[math.ceil(len(ordered) * 0.95) - 1],
            'above_512_tokens': sum(n > 512 for n in lengths),
            'above_1024_tokens': sum(n > 1024 for n in lengths),
            'above_2048_tokens': sum(n > 2048 for n in lengths)}
        report['splits'][name]['_prompts'] = prompts
    train_prompts = report['splits']['train.jsonl'].pop('_prompts')
    dev_prompts = report['splits']['dev.jsonl'].pop('_prompts')
    report['train_dev_prompt_overlap'] = len(train_prompts & dev_prompts)
    (REPORTS / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
