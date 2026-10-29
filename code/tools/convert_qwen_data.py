"""Convert course context/target JSONL into Qwen3.5 SFT messages.

Only transforms data; does not load model weights or start training.
"""

from pathlib import Path as _RuntimePath
import sys as _runtime_sys
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
from runtime import ensure_project_python
ensure_project_python(__file__)

import argparse
import hashlib
import json
import re
from pathlib import Path

CONTEXT = re.compile(r'Instruction: (.*?)\nInput: (.*)\nAnswer: *', re.S)
FENCE = re.compile(r'```(?:json)?\s*\n(.*?)\n```', re.S)
SPLITS = {'mixed_train_dataset.jsonl': 'train.jsonl', 'mixed_dev_dataset.jsonl': 'dev.jsonl'}


def read_rows(path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {'context', 'target'}:
            raise ValueError(f'{path.name}:{line_no}: unexpected schema')
        if not all(isinstance(row[key], str) for key in ('context', 'target')):
            raise ValueError(f'{path.name}:{line_no}: fields must be strings')
        rows.append((line_no, row))
    if not rows:
        raise ValueError(f'{path}: empty dataset')
    return rows


def convert(row):
    match = CONTEXT.fullmatch(row['context'])
    if match is None:
        raise ValueError('Context does not match Instruction/Input/Answer format')
    system, user = match.groups()
    target = row['target'].strip()
    fence = FENCE.fullmatch(target)
    answer = fence.group(1).strip() if fence else target
    json.loads(answer)  # Validate the label without altering its values or spacing.
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
        {'role': 'assistant', 'content': answer},
    ]
    if any(not message['content'].strip() for message in messages):
        raise ValueError('Empty message content')
    for message in messages:
        if any(token in message['content'] for token in ('<|im_start|>', '<|im_end|>', '<think>', '</think>')):
            raise ValueError('Source contains chat control tokens; manual inspection required')
    # These exact strings reconstruct the source, so no instruction/input was lost.
    rebuilt = f'Instruction: {system}\nInput: {user}\nAnswer: '
    if rebuilt.rstrip() != row['context'].rstrip():
        raise ValueError('Context reconstruction mismatch')
    return {'messages': messages}, bool(fence)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'artifacts' / 'converted_data')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        raise ValueError('Use a separate output directory to preserve original data')
    converted = {}
    report = {'format': 'messages JSONL', 'source_directory': str(source),
              'output_directory': str(output), 'training_started': False, 'splits': {}}
    # Validate both splits completely before writing any output.
    for source_name, target_name in SPLITS.items():
        source_path = source / source_name
        before_hash = digest(source_path)
        rows = read_rows(source_path)
        result, unfenced = [], 0
        for line_no, row in rows:
            try:
                item, removed = convert(row)
            except Exception as exc:
                raise ValueError(f'{source_name}:{line_no}: {exc}') from exc
            result.append(item)
            unfenced += removed
        converted[target_name] = result
        report['splits'][target_name] = {'source_file': source_name, 'rows': len(result),
            'json_fences_removed': unfenced, 'source_sha256': before_hash}

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in converted.items():
        path = output / name
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        reloaded = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if rows != reloaded:
            raise RuntimeError(f'{name}: output roundtrip mismatch')
        report['splits'][name]['output_sha256'] = digest(path)
        info = report['splits'][name]
        if digest(source / info['source_file']) != info['source_sha256']:
            raise RuntimeError('Source changed during conversion')
        print(f'{name}: {len(rows)} rows; removed {info["json_fences_removed"]} JSON code fences')
    train = {json.dumps(x, ensure_ascii=False, sort_keys=True) for x in converted['train.jsonl']}
    dev = {json.dumps(x, ensure_ascii=False, sort_keys=True) for x in converted['dev.jsonl']}
    report['exact_train_dev_overlap'] = len(train & dev)
    report['rows_removed'] = 0
    (output / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Conversion complete. Source unchanged; no training started.')


if __name__ == '__main__':
    main()
