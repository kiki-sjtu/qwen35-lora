import json
import torch
from torch.utils.data import DataLoader, Dataset
from .data_preprocess import convert_example_qwen35


class QwenDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_length, limit=0):
        self.examples = []
        self.path = str(path)
        with open(path, encoding='utf-8-sig') as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip():
                    continue
                try:
                    self.examples.append(convert_example_qwen35(json.loads(line),tokenizer,max_seq_length))
                except Exception as exc:
                    raise ValueError(f'{path}:{line_number}: {exc}') from exc
                if limit and len(self.examples)>=limit:
                    break
        if not self.examples:
            raise ValueError(f'Empty dataset: {path}')

    def __len__(self): return len(self.examples)

    def __getitem__(self,index): return self.examples[index]


class QwenCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        longest = max(len(row['input_ids']) for row in examples)
        batch = {}
        for field,pad in [('input_ids',self.pad_token_id),('attention_mask',0),('labels',-100)]:
            batch[field] = torch.tensor([row[field]+[pad]*(longest-len(row[field])) for row in examples],dtype=torch.long)
        return batch


def get_qwen_data_loader(dataset, tokenizer, config, epoch=0, shuffle=False):
    # A dedicated generator makes resumed epoch ordering reproducible on Windows.
    generator = torch.Generator().manual_seed(config.seed+epoch)
    return DataLoader(dataset,batch_size=config.batch_size,shuffle=shuffle,
                      collate_fn=QwenCollator(tokenizer.pad_token_id),num_workers=0,
                      pin_memory=torch.cuda.is_available(),generator=generator)
