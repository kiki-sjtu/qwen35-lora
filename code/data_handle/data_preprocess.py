"""Use the actual Qwen template; supervise only the assistant's JSON and EOS."""
import json


def convert_example_qwen35(example, tokenizer, max_seq_length):
    messages = example.get('messages')
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError('Expected three messages: system/user/assistant')
    if [m.get('role') for m in messages] != ['system','user','assistant']:
        raise ValueError('Expected system/user/assistant roles')
    if any(not isinstance(m.get('content'),str) or not m['content'].strip() for m in messages):
        raise ValueError('Message contents must be nonempty strings')
    json.loads(messages[-1]['content'])
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(prompt):
        raise ValueError('Template prefix mismatch: cannot safely mask the prompt')
    # Offsets avoid assuming that separate tokenization of prompt+answer is additive.
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    if len(ids) > max_seq_length:
        raise ValueError(f'Sample has {len(ids)} tokens, exceeds max_seq_length={max_seq_length}; no silent truncation')
    boundary = len(prompt)
    labels = [token if start >= boundary and end > start else -100
              for token,(start,end) in zip(ids,encoded['offset_mapping'])]
    eos = tokenizer.eos_token_id
    if eos not in [token for token in labels if token != -100]:
        raise ValueError('Assistant EOS not supervised')
    # Do not train on template whitespace after the assistant EOS.
    last_eos = max(i for i,token in enumerate(labels) if token == eos)
    labels[last_eos+1:] = [-100] * (len(labels)-last_eos-1)
    if not any(token != -100 for token in labels[1:]):
        raise ValueError('Sample has no causal prediction target')
    return {'input_ids':ids, 'attention_mask':[1]*len(ids), 'labels':labels}
