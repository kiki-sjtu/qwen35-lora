"""Single-GPU Qwen3.5 LoRA training, adapted from the course train.py."""

from runtime import ensure_project_python
ensure_project_python(__file__)

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import torch
from qwen_config import QwenConfig, ROOT
from data_handle.data_loader import QwenDataset, get_qwen_data_loader
from utils.modeling import qwen_answer_loss_sum, load_qwen_model, load_qwen_tokenizer
from utils.common_utils import file_sha, restore_state, safe_perplexity, save_checkpoint, seed_everything, write_json


def evaluate_qwen_model(model,loader,device,precision,chunk_size):
    was_training = model.training
    model.eval()
    total, tokens = 0., 0
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = {k:v.to(device) for k,v in batch.items()}
                with torch.autocast(device.type,dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision!='fp32' and device.type=='cuda'):
                    value,count = qwen_answer_loss_sum(model,batch,chunk_size)
                total += float(value); tokens += count
        if not tokens: raise ValueError('Evaluation has no target tokens')
        loss = total/tokens
        if not math.isfinite(loss): raise FloatingPointError('Non-finite evaluation loss')
        return {'loss':loss,'perplexity':safe_perplexity(loss),'answer_tokens':tokens,'samples':len(loader.dataset)}
    finally:
        model.train(was_training)


def build_qwen_datasets(config,tokenizer):
    train = QwenDataset(config.path('train_path'),tokenizer,config.max_seq_length,config.train_limit)
    dev = QwenDataset(config.path('dev_path'),tokenizer,config.max_seq_length,config.eval_limit)
    original = QwenDataset(config.path('original_dev_path'),tokenizer,config.max_seq_length,config.eval_limit) if config.path('original_dev_path').exists() else None
    return train,dev,original


def check_qwen_data(config,tokenizer,datasets):
    report = {}
    for name,data in zip(('train','dev','dev_original'),datasets):
        if data is None: continue
        lengths = [len(row['input_ids']) for row in data.examples]
        targets = [sum(x!=-100 for x in row['labels'][1:]) for row in data.examples]
        report[name] = {'rows':len(data),'max_tokens':max(lengths),'min_answer_tokens':min(targets),'answer_tokens':sum(targets)}
        assert min(targets)>0
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--check-data',action='store_true',help='Validate all labels/masks; do not load model weights')
    parser.add_argument('--resume',type=Path,help='Exact checkpoint directory, including optimizer/RNG state')
    for argument in ('max-steps','train-limit','eval-limit','epochs'):
        parser.add_argument('--'+argument,type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--stop-after-step',type=int,help='Save and stop at this update; omit this option when resuming')
    args=parser.parse_args()
    if args.stop_after_step is not None and args.stop_after_step<=0: parser.error('--stop-after-step must be positive')
    config=QwenConfig.load(args.config)
    for name in ('max_steps','train_limit','eval_limit','epochs','output_dir'):
        value=getattr(args,name)
        if value is not None: setattr(config,name,value)
    config.validate()
    seed_everything(config.seed)
    tokenizer=load_qwen_tokenizer(config)
    train,dev,original=build_qwen_datasets(config,tokenizer)
    data_report=check_qwen_data(config,tokenizer,(train,dev,original))
    if args.check_data: return
    output=config.path('output_dir')
    if args.resume:
        resume=args.resume.resolve()
        if not (resume/'training_state.pt').is_file(): raise ValueError('Resume needs a full checkpoint, not only a LoRA adapter')
    else:
        resume=None
        if output.exists() and any(output.iterdir()):
            raise ValueError(f'Output is nonempty: {output}. Choose a new output_dir or use --resume.')
    output.mkdir(parents=True,exist_ok=True)
    model,precision,device=load_qwen_model(config,resume,training=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.learning_rate,weight_decay=config.weight_decay)
    batches=math.ceil(len(train)/config.batch_size)
    steps_per_epoch=math.ceil(batches/config.gradient_accumulation_steps)
    total_steps=steps_per_epoch*config.epochs
    if config.max_steps: total_steps=min(total_steps,config.max_steps)
    warmup=math.ceil(total_steps*config.warmup_ratio)
    def lr_scale(step):
        if warmup and step<warmup: return float(step+1)/warmup
        return max(0.,float(total_steps-step)/max(1,total_steps-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale)
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    signature={'config':config.dictionary(),'train_sha256':file_sha(config.path('train_path')),
               'dev_sha256':file_sha(config.path('dev_path')),'model_config_sha256':file_sha(config.path('model_path')/'config.json')}
    state={'global_step':0,'epoch_index':0,'next_batch_index':0,'best_eval_loss':float('inf'),'signature':signature}
    if resume:
        state=restore_state(resume,optimizer,scheduler,scaler)
        if state['signature']!=signature:
            raise ValueError('Config/data differ from the checkpoint; exact resume rejected')
    write_json(output/'run_config.json',config.dictionary())
    write_json(output/'data_check.json',data_report)
    dev_loader=get_qwen_data_loader(dev,tokenizer,config)
    original_loader=get_qwen_data_loader(original,tokenizer,config) if original is not None else None
    print(f'precision={precision}; device={device}; effective_batch={config.batch_size*config.gradient_accumulation_steps}; optimizer_steps={total_steps}',flush=True)
    started=time.monotonic()
    for epoch in range(state['epoch_index'],config.epochs):
        if state['global_step']>=total_steps: break
        loader=get_qwen_data_loader(train,tokenizer,config,epoch,shuffle=True)
        iterator=iter(loader)
        consumed=state['next_batch_index'] if epoch==state['epoch_index'] else 0
        for _ in range(consumed): next(iterator)
        model.train()
        while consumed<len(loader) and state['global_step']<total_steps:
            group=list(itertools.islice(iterator,config.gradient_accumulation_steps))
            group_tokens=sum(int(batch['labels'][:,1:].ne(-100).sum()) for batch in group)
            if not group_tokens: raise ValueError('Accumulation group has no answer tokens')
            optimizer.zero_grad(set_to_none=True)
            group_loss=0.
            for cpu_batch in group:
                batch={k:v.to(device,non_blocking=True) for k,v in cpu_batch.items()}
                with torch.autocast(device.type,dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision!='fp32'):
                    value,_=qwen_answer_loss_sum(model,batch,config.loss_chunk_size)
                    loss=value/group_tokens
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
                group_loss+=float(value.detach())
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config.max_grad_norm,error_if_nonfinite=True)
            before_scale=scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale()>=before_scale: scheduler.step()
            consumed+=len(group)
            state.update(global_step=state['global_step']+1,epoch_index=epoch,next_batch_index=consumed)
            step=state['global_step']
            record={'step':step,'epoch':epoch+1,'loss':group_loss/group_tokens,'lr':optimizer.param_groups[0]['lr'],
                    'grad_norm':float(norm),'answer_tokens':group_tokens,'elapsed_seconds':round(time.monotonic()-started,2)}
            with (output/'training_log.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(record)+'\n')
            if step%config.logging_steps==0 or step==1: print(json.dumps(record),flush=True)
            if step%config.save_steps==0:
                save_checkpoint(output/'latest',model,tokenizer,config,optimizer,scheduler,scaler,state)
            if args.stop_after_step and step>=args.stop_after_step:
                save_checkpoint(output/'latest',model,tokenizer,config,optimizer,scheduler,scaler,state)
                print(f'Saved and stopped after step {step}. Resume from {output / "latest"}.',flush=True)
                return
        # Full dev by default; a short run can explicitly set --eval-limit.
        metrics={'step':state['global_step'],'epoch':epoch+1,'dev':evaluate_qwen_model(model,dev_loader,device,precision,config.loss_chunk_size)}
        if original_loader:
            metrics['dev_original']=evaluate_qwen_model(model,original_loader,device,precision,config.loss_chunk_size)
        if metrics['dev']['loss']<state['best_eval_loss']:
            state['best_eval_loss']=metrics['dev']['loss']
            save_checkpoint(output/'best',model,tokenizer,config)
        write_json(output/f'eval_step_{state["global_step"]}.json',metrics)
        print(json.dumps(metrics),flush=True)
        if consumed==len(loader): state.update(epoch_index=epoch+1,next_batch_index=0)
        save_checkpoint(output/'latest',model,tokenizer,config,optimizer,scheduler,scaler,state)
    print(f'Finished. Adapter: {output / "latest"}',flush=True)


if __name__=='__main__':
    main()
