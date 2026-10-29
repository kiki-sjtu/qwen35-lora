import hashlib
import json
import math
import random
from pathlib import Path
import torch


def seed_everything(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def save_checkpoint(path,model,tokenizer,config,optimizer=None,scheduler=None,scaler=None,state=None):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    # Save adapters, never deepcopy/merge an extra base model onto the 6GB GPU.
    model.save_pretrained(path,safe_serialization=True)
    tokenizer.save_pretrained(path)
    write_json(path/'run_config.json',config.dictionary())
    if optimizer is not None:
        payload={'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'scaler':scaler.state_dict(),'state':state,'rng_cpu':torch.get_rng_state(),
                 'rng_cuda':torch.cuda.get_rng_state_all(),'rng_python':random.getstate()}
        torch.save(payload,path/'training_state.tmp')
        (path/'training_state.tmp').replace(path/'training_state.pt')


def restore_state(path,optimizer,scheduler,scaler):
    payload=torch.load(Path(path)/'training_state.pt',map_location='cpu',weights_only=True)
    optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])
    torch.set_rng_state(payload['rng_cpu']); torch.cuda.set_rng_state_all(payload['rng_cuda'])
    random.setstate(payload['rng_python'])
    return payload['state']


def safe_perplexity(loss):
    return math.exp(loss) if loss<700 else None
