"""Verify the saved cloud adapter and frozen base without further optimisation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import subprocess


def verify_flash_pilot():
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Cloud-only saved-adapter verification')
    work=Path('/content/anyanwu-flash-lora-pilot-v1')
    child=r'''
from pathlib import Path
import hashlib,importlib.metadata,json,math,sys
import torch
from safetensors.torch import load_file
w=Path('/content/anyanwu-flash-lora-pilot-v1')
r=json.loads((w/'training-result.json').read_text())
p=json.loads((w/'plan.json').read_text())
assert r['optimizer_updates']==5 and len(r['losses'])==5
assert all(math.isfinite(x['speech_cross_entropy']) and math.isfinite(x['gradient_norm']) for x in r['losses'])
f=Path(r['adapter_path'])
assert f.parent==w and f.stat().st_size==r['adapter_bytes'] and f.stat().st_size<16*1024**2
assert hashlib.sha256(f.read_bytes()).hexdigest()==r['adapter_sha256']
state=load_file(str(f),device='cpu')
assert len(state)==97 and state['source_vectors'].shape==(3,896)
assert sum(t.numel() for t in state.values())==r['trainable_parameters']==273024
assert all(torch.isfinite(t).all().item() for t in state.values())
assert all(n=='source_vectors' or n.endswith(('.A','.B')) for n in state)
revision=p['models'][0]['revision']
base=w/'cache/hf/hub/models--Modularcomputing--dynamic-accent-preview/snapshots'/revision/'checkpoints/flash/flash_dpo/best.pt'
h=hashlib.sha256()
with base.open('rb') as handle:
    for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
assert h.hexdigest()==r['base_checkpoint_sha256']
owned_bytes=sum(f.lstat().st_size for f in w.rglob('*') if f.is_file() and not f.is_symlink())
verification={'adapter_reloaded':True,'finite_tensors':97,'parameter_count':273024,'adapter_hash_matches':True,
              'base_hash_unchanged':True,'additional_optimizer_updates':0,'cloud_owned_file_bytes':owned_bytes,
              'python':sys.version.split()[0],
              'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','onnxruntime','numpy','huggingface_hub','safetensors')}}
r['post_training_integrity']=verification
(w/'training-result.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(r,ensure_ascii=False))
'''
    process=subprocess.run([str(work/'venv/bin/python'),'-c',child],capture_output=True,text=True,timeout=120)
    if process.returncode or len(process.stdout)>24000:
        raise RuntimeError('Saved-adapter verification failed: '+process.stderr[-900:])
    return json.loads(process.stdout)
