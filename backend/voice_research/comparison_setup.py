"""Bounded synthesis dependencies and selected acoustic files in existing Colab."""
from pathlib import Path
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import time


def prepare_comparison(plan,runtime,publish):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Existing hosted Colab only; no laptop audio/model installation')
    train=Path('/content/anyanwu-waxal-female5-lora-v1')
    python=train/'venv/bin/python'
    if not python.is_file() or not (train/'training-result.json').is_file():
        raise RuntimeError('Warm training environment is missing; restore the verified training archive before synthesis')
    result=json.loads((train/'training-result.json').read_text())
    if result['adapter_sha256']!=plan['adapter_sha256'] or result['optimizer_updates']!=60:
        raise RuntimeError('Unexpected trained adapter')
    root=Path('/content/anyanwu-waxal-female5-comparison-v1');root.mkdir(exist_ok=True)
    receipt=root/'setup-result.json'
    if receipt.exists():
        if json.loads((root/'plan.json').read_text())!=plan:raise RuntimeError('Comparison plan changed; no implicit repeat')
        return json.loads(receipt.read_text())
    if (root/'setup-attempt.json').exists():raise RuntimeError('An earlier setup attempt exists; inspect before a declared repair')
    if shutil.disk_usage('/content').free<12*1024**3:raise RuntimeError('Twelve GiB cloud free disk required')
    (root/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (root/'cv3_runtime.py').write_text(runtime)
    (root/'setup-attempt.json').write_text(json.dumps({'phase':'started','minutes_cap':15,'optimizer_updates':0}))
    constraints='\n'.join(['torch==2.8.0','transformers==4.51.3','huggingface_hub==0.36.0','numpy==1.26.4','onnxruntime==1.23.2','safetensors==0.5.3'])+'\n'
    (root/'constraints.txt').write_text(constraints)
    excluded={'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','OPENAI_API_KEY','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}
    env={k:v for k,v in os.environ.items() if k not in excluded}
    env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
       HF_HOME=str(train/'cache/hf'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',PIP_CONFIG_FILE='/dev/null')
    started=time.monotonic();initial_free=shutil.disk_usage('/content').free;phase='initialising'
    def bounded(command,name):
        nonlocal phase
        phase=name;publish({'phase':name,'optimizer_updates_this_stage':0})
        log=root/(name+'.log')
        with log.open('w') as handle:
            process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while process.poll() is None:
                    free=shutil.disk_usage('/content').free
                    statm=Path(f'/proc/{process.pid}/statm');rss=int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0
                    if time.monotonic()-started>900 or initial_free-free>8*1024**3 or free<5*1024**3 or rss>9*1024**3 or log.stat().st_size>512*1024:
                        raise RuntimeError('Synthesis setup time/disk/memory/log cap')
                    time.sleep(2)
                if process.returncode:raise RuntimeError(f'{name} exited {process.returncode}; inspect bounded log')
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:
        bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','--constraint',str(root/'constraints.txt'),'torchaudio==2.8.0','--index-url','https://download.pytorch.org/whl/cu128'],'matching_torchaudio')
        bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','--constraint',str(root/'constraints.txt'),
             'x-transformers==2.11.24','einops==0.8.1','einx==0.3.0','omegaconf==2.3.0','scipy==1.11.4','librosa==0.10.2','--index-url','https://pypi.org/simple'],'acoustic_dependencies')
        source='''
from pathlib import Path
import hashlib,json
from huggingface_hub import hf_hub_download
root=Path('/content/anyanwu-waxal-female5-comparison-v1')
plan=json.loads((root/'plan.json').read_text());files={}
for f in plan['acoustic_files']:
    p=Path(hf_hub_download(repo_id=plan['acoustic_repo'],revision=plan['acoustic_revision'],filename=f['name'],token=False,cache_dir='/content/anyanwu-waxal-female5-lora-v1/cache/hf/hub'))
    if p.stat().st_size!=f['size']:raise RuntimeError('Acoustic file size mismatch')
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=f['sha256']:raise RuntimeError('Acoustic file checksum mismatch')
    files[f['name']]=str(p)
(root/'acoustic-files.json').write_text(json.dumps(files,indent=2)+'\\n')
import cv3_runtime
print('Verified acoustic model files and imported pinned inference classes.')
'''
        (root/'prepare_acoustic.py').write_text(source)
        bounded([str(python),str(root/'prepare_acoustic.py')],'verified_acoustic_models')
        report={'phase':'comparison_synthesis_setup_ready','elapsed_seconds':round(time.monotonic()-started,2),'selected_acoustic_bytes':sum(f['size'] for f in plan['acoustic_files']),
                'cloud_disk_increase_bytes':initial_free-shutil.disk_usage('/content').free,'optimizer_updates_this_stage':0,'laptop_media_or_models':False}
        receipt.write_text(json.dumps(report,indent=2)+'\n');return report
    except Exception as error:
        report={'phase':'comparison_setup_stopped','failed_phase':phase,'error_type':type(error).__name__,'error':str(error)[:400],'optimizer_updates_this_stage':0}
        log=root/(phase+'.log')
        if log.exists():report['log_tail']=log.read_text(errors='replace')[-2200:]
        (root/'setup-attempt.json').write_text(json.dumps(report,indent=2)+'\n');return report
