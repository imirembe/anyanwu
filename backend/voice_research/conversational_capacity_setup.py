"""Bounded token-capacity preparation in a separate cloud dependency directory."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import time


def run_capacity(plan,worker,publish):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Hosted Colab CPU only; no laptop media/models or installation')
    root=Path('/content/anyanwu-conversational-capacity-v1')
    python=Path('/content/anyanwu-conversational-transcripts-v1/venv/bin/python')
    if not python.is_file(): raise RuntimeError('Existing CPU environment missing; no automatic cold reinstall')
    if not Path('/content/drive/MyDrive').is_dir(): raise RuntimeError('Connected research Drive required')
    if shutil.disk_usage('/content').free<8*1024**3: raise RuntimeError('Eight GiB free cloud disk required')
    root.mkdir(exist_ok=True)
    result_path=root/'capacity-result.json'
    if result_path.exists():
        if json.loads((root/'plan.json').read_text())!=plan: raise RuntimeError('Existing capacity plan differs')
    else:
        if (root/'attempt.json').exists(): raise RuntimeError('An attempt already exists; inspect before a declared repair')
        if hashlib.sha256(worker.encode()).hexdigest()!=plan['worker_sha256']: raise RuntimeError('Capacity worker source changed')
        (root/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');(root/'worker.py').write_text(worker)
        (root/'attempt.json').write_text('{"phase":"started","optimizer_updates":0}\n')
        env={k:v for k,v in os.environ.items() if k not in {'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','OPENAI_API_KEY','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}}
        env.update(ANYANWU_CONVERSATIONAL_CAPACITY='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                   HF_HOME=str(root/'cache/hf'),HF_HUB_DISABLE_XET='1',XDG_CACHE_HOME=str(root/'cache'),PIP_CONFIG_FILE='/dev/null',
                   HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        started=time.monotonic();initial_free=shutil.disk_usage('/content').free;phase='dependencies';peak=0
        def bounded(command,name,child_env):
            nonlocal phase,peak
            phase=name;log=root/(name+'.log');publish({'phase':'conversational_asr_capacity_'+name,'optimizer_updates':0})
            with log.open('w') as out:
                process=subprocess.Popen(command,env=child_env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                last=None;heartbeat=0
                try:
                    while process.poll() is None:
                        free=shutil.disk_usage('/content').free;statm=Path(f'/proc/{process.pid}/statm')
                        rss=int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0;peak=max(peak,rss)
                        if time.monotonic()-started>600 or initial_free-free>4*1024**3 or free<5*1024**3 or rss>6*1024**3 or log.stat().st_size>512*1024:
                            raise RuntimeError('Capacity audit time/disk/memory/log cap reached')
                        progress=root/'progress.json'
                        if progress.exists():
                            current=progress.read_text()
                            if current!=last:last=current;publish(json.loads(current))
                        if time.monotonic()-heartbeat>25:
                            heartbeat=time.monotonic();publish({'phase':'conversational_asr_capacity_'+name,'elapsed_seconds':round(time.monotonic()-started),'rss_bytes':rss,'optimizer_updates':0})
                        time.sleep(2)
                    if process.returncode: raise RuntimeError(f'{name} failed ({process.returncode}); inspect bounded log')
                finally:
                    if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
        try:
            # Isolated target: keep the existing ASR environment and live callbacks untouched.
            bounded([str(python),'-m','pip','install','--quiet','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:',
                     '--target',str(root/'dependencies'),'numpy==1.26.4','onnxruntime==1.23.2','transformers==4.51.3','huggingface_hub==0.36.0',
                     '--index-url','https://pypi.org/simple'],'dependencies',env)
            dependencies=sum(p.stat().st_size for p in (root/'dependencies').rglob('*') if p.is_file() and not p.is_symlink())
            if dependencies>512*1024**2:raise RuntimeError('Isolated dependency cap reached')
            bounded([str(python),str(root/'worker.py')],'encoding',{**env,'PYTHONPATH':str(root/'dependencies')})
            result=json.loads(result_path.read_text());result.update(peak_worker_rss_bytes=peak,total_setup_seconds=round(time.monotonic()-started,2),
                                                                    cloud_disk_increase_bytes=initial_free-shutil.disk_usage('/content').free,dependency_bytes=dependencies)
            result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            (root/'attempt.json').write_text(json.dumps({'phase':'complete','optimizer_updates':0})+'\n')
        except Exception as error:
            report={'phase':'conversational_asr_capacity_stopped','failed_phase':phase,'error_type':type(error).__name__,'error':str(error)[:350],'optimizer_updates':0}
            (root/'attempt.json').write_text(json.dumps(report,indent=2)+'\n');return report
    report=json.loads(result_path.read_text());targets=root/'speech-targets.json'
    if targets.stat().st_size>256*1024 or hashlib.sha256(targets.read_bytes()).hexdigest()!=report['speech_targets_sha256']:
        raise RuntimeError('Encoded target integrity failure')
    backup=Path('/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research/experiments/conversational-style-v1/preparation')
    backup.mkdir(exist_ok=True,parents=True);receipts={}
    for name in ('plan.json','capacity-result.json','speech-targets.json','attempt.json','worker.py'):
        raw=(root/name).read_bytes()
        if len(raw)>256*1024:raise RuntimeError('Small metadata backup cap')
        destination=backup/name
        if destination.exists() and destination.read_bytes()!=raw:raise RuntimeError('Different capacity backup preserved')
        if not destination.exists():
            with destination.open('xb') as handle:handle.write(raw)
        if destination.read_bytes()!=raw:raise RuntimeError('Capacity backup verification failed')
        receipts[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    return {**report,'drive_backup_verified':True,'drive_backup':str(backup),'backup_files':receipts}
