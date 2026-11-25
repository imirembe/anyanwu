"""GPU-gated fixed trial with verified private Drive backup; no automatic allocation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import zipfile


def run_conversational_trial(plan,worker,adapter,publish):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Hosted Colab GPU only; no laptop models/media or installs')
    executable=shutil.which('nvidia-smi')
    if not executable:raise RuntimeError('Hosted GPU required; current CPU is preserved, no allocation or install attempted')
    gpu=subprocess.run([executable,'--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if gpu.returncode or len(gpu.stdout.strip().splitlines())!=1 or int(gpu.stdout.strip().rsplit(',',1)[1])<14000:
        raise RuntimeError('One hosted GPU with at least14000MiB required; no automatic upgrade')
    root=Path('/content/anyanwu-conversational-style-lora-v1');root.mkdir(exist_ok=True)
    result_path=root/'training-result.json'
    if result_path.exists():
        if json.loads((root/'plan.json').read_text())!=plan:raise RuntimeError('Different completed trial preserved')
        result=json.loads(result_path.read_text())
        if result.get('optimizer_updates')!=30 or hashlib.sha256((root/'adapter.safetensors').read_bytes()).hexdigest()!=result['adapter_sha256']:
            raise RuntimeError('Completed trial integrity failure')
    else:
        if (root/'attempt.json').exists():raise RuntimeError('A trial attempt exists; inspect it before a declared repair')
        if not Path('/content/drive/MyDrive').is_dir() or shutil.disk_usage('/content').free<25*1024**3:raise RuntimeError('Research Drive and25GiB free cloud disk required')
        (root/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');(root/'worker.py').write_text(worker);(root/'flash_adapter.py').write_text(adapter)
        (root/'attempt.json').write_text('{"phase":"started","optimizer_updates":0}\n')
        env={k:v for k,v in os.environ.items() if k not in {'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','OPENAI_API_KEY','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}}
        env.update(ANYANWU_CONVERSATIONAL_TRIAL='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                   HF_HOME=str(root/'cache/hf'),XDG_CACHE_HOME=str(root/'cache'),UV_CACHE_DIR=str(root/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(root/'python'),
                   PIP_CONFIG_FILE='/dev/null',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',
                   TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        started=time.monotonic();initial_free=shutil.disk_usage('/content').free;latest={};phase='source_restore';peak=0
        def bounded(command,name):
            nonlocal latest,phase,peak
            phase=name;log=root/(name+'.log');publish({'phase':'conversational_asr_trial_'+name,'optimizer_updates':latest.get('optimizer_updates',0)})
            with log.open('w') as out:
                process=subprocess.Popen(command,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);last=None
                try:
                    while process.poll() is None:
                        free=shutil.disk_usage('/content').free;statm=Path(f'/proc/{process.pid}/statm')
                        rss=int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0;peak=max(peak,rss)
                        if time.monotonic()-started>1800 or initial_free-free>20*1024**3 or free<5*1024**3 or rss>9*1024**3 or log.stat().st_size>512*1024:
                            raise RuntimeError('Trial time/disk/memory/log cap')
                        progress=root/'progress.json'
                        if progress.exists():
                            current=progress.read_text()
                            if current!=last:last=current;latest=json.loads(current);publish({**latest,'phase':'conversational_asr_'+latest['phase']})
                        time.sleep(2)
                    if process.returncode:raise RuntimeError(f'{name} failed ({process.returncode}); inspect bounded log')
                finally:
                    if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
        try:
            publish(restore_conversational_trial(plan))
            bounded([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:',
                     '--target',str(root/'bootstrap'),'uv==0.9.13','--index-url','https://pypi.org/simple'],'bootstrap')
            uv=root/'bootstrap/bin/uv'
            bounded([str(uv),'venv','--python','3.11.14','--managed-python','--seed',str(root/'venv')],'python311')
            python=root/'venv/bin/python'
            bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cu128'],'torch')
            bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','transformers==4.51.3','huggingface_hub==0.36.0',
                     'numpy==1.26.4','safetensors==0.5.3','--index-url','https://pypi.org/simple'],'libraries')
            bounded([str(python),str(root/'worker.py')],'training')
            result=json.loads(result_path.read_text());result.update(peak_worker_rss_bytes=peak,total_setup_seconds=round(time.monotonic()-started,2),
                                                                    cloud_disk_increase_bytes=initial_free-shutil.disk_usage('/content').free)
            result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            (root/'attempt.json').write_text('{"phase":"complete","optimizer_updates":30}\n')
        except Exception as error:
            report={**latest,'phase':'conversational_asr_trial_stopped','failed_phase':phase,'error_type':type(error).__name__,'error':str(error)[:350],
                    'optimizer_updates':latest.get('optimizer_updates',0)}
            (root/'attempt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return report
    names=['adapter.safetensors','adapter-two-source.safetensors','training-result.json','plan.json','training-cases.json','user-transcripts.submitted.json','worker.py','flash_adapter.py']
    manifest={'files':{}}
    for name in names:
        path=root/name;raw=path.read_bytes()
        if path.is_symlink() or len(raw)>2*1024**2:raise RuntimeError('Trial archive member size cap')
        manifest['files'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    archive=root/'anyanwu-conversational-style-lora-v1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for name in names+['manifest.json']:
            raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode() if name=='manifest.json' else (root/name).read_bytes()
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o600<<16;package.writestr(info,raw)
    if archive.stat().st_size>16*1024**2:raise RuntimeError('Private trial archive exceeds16MiB')
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:raise RuntimeError('Trial archive CRC verification failed')
        for name,meta in manifest['files'].items():
            if hashlib.sha256(package.read(name)).hexdigest()!=meta['sha256']:raise RuntimeError('Trial archive member verification failed')
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Research Drive disconnected; completed adapter preserved in cloud')
    backup=Path('/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research/checkpoints/conversational-style-lora-v1');backup.mkdir(parents=True,exist_ok=True)
    destination=backup/archive.name;raw=archive.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if destination.exists() and destination.read_bytes()!=raw:raise RuntimeError('Different existing private trial archive preserved')
    if not destination.exists():
        with destination.open('xb') as handle:handle.write(raw)
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise RuntimeError('Final trial Drive backup verification failed')
    return {**result,'phase':'conversational_asr_trial_complete','drive_backup_verified':True,'drive_archive':str(destination),'archive_bytes':len(raw),'archive_sha256':digest}
