"""Bounded setup of the separately registered expanded adapter run."""
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


def run_expanded_training(plan,worker,adapter,publish):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Existing hosted Colab runtime only; no local model/audio install')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if gpu.returncode or len(gpu.stdout.strip().splitlines())!=1 or int(gpu.stdout.strip().rsplit(',',1)[1])<14000:raise RuntimeError('A hosted GPU with14000MiB is required; no CPU fallback or paid upgrade')
    root=Path('/content/anyanwu-waxal-female5-expanded-lora-v2');root.mkdir(exist_ok=True)
    result_path=root/'training-result.json'
    if result_path.exists():
        if json.loads((root/'plan.json').read_text())!=plan:raise RuntimeError('Different completed run exists')
        result=json.loads(result_path.read_text())
        if result.get('optimizer_updates')!=60 or hashlib.sha256((root/'adapter.safetensors').read_bytes()).hexdigest()!=result['adapter_sha256'] or hashlib.sha256((root/'baseline/adapter.safetensors').read_bytes()).hexdigest()!=plan['starting_adapter_sha256']:
            raise RuntimeError('Existing completed adapter/baseline integrity failure')
        return result
    if (root/'attempt.json').exists():raise RuntimeError('An attempt exists; inspect it before a declared repair')
    if shutil.disk_usage('/content').free<25*1024**3:raise RuntimeError('Twenty-five GiB cloud free disk required')
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive must be connected for the final adapter backup')
    started=time.monotonic();initial_free=shutil.disk_usage('/content').free
    (root/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (root/'worker.py').write_text(worker);(root/'flash_adapter.py').write_text(adapter)
    (root/'attempt.json').write_text(json.dumps({'phase':'started','optimizer_updates':0,'minutes':30}))
    excluded={'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','OPENAI_API_KEY','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}
    env={k:v for k,v in os.environ.items() if k not in excluded}
    env.update(ANYANWU_EXPANDED_TRAIN='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
       HF_HOME=str(root/'cache/hf'),XDG_CACHE_HOME=str(root/'cache'),UV_CACHE_DIR=str(root/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(root/'python'),
       PIP_CONFIG_FILE='/dev/null',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',
       TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    phase='data_acquisition';peak=0;latest={}
    def bounded(command,name):
        nonlocal phase,peak,latest
        phase=name;publish({'phase':name,'optimizer_updates':latest.get('optimizer_updates',0)})
        log=root/(name+'.log')
        with log.open('w') as output:
            process=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            last=None
            try:
                while process.poll() is None:
                    free=shutil.disk_usage('/content').free
                    statm=Path(f'/proc/{process.pid}/statm')
                    rss=int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0
                    peak=max(peak,rss)
                    if time.monotonic()-started>1800 or initial_free-free>20*1024**3 or free<5*1024**3 or rss>9*1024**3 or log.stat().st_size>512*1024:
                        raise RuntimeError('Training time/disk/RSS/log cap')
                    progress=root/'progress.json'
                    if name=='training_worker' and progress.exists():
                        value=progress.read_text()
                        if value!=last:
                            last=value;latest=json.loads(value);publish(latest)
                    time.sleep(2)
                if process.returncode:raise RuntimeError(f'{name} exited {process.returncode}; inspect bounded log')
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:
        restore_expanded_training(plan,publish)
        bounded([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:',
                 '--target',str(root/'bootstrap'),'uv==0.9.13','--index-url','https://pypi.org/simple'],'bootstrap_uv')
        uv=root/'bootstrap/bin/uv'
        bounded([str(uv),'venv','--python','3.11.14','--managed-python','--seed',str(root/'venv')],'isolated_python311')
        python=root/'venv/bin/python'
        bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cu128'],'pinned_gpu_torch')
        bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','transformers==4.51.3','huggingface_hub==0.36.0',
                 'onnxruntime==1.23.2','numpy==1.26.4','safetensors==0.5.3','--index-url','https://pypi.org/simple'],'pinned_tts_dependencies')
        bounded([str(python),str(root/'worker.py')],'training_worker')
        result=json.loads(result_path.read_text())
        result.update(peak_worker_rss_bytes=peak,cloud_disk_increase_bytes=initial_free-shutil.disk_usage('/content').free,
                      total_setup_training_seconds=round(time.monotonic()-started,2))
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (root/'attempt.json').write_text(json.dumps({'phase':'complete','optimizer_updates':result['optimizer_updates']}))
        return result
    except Exception as error:
        progress=root/'progress.json'
        if progress.exists():latest=json.loads(progress.read_text())
        report={**latest,'phase':'expanded_training_attempt_stopped','failed_phase':phase,'error_type':type(error).__name__,
                'error':str(error)[:400],'optimizer_updates':latest.get('optimizer_updates',0),'peak_worker_rss_bytes':peak}
        log=root/(phase+'.log')
        if log.exists():report['log_tail']=log.read_text(errors='replace')[-2400:]
        (root/'attempt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        return report
