"""Colab supervisor for a bounded, real CPU LoRA pilot. Standard library only."""
from pathlib import Path
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time


def run_flash_training(plan, worker, adapter, publish):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Use the existing hosted Colab CPU runtime; no laptop installation or weights')
    work=Path('/content/anyanwu-flash-lora-pilot-v1')
    work.mkdir(exist_ok=True)
    result_path=work/'training-result.json'
    if result_path.exists():
        saved_plan=json.loads((work/'plan.json').read_text())
        if saved_plan!=plan:
            raise RuntimeError('Existing completed pilot has a different plan; no overwrite')
        return json.loads(result_path.read_text())
    repair=plan.get('repair',{}).get('id')=='rope-buffer-materialisation-v1'
    if (work/'attempt.json').exists():
        prior=json.loads((work/'attempt.json').read_text())
        if not repair or prior.get('optimizer_updates')!=0 or prior.get('error')!='Cannot copy out of meta tensor; no data!' or (work/'attempt-1').exists():
            raise RuntimeError('A training attempt already exists; no unregistered repeat or overwrite')
        archive=work/'attempt-1'
        archive.mkdir()
        for name in ('attempt.json','plan.json','progress.json','failure.json','training_worker.log','worker.py','flash_adapter.py'):
            source=work/name
            if source.exists():
                shutil.copyfile(source,archive/name)
    if shutil.disk_usage('/content').free<20*1024**3:
        raise RuntimeError('Twenty GiB cloud free disk required before setup')
    uv=Path('/content/anyanwu-cpu-transcripts-v1/bootstrap/bin/uv')
    python_base=Path('/content/anyanwu-cpu-transcripts-v1/venv/bin/python')
    if not uv.is_file() or not python_base.is_file():
        raise RuntimeError('The existing managed cloud Python/uv installation is unavailable')
    started=time.monotonic()
    initial_free=shutil.disk_usage('/content').free
    (work/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (work/'worker.py').write_text(worker,encoding='utf-8')
    (work/'flash_adapter.py').write_text(adapter,encoding='utf-8')
    (work/'attempt.json').write_text(json.dumps({'status':'started','plan':plan['id'],'optimizer_updates':0}))
    excluded={'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','OPENAI_API_KEY','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}
    env={k:v for k,v in os.environ.items() if k not in excluded}
    env.update(ANYANWU_FLASH_PILOT='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
               XDG_CACHE_HOME=str(work/'cache'),HF_HOME=str(work/'cache/hf'),
               UV_CACHE_DIR=str(work/'uv-cache'),PIP_CONFIG_FILE='/dev/null',PIP_DISABLE_PIP_VERSION_CHECK='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',
               TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    phase='initialising'
    peak_rss=0
    progress_value={}
    try:
        def bounded(command,name):
            nonlocal phase,peak_rss,progress_value
            phase=name
            publish({'phase':name,'optimizer_updates':progress_value.get('optimizer_updates',0),
                     'elapsed_seconds':round(time.monotonic()-started),'cloud_only':True})
            log=work/(name+'.log')
            with log.open('w') as handle:
                process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                last_progress=None
                try:
                    while process.poll() is None:
                        free=shutil.disk_usage('/content').free
                        # The worker performs inference/training in its own process, not a daemon.
                        statm=Path(f'/proc/{process.pid}/statm')
                        rss=int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0
                        peak_rss=max(peak_rss,rss)
                        if time.monotonic()-started>(15 if repair else 30)*60 or initial_free-free>20*1024**3 or free<5*1024**3 or rss>9*1024**3 or log.stat().st_size>512*1024:
                            raise RuntimeError('Cloud training time/disk/memory/log cap reached')
                        progress=work/'progress.json'
                        if name=='training_worker' and progress.exists():
                            value=progress.read_text()
                            if value!=last_progress:
                                last_progress=value
                                progress_value=json.loads(value)
                                publish(progress_value)
                        time.sleep(2)
                    if process.returncode:
                        raise RuntimeError(f'Cloud {name} exited {process.returncode}; inspect bounded log')
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
        python=work/'venv/bin/python'
        if not repair:
            bounded([str(uv),'venv','--python',str(python_base),'--seed',str(work/'venv')],'isolated_tts_python')
            bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','torch==2.8.0',
                     '--index-url','https://download.pytorch.org/whl/cpu'],'cpu_tts_torch')
            bounded([str(python),'-m','pip','install','--quiet','--no-cache-dir','transformers==4.51.3',
                     'huggingface_hub==0.36.0','onnxruntime==1.23.2','numpy==1.26.4',
                     'safetensors==0.5.3','--index-url','https://pypi.org/simple'],'cpu_tts_dependencies')
        elif not python.is_file():
            raise RuntimeError('Cached training Python unavailable; no reinstall in the repair attempt')
        bounded([str(python),str(work/'worker.py')],'training_worker')
        result=json.loads(result_path.read_text())
        result.update(peak_worker_rss_bytes=peak_rss,cloud_disk_increase_bytes=initial_free-shutil.disk_usage('/content').free,
                      total_setup_training_seconds=round(time.monotonic()-started,2))
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (work/'attempt.json').write_text(json.dumps({'status':'complete','optimizer_updates':result['optimizer_updates']}))
        return result
    except Exception as error:
        progress=work/'progress.json'
        if progress.exists():
            progress_value=json.loads(progress.read_text())
        result={**progress_value,'phase':'flash_pilot_attempt_stopped','supervisor_phase':phase,
                'error_type':type(error).__name__,'supervisor_error':str(error)[:350],
                'peak_worker_rss_bytes':peak_rss,'elapsed_seconds':round(time.monotonic()-started,2)}
        log=work/(phase+'.log')
        if log.exists():
            result['log_tail']=log.read_text(errors='replace')[-2400:]
        (work/'attempt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        return result
