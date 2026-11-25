"""Supervise one declared six-clip GPU inference run, without training."""
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


def run_comparison(worker,generator,publish):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Hosted Colab only; no laptop models or audio')
    root=Path('/content/anyanwu-waxal-female5-comparison-v1')
    train=Path('/content/anyanwu-waxal-female5-lora-v1')
    if not (root/'setup-result.json').is_file():raise RuntimeError('Verified acoustic setup is missing')
    result=root/'generation-result.json'
    if result.exists():
        value=json.loads(result.read_text())
        if len(value['clips'])!=6:raise RuntimeError('Incomplete saved comparison')
        for clip in value['clips']:
            path=root/clip['audio_filename']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=clip['audio_sha256']:raise RuntimeError('Saved audio changed')
        return value
    if (root/'generation-attempt.json').exists():raise RuntimeError('Prior synthesis attempt exists; inspect before a declared technical repair')
    if shutil.disk_usage('/content').free<8*1024**3:raise RuntimeError('Eight GiB cloud disk reserve required')
    (root/'comparison_worker.py').write_text(worker)
    (root/'flash_generation.py').write_text(generator)
    (root/'generation-attempt.json').write_text(json.dumps({'phase':'started','minutes_cap':15,'optimizer_updates':0}))
    excluded={'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','OPENAI_API_KEY','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}
    env={k:v for k,v in os.environ.items() if k not in excluded}
    env.update(ANYANWU_COMPARE_VOICE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
      HF_HOME=str(train/'cache/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1')
    log=root/'generation.log';started=time.monotonic();initial=shutil.disk_usage('/content').free;peak=0;previous=None
    try:
        with log.open('w') as handle:
            process=subprocess.Popen([str(train/'venv/bin/python'),str(root/'comparison_worker.py')],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while process.poll() is None:
                    statm=Path(f'/proc/{process.pid}/statm');rss=int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0;peak=max(peak,rss)
                    free=shutil.disk_usage('/content').free
                    if time.monotonic()-started>900 or rss>9*1024**3 or log.stat().st_size>512*1024 or free<5*1024**3 or initial-free>64*1024**2:
                        raise RuntimeError('Generation time/memory/log/disk cap reached')
                    progress=root/'generation-progress.json'
                    if progress.exists():
                        value=json.loads(progress.read_text());brief={k:v for k,v in value.items() if k!='clips'}
                        brief['completed_wavs']=len(value['clips'])
                        signature=(brief.get('phase'),brief.get('variant'),brief.get('text_id'),brief['completed_wavs'])
                        if signature!=previous:publish(brief);previous=signature
                    time.sleep(2)
                if process.returncode:raise RuntimeError(f'Comparison worker exited {process.returncode}')
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
        value=json.loads(result.read_text());value.update(supervisor_seconds=round(time.monotonic()-started,2),peak_worker_rss_bytes=peak)
        if value['completed_wavs']!=6 or value['optimizer_updates_this_stage']!=0:raise RuntimeError('Output count/inference-only check failed')
        result.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');return value
    except Exception as error:
        value={'phase':'comparison_generation_stopped','error':str(error)[:400],'peak_worker_rss_bytes':peak,'optimizer_updates_this_stage':0}
        progress=root/'generation-progress.json'
        if progress.exists():value['worker_progress']=json.loads(progress.read_text())
        value['log_tail']=log.read_text(errors='replace')[-2600:] if log.exists() else ''
        (root/'generation-attempt.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');return value
