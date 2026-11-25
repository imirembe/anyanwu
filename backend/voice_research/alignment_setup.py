"""Bounded offline CPU caption alignment in the existing speech environment."""
from pathlib import Path
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import time


def run_alignment(plan,worker,publish):
    if platform.system()!="Linux" or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Hosted Colab only; no laptop model/audio access.')
    if shutil.disk_usage('/content').free<5*1024**3:
        raise RuntimeError('Five GiB free cloud disk required.')
    name=plan['id']
    if name not in ('anyanwu-caption-alignment-v1','anyanwu-caption-alignment-v2-context','anyanwu-expanded-caption-alignment-v1','anyanwu-expanded-caption-joint-alignment-v1','anyanwu-conversational-alignment-v1'):
        raise RuntimeError('Unexpected alignment experiment identifier.')
    root=Path('/content')/name
    summary=root/'alignment-summary.json'
    if summary.exists():
        import hashlib
        if json.loads((root/'plan.json').read_text())!=plan:
            raise RuntimeError('Existing alignment plan differs; no overwrite or implicit reuse.')
        report=json.loads(summary.read_text())
        for case in report['cases']:
            if case.get('cloud_audio_path'):
                path=Path(case['cloud_audio_path'])
                if path.parent!=root or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=case['candidate_audio_sha256']:
                    raise RuntimeError('Existing aligned candidate failed integrity; no overwrite.')
        return report
    if root.exists():
        raise RuntimeError('An earlier alignment attempt exists; inspect it before a declared repair.')
    environment=Path('/content/anyanwu-conversational-transcripts-v1') if name=='anyanwu-conversational-alignment-v1' else Path('/content/anyanwu-cpu-transcripts-v1')
    python=environment/'venv/bin/python'
    if not python.is_file():
        raise RuntimeError('The existing CPU speech environment is missing; no automatic install.')
    root.mkdir()
    (root/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (root/'worker.py').write_text(worker,encoding='utf-8')
    (root/'attempt.json').write_text(json.dumps({'status':'alignment_started','minutes_cap':10,'rss_gib_cap':6,'output_mib_cap':12,'network_requests_allowed':0,'trained':False}))
    cache=environment/'cache'
    env={k:v for k,v in os.environ.items() if k not in {'OPENAI_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}}
    env.update(ANYANWU_CLOUD_ALIGNMENT='1',ANYANWU_ALIGNMENT_WORK=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XDG_CACHE_HOME=str(cache),TORCH_HOME=str(cache/'torch'),HF_HOME=str(cache/'hf'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    started=time.monotonic()
    log=root/'alignment.log'
    publish({'phase':'loading_cached_alignment_model','trained':False,'optimizer_updates':0})
    process=None
    try:
        with log.open('w') as handle:
            process=subprocess.Popen([str(python),str(root/'worker.py')],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            last=None
            while process.poll() is None:
                statm=Path(f'/proc/{process.pid}/statm')
                rss=int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0
                used=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
                if time.monotonic()-started>600 or rss>6*1024**3 or used>12*1024**2 or log.stat().st_size>256*1024 or shutil.disk_usage('/content').free<5*1024**3:
                    raise RuntimeError('Offline alignment time/memory/output/log limit reached.')
                progress=root/'progress.json'
                if progress.exists():
                    value=progress.read_text()
                    if value!=last:
                        last=value
                        publish({**json.loads(value),'trained':False})
                time.sleep(2)
        if process.returncode:
            raise RuntimeError(f'Alignment worker exited {process.returncode}; inspect the bounded cloud log.')
        report=json.loads(summary.read_text())
        (root/'attempt.json').write_text(json.dumps({'status':'alignment_complete','elapsed_seconds':round(time.monotonic()-started,2),'candidate_cuts':report['candidate_cuts'],'trained':False}))
        return report
    except Exception as error:
        report={'phase':'conversational_asr_alignment_stopped' if name=='anyanwu-conversational-alignment-v1' else 'caption_alignment_stopped','error_type':type(error).__name__,'error':str(error)[:220] if isinstance(error,RuntimeError) else 'Inspect the cloud alignment log','trained':False,'optimizer_updates':0}
        (root/'attempt.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
