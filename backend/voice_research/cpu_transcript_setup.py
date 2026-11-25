"""Bounded Colab CPU setup for source transcript drafts; not TTS training."""
from pathlib import Path
import ctypes.util
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time


def run(plan, worker_source, publish):
    if platform.system()!="Linux" or not Path("/content").is_dir() or importlib.util.find_spec("google.colab") is None:
        raise RuntimeError("Use the existing hosted Colab CPU runtime; no laptop models or installs.")
    if not ctypes.util.find_library("sndfile"):
        raise RuntimeError("Cloud libsndfile is unavailable; no system-package change attempted.")
    if shutil.disk_usage("/content").free<15*1024**3:
        raise RuntimeError("Fifteen GiB free cloud disk required.")
    work=Path("/content/anyanwu-cpu-transcripts-v1")
    work.mkdir(exist_ok=True)
    if (work/"draft-transcripts.json").exists():
        return json.loads((work/"draft-transcripts.json").read_text())
    if (work/"attempt.json").exists():
        raise RuntimeError("A CPU transcription attempt already exists; inspect it before explicitly resuming.")
    started=time.monotonic()
    initial_free=shutil.disk_usage("/content").free
    (work/"plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n")
    (work/"worker.py").write_text(worker_source,encoding="utf-8")
    (work/"attempt.json").write_text(json.dumps({"status":"started","model":"omniASR_CTC_300M_v2","minutes_cap":20,"disk_increase_gib_cap":8,"model_cache_gib_cap":2,"rss_gib_cap":6,"input_clips":5,"trained":False}))
    env={k:v for k,v in os.environ.items() if k not in {"HF_TOKEN","HUGGING_FACE_HUB_TOKEN","OPENAI_API_KEY","PYTHONPATH","PYTHONHOME","VIRTUAL_ENV"}}
    env.update(ANYANWU_CLOUD_DRAFTS="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
               XDG_CACHE_HOME=str(work/"cache"),TORCH_HOME=str(work/"cache/torch"),HF_HOME=str(work/"cache/hf"),
               UV_PYTHON_INSTALL_DIR=str(work/"python"),UV_CACHE_DIR=str(work/"uv-cache"),PIP_CONFIG_FILE="/dev/null",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1",HF_HUB_DISABLE_TELEMETRY="1",PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
    phase="initialising"
    try:
        def bounded(command, name):
            nonlocal phase
            phase=name
            publish({"phase":name,"trained":False,"optimizer_updates":0,"elapsed_seconds":round(time.monotonic()-started)})
            log=work/(name+".log")
            with log.open("w") as handle:
                process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                last_progress=None
                try:
                    while process.poll() is None:
                        free=shutil.disk_usage("/content").free
                        statm=Path(f"/proc/{process.pid}/statm")
                        rss=int(statm.read_text().split()[1])*os.sysconf("SC_PAGE_SIZE") if statm.exists() else 0
                        if time.monotonic()-started>20*60 or initial_free-free>8*1024**3 or free<5*1024**3 or rss>6*1024**3 or log.stat().st_size>256*1024:
                            raise RuntimeError("CPU draft time/disk/memory/log cap reached.")
                        cache=work/"cache"
                        if cache.exists() and sum(p.stat().st_size for p in cache.rglob("*.pt") if p.is_file())>2*1024**3:
                            raise RuntimeError("Two GiB speech-model cache cap reached.")
                        progress=work/"progress.json"
                        if name=="transcribing" and progress.exists():
                            value=progress.read_text()
                            if value!=last_progress:
                                last_progress=value
                                publish({**json.loads(value),"trained":False,"optimizer_updates":0})
                        time.sleep(2)
                    if process.returncode:
                        raise RuntimeError(f"Cloud phase {name} failed with exit code {process.returncode}; inspect its bounded log.")
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
        bootstrap=work/"bootstrap"
        bounded([sys.executable,"-m","pip","install","--quiet","--disable-pip-version-check","--no-cache-dir","--only-binary=:all:","--target",str(bootstrap),"uv==0.9.13","--index-url","https://pypi.org/simple"],"bootstrap_uv")
        uv=bootstrap/"bin/uv"
        bounded([str(uv),"venv","--python","3.11.14","--managed-python","--seed",str(work/"venv")],"python_311")
        python=work/"venv/bin/python"
        bounded([str(python),"-m","pip","install","--quiet","--disable-pip-version-check","--no-cache-dir","torch==2.8.0","torchaudio==2.8.0","--index-url","https://download.pytorch.org/whl/cpu"],"cpu_torch")
        bounded([str(python),"-m","pip","install","--quiet","--disable-pip-version-check","--no-cache-dir","numpy==1.26.4","pyarrow==20.0.0","fairseq2==0.6.0","omnilingual-asr==0.2.0","--extra-index-url","https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu"],"cpu_asr")
        bounded([str(python),str(work/"worker.py")],"transcribing")
        result=json.loads((work/"draft-transcripts.json").read_text())
        result["cloud_disk_increase_bytes"]=initial_free-shutil.disk_usage("/content").free
        (work/"attempt.json").write_text(json.dumps({"status":"complete","trained":False,"transcribed_clips":result["transcribed_clips"],"elapsed_seconds":round(time.monotonic()-started)}))
        return result
    except Exception as error:
        result={"phase":"cpu_transcript_drafts_stopped","failed_phase":phase,"trained":False,"optimizer_updates":0,
                "error_type":type(error).__name__,"error":str(error)[:220] if isinstance(error,RuntimeError) else "Inspect the cloud attempt log"}
        (work/"attempt.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        return result
