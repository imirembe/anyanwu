"""One declared cached cloud attempt after identifying the PyArrow ABI error."""
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


def resume(expected_worker_sha256,publish):
    if platform.system()!="Linux" or not Path("/content").is_dir() or importlib.util.find_spec("google.colab") is None:
        raise RuntimeError("Hosted Colab CPU only; no laptop model work.")
    work=Path("/content/anyanwu-cpu-transcripts-v1")
    result_file=work/"draft-transcripts.json"
    if result_file.exists():
        return json.loads(result_file.read_text())
    receipt=work/"resume-pyarrow20.json"
    if receipt.exists():
        raise RuntimeError("This declared cached attempt already exists; no automatic retry.")
    if hashlib.sha256((work/"worker.py").read_bytes()).hexdigest()!=expected_worker_sha256:
        raise RuntimeError("Cached transcript worker does not match the reviewed source.")
    prior=json.loads((work/"attempt.json").read_text())
    if prior.get("phase")!="cpu_transcript_drafts_stopped" or prior.get("failed_phase")!="transcribing":
        raise RuntimeError("The prior attempt does not match this repair.")
    if not (work/"attempt-1.json").exists():
        (work/"attempt-1.json").write_text(json.dumps(prior,indent=2)+"\n")
    started=time.monotonic()
    env={k:v for k,v in os.environ.items() if k not in {"HF_TOKEN","HUGGING_FACE_HUB_TOKEN","OPENAI_API_KEY","PYTHONPATH","PYTHONHOME","VIRTUAL_ENV"}}
    env.update(ANYANWU_CLOUD_DRAFTS="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",XDG_CACHE_HOME=str(work/"cache"),TORCH_HOME=str(work/"cache/torch"),HF_HOME=str(work/"cache/hf"),PIP_CONFIG_FILE="/dev/null",HF_HUB_DISABLE_IMPLICIT_TOKEN="1",HF_HUB_DISABLE_TELEMETRY="1",PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
    phase="fixing_arrow"
    receipt.write_text(json.dumps({"phase":"cached_attempt_started","time_cap_minutes":10,"total_work_gib_cap":8,"trained":False}))
    try:
        def bounded(command,name):
            nonlocal phase
            phase=name
            publish({"phase":name,"trained":False,"optimizer_updates":0})
            log=work/(name+".log")
            with log.open("w") as handle:
                process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                last=None
                last_heartbeat=0
                try:
                    while process.poll() is None:
                        free=shutil.disk_usage("/content").free
                        statm=Path(f"/proc/{process.pid}/statm")
                        rss=int(statm.read_text().split()[1])*os.sysconf("SC_PAGE_SIZE") if statm.exists() else 0
                        used=sum(p.stat().st_size for p in work.rglob('*') if p.is_file() and not p.is_symlink())
                        if time.monotonic()-started>600 or free<5*1024**3 or used>8*1024**3 or rss>6*1024**3 or log.stat().st_size>256*1024:
                            raise RuntimeError("Cached CPU attempt time/disk/memory/log cap reached.")
                        progress=work/"progress.json"
                        if progress.exists():
                            value=progress.read_text()
                            if value!=last:
                                last=value
                                publish({**json.loads(value),"trained":False,"optimizer_updates":0})
                        if time.monotonic()-last_heartbeat>=30:
                            last_heartbeat=time.monotonic()
                            publish({"phase":name,"elapsed_seconds":round(time.monotonic()-started),"cloud_work_gib":round(used/1024**3,2),"rss_gib":round(rss/1024**3,2),"trained":False})
                        time.sleep(2)
                    if process.returncode:
                        raise RuntimeError(f"Cached phase {name} failed with exit code {process.returncode}.")
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
        python=work/"venv/bin/python"
        bounded([str(python),"-m","pip","install","--quiet","--disable-pip-version-check","--no-cache-dir","numpy==1.26.4","pyarrow==20.0.0"],"pin_arrow20")
        bounded([str(python),str(work/"worker.py")],"transcribing_arrow20")
        result=json.loads(result_file.read_text())
        receipt.write_text(json.dumps({"phase":"cached_attempt_complete","trained":False,"elapsed_seconds":round(time.monotonic()-started),"transcribed_clips":result["transcribed_clips"]}))
        return result
    except Exception as error:
        result={"phase":"cpu_transcript_cached_attempt_stopped","failed_phase":phase,"error_type":type(error).__name__,"error":str(error)[:200] if isinstance(error,RuntimeError) else "Inspect cloud log","trained":False,"optimizer_updates":0}
        receipt.write_text(json.dumps(result,indent=2)+"\n")
        return result
