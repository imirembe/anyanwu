"""Notebook setup cell. Explicit Colab/GPU guards precede all package downloads."""
from pathlib import Path
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time

if (platform.system() != "Linux" or not Path("/content").is_dir()
        or importlib.util.find_spec("google.colab") is None):
    raise RuntimeError("Use a Google Colab hosted GPU runtime. Do not connect a local laptop runtime.")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                     check=True, capture_output=True, text=True, timeout=10).stdout.strip()
if len(gpu.splitlines()) != 1 or int(gpu.rsplit(",", 1)[1]) < 14000:
    raise RuntimeError("Use one NVIDIA GPU with at least 14,000 MiB VRAM. No CPU fallback or paid upgrade is started.")
if sys.version_info[:2] < (3, 10):
    raise RuntimeError("The notebook host needs Python 3.10 or newer; speech dependencies use their own Python 3.11.")
if shutil.disk_usage("/content").free < 30 * 1024**3:
    raise RuntimeError("At least 30 GiB of free cloud disk is required.")
WORK = Path("/content/anyanwu-voice-audition-v1")
WORK.mkdir(exist_ok=True)
if (WORK / "outputs").exists():
    raise RuntimeError("This run already has outputs. Listen to them; this notebook will not overwrite or retry them.")
DEADLINE = time.monotonic() + 30 * 60
INITIAL_FREE = shutil.disk_usage("/content").free
ENV = {k: v for k, v in os.environ.items() if k not in {
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
ENV.update(ANYANWU_CLOUD_AUDITION="1", HF_HOME=str(WORK / "cache"),
           UV_PYTHON_INSTALL_DIR=str(WORK / "python"), UV_CACHE_DIR=str(WORK / "uv-cache"),
           HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
           PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")


def bounded(command):
    """Stop child work after 30 minutes total or a 20 GiB cloud-disk increase."""
    import signal
    process = subprocess.Popen(command, env=ENV, start_new_session=True)
    try:
        while process.poll() is None:
            if (time.monotonic() >= DEADLINE
                    or INITIAL_FREE - shutil.disk_usage("/content").free > 20 * 1024**3
                    or shutil.disk_usage("/content").free < 5 * 1024**3):
                raise RuntimeError("Cloud time/disk limit reached. Stop here; no automatic retry or paid fallback.")
            time.sleep(2)
        if process.returncode:
            raise RuntimeError(f"Cloud command failed with exit code {process.returncode}; no automatic retry.")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


SETUP_REVISION = 2
PYTHON = WORK / "venv-py311/bin/python"
if not PYTHON.exists():
    # The host kernel can be newer than the speech libraries. Keep their tested
    # interpreter separate, and put every bootstrap/cache file on the cloud VM.
    bootstrap = WORK / "bootstrap"
    uv = bootstrap / "bin/uv"
    if not uv.exists():
        bounded([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:",
                 "--target", str(bootstrap), "uv==0.9.13", "--index-url", "https://pypi.org/simple"])
    if not uv.is_file():
        raise RuntimeError("The cloud Python bootstrap did not provide uv; no speech model was downloaded.")
    bounded([str(uv), "venv", "--python", "3.11.14", "--managed-python", "--seed", str(WORK / "venv-py311")])
version = subprocess.run([str(PYTHON), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
                         capture_output=True, text=True, check=True, env=ENV, timeout=10).stdout.strip()
if version != "3.11.14":
    raise RuntimeError("The isolated speech environment has the wrong Python version; no model download started.")
bounded([str(PYTHON), "-m", "pip", "install", "--no-cache-dir", "chatterbox-tts==0.1.7",
         "resemble-perth==1.0.1", "setuptools==80.9.0"])
bounded([str(PYTHON), "-c", "import perth; assert callable(perth.PerthImplicitWatermarker), 'Perth watermark import is unavailable'"])
with (WORK / "environment.txt").open("w") as handle:
    subprocess.run([str(PYTHON), "-m", "pip", "freeze"], check=True, stdout=handle, env=ENV, timeout=30)
print("Cloud environment ready:", gpu, "· speech Python", version, "· setup revision", SETUP_REVISION)
