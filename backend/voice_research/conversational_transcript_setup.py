"""Isolated, bounded CPU ASR preparation; preserves all original source recordings."""
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
import wave


def run_conversational_asr(plan, worker_source, publish):
    if (platform.system() != 'Linux' or not Path('/content').is_dir()
            or importlib.util.find_spec('google.colab') is None):
        raise RuntimeError('Hosted Colab CPU only; no laptop audio, model or dependency install')
    root = Path('/content/anyanwu-conversational-transcripts-v1')
    reference = Path('/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research/references/conversational-igbo-v1')
    backup = reference/'transcription-v1'
    if not reference.is_dir():
        raise RuntimeError('Existing research Drive connection required before model work')
    if hashlib.sha256(worker_source.encode()).hexdigest() != plan['worker_sha256']:
        raise RuntimeError('Worker source mismatch')
    root.mkdir(exist_ok=True)
    result_path = root/'draft-transcripts.json'

    def write_once(path, raw):
        if path.is_symlink():
            raise RuntimeError('Unexpected symlink')
        if path.exists():
            if path.read_bytes() != raw:
                raise RuntimeError('Different existing file preserved: '+path.name)
        else:
            with path.open('xb') as handle:
                handle.write(raw)
        if path.read_bytes() != raw:
            raise RuntimeError('Saved file verification failed')

    if not result_path.exists() and (backup/'draft-transcripts.json').exists():
        if (backup/'draft-transcripts.json').stat().st_size > 128*1024:
            raise RuntimeError('Saved draft size cap')
        saved = json.loads((backup/'draft-transcripts.json').read_text())
        if saved.get('plan_id') != plan['id'] or [r['id'] for r in saved.get('records', [])] != [c['id'] for c in plan['clips']]:
            raise RuntimeError('Saved drafts selection mismatch')
        write_once(result_path, (backup/'draft-transcripts.json').read_bytes())
    if not result_path.exists():
        if (root/'attempt.json').exists():
            raise RuntimeError('Prior transcription attempt exists; inspect before an explicit repair')
        if shutil.disk_usage('/content').free < 15*1024**3:
            raise RuntimeError('15 GiB free cloud disk required before isolated setup')
        started = time.monotonic()
        initial_free = shutil.disk_usage('/content').free
        env = {k: v for k, v in os.environ.items() if k not in {
            'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'}}
        env.update(ANYANWU_CONVERSATIONAL_ASR='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   XDG_CACHE_HOME=str(root/'cache'), TORCH_HOME=str(root/'cache/torch'), HF_HOME=str(root/'cache/hf'),
                   UV_PYTHON_INSTALL_DIR=str(root/'python'), UV_CACHE_DIR=str(root/'uv-cache'), PIP_CONFIG_FILE='/dev/null',
                   HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        write_once(root/'attempt.json', b'{"status":"started","optimizer_updates":0}\n')
        stage = 'input_verification'
        try:
            for selected in plan['clips']:
                source = reference/(selected['id']+'.wav')
                if (source.is_symlink() or source.stat().st_size > 2*1024**2
                        or hashlib.sha256(source.read_bytes()).hexdigest() != selected['listening_sha256']):
                    raise RuntimeError('Selected whole source integrity failure')
                dest = root/(selected['id']+'.analysis.wav')
                subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(source),
                                '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(dest)], check=True, timeout=45,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                with wave.open(str(dest)) as wav:
                    if ((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000)
                            or abs(wav.getnframes()/16000-selected['seconds']) > 1/16000):
                        raise RuntimeError('Whole-response analysis copy timing changed')
                selected['analysis_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
            write_once(root/'plan.json', (json.dumps(plan, ensure_ascii=False, indent=2)+'\n').encode())
            write_once(root/'worker.py', worker_source.encode())

            def bounded(command, phase):
                nonlocal stage
                stage = phase
                log = root/(phase+'.log')
                publish({'phase': 'conversational_asr_'+phase, 'optimizer_updates': 0})
                with log.open('w') as handle:
                    process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    heartbeat = 0
                    try:
                        while process.poll() is None:
                            free = shutil.disk_usage('/content').free
                            statm = Path(f'/proc/{process.pid}/statm')
                            rss = int(statm.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0
                            cache = root/'cache'
                            model_bytes = sum(p.stat().st_size for p in cache.rglob('*.pt') if p.is_file()) if cache.exists() else 0
                            if (time.monotonic()-started > 1200 or initial_free-free > 8*1024**3 or free < 5*1024**3
                                    or rss > 6*1024**3 or log.stat().st_size > 256*1024 or model_bytes > 2*1024**3):
                                raise RuntimeError('CPU transcription time/disk/model-cache/memory/log budget reached')
                            if time.monotonic()-heartbeat > 25:
                                heartbeat = time.monotonic()
                                publish({'phase': 'conversational_asr_'+phase, 'elapsed_seconds': round(time.monotonic()-started),
                                         'cloud_disk_increase_bytes': initial_free-free, 'rss_bytes': rss, 'optimizer_updates': 0})
                            time.sleep(2)
                        if process.returncode:
                            raise RuntimeError(f'Phase {phase} failed ({process.returncode}); inspect bounded cloud log')
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            bootstrap = root/'bootstrap'
            bounded([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--no-cache-dir',
                     '--only-binary=:all:', '--target', str(bootstrap), 'uv==0.9.13', '--index-url', 'https://pypi.org/simple'], 'bootstrap')
            uv = bootstrap/'bin/uv'
            bounded([str(uv), 'venv', '--python', '3.11.14', '--managed-python', '--seed', str(root/'venv')], 'python311')
            python = root/'venv/bin/python'
            bounded([str(python), '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--no-cache-dir',
                     'torch==2.8.0', 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cpu'], 'torch')
            bounded([str(python), '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--no-cache-dir',
                     'numpy==1.26.4', 'pyarrow==20.0.0', 'fairseq2==0.6.0', 'omnilingual-asr==0.2.0',
                     '--extra-index-url', 'https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu'], 'packages')
            bounded([str(python), str(root/'worker.py')], 'transcribing')
            (root/'attempt.json').write_text(json.dumps({'status': 'complete', 'elapsed_seconds': round(time.monotonic()-started),
                                                        'cloud_disk_increase_bytes': initial_free-shutil.disk_usage('/content').free,
                                                        'optimizer_updates': 0})+'\n')
        except Exception as error:
            report = {'phase': 'conversational_asr_stopped', 'stage': stage, 'error': str(error)[:350], 'optimizer_updates': 0}
            (root/'attempt.json').write_text(json.dumps(report, indent=2)+'\n')
            return report
    if result_path.stat().st_size > 128*1024:
        raise RuntimeError('Draft output size cap')
    result = json.loads(result_path.read_text())
    if (result.get('plan_id') != plan['id'] or result.get('transcribed_clips') != 2
            or [r['id'] for r in result['records']] != [c['id'] for c in plan['clips']]):
        raise RuntimeError('Draft output selection/count mismatch')
    backup.mkdir(exist_ok=True)
    for filename in ('draft-transcripts.json', 'plan.json', 'attempt.json', 'worker.py'):
        if (root/filename).exists():
            write_once(backup/filename, (root/filename).read_bytes())
    return {**result, 'drive_backup': str(backup), 'drive_backup_verified': True,
            'draft_file_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(), 'laptop_media_bytes': 0}
