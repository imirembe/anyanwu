"""Cold-cloud restore and bounded inference for the fixed v1/v2 comparison."""
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

COMPARISON_ROOT = Path('/content/anyanwu-waxal-female5-expanded-comparison-v2')


def comparison_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_comparison_archive(info, selected):
    path = Path(info['path'])
    if path.stat().st_size != info['bytes'] or path.stat().st_size > 32 * 1024**2 or comparison_sha(path) != info['sha256']:
        raise RuntimeError('Pinned adapter archive integrity failure')
    with zipfile.ZipFile(path) as package:
        manifest = json.loads(package.read('manifest.json'))
        files = manifest['files']
        if len(files) > 100 or sum(item['bytes'] for item in files.values()) > 32 * 1024**2:
            raise RuntimeError('Adapter archive manifest cap')
        result = {}
        for name in selected:
            meta = files[name]
            if meta['bytes'] > 2 * 1024**2:
                raise RuntimeError('Selected adapter archive member cap')
            body = package.read(name)
            if len(body) != meta['bytes'] or hashlib.sha256(body).hexdigest() != meta['sha256']:
                raise RuntimeError('Selected adapter archive member changed')
            result[name] = body
        return result


def restore_expanded_comparison(plan, root=COMPARISON_ROOT):
    assets = root / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    prompt_id = plan['prompt_id']
    required = ['plan.json', 'training-result.json', 'adapter.safetensors']
    previous = read_comparison_archive(plan['archives']['previous'], required + [
        'data-result.json', 'speech-targets.json', f'data/{prompt_id}.mp3', f'data/{prompt_id}.wav'])
    expanded = read_comparison_archive(plan['archives']['expanded'], required)
    old_plan = json.loads(previous['plan.json'])
    new_plan = json.loads(expanded['plan.json'])
    old_result = json.loads(previous['training-result.json'])
    new_result = json.loads(expanded['training-result.json'])
    excluded = {item['id'] for item in plan['texts']}
    if len(excluded) != 4 or excluded & ({c['id'] for c in old_plan['clips']} | {c['id'] for c in new_plan['cases']}):
        raise RuntimeError('Comparison texts overlap adapter training or validation')
    if old_result['optimizer_updates'] != 60 or new_result['optimizer_updates'] != 60 or new_result['cumulative_adapter_updates'] != 120:
        raise RuntimeError('Unexpected source adapter training counts')
    if new_result['starting_adapter_sha256'] != plan['adapter_hashes']['base']:
        raise RuntimeError('Expanded adapter has a different starting checkpoint')
    for variant, package in [('base', previous), ('adapter', expanded)]:
        if hashlib.sha256(package['adapter.safetensors']).hexdigest() != plan['adapter_hashes'][variant]:
            raise RuntimeError('Comparison adapter checksum mismatch')
    data = json.loads(previous['data-result.json'])
    source = next(c for c in data['clips'] if c['id'] == prompt_id)
    target = next(c for c in json.loads(previous['speech-targets.json']) if c['id'] == prompt_id)
    if (source['speaker_id'], source['gender'], source['adapter_split']) != ('5', 'Female', 'train'):
        raise RuntimeError('Unexpected common female reference')
    for suffix, key in [('.wav', 'audio_sha256'), ('.mp3', 'mp3_sha256')]:
        if hashlib.sha256(previous[f'data/{prompt_id}{suffix}']).hexdigest() != source[key]:
            raise RuntimeError('Common reference audio changed')
    outputs = {
        'plan.json': previous['plan.json'],
        'previous-adapter.safetensors': previous['adapter.safetensors'],
        'adapter.safetensors': expanded['adapter.safetensors'],
        'speech-targets.json': json.dumps([target], ensure_ascii=False).encode(),
        'data-result.json': json.dumps({'clips': [{**source, 'cloud_audio_path': str(assets / 'data' / (prompt_id + '.wav'))}]}, ensure_ascii=False).encode(),
        f'data/{prompt_id}.wav': previous[f'data/{prompt_id}.wav'],
        f'data/{prompt_id}.mp3': previous[f'data/{prompt_id}.mp3'],
    }
    for name, body in outputs.items():
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != body:
            raise RuntimeError('Different restored comparison input; no overwrite')
        if not path.exists():
            path.write_bytes(body)
    return {'phase': 'both_adapter_inputs_verified', 'adapter_hashes': plan['adapter_hashes'],
            'prompt_id': prompt_id, 'reserved_texts': sorted(excluded), 'optimizer_updates_this_stage': 0}


def run_expanded_comparison(plan, worker, generator, runtime, adapter_source, publish):
    if platform.system() != 'Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Hosted Colab only; no laptop model or audio installation')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if gpu.returncode or len(gpu.stdout.strip().splitlines()) != 1 or int(gpu.stdout.strip().rsplit(',', 1)[1]) < 14000:
        raise RuntimeError('One hosted GPU with at least14000MiB required; no CPU synthesis fallback or purchase')
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Connect Drive before comparison setup')
    root = COMPARISON_ROOT
    root.mkdir(exist_ok=True)
    stored_plan = root / 'plan.json'
    if stored_plan.exists() and json.loads(stored_plan.read_text()) != plan:
        raise RuntimeError('Different comparison plan exists')
    result_path = root / 'generation-result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get('completed_wavs') != 8 or result.get('optimizer_updates_this_stage') != 0:
            raise RuntimeError('Invalid completed comparison')
        for clip in result['clips']:
            if comparison_sha(root / clip['audio_filename']) != clip['audio_sha256']:
                raise RuntimeError('Existing comparison audio changed')
        return result
    if (root / 'attempt.json').exists():
        raise RuntimeError('An attempt already exists; inspect before any declared technical repair')
    if shutil.disk_usage('/content').free < 30 * 1024**3:
        raise RuntimeError('Thirty GiB free cloud disk required')
    if sum(f['size'] for f in plan['model_files']) > 4 * 1024**3:
        raise RuntimeError('Selected model download cap')
    stored_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    (root / 'attempt.json').write_text(json.dumps({'phase': 'setup_started', 'optimizer_updates_this_stage': 0}))
    assets = root / 'assets'
    started = time.monotonic()
    initial_free = shutil.disk_usage('/content').free
    excluded = {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'OPENAI_API_KEY', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'}
    env = {k: v for k, v in os.environ.items() if k not in excluded}
    env.update(ANYANWU_COMPARE_VOICE='1', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               HF_HOME=str(assets / 'cache/hf'), XDG_CACHE_HOME=str(root / 'cache'), UV_CACHE_DIR=str(root / 'uv-cache'),
               UV_PYTHON_INSTALL_DIR=str(root / 'python'), PIP_CONFIG_FILE='/dev/null', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
               HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    phase = 'restore_inputs'
    peak = 0
    latest = {}

    def bounded(command, name, offline=False):
        nonlocal phase, peak, latest
        phase = name
        publish({'phase': name, 'optimizer_updates_this_stage': 0})
        log = root / (name + '.log')
        child_env = {**env, **({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'} if offline else {})}
        stage_start = time.monotonic()
        with log.open('w') as output:
            process = subprocess.Popen(command, env=child_env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            last = None
            try:
                while process.poll() is None:
                    free = shutil.disk_usage('/content').free
                    statm = Path(f'/proc/{process.pid}/statm')
                    rss = int(statm.read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE') if statm.exists() else 0
                    peak = max(peak, rss)
                    if time.monotonic() - started > 1800 or (name == 'comparison_worker' and time.monotonic() - stage_start > 900) or initial_free - free > 20 * 1024**3 or free < 5 * 1024**3 or rss > 9 * 1024**3 or log.stat().st_size > 512 * 1024:
                        raise RuntimeError('Comparison time/disk/RSS/log resource cap')
                    progress = root / 'generation-progress.json'
                    if name == 'comparison_worker' and progress.exists():
                        value = progress.read_text()
                        if value != last:
                            last = value
                            latest = json.loads(value)
                            publish({**{k: v for k, v in latest.items() if k != 'clips'}, 'completed_wavs': len(latest['clips'])})
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError(f'{name} exited {process.returncode}; inspect bounded log')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    try:
        publish(restore_expanded_comparison(plan))
        for name, source in [('comparison_worker.py', worker), ('flash_generation.py', generator), ('cv3_runtime.py', runtime)]:
            (root / name).write_text(source)
        (assets / 'flash_adapter.py').write_text(adapter_source)
        constraints = '\n'.join(['torch==2.8.0', 'transformers==4.51.3', 'huggingface_hub==0.36.0', 'numpy==1.26.4', 'onnxruntime==1.23.2', 'safetensors==0.5.3']) + '\n'
        (root / 'constraints.txt').write_text(constraints)
        bounded([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', '--only-binary=:all:', '--target', str(root / 'bootstrap'), 'uv==0.9.13', '--index-url', 'https://pypi.org/simple'], 'comparison_bootstrap')
        bounded([str(root / 'bootstrap/bin/uv'), 'venv', '--python', '3.11.14', '--managed-python', '--seed', str(assets / 'venv')], 'comparison_python311')
        python = str(assets / 'venv/bin/python')
        bounded([python, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', 'torch==2.8.0', 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu128'], 'comparison_torch')
        bounded([python, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', '--constraint', str(root / 'constraints.txt'),
                 'transformers==4.51.3', 'huggingface_hub==0.36.0', 'numpy==1.26.4', 'onnxruntime==1.23.2', 'safetensors==0.5.3',
                 'x-transformers==2.11.24', 'einops==0.8.1', 'einx==0.3.0', 'omegaconf==2.3.0', 'scipy==1.11.4', 'librosa==0.10.2', '--index-url', 'https://pypi.org/simple'], 'comparison_dependencies')
        download_source = '''from pathlib import Path
import hashlib,json,shutil
from huggingface_hub import hf_hub_download
root=Path('/content/anyanwu-waxal-female5-expanded-comparison-v2');assets=root/'assets'
plan=json.loads((root/'plan.json').read_text());acoustic={}
for item in plan['model_files']:
 p=Path(hf_hub_download(repo_id=item['repo'],revision=item['revision'],filename=item['name'],token=False,cache_dir=str(assets/'cache/hf/hub')))
 if p.stat().st_size!=item['size']:raise RuntimeError('Selected model size changed')
 digest=hashlib.sha256() if 'sha256' in item else hashlib.sha1()
 if 'git_blob_sha1' in item:digest.update(('blob '+str(p.stat().st_size)+'\\0').encode())
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
 if digest.hexdigest()!=item.get('sha256',item.get('git_blob_sha1')):raise RuntimeError('Selected model hash changed')
 if item['name'] in {'flow.pt','hift.pt','campplus.onnx'}:acoustic[item['name']]=str(p)
 if item['name'].split('/')[-1] in {'vocab.json','merges.txt','tokenizer_config.json'}:
  target=assets/'text-tokenizer'/item['name'].split('/')[-1];target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
(root/'acoustic-files.json').write_text(json.dumps(acoustic,indent=2)+'\\n')
import cv3_runtime
print('Selected pinned files verified; acoustic runtime imports successfully.')
'''
        (root / 'download_models.py').write_text(download_source)
        bounded([python, str(root / 'download_models.py')], 'comparison_selected_models')
        (root / 'setup-result.json').write_text(json.dumps({'phase': 'comparison_setup_ready', 'selected_model_bytes': sum(f['size'] for f in plan['model_files']), 'optimizer_updates_this_stage': 0}))
        bounded([python, str(root / 'comparison_worker.py')], 'comparison_worker', offline=True)
        result = json.loads(result_path.read_text())
        if result['completed_wavs'] != 8 or result['optimizer_updates_this_stage'] != 0:
            raise RuntimeError('Comparison output count or inference-only guard')
        result.update(peak_worker_rss_bytes=peak, total_setup_generation_seconds=round(time.monotonic() - started, 2), cloud_disk_increase_bytes=initial_free - shutil.disk_usage('/content').free)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        (root / 'attempt.json').write_text(json.dumps({'phase': 'complete', 'optimizer_updates_this_stage': 0, 'completed_wavs': 8}))
        return result
    except Exception as error:
        failure = {'phase': 'expanded_comparison_stopped', 'failed_phase': phase, 'error': str(error)[:450], 'optimizer_updates_this_stage': 0, 'peak_worker_rss_bytes': peak}
        log = root / (phase + '.log')
        if log.exists():
            failure['log_tail'] = log.read_text(errors='replace')[-2600:]
        if (root / 'generation-progress.json').exists():
            failure['worker_progress'] = json.loads((root / 'generation-progress.json').read_text())
        (root / 'attempt.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2) + '\n')
        return failure
