"""Prepare the registered eight-clip comparison without importing tensor libraries."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    lab = ROOT / '.build/VoiceResearch'
    folder = lab / 'auditions/waxal-female5-expanded-comparison-v2'
    folder.mkdir(parents=True, exist_ok=True)
    training = json.loads((lab / 'training/waxal-female5-expanded-lora-v2/plan.json').read_text())
    completed = json.loads((lab / 'training/waxal-female5-expanded-lora-v2/training-completion-summary.json').read_text())
    original = json.loads((lab / 'training/waxal-female5-lora-v1/plan.json').read_text())
    prior_folder = lab / 'auditions/waxal-female5-adapter-comparison-v1'
    prior = json.loads((prior_folder / 'plan.json').read_text())
    runtime = (prior_folder / 'cv3_inference_runtime.py').read_text()
    if hashlib.sha256(runtime.encode()).hexdigest() != prior['runtime_sha256']:
        raise RuntimeError('Pinned acoustic runtime changed')
    if completed['optimizer_updates'] != 60 or not completed['backup']['durable_backup_verified'] or not completed['baseline_preserved']:
        raise RuntimeError('Completed, backed-up expanded adapter required')
    comparison = training['fresh_synthesis_comparison']
    if comparison['outputs'] != 8 or comparison['seed'] != 17 or comparison['prompt_id'] != 'ibo_616':
        raise RuntimeError('Prospective comparison settings changed')
    texts = comparison['texts']
    if len(texts) != 4 or {t['id'] for t in texts} & {c['id'] for c in training['cases']}:
        raise RuntimeError('Comparison texts overlap training')
    drive_root = '/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research'
    base_archive = training['archives']['baseline']
    previous_path = base_archive.get('drive_archive_path') or base_archive['drive_archive']
    previous_path = previous_path.replace('/content/drive/MyDrive/Anyanwu Voice Research', drive_root, 1)
    model_files = []
    for model in original['models']:
        for item in model['files']:
            if item['rfilename'] == 'speech_tokenizer_v3.onnx':
                continue  # The common prompt tokens are already in the verified archive.
            model_files.append({**{k: v for k, v in item.items() if k != 'rfilename'}, 'name': item['rfilename'], 'repo': model['repo'], 'revision': model['revision']})
    for item in prior['acoustic_files']:
        model_files.append({**item, 'repo': prior['acoustic_repo'], 'revision': prior['acoustic_revision']})
    if sum(f['size'] for f in model_files) > 4 * 1024**3:
        raise RuntimeError('Selected model byte budget exceeded')
    plan = {
        'id': 'anyanwu-waxal-female5-expanded-comparison-v2',
        'texts': texts, 'variants': ['base', 'adapter'],
        'variant_labels': {'base': 'Previous voice · v1 (60 updates)', 'adapter': 'Expanded voice · v2 (120 cumulative updates)'},
        'adapter_hashes': {'base': completed['starting_adapter_sha256'], 'adapter': completed['adapter_sha256']},
        'adapter_sha256': completed['adapter_sha256'],
        'archives': {
            'previous': {'path': previous_path, 'bytes': base_archive['archive_bytes'], 'sha256': base_archive['archive_sha256']},
            'expanded': {'path': completed['backup']['drive_archive'], 'bytes': completed['backup']['archive_bytes'], 'sha256': completed['backup']['archive_sha256']},
        },
        'base_repo': prior['base_repo'], 'base_revision': prior['base_revision'], 'base_checkpoint_sha256': prior['base_checkpoint_sha256'],
        'acoustic_repo': prior['acoustic_repo'], 'acoustic_revision': prior['acoustic_revision'],
        'model_files': model_files, 'runtime_sha256': prior['runtime_sha256'],
        'prompt_id': 'ibo_616', 'seed': 17, 'outputs': 8, 'max_speech_tokens': 400,
        'policy': comparison['policy'], 'optimizer_updates_this_stage': 0,
        'limits': {'total_minutes': 30, 'generation_minutes': 15, 'cloud_disk_increase_gib': 20, 'cloud_disk_floor_gib': 5,
                   'selected_model_bytes': sum(f['size'] for f in model_files), 'worker_rss_gib': 9, 'gpu_reserved_gib': 12,
                   'seconds_per_wav': 20, 'bytes_per_wav': 2 * 1024**2, 'archive_bytes': 16 * 1024**2,
                   'paid_upgrade': False, 'laptop_media_or_models': False},
        'attribution': prior['attribution'],
        'limitations': ['Four texts are excluded from both adapter runs; base pretraining exposure remains possible',
                        'Published requested text is not a verified transcript of generated speech',
                        'Same female reference, decoder, seed and sampler; one take each, no retakes or pitch changes',
                        'Single-speaker research comparison; no established unique voice, broad dialect coverage or native app adoption',
                        'Only actual listening can establish preference; no automatic training from feedback'],
    }
    plan_path = folder / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Different comparison plan exists; no overwrite')
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    modules = ['expanded_comparison_cloud.py', 'expanded_comparison_archive.py', 'expanded_comparison_review.py']
    sources = {name: (ROOT / 'backend/voice_research' / name).read_text() for name in modules}
    worker = (ROOT / 'backend/voice_research/expanded_comparison_worker.py').read_text()
    generator = (ROOT / 'backend/voice_research/flash_generation.py').read_text()
    adapter = (ROOT / 'backend/voice_research/flash_adapter.py').read_text()
    extras = {'runtime-source-receipt.json': (prior_folder / 'runtime-source-receipt.json').read_text(),
              'COSYVOICE-LICENSE': (prior_folder / 'reference/LICENSE').read_text(),
              'MATCHA-LICENSE': (prior_folder / 'reference/MATCHA-LICENSE').read_text(),
              'comparison_review.py': sources['expanded_comparison_review.py']}
    tail = '''
from IPython.display import Javascript,display,Markdown
def publish(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Previous and expanded female Igbo voices\\nFour reserved texts, one take per adapter. Same reference, seed and decoder. This cell generates speech only; it never trains.'))
try:
 gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 if gpu.returncode or len(gpu.stdout.strip().splitlines())!=1 or int(gpu.stdout.strip().rsplit(',',1)[1])<14000:raise RuntimeError('Hosted T4-class GPU required before setup; no CPU synthesis fallback or purchase')
 publish({'phase':'expanded_comparison_gpu_verified','gpu':gpu.stdout.strip(),'optimizer_updates_this_stage':0})
 from google.colab import drive
 if not Path('/content/drive/MyDrive').is_dir():
  publish({'phase':'expanded_comparison_waiting_for_drive','optimizer_updates_this_stage':0})
  drive.mount('/content/drive',timeout_ms=300000)
 report=run_expanded_comparison(PLAN,WORKER,GENERATOR,RUNTIME,ADAPTER_SOURCE,publish)
 publish(report)
 if report.get('completed_wavs')==8 and report.get('source_checkpoints_unchanged'):
  receipt=backup_comparison(EXTRAS)
  publish(receipt)
  show_comparison(receipt)
  print('Eight generated clips verified and backed up to Drive. Listen and save each pair; no further training starts automatically.')
except Exception as error:
 publish({'phase':'expanded_comparison_cell_stopped','error_type':type(error).__name__,'error':str(error)[:500],'optimizer_updates_this_stage':0})
 raise
'''
    cell = '# Anyanwu bounded cloud diagnostics: compare preserved v1 and expanded v2 voices\n' + '\n'.join(sources.values())
    for name, value in [('PLAN', plan), ('WORKER', worker), ('GENERATOR', generator), ('RUNTIME', runtime), ('ADAPTER_SOURCE', adapter), ('EXTRAS', extras)]:
        cell += '\n' + name + '=' + repr(value) + '\n'
    cell += tail
    for name, source in {**sources, 'worker': worker, 'cell': cell}.items():
        compile(source, name, 'exec')
    (folder / 'comparison-cell.py').write_text(cell)
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'accelerator': 'GPU', 'colab': {'gpuType': 'T4'}},
                'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# Anyanwụ: v1/v2 listening comparison\nRun only the comparison cell on a hosted free T4. Models/audio remain in Colab and Drive. No training or payment.\n']},
                          {'cell_type': 'code', 'metadata': {}, 'source': cell.splitlines(True), 'outputs': [], 'execution_count': None}]}
    (folder / 'Anyanwu-Expanded-Voice-Comparison.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    (ROOT / 'Voice Research/Anyanwu-Expanded-Voice-Comparison.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    receipt = {'phase': 'comparison_prepared_not_run', 'cell_sha256': hashlib.sha256(cell.encode()).hexdigest(), 'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
               'outputs': 8, 'pairs': 4, 'optimizer_updates_this_stage': 0, 'selected_model_bytes': sum(f['size'] for f in model_files),
               'sources': {name: hashlib.sha256(source.encode()).hexdigest() for name, source in {**sources, 'worker': worker, 'generator': generator}.items()}}
    (folder / 'source-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
