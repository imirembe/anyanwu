"""Verified Drive backup of the female Igbo adapter and small source sample."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile


def backup_waxal_training():
    root=Path('/content/anyanwu-waxal-female5-lora-v1')
    result=json.loads((root/'training-result.json').read_text())
    if result['optimizer_updates']!=60 or not result['adapter_reloaded'] or not result['base_hash_unchanged']:
        raise RuntimeError('Completed verified sixty-update model required')
    files={name:root/name for name in ('adapter.safetensors','plan.json','data-result.json','speech-targets.json','training-result.json','training-steps.jsonl','worker.py','flash_adapter.py')}
    for path in (root/'data').iterdir():
        if path.is_file() and path.suffix in ('.mp3','.wav'):files['data/'+path.name]=path
    if len(files)!=56 or sum(p.stat().st_size for p in files.values())>32*1024**2:
        raise RuntimeError('Unexpected training archive member/byte count')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(root/'adapter.safetensors')!=result['adapter_sha256']:raise RuntimeError('Adapter checksum mismatch before backup')
    manifest={'files':{name:{'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in files.items()},'base_weights_included':False,
              'license':'Flash/WAXAL Media Trust Igbo: CC BY-SA 4.0; CV3 code Apache-2.0; Whisper mel frontend MIT',
              'changes':'Rank4 q/v LoRA and one speaker vector;60 updates; no flow/vocoder training',
              'purpose':'Research only; source transcripts supplied by publisher; naturalness not evaluated'}
    archive=root/'anyanwu-waxal-female5-lora-v1.zip'
    if not archive.exists():
        temporary=root/'anyanwu-waxal-female5-lora-v1.zip.partial'
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as package:
            for name,p in files.items():package.write(p,name)
            package.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
            package.writestr('ATTRIBUTION.txt','Modular Dynamic Accent Flash; WAXAL / Media Trust Igbo TTS speaker5.\nCC BY-SA4.0: https://creativecommons.org/licenses/by-sa/4.0/\nhttps://huggingface.co/datasets/google/WaxalNLP\nhttps://huggingface.co/Modularcomputing/dynamic-accent-preview\nFunAudioLLM CosyVoice3 architecture/code: Apache-2.0. OpenAI Whisper mel frontend: MIT.\nSource media are whole lossy viewer MP3 previews, not lossless studio originals.\nPublisher terms/metadata are retained; independent speaker consent/native quality are not certified by this experiment.\nThe validation utterances are held out from this adapter run only; the pretrained base may already have seen them.\nNo rejected BBC pairs, app history, credentials or base model weights are included.\n')
        if temporary.stat().st_size>32*1024**2:raise RuntimeError('Training archive cap exceeded')
        temporary.rename(archive)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or json.loads(package.read('manifest.json'))!=manifest:raise RuntimeError('Training archive verification failed')
        for name,item in manifest['files'].items():
            if hashlib.sha256(package.read(name)).hexdigest()!=item['sha256']:raise RuntimeError('Training archive member checksum mismatch')
    directory=Path('/content/drive/MyDrive/Anyanwu Voice Research/checkpoints/waxal-female5-lora-v1')
    if not directory.parent.parent.parent.is_dir():raise RuntimeError('Drive unavailable; cloud checkpoint retained for backup retry')
    directory.mkdir(parents=True,exist_ok=True);destination=directory/archive.name;digest=sha(archive)
    if not destination.exists():
        temporary=directory/(archive.name+'.partial')
        if temporary.exists():raise RuntimeError('Partial backup exists; inspect before retry')
        shutil.copyfile(archive,temporary)
        if sha(temporary)!=digest:raise RuntimeError('Drive training backup checksum mismatch')
        temporary.rename(destination)
    if sha(destination)!=digest:raise RuntimeError('Different existing Drive training backup; no overwrite')
    report={'phase':'female_igbo_training_backup_verified','drive_archive_path':str(destination),'archive_bytes':archive.stat().st_size,
            'archive_sha256':digest,'adapter_sha256':result['adapter_sha256'],'durable_backup_verified':True,
            'base_weights_included':False,'local_model_or_audio':False}
    (root/'backup-result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
