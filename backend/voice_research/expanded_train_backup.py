"""Archive the expanded adapter and provenance without touching the v1 baseline."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile


def backup_expanded_training():
    root=Path('/content/anyanwu-waxal-female5-expanded-lora-v2')
    report=json.loads((root/'training-result.json').read_text());plan=json.loads((root/'plan.json').read_text())
    if report['optimizer_updates']!=60 or not report['warm_start_equal'] or not report['adapter_reloaded'] or not report['base_hash_unchanged'] or not report['starting_adapter_unchanged']:
        raise RuntimeError('Completed verified expanded training required before backup')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(root/'baseline/adapter.safetensors')!=plan['starting_adapter_sha256']:raise RuntimeError('Preserved v1 adapter changed')
    if sha(root/'adapter.safetensors')!=report['adapter_sha256']:raise RuntimeError('Expanded adapter checksum failure')
    names=['adapter.safetensors','plan.json','data-result.json','speech-targets.json','training-result.json','training-steps.jsonl','worker.py','flash_adapter.py','baseline-plan.json','source-reviews.submitted-v1.json']
    files={name:root/name for name in names}
    for clip in json.loads((root/'data-result.json').read_text())['clips']:
        for suffix in ('.mp3','.wav'):
            name='data/'+clip['id']+suffix;files[name]=root/name
    if len(files)!=86 or sum(p.stat().st_size for p in files.values())>32*1024**2:raise RuntimeError('Expanded archive member/size cap')
    manifest={'experiment':plan['id'],'files':{name:{'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in files.items()},'contains_base_weights':False,
      'starting_adapter_sha256':plan['starting_adapter_sha256'],'additional_optimizer_updates':60,'label_policy':'14 new clips user-accepted; original24 publisher labels retain original status',
      'quality_evaluated':False,'attribution':plan['attribution']}
    archive=root/'anyanwu-waxal-female5-expanded-lora-v2.zip'
    if not archive.exists():
        stage=root/(archive.name+'.staging')
        if stage.exists():raise RuntimeError('Incomplete expanded archive exists; inspect before retry')
        with zipfile.ZipFile(stage,'x',compression=zipfile.ZIP_DEFLATED) as package:
            for name,path in files.items():package.write(path,name)
            package.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
            package.writestr('ATTRIBUTION.txt','Google WAXAL / Media Trust Igbo TTS, CC BY-SA4.0. Modular Dynamic Accent Flash, CC BY-SA4.0.\nFunAudioLLM CosyVoice3 code Apache2.0; OpenAI Whisper mel frontend MIT.\nhttps://huggingface.co/datasets/google/WaxalNLP\nhttps://huggingface.co/Modularcomputing/dynamic-accent-preview\nChanges:60 additional rank4 q/v LoRA updates/source vector from verified v1 weights. No flow/vocoder training or synthetic dialogue.\n14 new source utterances user accepted, old24 publisher labels unchanged.38 whole lossy-preview MP3+WAV pairs. Four fresh evaluation texts excluded.\nNo rejected BBC cuts, private app data, credentials, base weights or original v1 adapter included. V1 archive remains preserved separately.\nResearch-only candidate; accent, naturalness, broad dialect coverage and unique identity require actual generated-speech listening.\n')
        if stage.stat().st_size>32*1024**2:raise RuntimeError('Expanded archive byte cap')
        stage.replace(archive)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or json.loads(package.read('manifest.json'))!=manifest or set(package.namelist())!=set(files)|{'manifest.json','ATTRIBUTION.txt'}:raise RuntimeError('Expanded archive integrity failure')
        for name,meta in manifest['files'].items():
            if hashlib.sha256(package.read(name)).hexdigest()!=meta['sha256']:raise RuntimeError('Expanded archive member hash mismatch')
    directory=Path('/content/drive/MyDrive/Anyanwu Voice Research/checkpoints/waxal-female5-expanded-lora-v2')
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive unavailable for expanded adapter backup')
    directory.mkdir(parents=True,exist_ok=True);destination=directory/archive.name;digest=sha(archive)
    if destination.exists():
        if sha(destination)!=digest:raise RuntimeError('Different expanded Drive backup exists; no overwrite')
    else:
        stage=directory/(archive.name+'.staging');shutil.copyfile(archive,stage)
        if sha(stage)!=digest:raise RuntimeError('Expanded Drive backup copy verification failed')
        stage.replace(destination)
    if sha(destination)!=digest:raise RuntimeError('Final expanded Drive archive changed')
    result={'phase':'expanded_adapter_drive_backup_verified','drive_archive':str(destination),'archive_bytes':destination.stat().st_size,'archive_sha256':digest,
      'adapter_sha256':report['adapter_sha256'],'starting_adapter_sha256':plan['starting_adapter_sha256'],'baseline_preserved':True,'durable_backup_verified':True,'base_weights_included':False}
    (root/'backup-result.json').write_text(json.dumps(result,indent=2)+'\n');return result
