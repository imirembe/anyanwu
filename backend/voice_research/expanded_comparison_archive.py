"""Archive eight generated clips and provenance directly to the mounted Drive."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile


def backup_comparison(extras):
    root=Path('/content/anyanwu-waxal-female5-expanded-comparison-v2')
    drive=Path('/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research/auditions/waxal-female5-expanded-comparison-v2')
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Google Drive is not connected; audio remains in the current runtime')
    existing=root/'backup-result.json'
    if existing.exists():
        prior=json.loads(existing.read_text());path=Path(prior['drive_archive'])
        if path.stat().st_size!=prior['archive_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=prior['archive_sha256']:raise RuntimeError('Existing comparison backup changed')
        return prior
    result=json.loads((root/'generation-result.json').read_text())
    if result.get('completed_wavs')!=8 or result.get('optimizer_updates_this_stage')!=0 or not result.get('source_checkpoints_unchanged'):raise RuntimeError('Eight completed waveforms required before backup')
    names=['plan.json','generation-result.json','token-result.json','setup-result.json','comparison_worker.py','flash_generation.py','cv3_runtime.py']
    for clip in result['clips']:
        names.extend([clip['filename'],clip['audio_filename']])
        p=root/clip['audio_filename']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=clip['audio_sha256']:raise RuntimeError('Audio integrity failure')
    for name,source in extras.items():
        if name not in {'runtime-source-receipt.json','COSYVOICE-LICENSE','MATCHA-LICENSE','comparison_review.py'}:raise RuntimeError('Unexpected archive extra')
        (root/name).write_text(source);names.append(name)
    files={name:{'bytes':(root/name).stat().st_size,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in names}
    if sum(v['bytes'] for v in files.values())>16*1024**2:raise RuntimeError('Sixteen MiB archive input cap')
    manifest={'experiment':result['experiment'],'files':files,'contains_weights':False,'contains_private_conversations':False}
    (root/'archive-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive=root/'anyanwu-female-expanded-comparison-v2.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as out:
        for name in names+['archive-manifest.json']:out.write(root/name,name)
    if archive.stat().st_size>16*1024**2:raise RuntimeError('Archive byte cap')
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None or set(check.namelist())!=set(names+['archive-manifest.json']):raise RuntimeError('Archive validation failed')
        for name,value in files.items():
            if hashlib.sha256(check.read(name)).hexdigest()!=value['sha256']:raise RuntimeError('Archive member integrity failure')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();drive.mkdir(parents=True,exist_ok=True);target=drive/archive.name
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('Different saved archive exists; no overwrite')
    else:
        stage=drive/(archive.name+'.staging');shutil.copyfile(archive,stage)
        if hashlib.sha256(stage.read_bytes()).hexdigest()!=digest:raise RuntimeError('Drive copy checksum failure')
        stage.replace(target)
    receipt={'phase':'comparison_drive_backup_verified','archive_bytes':target.stat().st_size,'archive_sha256':digest,
             'drive_archive':str(target),'files':len(names)+1,'audio_clips':8,'contains_weights':False,'durable_backup_verified':True}
    (root/'backup-result.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
