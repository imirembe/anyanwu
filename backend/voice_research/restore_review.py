"""Restore only review audio/metadata from the verified Drive archive."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import shutil
import zipfile


def restore_review_files(receipt,publish):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Existing hosted Colab runtime only; no laptop media/models')
    if shutil.disk_usage('/content').free<5*1024**3:
        raise RuntimeError('Five GiB cloud free disk required')
    root=Path('/content/anyanwu-bbc-expansion-v1')
    alignment_root=Path('/content/anyanwu-expanded-caption-joint-alignment-v1')
    previous={'review_files_present':root.exists(),'alignment_present':(alignment_root/'alignment-summary.json').exists(),
              'drive_mounted':Path('/content/drive/MyDrive').is_dir()}
    if not previous['drive_mounted']:
        publish({'phase':'review_restore_needs_drive_connection','previous_runtime_files':previous,
                 'current_form_untouched':True,'model_downloads':0,'training_started':False})
        from google.colab import drive
        drive.mount('/content/drive',timeout_ms=300000)
    archive=Path('/content/drive/MyDrive/Anyanwu Voice Research/datasets/bbc-expansion-v1/anyanwu-bbc-expansion-v1.zip')
    if archive.stat().st_size!=receipt['archive_bytes'] or archive.stat().st_size>32*1024**2:
        raise RuntimeError('Drive dataset archive size mismatch')
    digest=hashlib.sha256()
    with archive.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    if digest.hexdigest()!=receipt['archive_sha256']:
        raise RuntimeError('Drive dataset archive checksum mismatch')
    root.mkdir(exist_ok=True); alignment_root.mkdir(exist_ok=True)
    total=0
    def restore(package,manifest,name,destination):
        nonlocal total
        meta=manifest['files'][name]
        if meta['bytes']>512*1024:
            raise RuntimeError('Individual restored review member cap')
        data=package.read(name)
        if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']:
            raise RuntimeError('Restored review member checksum mismatch')
        total+=len(data)
        if total>4*1024**2:
            raise RuntimeError('Four MiB review-only restore cap')
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=meta['sha256']:
                raise RuntimeError('Different cloud review file exists; no overwrite')
        else:
            destination.write_bytes(data)
    with zipfile.ZipFile(archive) as package:
        if len(package.infolist())>180:
            raise RuntimeError('Unexpected archive member count')
        manifest=json.loads(package.read('manifest.json'))
        restore(package,manifest,'joint-alignment-summary.json',alignment_root/'alignment-summary.json')
        alignment=json.loads((alignment_root/'alignment-summary.json').read_text())
        for case in alignment['cases']:
            if case.get('cloud_audio_path') and not case.get('timing_warnings'):
                destination=Path(case['cloud_audio_path'])
                if destination.parent!=alignment_root or destination.suffix!='.wav':
                    raise RuntimeError('Unexpected review audio destination')
                restore(package,manifest,'candidate-cuts/'+destination.name,destination)
    (root/'dataset-backup-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return alignment,{'phase':'review_audio_restored','restored_bytes':total,'previous_runtime_files':previous,
                       'archive_hash_verified':True,'model_downloads':0,'current_form_untouched':True,'training_started':False}
