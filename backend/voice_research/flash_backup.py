"""Package only the small trained adapter and provenance in the cloud.

No base weights, BBC media, app history or credentials are included. Drive
access is left to Colab's normal user-mediated authorisation flow.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import shutil
import zipfile

WORK=Path('/content/anyanwu-flash-lora-pilot-v1')
DRIVE=Path('/content/drive/MyDrive/Anyanwu Voice Research/checkpoints/flash-lora-pilot-v1')


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup_flash_adapter(sources, publish, authorise=False):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Cloud-only adapter backup; no laptop model files')
    result=json.loads((WORK/'training-result.json').read_text())
    if result['optimizer_updates']!=5 or not result.get('post_training_integrity',{}).get('adapter_reloaded'):
        raise RuntimeError('Verified five-update training result required')
    adapter=WORK/'adapter.safetensors'
    if adapter.stat().st_size!=1102256 or checksum(adapter)!=result['adapter_sha256']:
        raise RuntimeError('Adapter failed its saved integrity check')
    archive=WORK/'anyanwu-flash-lora-pilot-v1.zip'
    if not archive.exists():
        files={name:(WORK/name).read_bytes() for name in
               ('adapter.safetensors','plan.json','training-result.json','worker.py','flash_adapter.py','speech-targets.json')}
        files['bbc-source-permission.json']=json.dumps(sources,ensure_ascii=False,indent=2).encode()
        files['README.txt']=('Anyanwu first speech LM LoRA pilot: five updates on 17.989625 seconds of provisional BBC cuts.\n'
          'This archive contains the small adapter and reproducibility metadata, not base weights or media.\n'
          'No naturalness, speaker identity, broad intelligibility or original voice is established.\n'
          'Use pinned base revision and rank-4/alpha-8 q/v LoRA from plan.json; retain three separate source vectors.\n'
          'Modular / Dynamic Accent Flash and WAXAL / Media Trust: CC BY-SA 4.0; changes: LoRA pilot and separate source vectors.\n'
          'https://huggingface.co/Modularcomputing/dynamic-accent-preview/tree/80a4e9520ea91b1454c3ce91a2c750786c4cf777\n'
          'https://creativecommons.org/licenses/by-sa/4.0/\n'
          'FunAudioLLM CosyVoice3: Apache-2.0. OpenAI Whisper mel frontend: MIT.\n'
          'BBC audio/subtitle training permission is user-attested; it is not a declaration that BBC content is CC-licensed.\n'
          'Research pilot only; source captions matched per user report, exact cuts remain provisional.\n').encode()
        files['manifest.json']=json.dumps({'files':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()},
                                          'adapter_only':True,'base_weights_included':False,'media_included':False},indent=2).encode()
        temporary=WORK/'anyanwu-flash-lora-pilot-v1.zip.partial'
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as package:
            for name,data in files.items():
                package.writestr(name,data)
        if temporary.stat().st_size>2*1024**2:
            raise RuntimeError('Two MiB archive cap exceeded')
        temporary.rename(archive)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or hashlib.sha256(package.read('adapter.safetensors')).hexdigest()!=result['adapter_sha256']:
            raise RuntimeError('Adapter archive failed round-trip verification')
        manifest=json.loads(package.read('manifest.json'))
        for name,meta in manifest['files'].items():
            data=package.read(name)
            if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']:
                raise RuntimeError('Archive provenance failed verification')
    report={'phase':'adapter_archive_ready','archive_cloud_path':str(archive),'archive_bytes':archive.stat().st_size,
            'archive_sha256':checksum(archive),'adapter_sha256':result['adapter_sha256'],
            'media_included':False,'base_weights_included':False,'downloaded_to_laptop':False,
            'durable_backup_verified':False,'drive_authorisation_required':not Path('/content/drive/MyDrive').is_dir()}
    publish(report)
    if report['drive_authorisation_required'] and authorise:
        from google.colab import drive
        drive.mount('/content/drive',timeout_ms=120000)
    if Path('/content/drive/MyDrive').is_dir():
        DRIVE.mkdir(parents=True,exist_ok=True)
        destination=DRIVE/archive.name
        if destination.exists():
            if checksum(destination)!=report['archive_sha256']:
                raise RuntimeError('Different Drive backup exists; refusing overwrite')
        else:
            temporary=DRIVE/(archive.name+'.partial')
            if temporary.exists():
                raise RuntimeError('Prior partial Drive backup exists; preserve for inspection')
            shutil.copyfile(archive,temporary)
            if checksum(temporary)!=report['archive_sha256']:
                raise RuntimeError('Drive copy checksum mismatch')
            temporary.rename(destination)
        if checksum(destination)!=report['archive_sha256']:
            raise RuntimeError('Drive backup final checksum mismatch')
        report.update(phase='adapter_drive_backup_verified',durable_backup_verified=True,
                      drive_archive_path=str(destination),drive_authorisation_required=False)
    (WORK/'backup-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
