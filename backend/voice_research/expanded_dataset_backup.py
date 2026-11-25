"""Persist the small published speech sample and provenance directly to Drive."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import shutil
import zipfile

SAMPLE=Path('/content/anyanwu-bbc-expansion-v1')
ALIGN=Path('/content/anyanwu-expanded-caption-joint-alignment-v1')
DESTINATION=Path('/content/drive/MyDrive/Anyanwu Voice Research/datasets/bbc-expansion-v1')


def hash_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_expanded_dataset(captions):
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Hosted Colab only; no laptop corpus/media acquisition')
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Authorised Drive mount unavailable; no laptop backup fallback')
    review=json.loads((SAMPLE/'expansion-review-result.json').read_text())
    alignment=json.loads((ALIGN/'alignment-summary.json').read_text())
    if captions['caption_count']!=24 or review['new_audio_seconds']!=100 or alignment['experiment_id']!=ALIGN.name:
        raise RuntimeError('Unexpected expanded corpus snapshot')
    local_captions=SAMPLE/'caption-text.json'
    encoded=json.dumps(captions,ensure_ascii=False,indent=2)+'\n'
    if local_captions.exists() and local_captions.read_text()!=encoded:
        raise RuntimeError('Different expanded caption snapshot already exists')
    local_captions.write_text(encoded)
    files={'captions.json':local_captions,'source-acquisition.json':SAMPLE/'expansion-result.json',
           'source-review.json':SAMPLE/'expansion-review-result.json','joint-alignment-plan.json':ALIGN/'plan.json',
           'joint-alignment-summary.json':ALIGN/'alignment-summary.json',
           'independent-alignment-summary.json':Path('/content/anyanwu-expanded-caption-alignment-v1/alignment-summary.json')}
    for row in review['sources']:
        for entry in [row['audio']]+row['frames']+row['sheets']:
            path=Path(entry['path'])
            if not path.is_relative_to(SAMPLE/row['id']) or not path.is_file() or hash_file(path)!=entry['sha256']:
                raise RuntimeError('Source artifact failed hash or path validation')
            files['sources/'+row['id']+'/'+path.relative_to(SAMPLE/row['id']).as_posix()]=path
    for case in alignment['cases']:
        if case.get('cloud_audio_path'):
            path=Path(case['cloud_audio_path'])
            if path.parent!=ALIGN or hash_file(path)!=case['candidate_audio_sha256']:
                raise RuntimeError('Aligned candidate failed integrity')
            files['candidate-cuts/'+path.name]=path
    if sum(p.stat().st_size for p in files.values())>32*1024**2:
        raise RuntimeError('Thirty-two MiB source-backup cap')
    archive=SAMPLE/'anyanwu-bbc-expansion-v1.zip'
    manifest={'files':{name:{'bytes':path.stat().st_size,'sha256':hash_file(path)} for name,path in files.items()},
              'models_included':False,'full_video_included':False,'human_approvals':0,
              'license':'BBC user-attested training permission; no assertion of an open BBC licence',
              'limitations':'AI caption copies and machine timing candidates; no ready training corpus, native model quality, or speaker identity established'}
    if not archive.exists():
        temporary=SAMPLE/'anyanwu-bbc-expansion-v1.zip.partial'
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as package:
            for name,path in files.items():
                package.write(path,name)
            package.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
            package.writestr('README.txt','Anyanwu research sample: 100 seconds in five fixed BBC windows, nominal 24–44s.\n24 AI visual caption copies. New candidate cuts require exact-boundary and clean-voice listening review.\nNo model weights, full video, app conversation or new synthetic dialogue is included.\nSource attribution and user-attested training permission are in source-acquisition.json.\nThe original five-update adapter is backed up separately.\n')
        if temporary.stat().st_size>32*1024**2:
            raise RuntimeError('Compressed corpus archive cap')
        temporary.rename(archive)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or json.loads(package.read('manifest.json'))!=manifest:
            raise RuntimeError('Corpus archive snapshot/CRC verification failed')
        for name,item in manifest['files'].items():
            if hashlib.sha256(package.read(name)).hexdigest()!=item['sha256']:
                raise RuntimeError('Corpus archive member checksum mismatch')
    DESTINATION.mkdir(parents=True,exist_ok=True)
    target=DESTINATION/archive.name
    digest=hash_file(archive)
    if not target.exists():
        temporary=DESTINATION/(archive.name+'.partial')
        if temporary.exists():
            raise RuntimeError('Partial Drive corpus archive exists; inspect before retry')
        shutil.copyfile(archive,temporary)
        if hash_file(temporary)!=digest:
            raise RuntimeError('Drive corpus copy checksum mismatch')
        temporary.rename(target)
    if hash_file(target)!=digest:
        raise RuntimeError('Existing Drive corpus differs; no overwrite')
    report={'phase':'expanded_dataset_drive_backup_verified','drive_archive_path':str(target),'archive_bytes':archive.stat().st_size,
            'archive_sha256':digest,'file_count':len(files),'new_audio_seconds':100,'caption_count':24,
            'models_included':False,'downloaded_to_laptop':False,'durable_backup_verified':True,
            'candidate_cuts':alignment['candidate_cuts'],'flagged_cuts':alignment['flagged_cuts']}
    (SAMPLE/'dataset-backup-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report,alignment
