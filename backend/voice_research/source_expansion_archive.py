"""Verify and back up the new whole source sample directly to Google Drive."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile


def backup_source_expansion(extras):
    root=Path('/content/anyanwu-waxal-female5-expansion-v1')
    drive=Path('/content/drive/MyDrive/Anyanwu Voice Research/datasets/waxal-female5-expansion-v1')
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive must be connected to preserve the source sample')
    result=json.loads((root/'data-result.json').read_text());plan=json.loads((root/'plan.json').read_text())
    if result['optimizer_updates']!=0 or not 1<=len(result['clips'])<=24:raise RuntimeError('Unexpected source sample scope')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    files={'plan.json':root/'plan.json','data-result.json':root/'data-result.json'}
    for c in result['clips']:
        for extension,key in (('.mp3','mp3_sha256'),('.wav','audio_sha256')):
            name='data/'+c['id']+extension;path=root/name
            if path.stat().st_size>1024**2 or sha(path)!=c[key]:raise RuntimeError('Source archive hash/size failure')
            files[name]=path
    # Duration-held MP3s are retained in the current VM for the audit, never silently included as training pairs.
    for name in ('link-refresh-receipt.json','acquisition-attempt-1-expired-link.json'):
        if (root/name).is_file():files[name]=root/name
    for name,content in extras.items():
        if name not in {'SOURCE_CARD.txt','ATTRIBUTION.txt','source_expansion_review.py','waxal_expansion_data.py'}:raise RuntimeError('Unexpected source archive extra')
        path=root/name;path.write_text(content);files[name]=path
    if sha(root/'SOURCE_CARD.txt')!=plan['source_card_sha256']:raise RuntimeError('Source card fingerprint changed')
    if sum(p.stat().st_size for p in files.values())>16*1024**2:raise RuntimeError('Source archive input cap')
    manifest={'experiment':plan['id'],'files':{name:{'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in files.items()},
      'contains_weights':False,'human_review_pending':True,'source':'WAXAL / Media Trust Igbo TTS, CC BY-SA 4.0','representation':'Whole lossy preview MP3s and decoded16k WAVs; no cropped passages'}
    archive=root/'anyanwu-female-source-expansion-v1.zip'
    if not archive.exists():
        stage=root/(archive.name+'.staging')
        if stage.exists():raise RuntimeError('Partial source archive exists; inspect before retry')
        with zipfile.ZipFile(stage,'x',compression=zipfile.ZIP_DEFLATED) as out:
            for name,path in files.items():out.write(path,name)
            out.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        if stage.stat().st_size>16*1024**2:raise RuntimeError('Source archive byte cap')
        stage.replace(archive)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or json.loads(package.read('manifest.json'))!=manifest or set(package.namelist())!=set(files)|{'manifest.json'}:raise RuntimeError('Source archive validation failed')
        for name,meta in manifest['files'].items():
            if hashlib.sha256(package.read(name)).hexdigest()!=meta['sha256']:raise RuntimeError('Source archive member hash failure')
    drive.mkdir(parents=True,exist_ok=True);destination=drive/archive.name;digest=sha(archive)
    if destination.exists():
        if sha(destination)!=digest:raise RuntimeError('Different saved source archive exists; no overwrite')
    else:
        stage=drive/(archive.name+'.staging');shutil.copyfile(archive,stage)
        if sha(stage)!=digest:raise RuntimeError('Drive source archive copy failed verification')
        stage.replace(destination)
    if sha(destination)!=digest:raise RuntimeError('Final Drive archive hash failure')
    report={'phase':'expanded_source_drive_backup_verified','drive_archive':str(destination),'archive_bytes':destination.stat().st_size,'archive_sha256':digest,
      'files':len(files)+1,'source_clips':len(result['clips']),'audio_seconds':result['audio_seconds'],'technical_holds':result['technical_holds'],
      'durable_backup_verified':True,'contains_weights':False,'laptop_media_or_models':False}
    (root/'backup-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return report
