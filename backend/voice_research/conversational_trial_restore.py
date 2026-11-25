"""Restore exact reviewed, pre-encoded targets and the preserved v1 adapter."""
from pathlib import Path
import hashlib
import json
import zipfile


def restore_conversational_trial(plan, *, root=None):
    root=Path(root) if root is not None else Path('/content/anyanwu-conversational-style-lora-v1');root.mkdir(exist_ok=True)
    info=plan['baseline_archive'];archive=Path(info['path'])
    if archive.stat().st_size!=info['bytes'] or hashlib.sha256(archive.read_bytes()).hexdigest()!=info['sha256']:
        raise RuntimeError('Preserved v1 archive integrity failure')
    baseline=root/'baseline';baseline.mkdir(exist_ok=True)
    selected={'plan.json','training-result.json','adapter.safetensors','speech-targets.json','data-result.json',
              'data/ibo_616.wav','data/ibo_616.mp3'}
    with zipfile.ZipFile(archive) as package:
        manifest=json.loads(package.read('manifest.json'))
        if len(manifest['files'])>60 or not selected<=set(manifest['files']):raise RuntimeError('Unexpected baseline archive manifest')
        for name in sorted(selected):
            meta=manifest['files'][name]
            if meta['bytes']>2*1024**2 or package.getinfo(name).file_size!=meta['bytes']:raise RuntimeError('Baseline member size cap')
            raw=package.read(name)
            if len(raw)!=meta['bytes'] or hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise RuntimeError('Baseline member hash mismatch')
            dest=baseline/name;dest.parent.mkdir(exist_ok=True)
            if dest.exists() and dest.read_bytes()!=raw:raise RuntimeError('Different existing baseline preserved')
            if not dest.exists():dest.write_bytes(raw)
    if hashlib.sha256((baseline/'adapter.safetensors').read_bytes()).hexdigest()!=plan['starting_adapter_sha256']:
        raise RuntimeError('Starting v1 adapter changed')
    snapshot=Path(plan['submitted_snapshot']);targets_path=Path(plan['encoded_targets']['path'])
    if (snapshot.stat().st_size>65536 or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=plan['submitted_review_sha256']
            or targets_path.stat().st_size>256*1024 or hashlib.sha256(targets_path.read_bytes()).hexdigest()!=plan['encoded_targets']['sha256']):
        raise RuntimeError('Reviewed text or whole-response acoustic targets changed')
    review=json.loads(snapshot.read_text())['records'];new=json.loads(targets_path.read_text())
    old=json.loads((baseline/'speech-targets.json').read_text());meta={r['id']:r for r in json.loads((baseline/'data-result.json').read_text())['clips']}
    if len(old)!=24 or len(new)!=2 or set(review)!={r['id'] for r in new}:raise RuntimeError('Source counts/coverage mismatch')
    cases=[]
    for row in old:
        cases.append({**row,'text':meta[row['id']]['text'],'source_index':0,'source_group':'waxal:ibo:5'})
    for row in new:
        record=review[row['id']]
        if (record['words_match']!=record['clean_complete_speech'] or record['words_match']!='yes'
                or row['text']!=record['transcript'] or hashlib.sha256(row['text'].encode()).hexdigest()!=row['transcript_sha256']):
            raise RuntimeError('Conversational targets differ from exact user-reviewed text')
        cases.append({**row,'split':'train','source_index':1,'source_group':'silencio:IGB_005'})
    if len([c for c in cases if c['split']=='train'])!=22 or len([c for c in cases if c['split']=='validation'])!=4:
        raise RuntimeError('Registered22-train/4-diagnostic source pool mismatch')
    ids={c['id'] for c in cases}
    if ids & {t['id'] for t in plan['comparison']['texts']}:raise RuntimeError('Evaluation text entered training')
    for c in cases:
        if (not 0<len(c['speech_ids'])<=800 or len(c['text_ids'])+len(c['speech_ids'])+10>1024
                or any(type(i) is not int or i<0 for i in c['text_ids']+c['speech_ids'])):
            raise RuntimeError('Whole target exceeds registered limits; no truncation')
    if len(plan['schedule'])!=30 or any(i not in ids or next(c for c in cases if c['id']==i)['split']!='train' for i in plan['schedule']):
        raise RuntimeError('Training schedule does not match accepted targets')
    (root/'training-cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    (root/'user-transcripts.submitted.json').write_bytes(snapshot.read_bytes())
    return {'phase':'conversational_trial_sources_restored','source_groups':2,'whole_conversational_responses':2,
            'train_pool':22,'diagnostic_validation':4,'optimizer_updates':0}
