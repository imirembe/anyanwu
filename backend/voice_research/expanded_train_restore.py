"""Restore only pinned speech artifacts and accepted data into a separate run."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile


def restore_expanded_training(plan,publish):
    root=Path('/content/anyanwu-waxal-female5-expanded-lora-v2');root.mkdir(exist_ok=True)
    previous_plan=root/'plan.json'
    if previous_plan.exists() and json.loads(previous_plan.read_text())!=plan:raise RuntimeError('Different expanded plan exists; no overwrite')
    previous_plan.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    def sha(body):return hashlib.sha256(body).hexdigest()
    def restore_archive(info,baseline):
        path=Path(info.get('drive_archive_path') or info['drive_archive'])
        if path.stat().st_size!=info['archive_bytes'] or path.stat().st_size>32*1024**2 or sha(path.read_bytes())!=info['archive_sha256']:raise RuntimeError('Pinned Drive archive checksum/size failure')
        target=root/('baseline' if baseline else 'expansion');target.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as package:
            manifest=json.loads(package.read('manifest.json'))
            if len(manifest['files'])>60 or sum(m['bytes'] for m in manifest['files'].values())>32*1024**2:raise RuntimeError('Source archive manifest cap')
            selected=[]
            for name,meta in manifest['files'].items():
                metadata={'plan.json','data-result.json','speech-targets.json','training-result.json','adapter.safetensors'} if baseline else {'plan.json','data-result.json'}
                if name not in metadata and not re.fullmatch(r'data/ibo_[0-9]+\.(wav|mp3)',name):continue
                if meta['bytes']>2*1024**2:raise RuntimeError('Selected source archive member cap')
                body=package.read(name)
                if len(body)!=meta['bytes'] or sha(body)!=meta['sha256']:raise RuntimeError('Selected source member integrity failure')
                destination=target/name;destination.parent.mkdir(exist_ok=True)
                if destination.exists() and destination.read_bytes()!=body:raise RuntimeError('Different restored baseline/source file exists')
                if not destination.exists():destination.write_bytes(body)
                selected.append(name)
        return target
    baseline=restore_archive(plan['archives']['baseline'],True)
    expansion=restore_archive(plan['archives']['new_source'],False)
    if sha((baseline/'adapter.safetensors').read_bytes())!=plan['starting_adapter_sha256']:raise RuntimeError('Starting v1 adapter checksum mismatch')
    feedback_path=Path(plan['source_feedback_snapshot'])
    if feedback_path.stat().st_size>64*1024:raise RuntimeError('Source feedback size cap')
    raw=feedback_path.read_bytes()
    if sha(raw)!=plan['feedback_sha256']:raise RuntimeError('Accepted source feedback changed')
    feedback=json.loads(raw)
    old=json.loads((baseline/'data-result.json').read_text());new=json.loads((expansion/'data-result.json').read_text())
    if len(old['clips'])!=24 or len(new['clips'])!=14 or set(feedback['records'])!={c['id'] for c in new['clips']}:raise RuntimeError('Registered source counts/feedback coverage changed')
    if feedback.get('experiment')!='anyanwu-waxal-female5-expansion-v1':raise RuntimeError('Source feedback belongs to another experiment')
    all_cases=[];folder=root/'data';folder.mkdir(exist_ok=True)
    expected={c['id']:c for c in plan['cases']}
    for origin,items in ((baseline,old['clips']),(expansion,new['clips'])):
        for item in items:
            case=expected[item['id']]
            if any(item[key]!=case[key] for key in ('id','text','speaker_id','gender','row_idx')):raise RuntimeError('Source metadata changed from prospective plan')
            if origin==expansion:
                record=feedback['records'][item['id']]
                if record['audio_sha256']!=item['audio_sha256'] or item['audio_sha256']!=case['audio_sha256'] or any(record.get(k)!='yes' for k in ('words_match','complete_utterance','clean_female_voice')):raise RuntimeError('A new source was not explicitly accepted by the user')
            for suffix,key in (('.wav','audio_sha256'),('.mp3','mp3_sha256')):
                source=origin/'data'/(item['id']+suffix);destination=folder/source.name
                if sha(source.read_bytes())!=item[key]:raise RuntimeError('Source pair changed during restore')
                if destination.exists() and sha(destination.read_bytes())!=item[key]:raise RuntimeError('Different existing expanded audio')
                if not destination.exists():shutil.copyfile(source,destination)
            all_cases.append({**item,'adapter_split':case['adapter_split'],'cloud_audio_path':str(folder/(item['id']+'.wav')),
                              'source_review_status':case['source_review_status']})
    train=[c for c in all_cases if c['adapter_split']=='train'];validation=[c for c in all_cases if c['adapter_split']=='validation']
    reserved={c['id'] for c in plan['fresh_synthesis_comparison']['texts']}
    if (len(train),len(validation))!=(34,4) or reserved&set(expected):raise RuntimeError('Registered source/evaluation separation failed')
    if round(sum(c['seconds'] for c in train),6)!=plan['combined_train_seconds']:raise RuntimeError('Expanded training duration changed')
    receipt={'phase':'expanded_reviewed_training_data_restored','clips':all_cases,'train_clips':34,'validation_clips':4,
      'audio_seconds':round(sum(c['seconds'] for c in all_cases),6),'new_user_accepted_clips':14,'feedback_sha256':plan['feedback_sha256'],
      'starting_adapter_sha256':plan['starting_adapter_sha256'],'reserved_evaluation_ids':sorted(reserved),'optimizer_updates':0}
    (root/'data-result.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    (root/'source-reviews.submitted-v1.json').write_bytes(raw)
    (root/'baseline-plan.json').write_bytes((baseline/'plan.json').read_bytes())
    publish({k:v for k,v in receipt.items() if k!='clips'})
    return receipt
