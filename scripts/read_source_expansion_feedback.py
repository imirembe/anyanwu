"""Read and optionally freeze exact submitted source feedback; never submit it."""
from pathlib import Path
import base64
import json
import sys

root=Path(__file__).resolve().parents[1];folder=root/'.build/VoiceResearch/datasets/waxal-female5-expansion-v1'
data=json.loads((folder/'data-result.json').read_text());allowed={c['id']:c['audio_sha256'] for c in data['clips']}
validator=(root/'backend/voice_research/source_expansion_review.py').read_text().split('\ndef restore_source_expansion',1)[0]
freeze=sys.argv[1:]==['freeze']
expected=json.loads((folder/'submitted-source-feedback-summary.json').read_text())['feedback_sha256'] if freeze else None
tail='''
from IPython.display import Javascript,display
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
    path=Path('/content/drive/MyDrive/Anyanwu Voice Research/datasets/waxal-female5-expansion-v1/source-reviews.json')
    if not path.is_file():raise RuntimeError('No saved source feedback file on the connected Drive')
    if path.stat().st_size>64*1024:raise RuntimeError('Source feedback size cap')
    raw=path.read_bytes();saved=json.loads(raw)
    if saved.get('experiment')!='anyanwu-waxal-female5-expansion-v1' or not isinstance(saved.get('records'),dict) or len(saved['records'])>len(ALLOWED):raise RuntimeError('Source feedback experiment/count mismatch')
    records=[]
    for key,value in saved['records'].items():
        if key!=value.get('id'):raise RuntimeError('Source feedback ID mismatch')
        checked=validate_source_feedback({**value,'nonce':'read-only'},ALLOWED,'read-only')
        records.append({k:checked[k] for k in ('id','words_match','complete_utterance','clean_female_voice','status','notes')})
    digest=hashlib.sha256(raw).hexdigest();snapshot=None
    if FREEZE:
        if len(records)!=len(ALLOWED) or any(r['status']=='pending' for r in records) or digest!=EXPECTED:raise RuntimeError('Source feedback incomplete/changed; snapshot not written')
        snapshot=path.with_name('source-reviews.submitted-v1.json')
        if snapshot.exists():
            if snapshot.read_bytes()!=raw:raise RuntimeError('Different source submission snapshot exists; no overwrite')
        else:
            stage=path.with_name('source-reviews.submitted-v1.staging.json');stage.write_bytes(raw)
            if stage.read_bytes()!=raw:raise RuntimeError('Source feedback snapshot staging verification failed')
            stage.replace(snapshot)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Source submission snapshot verification failed')
    report={'phase':'saved_source_feedback_checked','saved_records':len(records),'expected_records':len(ALLOWED),
      'accepted_ids':[r['id'] for r in records if r['status']=='accepted_for_research'],'held_ids':[r['id'] for r in records if r['status']=='held_by_user'],
      'pending_ids':[r['id'] for r in records if r['status']=='pending'],'missing_ids':sorted(set(ALLOWED)-set(saved['records'])),
      'records':records,'feedback_sha256':digest,'feedback_bytes':len(raw),'submitted_snapshot':str(snapshot) if snapshot else None,
      'submitted_snapshot_verified':bool(snapshot),'feedback_changed':False,'training_started':False}
except Exception as error:report={'phase':'source_feedback_read_stopped','error':str(error)[:500],'feedback_changed':False,'training_started':False}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu review connection recovery: inspect saved source feedback\n'+validator+'\nALLOWED='+repr(allowed)+'\nFREEZE='+repr(freeze)+'\nEXPECTED='+repr(expected)+'\n'+tail
compile(cell,'read-source-feedback.py','exec');(folder/'read-feedback-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-review-recovery.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared bounded Drive source feedback inspection; existing form/callbacks remain untouched.')
