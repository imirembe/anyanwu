"""Stage a read-only check of real Drive feedback; keep the live form intact."""
from pathlib import Path
import base64
import json
import sys

root=Path(__file__).resolve().parents[1]
folder=root/'.build/VoiceResearch/auditions/waxal-female5-adapter-comparison-v1'
result=json.loads((folder/'generation-result.json').read_text())
allowed={item['text_id']:{c['variant']:c['audio_sha256'] for c in result['clips'] if c['text_id']==item['text_id']} for item in result['clips']}
validator=(root/'backend/voice_research/comparison_review.py').read_text().split('\ndef restore_comparison',1)[0]
freeze=sys.argv[1:]==['freeze']
expected=json.loads((folder/'submitted-feedback-summary.json').read_text())['feedback_sha256'] if freeze else None
tail='''
from IPython.display import Javascript,display
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
    path=Path('/content/drive/MyDrive/Anyanwu Voice Research/auditions/waxal-female5-adapter-comparison-v1/human-comparison.json')
    if not path.is_file():raise RuntimeError('Saved comparison feedback is not present on the connected Drive')
    if path.stat().st_size>16*1024:raise RuntimeError('Feedback file size cap')
    raw=path.read_bytes();value=json.loads(raw)
    if value.get('experiment')!='anyanwu-waxal-female5-comparison-v1' or not isinstance(value.get('records'),dict) or len(value['records'])>3:raise RuntimeError('Feedback snapshot mismatch')
    records=[]
    for key,record in value['records'].items():
        if key!=record.get('id'):raise RuntimeError('Feedback ID mismatch')
        checked=validate_comparison_feedback({**record,'nonce':'read-only'},ALLOWED,'read-only')
        records.append({k:checked[k] for k in ('id','preference','words_base','words_adapter','cadence_base','cadence_adapter','notes')})
    snapshot=None
    if FREEZE:
        if len(records)!=3 or hashlib.sha256(raw).hexdigest()!=EXPECTED:raise RuntimeError('Feedback changed since inspection; no snapshot written')
        snapshot=path.with_name('human-comparison.submitted-v1.json')
        if snapshot.exists():
            if snapshot.read_bytes()!=raw:raise RuntimeError('Different submitted feedback snapshot exists; no overwrite')
        else:
            stage=path.with_name('human-comparison.submitted-v1.staging.json');stage.write_bytes(raw)
            if stage.read_bytes()!=raw:raise RuntimeError('Snapshot verification failed')
            stage.replace(snapshot)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Submitted Drive snapshot changed')
    report={'phase':'saved_female_comparison_feedback_checked','saved_records':len(records),'expected_records':3,
      'missing_ids':sorted(set(ALLOWED)-set(value['records'])),'feedback_sha256':hashlib.sha256(raw).hexdigest(),
      'feedback_bytes':len(raw),'records':records,'feedback_changed':False,'training_started':False,
      'submitted_snapshot':str(snapshot) if snapshot else None,'submitted_snapshot_verified':bool(snapshot)}
except Exception as error:report={'phase':'comparison_feedback_read_stopped','error':str(error)[:500],'feedback_changed':False,'training_started':False}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu review connection recovery: read saved female comparison feedback\n'+validator+'\nALLOWED='+repr(allowed)+'\nFREEZE='+repr(freeze)+'\nEXPECTED='+repr(expected)+'\n'+tail
compile(cell,'read-comparison-feedback.py','exec')
(folder/'read-feedback-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-review-recovery.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared read-only Drive feedback check in the recovery cell; live form remains intact.')
