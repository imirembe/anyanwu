"""Build a bounded CPU audit and immutable snapshot of actual submitted feedback."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.build/VoiceResearch/auditions/waxal-female5-expanded-comparison-v2'
generated=json.loads((folder/'generation-result.json').read_text())
allowed={}
for clip in generated['clips']:
    allowed.setdefault(clip['text_id'],{})[clip['variant']]=clip['audio_sha256']
if len(allowed)!=4 or any(set(v)!={'base','adapter'} for v in allowed.values()):raise RuntimeError('Expected four exact comparison pairs')
validator=(ROOT/'backend/voice_research/expanded_comparison_review.py').read_text().split('\ndef restore_comparison',1)[0]
tail='''
import importlib.util,platform,math
from IPython.display import Javascript,display
def publish(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
 if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab CPU only; no laptop Drive or media access')
 if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive is disconnected; no feedback or snapshot changed')
 path=Path('/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research/auditions/waxal-female5-expanded-comparison-v2/human-comparison.json')
 if not path.is_file() or path.stat().st_size>16*1024:raise RuntimeError('Expected bounded saved comparison feedback file')
 raw=path.read_bytes();value=json.loads(raw)
 if value.get('experiment')!='anyanwu-waxal-female5-expanded-comparison-v2' or not isinstance(value.get('records'),dict) or len(value['records'])>4:raise RuntimeError('Feedback experiment/count mismatch')
 records=[];incomplete=[]
 for key,record in value['records'].items():
  if key!=record.get('id'):raise RuntimeError('Feedback record ID mismatch')
  checked=validate_comparison_feedback({**record,'nonce':'read-only-audit'},ALLOWED,'read-only-audit')
  if not isinstance(record.get('saved_at_ms'),(int,float)) or not math.isfinite(record['saved_at_ms']) or record['saved_at_ms']<=0:raise RuntimeError('Missing verified-save timestamp')
  fields=[k for k in ('preference','words_base','words_adapter','cadence_base','cadence_adapter') if checked[k]=='unreviewed']
  if fields:incomplete.append({'id':key,'fields':fields})
  records.append({**checked,'saved_at_ms':record['saved_at_ms']})
 missing=sorted(set(ALLOWED)-set(value['records']));complete=len(records)==4 and not missing and not incomplete
 digest=hashlib.sha256(raw).hexdigest();snapshot=None
 if complete:
  if path.read_bytes()!=raw:raise RuntimeError('Feedback changed during verification; no snapshot written')
  snapshot=path.with_name('human-comparison.submitted-v2.json')
  if snapshot.exists():
   if snapshot.read_bytes()!=raw:raise RuntimeError('A different submitted snapshot already exists; no overwrite')
  else:
   stage=path.with_name('human-comparison.submitted-v2.staging.json')
   if stage.exists() and stage.read_bytes()!=raw:raise RuntimeError('Different snapshot staging file exists; no overwrite')
   stage.write_bytes(raw)
   if stage.read_bytes()!=raw:raise RuntimeError('Submitted snapshot staging verification failed')
   if snapshot.exists():raise RuntimeError('Submitted snapshot appeared concurrently; no overwrite')
   stage.replace(snapshot)
  if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:raise RuntimeError('Submitted snapshot hash verification failed')
 preferences={v:sum(r['preference']==v for r in records) for v in ('base','adapter','tie','unreviewed')}
 cadence={v:[int(r['cadence_'+v]) for r in records if r['cadence_'+v]!='unreviewed'] for v in ('base','adapter')}
 report={'phase':'expanded_comparison_feedback_verified','saved_records':len(records),'expected_records':4,'complete':complete,
  'missing_ids':missing,'incomplete_records':incomplete,'feedback_sha256':digest,'feedback_bytes':len(raw),'records':records,
  'preference_counts':preferences,'cadence_scores':cadence,'word_match_yes':{v:sum(r['words_'+v]=='yes' for r in records) for v in ('base','adapter')},
  'submitted_snapshot':str(snapshot) if snapshot else None,'submitted_snapshot_verified':bool(snapshot),
  'master_feedback_changed':False,'ratings_submitted_by_agent':False,'training_started':False,'form_untouched':True}
except Exception as error:
 report={'phase':'expanded_comparison_feedback_audit_stopped','error_type':type(error).__name__,'error':str(error)[:500],'master_feedback_changed':False,'training_started':False,'form_untouched':True}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu CPU audit: verify submitted expanded voice comparison feedback\n'+validator+'\nALLOWED='+repr(allowed)+'\n'+tail
compile(cell,'expanded-feedback-audit','exec')
(folder/'feedback-audit-cell.py').write_text(cell)
receipt={'phase':'feedback_audit_prepared','cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),'expected_pairs':4,'audio_hashes':allowed,'master_feedback_changed':False,'snapshot_if_all_complete':True,'training':False}
(folder/'feedback-audit-source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'phase':receipt['phase'],'cell_sha256':receipt['cell_sha256'],'expected_pairs':4}))
