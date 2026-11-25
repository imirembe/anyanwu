"""Restore feedback callbacks in the old preview cell, preserving review inputs."""
from pathlib import Path
import base64
import json
import sys

root=Path(__file__).resolve().parents[1]
folder=root/'.build/VoiceResearch/training/anyanwu-female-v1/expansion-v1'
receipt=json.loads((folder/'dataset-backup-result.json').read_text())
source=(root/'backend/voice_research/restore_review.py').read_text()+'\n'+(root/'backend/voice_research/review_rules.py').read_text()+'\n'+(root/'backend/voice_research/review_connection.py').read_text().replace('from backend.voice_research.review_rules import validate_cut_review','# Validation embedded above')
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('### Restore review connection\\nRestoring only the seven source-audio cuts and feedback connection from your verified Drive backup. Your existing form below is left in place. No models are installed and no training starts.'))
try:
    alignment,report=restore_review_files(RECEIPT,publish)
    nonce,state=register_review_connection(alignment,recover_existing_form=True)
    publish({**report,'phase':'existing_review_save_connection_restored','registered_cases':7,'existing_saved_reviews':len(state['records'])})
    probe=secrets.token_urlsafe(18)
    display(Javascript("""(async()=>{try{const response=await google.colab.kernel.invokeFunction('anyanwu.expanded_cuts.check',PROBE,{});let r=response.data['application/json'];if(typeof r==='string')r=JSON.parse(r);window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:{phase:'existing_form_connection_verified',callback_verified:r.ready,cases:r.candidate_count,existing_saved_reviews:r.review_count,current_form_untouched:true,feedback_records_changed:r.feedback_records_changed}},'https://colab.research.google.com');}catch(error){window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:{phase:'review_callback_check_failed',current_form_untouched:true}},'https://colab.research.google.com');}})();""".replace('PROBE',json.dumps([probe]))))
    display(Markdown('**Connection restored.** Return to your existing choices below and press **Save feedback** again.'))
except Exception as error:
    report={'phase':'review_connection_recovery_stopped','error_type':type(error).__name__,'error':str(error)[:500],'current_form_untouched':True}
    publish(report)
    print(json.dumps(report,indent=2))
'''
cell='# Anyanwu review connection recovery: preserve existing form choices\n'+source+'\nRECEIPT='+repr(receipt)+'\n'+tail
if sys.argv[1:] in (['read-feedback'],['freeze-feedback']):
    cell='''# Anyanwu review connection recovery: read saved feedback status only
from pathlib import Path
import hashlib,json
from IPython.display import Javascript,display,Markdown
folder=Path('/content/anyanwu-bbc-expansion-v1')
drive=Path('/content/drive/MyDrive/Anyanwu Voice Research/datasets/bbc-expansion-v1/human-reviews.json')
report={'phase':'review_feedback_status','drive_reviews_exists':drive.is_file(),'training_started':False,'current_form_untouched':True}
if drive.is_file():
    if drive.stat().st_size>64*1024: raise RuntimeError('Review status file cap')
    data=drive.read_bytes()
    state=json.loads(data)
    report.update(saved_reviews=len(state['records']),review_file_sha256=hashlib.sha256(data).hexdigest(),
       reviews=[{k:v for k,v in row.items() if k not in ('notes',)} for row in state['records'].values()])
else: report['saved_reviews']=0
display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':report})+', "https://colab.research.google.com");'))
display(Markdown('Saved Drive reviews: **'+str(report['saved_reviews'])+'**. The existing review form below was not rerun.'))
'''
    if sys.argv[1:]==['freeze-feedback']:
        marker="display(Javascript('window.top.postMessage('"
        freeze='''
if report['saved_reviews']!=7: raise RuntimeError('Seven submitted records required before snapshot')
alignment=json.loads(Path('/content/anyanwu-expanded-caption-joint-alignment-v1/alignment-summary.json').read_text())
allowed={c['id']:c['candidate_audio_sha256'] for c in alignment['cases'] if not c.get('timing_warnings') and c.get('cloud_audio_path')}
if set(state['records'])!=set(allowed) or any(v['audio_sha256']!=allowed[k] for k,v in state['records'].items()): raise RuntimeError('Review snapshot audio IDs/hashes changed')
snapshot=drive.parent/'human-reviews.submitted-v1.json'
if snapshot.exists() and snapshot.read_bytes()!=data: raise RuntimeError('A different submitted snapshot exists; no overwrite')
if not snapshot.exists():
    temporary=drive.parent/'human-reviews.submitted-v1.tmp'
    temporary.write_bytes(data)
    temporary.replace(snapshot)
if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=report['review_file_sha256']: raise RuntimeError('Submitted snapshot checksum mismatch')
report.update(phase='seven_submitted_reviews_snapshot_verified',snapshot_path=str(snapshot),snapshot_sha256=report['review_file_sha256'],
              accepted=sum(v['status']=='accepted_by_user_for_pilot' for v in state['records'].values()),
              held=sum(v['status']=='held_by_user' for v in state['records'].values()),
              unchecked_fields=sum(v.get(k)=='unreviewed' for v in state['records'].values() for k in ('words_match','complete_edges','clean_female_voice')))
'''
        cell=cell.replace(marker,freeze+'\n'+marker)
compile(cell,'review-recovery-cell.py','exec')
(folder/'review-recovery-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-review-recovery.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared recovery in the unused publisher-preview cell; existing review form will not be rerun.')
