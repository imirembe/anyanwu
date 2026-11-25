"""Prepare CPU-only callback repair while leaving the existing four cards intact."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.build/VoiceResearch/auditions/waxal-female5-expanded-comparison-v2'
receipt=json.loads((folder/'backup-result.json').read_text())
if receipt['audio_clips']!=8 or not receipt['durable_backup_verified'] or receipt['archive_sha256']!='9c005ba6ff3d23c0d61e91880d33ea23ccb6d2b4bc239cb2258fc37033f832c4':raise RuntimeError('Exact verified audio archive required')
review=(ROOT/'backend/voice_research/expanded_comparison_review.py').read_text()
tail='''
import importlib.util,platform
if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab CPU only; no laptop media or models')
from google.colab import drive
from IPython.display import Javascript,display
def publish(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
if not Path('/content/drive/MyDrive').is_dir():
 publish({'phase':'existing_comparison_connection_waiting_for_drive','form_untouched':True,'feedback_changed':False})
 drive.mount('/content/drive',timeout_ms=300000)
restore_comparison(RECEIPT)
nonce,state,counts=register_expanded_comparison_connection(recover_existing_form=True)
publish({'phase':'existing_comparison_callbacks_registered','form_untouched':True,'feedback_changed':False,'optimizer_updates_this_stage':0,**counts(state)})
display(Javascript("""(async()=>{let value={phase:'existing_comparison_connection_check_failed',form_untouched:true,feedback_changed:false};try{const reply=await google.colab.kernel.invokeFunction('anyanwu.expanded_comparison.check',[{read_only_health:true}],{});let r=reply.data['application/json'];if(typeof r==='string')r=JSON.parse(r);value={phase:r.ready?'existing_comparison_connection_verified':'existing_comparison_connection_check_failed',callback_verified:r.ready,pairs:r.pairs,saved_records:r.saved_records,complete_reviews:r.complete_reviews,session_bound:r.session_bound,form_untouched:true,feedback_changed:r.feedback_changed,automatic_training:false};}catch(error){value.error=String(error.message||error).slice(0,220);}window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:value},'https://colab.research.google.com');})();"""))
print('Save connection restored. Retry Save feedback on your EXISTING cards below. No cards, choices or browser drafts were replaced; no generation or training ran.')
'''
cell='# Anyanwu CPU-only repair: reconnect the existing expanded voice review\n'+review+'\nRECEIPT='+repr(receipt)+'\n'+tail
compile(cell,'comparison-connection-repair','exec')
(folder/'connection-repair-cell.py').write_text(cell)
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':[{'cell_type':'markdown','metadata':{},'source':['# Restore the existing voice-review save connection\nHosted CPU is sufficient. Restores only the verified small audio archive and registers callbacks for the existing cards. It never renders a replacement form, generates speech or trains.\n']},{'cell_type':'code','metadata':{},'source':cell.splitlines(True),'execution_count':None,'outputs':[]}]}
(ROOT/'Voice Research/Anyanwu-Expanded-Voice-Connection.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
result={'phase':'connection_repair_prepared','cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),'review_source_sha256':hashlib.sha256(review.encode()).hexdigest(),'archive_sha256':receipt['archive_sha256'],'form_replaced':False,'generation':False,'training':False}
(folder/'connection-repair-source-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
