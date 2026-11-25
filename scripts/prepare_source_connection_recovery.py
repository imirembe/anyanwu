"""Reconnect a still-open source form without rerendering its choices."""
from pathlib import Path
import base64
import json

root=Path(__file__).resolve().parents[1]
folder=root/'.build/VoiceResearch/datasets/waxal-female5-expansion-v1'
receipt=json.loads((folder/'backup-result.json').read_text())
module=(root/'backend/voice_research/source_expansion_review.py').read_text()
cell='''# Anyanwu review connection recovery: restore existing source form connection
import platform
from pathlib import Path
if platform.system()!='Linux' or not Path('/content').is_dir():
    raise RuntimeError('Hosted Colab only; no local audio, models or new runtime')
from google.colab import drive
from IPython.display import Javascript,display,Markdown
import json
def publish_recovery(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('### Restore review connection\\nRestores only the verified source WAVs and review metadata from Drive, then reconnects saving to the still-open form below. Your selections and page are left intact. No model install, training or GPU work.'))
if not Path('/content/drive/MyDrive').is_dir():
    publish_recovery({'phase':'source_review_waiting_for_drive_authorization','form_untouched':True,'training_started':False})
    try:drive.mount('/content/drive',timeout_ms=300000)
    except Exception as error:
        publish_recovery({'phase':'source_review_drive_authorization_stopped','error':str(error)[:400],'form_untouched':True,'training_started':False})
        raise
'''+module+'\nRECEIPT='+repr(receipt)+'''
restore_source_expansion(RECEIPT)
nonce,state,counts=register_source_review_connection(recover_existing_form=True)
report={'phase':'existing_source_form_callbacks_registered','source_count':14,'drive_mounted':Path('/content/drive/MyDrive').is_dir(),'saved_reviews':len(state),'form_untouched':True,'feedback_changed':False,'training_started':False}
print(json.dumps(report,indent=2))
display(Javascript("""(async()=>{let value={phase:'existing_source_form_connection_check_failed',form_untouched:true};try{const reply=await google.colab.kernel.invokeFunction('anyanwu.waxal_expansion.check',[{read_only_health:true}],{});let result=reply.data['application/json'];if(typeof result==='string')result=JSON.parse(result);value={phase:result.ready?'existing_source_form_connection_verified':'existing_source_form_connection_check_failed',callback_verified:result.ready,source_count:result.source_count,saved_reviews:result.saved_records,session_bound:result.session_bound,form_untouched:true,feedback_changed:result.feedback_changed,training_started:false};}catch(error){value.error=String(error.message||error).slice(0,250);}window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:value},'https://colab.research.google.com');})();"""))
'''
compile(cell,'source-connection-recovery.py','exec')
(folder/'connection-recovery-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-review-recovery.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared existing-form callback recovery; no form rerender or feedback submission.')
