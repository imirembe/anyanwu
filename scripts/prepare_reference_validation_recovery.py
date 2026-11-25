"""Prepare model-free recovery only after the new archive is verified."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'.build/VoiceResearch/auditions/conversational-reference-validation-v2'


def prepare(*, original_account=False):
    receipt=json.loads((FOLDER/'backup-result.json').read_text())
    if not receipt.get('durable_backup_verified') or receipt.get('audio_clips')!=12:
        raise RuntimeError('Verified twelve-clip archive required')
    source=(FOLDER/('validation_review.admin.py' if original_account else 'validation_review.py')).read_text()
    if original_account and '/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research' in source:
        raise RuntimeError('Original-account recovery must use the verified single project root')
    common='''
import importlib.util,platform
from IPython.display import Javascript,display
def publish_validation_recovery(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab only; no laptop audio/models')
if not Path('/content/drive/MyDrive').is_dir():
 publish_validation_recovery({'phase':'reference_validation_review_waiting_for_drive','models_needed':False})
 from google.colab import drive
 drive.mount('/content/drive',timeout_ms=300000)
restore_reference_validation_listening(RECEIPT)
'''
    cells={
        'connection':source+'\nRECEIPT='+repr(receipt)+'\n'+common+'''
_,state,_=register_reference_validation_connection(recover_existing_form=True)
display(Javascript("""(async()=>{let ready=false;try{const reply=await google.colab.kernel.invokeFunction('anyanwu.reference_validation.check',[{read_only_health:true}],{});let value=reply.data['application/json'];if(typeof value==='string')value=JSON.parse(value);ready=value.ready&&value.pairs===6&&!value.feedback_changed;}catch(e){}window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:{phase:'reference_validation_connection_recovered',callback_verified:ready,form_untouched:true,feedback_changed:false,training_started:false}},'https://colab.research.google.com');})();"""))
print('Existing Save buttons reconnected. Original choices stay untouched. No models, generation or training.')
''',
        'review':source+'\nRECEIPT='+repr(receipt)+'\n'+common+'\nshow_reference_validation(RECEIPT)\n'}
    for name,cell in cells.items():
        compile(cell,'validation-'+name,'exec');(FOLDER/('audio-'+name+'-cell.py')).write_text(cell)
        notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':[
            {'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · six-pair validation '+name+'\n','Hosted CPU and research Drive only. Uses the existing matched-volume clips; no weights, synthesis or training. '+('Reconnects existing Save buttons without replacing choices.' if name=='connection' else 'Restores the six-page audio and feedback panel, saved reviews and valid browser drafts.')+'\n']},
            {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
        filename='Anyanwu-Reference-Validation-v2-'+('Connection' if name=='connection' else 'Review')+'.ipynb'
        (ROOT/'Voice Research'/filename).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    result={'phase':'reference_validation_audio_recovery_prepared','archive_sha256':receipt['archive_sha256'],
        'original_account':original_account,
        'review_cell_sha256':hashlib.sha256(cells['review'].encode()).hexdigest(),'connection_cell_sha256':hashlib.sha256(cells['connection'].encode()).hexdigest(),
        'models_needed':False,'new_generation':False,'training':False}
    (FOLDER/'audio-recovery-source-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
