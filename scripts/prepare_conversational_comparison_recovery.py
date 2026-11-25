"""Build audio-only review and connection-only recovery after verified generation."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def prepare():
    folder=ROOT/'.build/VoiceResearch/auditions/conversational-voice-comparison-v1'
    receipt=json.loads((folder/'backup-result.json').read_text())
    if not receipt.get('durable_backup_verified') or receipt.get('audio_clips')!=8:raise RuntimeError('Verified eight-clip backup required')
    source=(folder/'comparison_review.py').read_text()
    common='''
import importlib.util,platform
from IPython.display import Javascript,display
def publish(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab CPU only; no laptop media/models')
if not Path('/content/drive/MyDrive').is_dir():
 publish({'phase':'conversational_comparison_review_waiting_for_drive','models_needed':False})
 from google.colab import drive
 drive.mount('/content/drive',timeout_ms=300000)
restore_conversational_listening(RECEIPT)
'''
    cells={
        'connection':source+'\nRECEIPT='+repr(receipt)+'\n'+common+'''
_,state,_=register_conversational_comparison_connection(recover_existing_form=True)
display(Javascript("""(async()=>{let ready=false;try{const reply=await google.colab.kernel.invokeFunction('anyanwu.conversational_comparison.check',[{read_only_health:true}],{});let value=reply.data['application/json'];if(typeof value==='string')value=JSON.parse(value);ready=value.ready&&value.pairs===4&&!value.feedback_changed;}catch(e){}window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:{phase:'conversational_comparison_connection_recovered',callback_verified:ready,form_untouched:true,feedback_changed:false,training_started:false}},'https://colab.research.google.com');})();"""))
print('Existing Save connection restored; original form and user choices are untouched. No model install or generation.')
''',
        'review':source+'\nRECEIPT='+repr(receipt)+'\n'+common+'\nshow_conversational_comparison(RECEIPT)\n'}
    for name,cell in cells.items():
        compile(cell,'conversation-'+name,'exec');(folder/('audio-'+name+'-cell.py')).write_text(cell)
        notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
                  'cells':[{'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · conversational voice '+name+'\n','Use hosted CPU and connect research Drive. No speech models, synthesis or training. '+('This reconnects the existing form without changing it.' if name=='connection' else 'Restores the saved clips and reviews; browser drafts stay protected.')+'\n']},
                           {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
        filename='Anyanwu-Conversational-Voice-'+('Connection' if name=='connection' else 'Review')+'.ipynb'
        (ROOT/'Voice Research'/filename).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    result={'phase':'conversational_comparison_audio_recovery_prepared','archive_sha256':receipt['archive_sha256'],
            'review_cell_sha256':hashlib.sha256(cells['review'].encode()).hexdigest(),
            'connection_cell_sha256':hashlib.sha256(cells['connection'].encode()).hexdigest(),
            'models_needed':False,'new_generation':False,'training':False}
    (folder/'audio-recovery-source-receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
