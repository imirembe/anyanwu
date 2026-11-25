"""Build the ready-to-run expanded GPU experiment without local model imports."""
from pathlib import Path
import base64
import hashlib
import json

root=Path(__file__).resolve().parents[1];folder=root/'.build/VoiceResearch/training/waxal-female5-expanded-lora-v2'
plan=json.loads((folder/'plan.json').read_text())
worker=(root/'backend/voice_research/expanded_train_worker.py').read_text()
adapter=(root/'backend/voice_research/flash_adapter.py').read_text()
modules=['backend/voice_research/expanded_train_restore.py','backend/voice_research/expanded_train_setup.py','backend/voice_research/expanded_train_backup.py']
source='\n'.join((root/name).read_text() for name in modules)
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Anyanwụ: expanded female Igbo training\\nOne fixed60-update run from the preserved current adapter.34 training utterances,4 original validation utterances. Four fresh generated-test texts remain excluded. No model or source download to the laptop, purchase or automatic CPU fallback.'))
report={'optimizer_updates':0,'training_started':False}
try:
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if gpu.returncode or len(gpu.stdout.strip().splitlines())!=1 or int(gpu.stdout.strip().rsplit(',',1)[1])<14000:
        raise RuntimeError('A hosted GPU with at least14000MiB is required. No install, model download, CPU fallback or purchase started.')
    publish({'phase':'expanded_training_gpu_verified','gpu':gpu.stdout.strip(),'optimizer_updates':0})
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        publish({'phase':'expanded_training_waiting_for_drive_authorization','optimizer_updates':0})
        drive.mount('/content/drive',timeout_ms=300000)
    report=run_expanded_training(PLAN,WORKER,ADAPTER,publish)
    if report.get('optimizer_updates')==60 and report.get('adapter_reloaded'):
        report['backup']=backup_expanded_training()
        display(Markdown('**60 real additional updates completed and the new adapter is backed up in Drive.** The earlier voice remains preserved. Generated-speech listening is still needed to judge this candidate.'))
    publish(report)
except Exception as error:
    report.update(phase='expanded_training_or_backup_stopped',error_type=type(error).__name__,error=str(error)[:500])
    publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu bounded cloud diagnostics: train expanded female Igbo adapter\n'+source+'\nPLAN='+repr(plan)+'\nWORKER='+repr(worker)+'\nADAPTER='+repr(adapter)+'\n'+tail
for name,text in [('worker',worker),('adapter',adapter),('cell',cell)]:compile(text,name,'exec')
(folder/'training-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
metadata={'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'},'accelerator':'GPU','colab':{'gpuType':'T4'}}
intro='''# Anyanwụ: expanded female Igbo adapter

Select a hosted T4 GPU in this same notebook and run the cell once. It first checks GPU availability, then connects Drive, verifies both small source archives and the exact submitted feedback, and restores only permitted files. All media/models stay in Colab/Drive. No CPU fallback, paid upgrade or laptop model download is started.

Fixed plan:60 additional updates from the preserved v1 adapter,34 training utterances and four unchanged diagnostic validation utterances. Four fresh texts are reserved for a separate eight-clip current/candidate comparison. No quality improvement is assumed before listening.

Limits:30 minutes setup+training,15 minutes optimiser work,20GiB cloud-disk growth/5GiB floor,9GiB worker RSS,12GiB GPU reservation,4GiB selected models,16MiB adapter and32MiB Drive backup. Do not repeatedly run after a failure; inspect its receipt first. Never run the old audition cells or Run all in the existing mixed-purpose notebook.

The free GPU request on23 November was rejected by Colab's usage limit. This notebook is prepared, not evidence of a completed training run. No paid compute was purchased.
'''
notebook={'nbformat':4,'nbformat_minor':5,'metadata':metadata,'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},
  {'cell_type':'code','metadata':{},'source':cell.splitlines(True),'execution_count':None,'outputs':[]}]}
for path in (folder/'Anyanwu-Expanded-Female-Training.ipynb',root/'Voice Research/Anyanwu-Expanded-Female-Training.ipynb'):
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
receipt={'plan_sha256':hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest(),'cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),
  'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in modules+['backend/voice_research/expanded_train_worker.py','backend/voice_research/flash_adapter.py']},
  'registered_updates':60,'warm_start':plan['starting_adapter_sha256'],'initial_execution_status':'GPU quota denied before training; zero optimizer updates'}
(folder/'source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Prepared the isolated60-update GPU run and visible recovery notebook; no model imports/downloads on the Mac.')
