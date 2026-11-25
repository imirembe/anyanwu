"""Prepare the model-free review/recovery cell for the two selected transcripts."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'.build/VoiceResearch/conversational-igbo-v1'
module = (ROOT/'backend/voice_research/conversational_transcript_review.py').read_text()
tail = '''
import importlib.util,platform
from IPython.display import Javascript,display
def publish(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
 if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab only; no laptop audio/models')
 if not Path('/content/drive/MyDrive').is_dir():
  publish({'phase':'conversational_asr_review_waiting_for_drive','models_needed':False})
  from google.colab import drive
  drive.mount('/content/drive',timeout_ms=300000)
 show_transcript_review()
except Exception as error:
 publish({'phase':'conversational_asr_review_stopped','error_type':type(error).__name__,'error':str(error)[:350],'training_started':False})
 raise
'''
cell = '# Anyanwu selected transcript listening and Drive review — no models or training\n'+module+'\n'+tail
compile(cell, 'selected-transcript-review', 'exec')
(folder/'transcription-review-cell.py').write_text(cell)
nb = {'nbformat':4, 'nbformat_minor':5, 'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
      'cells':[{'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · review selected conversational transcripts\n','Hosted CPU, research Drive only. Restores the saved audio/drafts, reconnects Save, and preserves browser drafts. No ASR or speech-model downloads; no training.\n']},
               {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
(ROOT/'Voice Research/Anyanwu-Selected-Transcript-Review.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
receipt={'phase':'conversational_asr_review_prepared','cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),
         'models_needed':False,'training_started':False,'transcript_approval_submitted_by_agent':False}
(folder/'transcription-review-source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
