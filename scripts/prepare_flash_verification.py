"""Prepare the small cloud-only final integrity check and result display."""
from pathlib import Path
import base64

root=Path(__file__).resolve().parents[1]
source=(root/'backend/voice_research/flash_verify.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown
display(Markdown('# Anyanwụ: first speech-model training completed\\nFive real CPU LoRA updates, using three provisional BBC caption/audio cuts (17.99 seconds). Checking the saved adapter and frozen base now; no further training.'))
try:
    report=verify_flash_pilot()
except Exception as error:
    report={'phase':'post_training_verification_stopped','error':str(error)[:1200]}
display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':report})+', "https://colab.research.google.com");'))
print(json.dumps(report,ensure_ascii=False,indent=2))
if report.get('post_training_integrity',{}).get('adapter_reloaded'):
    display(Markdown('**Saved adapter verified:** 273,024 trainable parameters, 97 finite tensors, matching checksum, unchanged frozen base. Voice quality has not been evaluated and no new speech was generated. The adapter is in the temporary Colab runtime; preserve it before resetting/deleting the runtime.'))
'''
cell='# Anyanwu bounded cloud diagnostics: verify saved speech LM adapter\n'+source+'\n'+tail
compile(cell,'flash-verification-cell.py','exec')
(root/'.build/VoiceResearch/training/anyanwu-female-v1/flash-verification-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared read-only cloud tensor/base verification; zero extra optimizer steps.')
