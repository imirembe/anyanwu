"""Small standalone Colab review recovery; no model, training or local audio."""
from pathlib import Path
import base64
import json

root=Path(__file__).resolve().parents[1];folder=root/'.build/VoiceResearch/datasets/waxal-female5-expansion-v1'
receipt=json.loads((folder/'backup-result.json').read_text())
review=(root/'backend/voice_research/source_expansion_review.py').read_text()
cell='''# Anyanwu review connection recovery: expanded female source recordings
import platform
from pathlib import Path
if platform.system()!='Linux' or not Path('/content').is_dir():
    raise RuntimeError('Use hosted Colab; do not download this source sample to the laptop')
from google.colab import drive
if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive',timeout_ms=300000)
'''+review+'\nRECEIPT='+repr(receipt)+'''
restore_source_expansion(RECEIPT)
show_source_expansion(RECEIPT)
'''
compile(cell,'source-expansion-recovery.py','exec')
markdown='''# Anyanwụ: review the new female Igbo recordings

Fourteen whole published recordings, about two minutes. Listen across all three pages. Check words, complete beginnings/endings and clean single female speech. You may click **Accept all three checks and save** after listening, or save individual checks/notes. Any unchecked or rejected recording stays out of future training.

Run the cell in a hosted Colab runtime, connecting Google Drive if asked. CPU is sufficient for saved-audio review. The cell verifies and restores the ~5 MB Drive archive and reconnects feedback saves, preserving saved reviews and matching browser drafts. No models, GPU computation, training, new downloads to the laptop or purchases.

Source: Google WAXAL / Media Trust Igbo TTS, CC BY-SA 4.0. Whole lossy preview recordings; original text retained. This is source review, not generated voice evaluation.
'''
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}},'cells':[
 {'cell_type':'markdown','metadata':{},'source':markdown.splitlines(True)},
 {'cell_type':'code','metadata':{},'source':cell.splitlines(True),'execution_count':None,'outputs':[]}]}
for destination in (folder/'Anyanwu-New-Female-Recordings-Review.ipynb',root/'Voice Research/Anyanwu-New-Female-Recordings-Review.ipynb'):
    destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
# Standalone notebook renders its own form. The existing notebook's recovery
# action must only reconnect its still-open form, preserving every selection.
import prepare_source_connection_recovery
print('Created standalone review notebook; staged connection-only recovery for the existing form.')
