"""Create a small audio-review recovery notebook; no models or GPU required."""
from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
folder=root/'.build/VoiceResearch/auditions/waxal-female5-adapter-comparison-v1'
receipt=json.loads((folder/'backup-result.json').read_text())
review=(root/'backend/voice_research/comparison_review.py').read_text()
code='''# Anyanwu audio-only comparison recovery
import platform
from pathlib import Path
if platform.system()!='Linux' or not Path('/content').is_dir():
    raise RuntimeError('Use a hosted Colab runtime; no laptop audio/model download')
from google.colab import drive
if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive',timeout_ms=300000)
'''+review+'\nRECEIPT='+repr(receipt)+'''
restore_comparison(RECEIPT)
show_comparison(RECEIPT)
'''
compile(code,'comparison-recovery.py','exec')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}},'cells':[
 {'cell_type':'markdown','metadata':{},'source':('# Anyanwụ: female Igbo voice comparison\n\nRun the cell below in a hosted Colab runtime and connect Google Drive when asked. CPU is sufficient for **playing and reviewing** these six saved clips. This restores the verified ~1 MB archive directly from Drive, checks every file, and reconnects feedback saving. No models, new speech generation, training or laptop downloads.\n\nListen to each Original/Trained pair. Save feedback on each card. Browser drafts and saved Drive choices are restored where available. Ratings concern the generated voice, not the source recording.\n\nArchive: `MyDrive/Anyanwu Voice Research/auditions/waxal-female5-adapter-comparison-v1/anyanwu-female-comparison-v1.zip`.').splitlines(True)},
 {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
for path in (folder/'Anyanwu-Female-Voice-Comparison.ipynb',root/'Voice Research/Anyanwu-Female-Voice-Comparison.ipynb'):
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print('Created a small audio-only recovery notebook in Voice Research; models stay in Colab.')
