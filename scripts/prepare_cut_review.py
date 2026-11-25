"""Persist expanded source data to Drive, then show the actual listening form."""
from pathlib import Path
import base64
import json

root=Path(__file__).resolve().parents[1]
folder=root/'.build/VoiceResearch/training/anyanwu-female-v1/expansion-v1'
captions=json.loads((folder/'caption-text.json').read_text())
receipt=json.loads((folder/'dataset-backup-result.json').read_text())
source=(root/'backend/voice_research/restore_review.py').read_text()+'\n'+(root/'backend/voice_research/review_rules.py').read_text()+'\n'+(root/'backend/voice_research/review_connection.py').read_text()+'\n'+(root/'backend/voice_research/cut_review.py').read_text()
source=source.replace('from backend.voice_research.review_rules import validate_cut_review','# Bounded rules embedded above').replace('from backend.voice_research.review_connection import register_review_connection','# Connection embedded above')
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Review the expanded speech cuts\\nLoading only the seven small review clips from the verified Drive backup. No models or training setup are needed. Saved feedback is restored automatically.'))
try:
    alignment,restored=restore_review_files(RECEIPT,publish)
    backup=RECEIPT
    publish(restored)
    form=show_cut_review(alignment,backup,publish)
    (Path('/content/anyanwu-bbc-expansion-v1')/'listening-form-result.json').write_text(json.dumps(form,ensure_ascii=False,indent=2)+'\\n')
    print(json.dumps(form,ensure_ascii=False,indent=2))
except Exception as error:
    report={'phase':'expanded_listening_review_stopped','error_type':type(error).__name__,'error':str(error)[:500]}
    publish(report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu bounded cloud diagnostics: restored review with protected drafts\n'+source+'\nRECEIPT='+repr(receipt)+'\n'+tail
compile(cell,'expanded-listening-review-cell.py','exec')
(folder/'listening-review-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared Drive-restorable review with bounded browser drafts, no models/training and no full-corpus extraction.')
