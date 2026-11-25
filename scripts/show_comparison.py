"""Prepare a Drive backup and durable Colab audio review without local media."""
from pathlib import Path
import base64
import json
import sys

root=Path(__file__).resolve().parents[1]
folder=root/'.build/VoiceResearch/auditions/waxal-female5-adapter-comparison-v1'
archive=(root/'backend/voice_research/comparison_archive.py').read_text()
review=(root/'backend/voice_research/comparison_review.py').read_text()
extras={'runtime-source-receipt.json':(folder/'runtime-source-receipt.json').read_text(),
        'COSYVOICE-LICENSE':(folder/'reference/LICENSE').read_text(),
        'MATCHA-LICENSE':(folder/'reference/MATCHA-LICENSE').read_text(),'comparison_review.py':review}
tail='''
from IPython.display import Javascript,display
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
    backup=backup_comparison(EXTRAS)
    publish(backup)
    print('Six generated clips verified and saved to Google Drive. No model weights downloaded to the laptop.')
    show_comparison(backup)
except Exception as error:
    publish({'phase':'comparison_listening_stopped','error':str(error)[:500]})
    raise
'''
if sys.argv[1:]==['restore-feedback']:
    receipt=json.loads((folder/'backup-result.json').read_text())
    tail=tail.replace('backup=backup_comparison(EXTRAS)','backup=RECEIPT')
    cell='# Anyanwu bounded cloud diagnostics: restore comparison feedback confirmation\n'+review+'\nRECEIPT='+repr(receipt)+'\n'+tail
else:
    cell='# Anyanwu bounded cloud diagnostics: listen to female voice comparison\n'+archive+'\n'+review+'\nEXTRAS='+repr(extras)+'\n'+tail
compile(cell,'comparison-listen-cell.py','exec')
(folder/'listening-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared six-player comparison with hash-verified Drive archive and draft-protected feedback.')
