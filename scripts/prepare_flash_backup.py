"""Prepare an adapter-only cloud archive and optional Colab Drive authorisation."""
from pathlib import Path
import base64
import json
import sys

root=Path(__file__).resolve().parents[1]
source=json.loads((root/'.build/VoiceResearch/bbc-igbo-female-sources-v1/sources.json').read_text())
provenance={'permission':source['permission'],'sources':[{k:s[k] for k in ('id','url','title','video_id')} for s in source['sources']]}
code=(root/'backend/voice_research/flash_backup.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Preserve the trained adapter\\nPreparing a verified archive under 2 MiB containing only the adapter and provenance. Base weights and BBC media stay out. The archive can be copied directly to Google Drive without a laptop download.'))
try:
    report=backup_flash_adapter(SOURCES,publish,authorise=AUTHORISE)
except Exception as error:
    report={'phase':'adapter_backup_stopped','error_type':type(error).__name__,'error':str(error)[:500],'durable_backup_verified':False}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu bounded cloud diagnostics: preserve trained adapter\n'+code+'\nSOURCES='+repr(provenance)+'\nAUTHORISE='+repr(sys.argv[1:]==['authorise-drive'])+'\n'+tail
compile(cell,'flash-backup-cell.py','exec')
folder=root/'.build/VoiceResearch/training/anyanwu-female-v1'
(folder/'flash-backup-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared adapter-only cloud backup; Google Drive access uses normal Colab authorisation.')
