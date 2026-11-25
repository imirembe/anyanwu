"""Prepare offline repair and publish only explicitly selected caption previews."""
from pathlib import Path
import base64

root=Path(__file__).resolve().parents[1]
worker=(root/'backend/voice_research/expansion_review.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown,Audio
import base64
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Expanded BBC speech review\\nFive new 20-second windows and their source subtitle images. Smaller preview sheets use the existing frames only. These are candidate data, not a new trained voice or verified word boundaries.'))
try:
    report=build_expansion_review()
    for row in report['sources']:
        for sheet in row['sheets']:
            publish({'phase':'caption_sheet','key':row['id']+'/'+str(sheet['index']).zfill(2),
                     'mime':'image/jpeg','base64':base64.b64encode(Path(sheet['path']).read_bytes()).decode(),
                     'source_seconds':sheet['nominal_source_seconds'],'sha256':sheet['sha256']})
except Exception as error:
    report={'phase':'expansion_review_stopped','error_type':type(error).__name__,'error':str(error)[:350]}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
for row in report.get('sources',[]):
    display(Markdown('### '+row['id']+' — source 24–44 seconds (nominal)'))
    display(Audio(filename=row['audio']['path']))
'''
cell='# Anyanwu bounded cloud diagnostics: expanded speech caption review\n'+worker+'\n'+tail
compile(cell,'expansion-review-cell.py','exec')
(root/'.build/VoiceResearch/training/anyanwu-female-v1/expansion-v1/review-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared cloud-only two-frame caption sheets; no new downloads/training.')
