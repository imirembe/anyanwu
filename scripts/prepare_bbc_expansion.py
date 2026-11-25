"""Build the next separately bounded BBC corpus acquisition cell."""
from pathlib import Path
import base64
import json

root=Path(__file__).resolve().parents[1]
sources=json.loads((root/'.build/VoiceResearch/bbc-igbo-female-sources-v1/sources.json').read_text())
plan={'id':'anyanwu-bbc-expansion-v1','purpose':'Expand published speech/caption candidates after the five-update technical pilot',
      'sources':[{**{k:s[k] for k in ('id','url','video_id')},'source_duration_seconds':s['duration_seconds'],
                  'start_seconds':24,'duration_seconds':20} for s in sources['sources']],
      'permission':sources['permission'],'caption_speech_match':'User confirmed the BBC source subtitles; new copied text remains AI-transcribed until reviewed',
      'selection':'Fixed next non-overlapping 20-second windows; no score-guided selection or full-video fallback',
      'limits':{'cloud_only':True,'body_mib':64,'source_body_mib':12,'requests':100,'minutes':10,'free_floor_gib':5,
                'sources':5,'seconds_per_source':20,'frames_per_source':20,'caption_sheets_per_source':5},
      'frozen_artifacts':'Original audio/caption/alignment/training results unchanged; no additional optimisation'}
transport=(root/'backend/voice_research/bbc_probe.py').read_text().split('\nif __name__ == "__main__":')[0]
sample=(root/'backend/voice_research/bbc_samples.py').read_text().replace('from backend.voice_research.bbc_probe import SOURCES, Transport, attributes, checked_url, cloud_guard','# Bounded helpers embedded above')
sheet=(root/'backend/voice_research/caption_capture.py').read_text().split('\ndef contact_sheet')[1].split('\ndef capture')[0]
sheet='def contact_sheet'+sheet
worker=(root/'backend/voice_research/bbc_expansion.py').read_text()
for line in ('from backend.voice_research.bbc_probe import Transport, checked_url, cloud_guard',
             'from backend.voice_research.bbc_samples import master_selection, playlist_window, ffmpeg',
             'from backend.voice_research.caption_capture import contact_sheet'):
    worker=worker.replace(line,'# Bounded helpers embedded above')
tail='''
from IPython.display import Javascript,display,Markdown,Audio
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Next published Igbo speech sample\\nFive new 20-second BBC windows (nominal 24–44 seconds). Audio and subtitle frames remain in Colab. The trained adapter is backed up separately. This collection does not perform more training or certify voice quality.'))
try:
    report=expand_bbc_samples(PLAN,publish)
except Exception as error:
    report={'phase':'bbc_expansion_stopped','error_type':type(error).__name__,'error':str(error)[:250],'optimizer_updates_this_stage':0}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
for row in report.get('sources',[]):
    if row['status']=='new_audio_and_caption_frames_ready_for_text_capture':
        display(Markdown('### '+row['id']+' — new 20-second window'))
        display(Audio(filename=str(EXPANSION/row['id']/'speech-20s.wav')))
'''
cell='# Anyanwu bounded cloud diagnostics: expand BBC speech candidates\n'+transport+'\n'+sample+'\n'+sheet+'\n'+worker+'\nPLAN='+repr(plan)+'\n'+tail
compile(cell,'bbc-expansion-cell.py','exec')
folder=root/'.build/VoiceResearch/training/anyanwu-female-v1/expansion-v1'
folder.mkdir(exist_ok=True)
(folder/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
(folder/'expansion-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared 100-second cloud-only expansion: fixed 24–44s windows, 64MiB transfer cap, no further training.')
