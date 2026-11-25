"""Prepare offline cloud alignment of the five eligible source captions."""
from pathlib import Path
import base64
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.build/VoiceResearch/training/anyanwu-female-v1'
captions=json.loads((folder/'caption-text.json').read_text())
plan={'id':'anyanwu-caption-alignment-v1','captions':captions['entries'],
      'caption_text_sha256':hashlib.sha256((folder/'caption-text.json').read_bytes()).hexdigest(),
      'source_caption_review':captions['user_caption_review'],
      'asr_model_sha256':'8ce340ada22435d189908a8af67e3bb04899b6f2f5329d036248b9c3e38d2b50',
      'search_margin_seconds':1.0,'policy':{'cloud_only':True,'offline':True,'network_requests':0,'new_model_downloads':0,'maximum_captions':5,'minutes':10,'rss_gib':6,'output_mib':12,
                'candidate_selection':'Display-bounded captions with high visual-transcription confidence and no marked ambiguity',
                'search_window':'First display observation minus one second through last observation plus one second, bounded by existing WAV',
                'timing_flags':'Mean token probability below 0.10, path touches search-window edge, or implausible duration; do not retune on these cases',
                'review':'Machine cuts are not automatically training-ready; source subtitle match remains user-confirmed'}}
plan_filename='alignment-plan.json'
if sys.argv[1:]==['refine-context']:
    prior=json.loads((folder/'alignment-summary.json').read_text())
    plan['id']='anyanwu-caption-alignment-v2-context'
    plan['search_margin_seconds']=2.0
    plan['selected_ids']=[c['id'] for c in prior['cases'] if 'alignment_touches_search_window_edge' in c.get('timing_warnings',[])]
    plan['reason']='One declared timing repair: expand existing WAV context for window-edge flags. Original results and score thresholds unchanged; not a new language benchmark.'
    plan['policy']['search_window']='Two-second margins using existing source WAVs only; genuine sample edges remain flagged'
    plan_filename='alignment-context-plan.json'
(folder/plan_filename).write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
rules=(ROOT/'backend/voice_research/alignment_rules.py').read_text()
worker=(ROOT/'backend/voice_research/alignment_worker.py').read_text().replace('from backend.voice_research.alignment_rules import alignment_text,eligible_caption,evaluate_path,caption_token_ranges','# Pure alignment rules embedded above.')
worker=rules+'\n'+worker
compile(worker,'alignment-cloud-worker.py','exec')
setup=(ROOT/'backend/voice_research/alignment_setup.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown,Audio
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Aligning reviewed BBC captions\\nThe source subtitles were confirmed by the user. This offline CPU step proposes audio boundaries using cached ASR weights; it does not train a TTS model.'))
try:
    report=run_alignment(PLAN,WORKER,publish)
except Exception as error:
    report={'phase':'caption_alignment_stopped','error_type':type(error).__name__,'error':str(error)[:180] if isinstance(error,RuntimeError) else 'Inspect cloud alignment','trained':False}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
for case in report.get('cases',[]):
    display(Markdown('### '+case['id']))
    print(case['text'])
    if case.get('cloud_audio_path'):
        print('Timing flags:',case.get('timing_warnings',[]))
        display(Audio(filename=case['cloud_audio_path']))
'''
cell='# Anyanwu bounded cloud diagnostics: offline caption/audio alignment\n'+setup+'\nPLAN='+repr(plan)+'\nWORKER='+repr(worker)+'\n'+tail
compile(cell,'alignment-cell.py','exec')
(folder/'alignment-cell.py').write_text(cell,encoding='utf-8')
(ROOT/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print(json.dumps({'phase':'alignment_cell_prepared','eligible_captions':5,'network_requests':0,'new_weights':False,'trained':False}))
