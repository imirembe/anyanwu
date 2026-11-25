"""Declare one offline pass on nine new published-caption speech passages."""
from pathlib import Path
import base64
import hashlib
import json
import sys

root=Path(__file__).resolve().parents[1]
folder=root/'.build/VoiceResearch/training/anyanwu-female-v1/expansion-v1'
caption_report=json.loads((folder/'caption-text.json').read_text())
by_id={e['id']:e for e in caption_report['entries']}
targets=[]


def add(source,indices):
    parts=[by_id[f'bbc-igbo-{source:02d}-expanded-caption-{i:02d}'] for i in indices]
    if any(e['uncertainties'] or not e['display_bounded_candidate'] for e in parts):
        raise RuntimeError('Uncertain/edge caption cannot enter this declared pass')
    e={**parts[0]}
    if len(parts)>1:
        e.update(id=f'bbc-igbo-{source:02d}-expanded-passage-'+ '-'.join(f'{i:02d}' for i in indices),
                 text=' '.join(p['text'] for p in parts),
                 observed_last_frame_seconds=parts[-1]['observed_last_frame_seconds'])
    e['components']=[p['id'] for p in parts]
    e['join_rule']='Adjacent published captions concatenated with a space only; no added words. Full utterance/cut not automatically certified.'
    targets.append(e)


add(1,[2]);add(1,[3])
add(2,[2]);add(2,[3,4])
add(3,[2]);add(3,[3]);add(3,[4])
add(5,[1,2]);add(5,[3])
used={i for t in targets for i in t['components']}
holds=[{'id':e['id'],'reason':('Copied-text uncertainty' if e['uncertainties'] else
                             'Caption touches actual 20-second sample edge' if not e['display_bounded_candidate'] else
                             'Sentence continues across sample or is a dependent fragment; not selected for this pass')}
       for e in caption_report['entries'] if e['id'] not in used]
plan={'id':'anyanwu-expanded-caption-alignment-v1','captions':targets,'excluded_source_captions':holds,
      'asr_model_sha256':'8ce340ada22435d189908a8af67e3bb04899b6f2f5329d036248b9c3e38d2b50',
      'search_margin_seconds':2.0,'source_start_seconds':24,'caption_text_sha256':hashlib.sha256((folder/'caption-text.json').read_bytes()).hexdigest(),
      'selection':'Nine predeclared source passages from the next windows; no score-guided retuning or automatic expansion retry',
      'policy':{'cloud_only':True,'offline':True,'new_model_downloads':0,'network_requests':0,'maximum_captions':10,
                'minutes':10,'rss_gib':6,'output_mib':12,'ready_training_pairs':0,'optimizer_updates':0},
      'source_group_policy':'Keep source IDs separate and splits unassigned; no voice quality or independent-speaker evaluation claim'}
joint=sys.argv[1:]==['joint-source']
if joint:
    plan.update(id='anyanwu-expanded-caption-joint-alignment-v1',alignment_mode='ordered_source_captions',
                reason='Independent caption paths collided in time. One declared repair aligns the same ordered text jointly per source, then partitions exact token boundaries. No window/score threshold changes, new text, repeated training or downloads.',
                original_result='alignment-summary.json')
(folder/('joint-alignment-plan.json' if joint else 'alignment-plan.json')).write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
rules=(root/'backend/voice_research/alignment_rules.py').read_text()
worker=(root/'backend/voice_research/alignment_worker.py').read_text().replace('from backend.voice_research.alignment_rules import alignment_text,eligible_caption,evaluate_path,caption_token_ranges','# Pure rules embedded above')
worker=rules+'\n'+worker
setup=(root/'backend/voice_research/alignment_setup.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown,Audio
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Aligning the expanded published speech\\nOne offline CPU pass on nine new caption passages. Original words and source groups stay intact. Flagged cuts will be held; no more training runs on the old clips.'))
try:
    report=run_alignment(PLAN,WORKER,publish)
except Exception as error:
    report={'phase':'expanded_alignment_stopped','error_type':type(error).__name__,'error':str(error)[:350]}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
for case in report.get('cases',[]):
    display(Markdown('### '+case['id']))
    print(case['text'])
    if case.get('cloud_audio_path'):
        print('Timing flags:',case.get('timing_warnings',[]))
        display(Audio(filename=case['cloud_audio_path']))
'''
cell='# Anyanwu bounded cloud diagnostics: align expanded speech passages\n'+setup+'\nPLAN='+repr(plan)+'\nWORKER='+repr(worker)+'\n'+tail
compile(cell,'expanded-alignment-cell.py','exec')
(folder/('joint-alignment-cell.py' if joint else 'alignment-cell.py')).write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print(json.dumps({'declared_passages':len(targets),'source_captions_used':len(used),'held_captions':len(holds),'network_requests':0,'new_models':0,'training_updates':0}))
