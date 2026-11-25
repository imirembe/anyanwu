"""Build the two-response transcription job; no local audio/model work."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'.build/VoiceResearch/conversational-igbo-v1'
selection = json.loads((folder/'user-selection-v1.json').read_text())
manifest = json.loads((folder/'cloud-result.json').read_text())['manifest']
worker = (ROOT/'backend/voice_research/conversational_transcript_worker.py').read_text()
setup = (ROOT/'backend/voice_research/conversational_transcript_setup.py').read_text()
clips = [{k: c[k] for k in ('id', 'speaker_id', 'topic')} | {
    'seconds': c['listening']['seconds'], 'source_audio_sha256': c['source']['sha256'],
    'listening_sha256': c['listening']['sha256']}
    for c in manifest['clips'] if c['id'] in selection['reference_ids']]
if len(clips) != 2 or any(c['speaker_id'] != 'IGB_005' for c in clips):
    raise RuntimeError('Expected the two user-selected whole responses')
plan = {'id': 'conversational-igbo-transcription-v1', 'source': manifest['repository'],
        'source_revision': manifest['revision'], 'attribution': manifest['attribution'],
        'selection_sha256': hashlib.sha256((folder/'user-selection-v1.json').read_bytes()).hexdigest(),
        'worker_sha256': hashlib.sha256(worker.encode()).hexdigest(),
        'asr_model': 'omniASR_CTC_300M_v2', 'asr_model_license': 'Apache-2.0',
        'asr_model_bytes': 1304065508,
        'asr_model_sha256': '8ce340ada22435d189908a8af67e3bb04899b6f2f5329d036248b9c3e38d2b50',
        'model_hash_provenance': 'Previously observed project download; not a publisher-supplied checksum',
        'clips': clips, 'supervised_training_ready': False, 'optimizer_updates': 0,
        'scope': 'Source-derived unverified ASR drafts and acoustic pause candidates. No TTS training, model benchmark, source cutting or speech synthesis.',
        'limits': {'minutes': 20, 'cloud_disk_growth_gib': 8, 'cloud_free_floor_gib': 5,
                   'model_cache_gib': 2, 'child_rss_gib': 6, 'log_kib': 256, 'clips': 2,
                   'laptop_media_or_models_bytes': 0},
        'sources': ['https://github.com/facebookresearch/omnilingual-asr']}
(folder/'transcription-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
tail = '''
from IPython.display import Javascript,display,Markdown
def publish(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('## Preparing transcripts for references 3 and 4\\nWhole responses, cloud CPU only. Machine text and possible pauses need listening review. No voice training.'))
try:
 report=run_conversational_asr(PLAN,WORKER_SOURCE,publish)
except Exception as error:
 report={'phase':'conversational_asr_stopped','error_type':type(error).__name__,'error':str(error)[:350],'optimizer_updates':0}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell = '# Anyanwu selected conversational Igbo transcripts — CPU only, no training\n'+setup+'\nPLAN='+repr(plan)+'\nWORKER_SOURCE='+repr(worker)+'\n'+tail
compile(cell, 'conversational-transcript-cell', 'exec')
(folder/'transcription-cell.py').write_text(cell)
notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
            'cells':[{'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · selected conversational transcripts\n','Use the existing hosted CPU runtime and connected research Drive. Audio and ASR weights remain in Colab; no TTS training or download to laptop.\n']},
                     {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
(ROOT/'Voice Research/Anyanwu-Selected-Igbo-Transcripts.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
receipt={'phase':'conversational_asr_prepared','clips':2,'source_seconds':sum(c['seconds'] for c in clips),
         'cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),'worker_sha256':plan['worker_sha256'],
         'model_or_media_downloaded_to_laptop':False,'optimizer_updates':0}
(folder/'transcription-source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
