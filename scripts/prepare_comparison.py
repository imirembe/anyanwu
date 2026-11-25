"""Prepare a declared one-take original/adapter speech comparison in Colab."""
from pathlib import Path
import base64
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_cv3_runtime import build

folder=ROOT/'.build/VoiceResearch/auditions/waxal-female5-adapter-comparison-v1'
prior=ROOT/'.build/VoiceResearch/training/anyanwu-female-v1'
trainfolder=ROOT/'.build/VoiceResearch/training/waxal-female5-lora-v1'
trained=json.loads((trainfolder/'training-result.json').read_text())
runtime=build()
index=json.loads((prior/'cv3-index.json').read_text())
files=[]
for name in ('flow.pt','hift.pt','campplus.onnx'):
    row=next(f for f in index['files'] if f['rfilename']==name)
    files.append({'name':name,'size':row['size'],'sha256':row['lfs']['sha256']})
metadata=json.loads((prior/'waxal-female-short-rows.json').read_text())
selected=['ibo_658','ibo_655','ibo_651']
texts=[]
training=json.loads((trainfolder/'plan.json').read_text())
for source_id in selected:
    row=next(r for r in metadata['rows'] if r['row']['id']==source_id)
    if source_id in {c['id'] for c in training['clips']}:raise RuntimeError('Comparison text overlaps this adapter training/validation sample')
    texts.append({'id':source_id,'text':row['row']['text'],'source_row':row['row_idx'],'origin':'Published WAXAL text; not used in this adapter run, but base exposure possible'})
plan={'id':'anyanwu-waxal-female5-comparison-v1','adapter_sha256':trained['adapter_sha256'],'adapter_training_updates':60,
      'base_repo':training['models'][0]['repo'],'base_revision':training['models'][0]['revision'],
      'base_checkpoint_sha256':training['models'][0]['files'][0]['sha256'],'acoustic_repo':index['id'],'acoustic_revision':index['revision'],'acoustic_files':files,
      'runtime_sha256':hashlib.sha256(runtime.encode()).hexdigest(),'official_code_commit':'074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc',
      'prompt_id':'ibo_616','prompt_policy':'Whole published female source5 training utterance; same reference for both variants',
      'texts':texts,'variants':['base','adapter'],'outputs':6,'seed':17,'top_p':.8,'top_k':25,'win':10,'tau_r':.1,'max_speech_tokens':400,
      'sampling':'Published Flash SILEX-cap procedure; minimum2 acoustic tokens/text token; one take per text/variant, no seed search',
      'limits':{'setup_minutes':15,'generation_minutes':15,'additional_model_bytes':sum(f['size'] for f in files),'additional_disk_gib':8,
                'cpu_rss_gib':9,'gpu_reserved_gib':12,'wav_count':6,'seconds_per_wav':20,'bytes_per_wav':2*1024**2,'backup_mib':16,
                'optimizer_updates':0,'laptop_audio_or_models':False,'paid_upgrade':False},
      'limitations':['Six generated clips are a listening experiment, not native-reviewed transcripts or an independent benchmark',
                     'All text is published; not used in the adapter run, but the pretrained base may have seen it',
                     'Frozen flow/vocoder and same female source prompt in both variants; difference is the60-update LM adapter/source vector',
                     'Cap/EOS/repetition diagnostics are retained; no retakes or pitch changes to create a favourable comparison',
                     'Research only; female identity is publisher-labelled, not a validated unique Anyanwu voice'],
      'attribution':trained['attribution']}
(folder/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
setup=(ROOT/'backend/voice_research/comparison_setup.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Prepare the voice comparison\\nAdding only the pinned frozen acoustic decoder/vocoder and small matching libraries in the existing Colab environment. Three published sentences will each have one original-base and one trained-adapter take. No new training or paid resource.'))
try: report=prepare_comparison(PLAN,RUNTIME,publish)
except Exception as error: report={'phase':'comparison_setup_guard_stopped','error':str(error)[:450]}
publish(report)
print(json.dumps(report,indent=2))
'''
cell='# Anyanwu bounded cloud diagnostics: prepare generated voice comparison\n'+setup+'\nPLAN='+repr(plan)+'\nRUNTIME='+repr(runtime)+'\n'+tail
compile(cell,'comparison-setup-cell.py','exec')
(folder/'setup-cell.py').write_text(cell)
(ROOT/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print(json.dumps({'declared_outputs':6,'source_sentences':3,'new_acoustic_model_bytes':sum(f['size'] for f in files),'optimizer_updates':0,'local_models':False}))
