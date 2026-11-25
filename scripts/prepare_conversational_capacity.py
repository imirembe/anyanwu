"""Build the whole-response capacity audit; no local model/media work."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.build/VoiceResearch/conversational-igbo-v1'
baseline=json.loads((ROOT/'.build/VoiceResearch/training/waxal-female5-lora-v1/plan.json').read_text())
audit=json.loads((folder/'submitted-transcript-review-audit.json').read_text())
drafts=json.loads((folder/'transcription-cloud-result.json').read_text())
if not audit['snapshot_verified'] or not audit['all_marked_exact_and_clean']:raise RuntimeError('Two exact saved user reviews required')
model=next(m for m in baseline['models'] if m['repo']=='FunAudioLLM/Fun-CosyVoice3-0.5B-2512')
names={'CosyVoice-BlankEN/config.json','CosyVoice-BlankEN/vocab.json','CosyVoice-BlankEN/merges.txt','CosyVoice-BlankEN/tokenizer_config.json','speech_tokenizer_v3.onnx'}
assets=[f for f in model['files'] if f['rfilename'] in names]
if len(assets)!=5 or sum(f['size'] for f in assets)>1024**3:raise RuntimeError('Expected five bounded tokenizer assets, no LM weights')
worker=(ROOT/'backend/voice_research/conversational_capacity_worker.py').read_text()
cases=[]
for row in audit['records']:
    original=next(r for r in drafts['records'] if r['id']==row['id'])
    cases.append({'id':row['id'],'transcript_sha256':hashlib.sha256(row['transcript'].encode()).hexdigest(),
                  'analysis_sha256':original['analysis_sha256'],'listening_sha256':row['audio_sha256']})
plan={'id':'anyanwu-conversational-capacity-v1','submitted_snapshot':audit['submitted_snapshot'],
      'submitted_review_sha256':audit['saved_file_sha256'],'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),
      'model':{'repo':model['repo'],'revision':model['revision'],'files':assets},'mel_filter':baseline['mel_filter'],
      'cases':cases,'speech_token_cap':800,'packed_position_cap':1280,
      'scope':'CPU whole-response token measurement and training-target preparation only; no LM weight download, training, synthesis or cuts.',
      'limits':{'minutes':10,'cloud_disk_growth_gib':4,'cloud_free_floor_gib':5,'child_rss_gib':6,
                'log_kib':512,'dependencies_mib':512,'selected_model_assets_bytes':sum(f['size'] for f in assets)},
      'attribution':{'source':'Silencio Network, Igbo Speech Dataset, 2026. CC BY-NC 4.0.',
                     'speech_tokenizer':'FunAudioLLM / CosyVoice3, Apache-2.0; pinned source metadata retained',
                     'mel_filters':'OpenAI Whisper, MIT'},'optimizer_updates':0}
(folder/'capacity-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
setup=(ROOT/'backend/voice_research/conversational_capacity_setup.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('## Checking complete conversational responses\\nCPU token measurement only. Exact saved wording and full recordings are retained; no speech-model training or synthesis.'))
try:
 report=run_capacity(PLAN,WORKER,publish)
except Exception as error:
 report={'phase':'conversational_asr_capacity_stopped','error_type':type(error).__name__,'error':str(error)[:350],'optimizer_updates':0}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu whole-response conversational capacity check — CPU, no training\n'+setup+'\nPLAN='+repr(plan)+'\nWORKER='+repr(worker)+'\n'+tail
compile(cell,'capacity-cell','exec');(folder/'capacity-cell.py').write_text(cell)
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
    'cells':[{'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · conversational trial preparation\n','Existing hosted CPU and connected research Drive. Whole-response text/speech token measurement; no voice training or laptop model/media download.\n']},
             {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
(ROOT/'Voice Research/Anyanwu-Conversational-Trial-Preparation.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
receipt={'phase':'whole_response_capacity_prepared','cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),
         'clips':2,'selected_asset_bytes':plan['limits']['selected_model_assets_bytes'],'lm_weights_downloaded':False,'optimizer_updates':0}
(folder/'capacity-source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
