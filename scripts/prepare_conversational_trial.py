"""Register a fixed30-update whole-response trial and GPU-gated notebook."""
from pathlib import Path
import hashlib
import json
import random

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'.build/VoiceResearch/conversational-igbo-v1'
folder=ROOT/'.build/VoiceResearch/training/conversational-style-lora-v1';folder.mkdir(parents=True,exist_ok=True)
baseline=json.loads((ROOT/'.build/VoiceResearch/training/waxal-female5-lora-v1/plan.json').read_text())
comparison=json.loads((ROOT/'.build/VoiceResearch/auditions/waxal-female5-expanded-comparison-v2/plan.json').read_text())
capacity=json.loads((source/'capacity-cloud-result.json').read_text())
audit=json.loads((source/'submitted-transcript-review-audit.json').read_text())
if not capacity.get('drive_backup_verified') or not audit.get('all_marked_exact_and_clean'):raise RuntimeError('Verified full-response tokens and saved user review required')
if max(c['packed_positions'] for c in capacity['cases'])>1024:raise RuntimeError('Whole-response training cap exceeded; no automatic cuts')
train=[c['id'] for c in baseline['clips'] if c['adapter_split']=='train']
random.Random(23).shuffle(train)
schedule=[case for anchor in train[:10] for case in ('conversation-ref-03','conversation-ref-04',anchor)]
excluded=set()
for name in ('training/waxal-female5-lora-v1/plan.json','training/waxal-female5-expanded-lora-v2/plan.json','auditions/waxal-female5-adapter-comparison-v1/plan.json','auditions/waxal-female5-expanded-comparison-v2/plan.json'):
    prior=json.loads((ROOT/'.build/VoiceResearch'/name).read_text())
    for key in ('clips','cases','texts'):
        excluded.update(c['id'] for c in prior.get(key,[]) if isinstance(c,dict) and 'id' in c)
metadata_path=ROOT/'.build/VoiceResearch/datasets/waxal-female5-expansion-v1/source-rows-517-616.json'
rows={r['row']['id']:(r['row_idx'],r['row']) for r in json.loads(metadata_path.read_text())['rows']}
texts=[]
for key in ('ibo_1703','ibo_1715','ibo_1755','ibo_1790'):
    if key in excluded:raise RuntimeError('Previously trained/rated text selected for comparison')
    index,row=rows[key]
    texts.append({'id':key,'text':row['text'],'row_idx':index,'source':'google/WaxalNLP','source_config':'ibo_tts/train',
                  'revision':baseline['dataset_revision'],'role':'Published text only, no source audio acquired; not a factual assistant answer'})
models=[]
for model in baseline['models']:
    files=[f for f in model['files'] if f['rfilename']!='speech_tokenizer_v3.onnx']
    models.append({**model,'files':files})
plan={'id':'anyanwu-conversational-style-lora-v1','status':'registered_not_started','optimizer_updates_performed':0,
      'objective':'Explore transfer of connected conversational delivery from complete user-reviewed Igbo responses, with a preserved baseline comparison.',
      'starting_adapter_sha256':comparison['adapter_hashes']['base'],'baseline_archive':comparison['archives']['previous'],
      'base_checkpoint_sha256':comparison['base_checkpoint_sha256'],'models':models,
      'submitted_snapshot':audit['submitted_snapshot'],'submitted_review_sha256':audit['saved_file_sha256'],
      'encoded_targets':{'path':capacity['drive_backup']+'/speech-targets.json','sha256':capacity['speech_targets_sha256']},
      'capacity_cases':capacity['cases'],'source_groups':['waxal:ibo:5','silencio:IGB_005'],
      'initialization':'Exact v1 LoRA warm start, new source vector initialized from v1; keep anchor vector unchanged, fresh AdamW state.',
      'updates':30,'historical_baseline_updates':60,'learning_rate':1e-5,'rank':4,'alpha':8,'batch_size':1,'seed':23,'dtype':'float32',
      'schedule':schedule,'conversational_presentations':20,'anchor_presentations':10,'train_pool':22,'diagnostic_validation':4,
      'position_cap':1024,'speech_token_cap':800,'whole_responses':True,'source_audio_cut':False,'raw_text_changed':False,
      'comparison':{'texts':texts,'outputs':8,'variants':['preserved_v1','conversation_candidate'],
                    'prompt_id':'ibo_616','fixed_acoustic_speaker_reference':'WAXAL female5 for both variants',
                    'sampler_seed':17,'max_tokens':400,'max_seconds_per_output':20,'max_output_bytes':2*1024**2,
                    'one_take':True,'retakes':False,'automatic_adoption':False,'prior_comparison_texts_excluded':sorted(excluded)},
      'limits':{'setup_and_train_minutes':30,'optimizer_minutes':15,'cloud_growth_gib':20,'cloud_free_floor_gib':5,
                'child_rss_gib':9,'gpu_reserved_gib':12,'selected_model_bytes':sum(f['size'] for m in models for f in m['files']),
                'adapter_mib':16,'private_archive_mib':16,'log_kib':512,'laptop_model_or_media_bytes':0},
      'runtime':'Existing notebook only; hosted GPU>=14000MiB. Code never allocates/purchases compute or silently falls back to CPU.',
      'attribution':{'conversation':'Silencio Network, Igbo Speech Dataset, 2026. CC BY-NC 4.0.',
                     'read_speech':'Google WAXAL / Media Trust Igbo TTS, CC BY-SA4.0.',
                     'base':'Modular Dynamic Accent Flash, CC BY-SA4.0; CosyVoice3 Apache-2.0.'},
      'distribution':'Private research artifacts only; preserve all source conditions, no combined redistribution licence asserted.',
      'limitations':['Only two conversational responses from one source speaker; no conversational validation set or broad naturalness claim.',
                     'Four original same-speaker read-speech validation examples are diagnostic only, with possible base-model exposure.',
                     'New published synthesis texts are excluded from adapter training; base pretraining exposure remains possible.',
                     'Source groups remain distinct. Candidate LM source conditioning changes while comparison decoder reference stays fixed.',
                     'Training loss and token-capacity checks do not establish natural Igbo delivery. Human comparison is required.',
                     'Prior v2 regression remains preserved; do not repeat this trial or tune seeds/steps to chase rated outputs.']}
raw=json.dumps(plan,ensure_ascii=False,indent=2)+'\n'
existing=folder/'plan.json'
if existing.exists() and existing.read_text()!=raw:raise RuntimeError('Different registered trial already exists; preserve it')
existing.write_text(raw)
restore=(ROOT/'backend/voice_research/conversational_trial_restore.py').read_text()
setup=(ROOT/'backend/voice_research/conversational_trial_setup.py').read_text()
worker=(ROOT/'backend/voice_research/conversational_trial_worker.py').read_text()
adapter=(ROOT/'backend/voice_research/flash_adapter.py').read_text()
tail='''
from IPython.display import Javascript,display
def publish(value):
 if not str(value.get('phase','')).startswith('conversational_asr_'):value={**value,'phase':'conversational_asr_'+value.get('phase','trial')}
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
 if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab only; no laptop model/audio work')
 if not shutil.which('nvidia-smi'):raise RuntimeError('GPU required. Current CPU is preserved; no model install, allocation or training attempted.')
 if not Path('/content/drive/MyDrive').is_dir():
  publish({'phase':'conversational_asr_trial_waiting_for_drive','optimizer_updates':0})
  from google.colab import drive
  drive.mount('/content/drive',timeout_ms=300000)
 report=run_conversational_trial(PLAN,WORKER,ADAPTER,publish)
except Exception as error:
 report={'phase':'conversational_asr_trial_stopped','error_type':type(error).__name__,'error':str(error)[:350],'optimizer_updates':0}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu registered whole-response conversational trial — GPU only,30 updates\n'+restore+'\n'+setup+'\nPLAN='+repr(plan)+'\nWORKER='+repr(worker)+'\nADAPTER='+repr(adapter)+'\n'+tail
compile(cell,'conversational-trial-cell','exec');compile(worker,'conversational-trial-worker','exec')
(folder/'training-cell.py').write_text(cell)
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
    'cells':[{'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · registered conversational delivery trial\n','30 fixed GPU updates, whole responses and exact user-corrected wording. Preserved v1 baseline; two source groups. Private research. No automatic runtime allocation/purchase. Backup excludes base weights. Fresh listening comparison follows; no automatic adoption.\n']},
             {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
(ROOT/'Voice Research/Anyanwu-Conversational-Voice-Trial.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
receipt={'phase':'conversational_trial_prepared','plan_sha256':hashlib.sha256(raw.encode()).hexdigest(),
         'cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),'whole_responses':2,'registered_updates':30,
         'actual_updates':0,'comparison_text_ids':[t['id'] for t in texts],'selected_model_bytes':plan['limits']['selected_model_bytes']}
(folder/'source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
