"""Prepare a fixed next adapter experiment from real saved source reviews only."""
from pathlib import Path
import hashlib
import json
import unicodedata

root=Path(__file__).resolve().parents[1];lab=root/'.build/VoiceResearch'
old=lab/'training/waxal-female5-lora-v1';new=lab/'datasets/waxal-female5-expansion-v1'
previous=json.loads((old/'plan.json').read_text());trained=json.loads((old/'training-result.json').read_text())
source=json.loads((new/'plan.json').read_text());data=json.loads((new/'data-result.json').read_text())
feedback=json.loads((new/'submitted-source-feedback-receipt.json').read_text())
if not feedback['submitted_snapshot_verified'] or len(feedback['accepted_ids'])!=14 or feedback['held_ids'] or feedback['pending_ids'] or feedback['missing_ids']:
    raise RuntimeError('Verified complete source review is required')
accepted=set(feedback['accepted_ids']);cases=[]
for c in previous['clips']:
    cases.append({**c,'acquisition':'preserved_baseline_archive','source_review_status':'Existing publisher-paired baseline item; original source status unchanged'})
for c in data['clips']:
    if c['id'] not in accepted:continue
    cases.append({**c,'adapter_split':'train','acquisition':'verified_source_expansion_archive','source_review_status':'User accepted exact words, complete utterance and clean female speech'})
training=[c for c in cases if c['adapter_split']=='train'];validation=[c for c in cases if c['adapter_split']=='validation']
if (len(training),len(validation))!=(34,4) or len({c['id'] for c in cases})!=38:raise RuntimeError('Registered34/4 source split mismatch')
normalize=lambda text:unicodedata.normalize('NFC',text).strip()
reserved={c['id'] for c in source['reserved_evaluation_texts']};texts={normalize(c['text']) for c in cases}
if reserved&{c['id'] for c in cases} or texts&{normalize(c['text']) for c in source['reserved_evaluation_texts']}:raise RuntimeError('Fresh evaluation overlap')
comparison=json.loads((lab/'auditions/waxal-female5-adapter-comparison-v1/plan.json').read_text())
if {c['id'] for c in cases}&{c['id'] for c in comparison['texts']}:raise RuntimeError('Already-rated generated text entered training')
plan={'id':'anyanwu-waxal-female5-expanded-lora-v2','status':'prepared_not_started','optimizer_updates_performed':0,
 'starting_adapter_sha256':trained['adapter_sha256'],'baseline_preserved':True,'initialisation':'Warm-start the verified60-update v1 LoRA/source vector; fresh AdamW state, not an exact optimizer-state resume',
 'registered_additional_updates':60,'learning_rate':0.00002,'batch_size':1,'rank':4,'alpha':8,'seed':11,'precision':'float32',
 'train_count':34,'validation_count':4,'new_user_accepted_count':14,'cases':cases,
 'baseline_train_seconds':trained['train_audio_seconds'],'new_user_accepted_seconds':data['audio_seconds'],
 'combined_train_seconds':round(trained['train_audio_seconds']+data['audio_seconds'],6),
 'validation_policy':'Preserve the original four v1 same-speaker validation IDs; diagnostic only, no repeated tuning or checkpoint selection against them',
 'feedback_sha256':feedback['feedback_sha256'],'source_feedback_snapshot':feedback['submitted_snapshot'],
 'archives':{'baseline':json.loads((old/'backup-result.json').read_text()),'new_source':json.loads((new/'backup-result.json').read_text())},
 'fresh_synthesis_comparison':{'texts':source['reserved_evaluation_texts'],'variants':['preserved_v1_adapter','expanded_v2_adapter'],'outputs':8,'seed':17,
    'prompt_id':'ibo_616','policy':'One take per text/variant, same source reference, decoder and sampler; no retakes, pitch edits or seed search','minimum_evidence':'Actual user listening to words, pronunciation and cadence before adoption'},
 'limits':{'setup_and_training_minutes':30,'optimizer_minutes':15,'cloud_disk_growth_gib':20,'cpu_rss_gib':9,'gpu_reserved_gib':12,
   'selected_models_gib':4,'speech_tokens':400,'packed_positions':512,'adapter_mib':16,'archive_mib':32,'local_models_or_audio':False,'paid_upgrade':False},
 'runtime_policy':'Use an already selected hosted GPU only; current CPU remains for review. No automatic runtime switch, GPU purchase or local training.',
 'attribution':previous['attribution'],'limitations':['Fourteen new source items were reviewed by the user; older publisher labels retain their original status',
   'Single source speaker and possible base pretraining exposure; this is not an independent general Igbo benchmark',
   'Four fresh texts are excluded from this adapter training but may be familiar to the pretrained base',
   'Fixed prospective pilot; no quality improvement assumed from loss alone','No model install, new generation, training or native assistant integration has started']}
folder=lab/'training/waxal-female5-expanded-lora-v2';folder.mkdir(exist_ok=True)
path=folder/'plan.json'
if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Different next-stage plan exists; no overwrite')
path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':plan['status'],'train_count':34,'validation_count':4,'new_reviewed':14,'train_seconds':plan['combined_train_seconds'],'fresh_generated_tests':8,'training_started':False}))
