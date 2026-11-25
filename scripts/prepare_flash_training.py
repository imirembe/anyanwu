"""Prepare a cloud-only five-update Flash speech LM LoRA pilot (no laptop model IO)."""
from pathlib import Path
import base64
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.build/VoiceResearch/training/anyanwu-female-v1'
cases=[json.loads(line) for line in (folder/'pairing-candidates.jsonl').read_text().splitlines()]
assert len(cases)==3 and len({c['source_group'] for c in cases})==3
assert all(c['dataset_status']=='unflagged_machine_alignment_candidate' and not c['timing_warnings'] and c['split'] is None for c in cases)


def selected(filename,names):
    index=json.loads((folder/filename).read_text())
    files=[]
    for name in names:
        item=next(f for f in index['files'] if f['rfilename']==name)
        result={'rfilename':name,'size':item['size']}
        if item.get('lfs'):
            result['sha256']=item['lfs']['sha256']
        else:
            result['git_blob_sha1']=item['blobId']
        files.append(result)
    return {'repo':index['id'],'revision':index['revision'],'files':files}


plan={'id':'anyanwu-flash-lora-pilot-v1','purpose':'Real speech-model training-path test; not a voice-quality benchmark',
      'updates':5,'batch_size':1,'learning_rate':0.0001,'lora':{'rank':4,'alpha':8,'projections':['q_proj','v_proj']},
      'cases':cases,'case_use':'All three provisional cases train-only for this technical experiment; original readiness flags unchanged',
      'models':[
          selected('flash-blobs.json',['checkpoints/flash/flash_dpo/best.pt','checkpoints/flash/flash_dpo/config.json','checkpoints/flash/flash_dpo/spk2id.json']),
          selected('cv3-index.json',['CosyVoice-BlankEN/config.json','CosyVoice-BlankEN/tokenizer_config.json','CosyVoice-BlankEN/merges.txt',
                                     'CosyVoice-BlankEN/vocab.json','speech_tokenizer_v3.onnx'])],
      'mel_filter':json.loads((folder/'mel-filter-meta.json').read_text()),
      'attribution':{'flash':'Modular / Dynamic Accent Flash, CC BY-SA 4.0; female Igbo prior waxal_ibo_5 (WAXAL / Media Trust)',
                     'base':'FunAudioLLM / CosyVoice3, Apache-2.0','mel_frontend':'OpenAI Whisper, MIT',
                     'data':'BBC Igbo; five URLs/source provenance in case manifests; user-attested training permission',
                     'upstream_rights':'Publisher declarations and user attestation; not an independent upstream consent audit'},
      'limitations':['17.989625 seconds, three provisional machine cuts; source subtitles user-confirmed, exact cut boundaries not independently approved',
                     'Five updates test optimisation only; no held-out evaluation, intelligibility/cadence result or original-voice claim',
                     'LM adapters and separate source vectors only; frozen acoustic flow/vocoder not downloaded or trained',
                     'No synthesized output in this run; not integrated into the app or working voice agent'],
      'limits':{'cloud_only':True,'paid_compute':False,'model_gib':4,'disk_growth_gib':20,'free_floor_gib':5,
                'rss_gib':9,'minutes':30,'training_minutes':15,'max_sequence':512,'speech_tokens':400,
                'checkpoint_mib':16,'log_kib':512,'cpu_threads':2,'max_trainable_parameters':500000}}
if sys.argv[1:]==['repair-rope']:
    plan['repair']={'id':'rope-buffer-materialisation-v1','reason':'First actual pretrained forward failed before update 1 because nonpersistent rotary buffers remained on meta. Rebuild them from pinned config, validate parameters AND buffers, and exercise the same meta load in the cloud fixture.',
                    'minutes':15,'prior_optimizer_updates':0,'reuse_verified_models':True,'additional_updates_above_original_five':0}
    plan['limits']['minutes']=15
adapter=(ROOT/'backend/voice_research/flash_adapter.py').read_text()
worker=(ROOT/'backend/voice_research/flash_train_worker.py').read_text()
setup=(ROOT/'backend/voice_research/flash_train_setup.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Anyanwụ: first speech-model training pilot\\nFive CPU LoRA updates on three provisional BBC caption/audio cuts. Models and adapter stay in this cloud VM. This technical run does not establish natural Igbo voice quality.'))
try:
    report=run_flash_training(PLAN,WORKER,ADAPTER,publish)
except Exception as error:
    report={'phase':'flash_pilot_guard_stopped','error_type':type(error).__name__,'error':str(error)[:500]}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
if report.get('optimizer_updates')==5 and report.get('adapter_path'):
    display(Markdown('**Training completed: five real optimizer updates.** Adapter: `'+report['adapter_path']+'`. Voice quality has not been evaluated and no new speech was generated.'))
else:
    display(Markdown('**Training did not complete.** Inspect the actual update count and cloud error above before any repair.'))
'''
cell='# Anyanwu bounded cloud diagnostics: five-update speech LM LoRA pilot\n'+setup+'\nPLAN='+repr(plan)+'\nWORKER='+repr(worker)+'\nADAPTER='+repr(adapter)+'\n'+tail
compile(cell,'flash-training-cell.py','exec')
(folder/'flash-training-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
(folder/'flash-training-cell.py').write_text(cell,encoding='utf-8')
(ROOT/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print(json.dumps({'phase':'flash_training_cell_prepared','updates':5,'cases':len(cases),'model_bytes':sum(f['size'] for m in plan['models'] for f in m['files']),
                  'cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),'model_or_audio_on_laptop':False}))
