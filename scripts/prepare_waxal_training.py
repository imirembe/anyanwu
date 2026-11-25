"""Declare and prepare a paired-female Igbo60-update research run; no local media."""
from pathlib import Path
import base64
import hashlib
import json
import urllib.parse

ROOT=Path(__file__).resolve().parents[1]
prior=ROOT/'.build/VoiceResearch/training/anyanwu-female-v1'
folder=ROOT/'.build/VoiceResearch/training/waxal-female5-lora-v1';folder.mkdir(exist_ok=True)
metadata=prior/'waxal-female-short-rows.json'
response=json.loads(metadata.read_text());rows=response['rows'][:24]
if len(rows)!=24 or any(r['row']['gender']!='Female' or str(r['row']['speaker_id'])!='5' or r.get('truncated_cells') for r in rows):
    raise RuntimeError('Expected24 complete source-labelled female speaker5 rows')
card=(prior/'waxal-card.txt').read_bytes()
if hashlib.sha256(card).hexdigest()!='60b3ffe00b11d23790e3176d42500f7ea83d8e6e663adef9f560ff438f5c475e':
    raise RuntimeError('Pinned source licence/card fingerprint changed')
ids=[r['row']['id'] for r in rows]
if len(set(ids))!=24:raise RuntimeError('Duplicate source IDs')
validation=set(sorted(ids,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:4])
clips=[];urls={}
for r in rows:
    value=r['row'];url=value['audio'][0]['src']
    if '5f4d8ca24f2b9d168b2ee545f1febaaff4b40580' not in url or not 15<=len(value['text'])<=180:
        raise RuntimeError('Unexpected source text length or asset revision')
    urls[value['id']]=url
    clips.append({'id':value['id'],'text':value['text'],'speaker_id':'5','gender':'Female','source':'google/WaxalNLP',
         'source_config':'ibo_tts','source_split':'train','source_group':'waxal:ibo:5','row_idx':r['row_idx'],
         'source_row_url':'https://huggingface.co/datasets/google/WaxalNLP/viewer/ibo_tts/train?row='+str(r['row_idx']),
         'asset_path':urllib.parse.urlsplit(url).path,'truncated_cells':r.get('truncated_cells',[]),
         'adapter_split':'validation' if value['id'] in validation else 'train','independent_native_review':False})
old=json.loads((prior/'flash-training-plan.json').read_text())
plan={'id':'anyanwu-waxal-female5-lora-v1','updates':60,'learning_rate':.00005,'batch_size':1,'rank':4,'alpha':8,'seed':7,
      'dataset_revision':'5f4d8ca24f2b9d168b2ee545f1febaaff4b40580','clips':clips,'models':old['models'],'mel_filter':old['mel_filter'],
      'initialisation':'Fresh LoRA from published pinned Flash base; previous BBC adapter not loaded',
      'source_selection':'First24 short whole speaker5/Female rows at fixed485–508; four stable ID-hash selected validation items',
      'source_card_sha256':hashlib.sha256(card).hexdigest(),'viewer_metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),
      'attribution':{'data':'Google WAXAL / Media Trust Igbo TTS, CC BY-SA4.0, source speaker5/Female',
                     'flash':old['attribution']['flash'],'base':old['attribution']['base'],'mel_frontend':old['attribution']['mel_frontend'],
                     'source_card':'https://huggingface.co/datasets/google/WaxalNLP/blob/5f4d8ca24f2b9d168b2ee545f1febaaff4b40580/README.md'},
      'limitations':['Publisher-paired full utterances and declared female gender; no independent human word/boundary or speaker-consent audit for this new sample',
                     'Lossy viewer MP3 previews; no claim that they are native lossless studio WAVs',
                     'Single source speaker. Validation is held out from this adapter only; the base was trained on WAXAL and may have seen these utterances',
                     'Sixty-update research adapter only: no generated-speech naturalness/intelligibility result, novel voice identity or production readiness',
                     'Frozen flow/vocoder omitted. No BBC examples, previous BBC adapter or private app data are used'],
      'limits':{'minutes':30,'training_minutes':15,'cloud_disk_growth_gib':20,'cpu_rss_gib':9,'gpu_reserved_gib':12,'models_gib':4,
                'data_body_mib':32,'data_requests':90,'data_minutes':8,'clips':24,'seconds_per_clip':15,
                'speech_tokens':400,'packed_positions':512,'adapter_mib':16,'backup_mib':32,'local_media_or_models':False,'paid_compute':False}}
(folder/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
worker=(ROOT/'backend/voice_research/waxal_train_worker.py').read_text()
adapter=(ROOT/'backend/voice_research/flash_adapter.py').read_text()
setup=(ROOT/'backend/voice_research/waxal_training_data.py').read_text()+'\n'+(ROOT/'backend/voice_research/waxal_train_setup.py').read_text()+'\n'+(ROOT/'backend/voice_research/waxal_training_backup.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Anyanwụ: paired female Igbo training\\n60 GPU LoRA updates. Twenty whole publisher-paired utterances train the adapter; four are held out from this run for a limited same-speaker validation check. The rejected BBC cuts remain excluded. The sample and model stay in Colab/Drive; no paid resource is started.'))
report={}
try:
    report=run_waxal_training(PLAN,SOURCE_URLS,WORKER,ADAPTER,publish)
    if report.get('optimizer_updates')==60 and report.get('adapter_reloaded'):
        report['backup']=backup_waxal_training()
        publish(report)
        display(Markdown('**Sixty real training updates completed and the adapter is backed up to Drive.** Natural Igbo voice quality still needs generated-speech listening.'))
    else: publish(report)
except Exception as error:
    report.update(phase='female_training_or_backup_stopped',error_type=type(error).__name__,error=str(error)[:450])
    publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu bounded cloud diagnostics: paired female Igbo training\n'+setup+'\nPLAN='+repr(plan)+'\nSOURCE_URLS='+repr(urls)+'\nWORKER='+repr(worker)+'\nADAPTER='+repr(adapter)+'\n'+tail
compile(cell,'waxal-female-training-cell.py','exec')
(folder/'training-cell.py').write_text(cell)
(ROOT/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
(folder/'source-receipt.json').write_text(json.dumps({'plan_sha256':hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest(),
   'cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
    ['backend/voice_research/waxal_training_data.py','backend/voice_research/waxal_train_setup.py','backend/voice_research/waxal_train_worker.py','backend/voice_research/waxal_training_backup.py','backend/voice_research/flash_adapter.py']},
   'rejected_bbc_used':False,'previous_bbc_adapter_loaded':False,'new_source_labels':'Publisher-paired; not independently human verified'},indent=2)+'\n')
print(json.dumps({'registered_updates':60,'publisher_clips':24,'train':20,'validation':4,'speaker':'5','gender':'Female','models_and_audio_on_laptop':False}))
