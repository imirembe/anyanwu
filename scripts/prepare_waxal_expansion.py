"""Select a bounded separate source sample while reserving fresh evaluation text."""
from pathlib import Path
import base64
import hashlib
import json
import sys
import unicodedata
import urllib.parse

root=Path(__file__).resolve().parents[1];lab=root/'.build/VoiceResearch'
folder=lab/'datasets/waxal-female5-expansion-v1'
prior=json.loads((lab/'training/waxal-female5-lora-v1/plan.json').read_text())
comparison=json.loads((lab/'auditions/waxal-female5-adapter-comparison-v1/plan.json').read_text())
blocked=prior['clips']+comparison['texts'];ids={x['id'] for x in blocked};texts={unicodedata.normalize('NFC',x['text']).strip() for x in blocked}
paths=[lab/'training/anyanwu-female-v1/waxal-female-short-rows.json',folder/'source-rows-517-616.json']
selected={};decisions=[]
for path in paths:
    for row in json.loads(path.read_text())['rows']:
        v=row['row'];reason=None
        if str(v['speaker_id'])!='5' or v['gender']!='Female':reason='Different source speaker/gender'
        elif v['id'] in ids or unicodedata.normalize('NFC',v['text']).strip() in texts:reason='Prior adapter or already-rated development text'
        elif row.get('truncated_cells') or not 15<=len(v['text'])<=180:reason='Truncated or outside declared short-text length'
        if reason:decisions.append({'id':v['id'],'row_idx':row['row_idx'],'reason':reason});continue
        selected[v['id']]=row
reserve={'ibo_648','ibo_637','ibo_638','ibo_664'}
if not reserve<=set(selected):raise RuntimeError('Fixed fresh evaluation text missing')
clips=[];heldout=[];urls={}
for source_id,row in selected.items():
    v=row['row'];url=v['audio'][0]['src']
    if prior['dataset_revision'] not in urllib.parse.urlsplit(url).path:raise RuntimeError('Asset revision mismatch')
    item={'id':source_id,'text':v['text'],'speaker_id':'5','gender':'Female','row_idx':row['row_idx'],
          'source':'google/WaxalNLP','source_config':'ibo_tts','source_split':'train','source_group':'waxal:ibo:5',
          'source_row_url':'https://huggingface.co/datasets/google/WaxalNLP/viewer/ibo_tts/train?row='+str(row['row_idx']),
          'asset_path':urllib.parse.urlsplit(url).path,'truncated_cells':[],'adapter_split':None,'independent_human_review':False}
    if source_id in reserve:heldout.append(item)
    else:clips.append(item);urls[source_id]=url
plan={'id':'anyanwu-waxal-female5-expansion-v1','stage':'published_source_acquisition_and_review_only','revision':prior['dataset_revision'],
      'source_card_sha256':prior['source_card_sha256'],'clips':clips,'reserved_evaluation_texts':heldout,
      'prior_ids':sorted(ids),'prior_texts':sorted(texts),'baseline_adapter_sha256':comparison['adapter_sha256'],
      'selection':'All complete female speaker5 labels15–180 characters in fixed485–533 metadata, excluding all24 prior adapter rows and3 already-rated texts; four fixed fresh texts reserved without audio download',
      'metadata_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'attribution':prior['attribution'],
      'limits':{'optimizer_updates':0,'clips':len(clips),'seconds_per_clip':15,'mp3_bytes_per_clip':1024**2,'response_body_mib':32,
                'requests':90,'acquisition_minutes':10,'cloud_disk_growth_mib':64,'archive_mib':16,'laptop_media_or_models':False,'paid_upgrade':False},
      'review_policy':'Every newly acquired in-bound utterance has an explicit user review; any unchecked or rejected source stays excluded from a future registered training run',
      'limitations':['Publisher labels, gender and licence preserved; independent source-word and speaker-consent audit not established',
        'Source text is a speech pronunciation target, not verified current factual knowledge for assistant replies',
        'Whole lossy previews only; long recordings held without cropping, no forced alignment',
        'Same-speaker and possible base pretraining exposure; future reserved text is held out from adapter only',
        'No rejected BBC pairs, private conversations, new models or training updates']}
destination=folder/'plan.json'
if destination.exists() and json.loads(destination.read_text())!=plan:raise RuntimeError('Different expansion plan exists; no overwrite')
destination.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
repair=sys.argv[1:]==['--refresh-links']
if repair:
    refreshed=folder/'refreshed-rows-509-533.json'
    fresh={r['row']['id']:r for r in json.loads(refreshed.read_text())['rows']}
    for clip in clips:
        row=fresh[clip['id']];value=row['row'];url=value['audio'][0]['src']
        if row['row_idx']!=clip['row_idx'] or value['text']!=clip['text'] or str(value['speaker_id'])!='5' or value['gender']!='Female' or row.get('truncated_cells') or urllib.parse.urlsplit(url).path!=clip['asset_path']:
            raise RuntimeError('Refreshed public asset changes the declared source')
        urls[clip['id']]=url
    (folder/'link-refresh-receipt.json').write_text(json.dumps({'reason':'Old first source URL expired at2026-11-23T15:48:39Z; first request failed403 before any audio was saved',
      'metadata_sha256':hashlib.sha256(refreshed.read_bytes()).hexdigest(),'clips_unchanged':True,'count':len(clips),'optimizer_updates':0},indent=2)+'\n')
(folder/'selection-decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2)+'\n')
source=(root/'backend/voice_research/waxal_expansion_data.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Expand the female Igbo source sample\\nWhole published utterances from the same female source speaker. Four fresh texts remain separate for the next generated comparison. Data review only: no model install, training or paid resource.'))
try:
    if REPAIR_EXPIRED_LINKS:
        previous=ROOT/'acquisition-attempt.json';preserved=ROOT/'acquisition-attempt-1-expired-link.json'
        failed=json.loads(previous.read_text())
        if failed.get('error')!='HTTP Error 403: Forbidden' or failed.get('completed')!=0 or failed.get('requests')!=2 or list((ROOT/'data').iterdir()) or preserved.exists():
            raise RuntimeError('Expired-link repair preconditions changed; no automatic repeat')
        preserved.write_bytes(previous.read_bytes())
        (ROOT/'link-refresh-receipt.json').write_text(json.dumps({'reason':'One declared public URL refresh after403; exact same17 source rows; no previous audio saved','optimizer_updates':0},indent=2))
        previous.unlink();(ROOT/'data').rmdir()
    report=acquire_expansion(PLAN,SOURCE_URLS,publish)
    publish(report)
    print(json.dumps({k:v for k,v in report.items() if k!='clips'},ensure_ascii=False,indent=2))
except Exception as error:
    publish({'phase':'expansion_acquisition_guard_stopped','error':str(error)[:450],'training_started':False})
    raise
'''
cell='# Anyanwu bounded cloud diagnostics: expand whole female Igbo sources\n'+source+'\nPLAN='+repr(plan)+'\nSOURCE_URLS='+repr(urls)+'\nREPAIR_EXPIRED_LINKS='+repr(repair)+'\n'+tail
compile(cell,'expansion-acquire.py','exec');(folder/'acquisition-cell.py').write_text(cell)
(lab/'colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print(json.dumps({'new_source_candidates':len(clips),'fresh_reserved_texts':len(heldout),'prior_adapter_and_review_ids_excluded':len(ids),'optimizer_updates':0,'local_audio':False}))
