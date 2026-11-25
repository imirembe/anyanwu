"""Trusted Colab listening form with an explicit bounded feedback callback.

Uses Colab's documented output.register_callback / kernel.invokeFunction API:
https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb
"""
from pathlib import Path
import base64
import hashlib
import json
import secrets
import shutil
import time

from backend.voice_research.review_rules import validate_cut_review
from backend.voice_research.review_connection import register_review_connection


def show_cut_review(alignment,backup,publish):
    from google.colab import output
    from IPython.display import HTML,JSON,display
    folder=Path('/content/anyanwu-bbc-expansion-v1')
    drive=Path('/content/drive/MyDrive/Anyanwu Voice Research/datasets/bbc-expansion-v1')
    cases=[c for c in alignment['cases'] if c.get('cloud_audio_path') and not c.get('timing_warnings')]
    allowed={c['id']:c['candidate_audio_sha256'] for c in cases}
    if not cases or len(cases)>9:
        raise RuntimeError('No unflagged candidates or unexpected review count')
    # Joint target ordering must give disjoint acoustic spans. Padding may overlap.
    for index,a in enumerate(cases):
        for b in cases[index+1:]:
            if a['source_id']==b['source_id'] and min(a['relative_audio_end_seconds'],b['relative_audio_end_seconds'])>max(a['relative_audio_start_seconds'],b['relative_audio_start_seconds'])+0.001:
                raise RuntimeError('Overlapping acoustic targets cannot enter the listening form')
    nonce,state=register_review_connection(alignment)
    data=[]
    total=0
    for case in cases:
        path=Path(case['cloud_audio_path'])
        if path.parent!=Path('/content/anyanwu-expanded-caption-joint-alignment-v1') or hashlib.sha256(path.read_bytes()).hexdigest()!=case['candidate_audio_sha256']:
            raise RuntimeError('Review audio integrity/path failure')
        audio=path.read_bytes()
        total+=len(audio)
        if total>4*1024**2:
            raise RuntimeError('Four MiB inline listening-audio cap')
        data.append({'id':case['id'],'text':case['text'],'seconds':case['candidate_seconds'],'audio_sha256':case['candidate_audio_sha256'],
                     'audio':'data:audio/wav;base64,'+base64.b64encode(audio).decode(),
                     'review':state['records'].get(case['id'],{})})
    template='''
<div id="anyanwu-cuts" style="background:#f7f5eb;color:#173f34;padding:24px;font:16px/1.6 system-ui;max-width:850px;border-radius:16px">
<h2 style="margin-top:0">Review the new speech cuts</h2>
<p>Listen to each clip and check its exact words, beginning and ending, and clean single female voice. Saved choices or browser drafts are restored when available. Only Save feedback writes to Google Drive; it does not start training.</p>
<p id="cut-summary"></p><p id="connection-status" role="status"></p><div id="cut-list"></div><p style="font-size:14px">BBC source recordings, used under your stated training permission. The captions were copied by AI. No generated Anyanwụ voice is being evaluated here.</p></div>
<script>
(async()=>{
 const data=__DATA__,nonce=__NONCE__,backup=__BACKUP__;
 const root=document.getElementById('anyanwu-cuts'),list=root.querySelector('#cut-list');
 const banner=root.querySelector('#connection-status');
 const draftKey='anyanwu:expanded-cut-drafts:v1';
 let draftStorage=true,drafts={};
 const allowedChoices=new Set(['unreviewed','yes','no']);
 try{const raw=localStorage.getItem(draftKey);if(raw&&raw.length<=32768){const saved=JSON.parse(raw);for(const item of data){const d=saved[item.id];if(d&&d.audio_sha256===item.audio_sha256&&typeof d.notes==='string'&&d.notes.length<=500&&Number.isFinite(d.updated_at_ms)&&['words_match','complete_edges','clean_female_voice'].every(k=>allowedChoices.has(d[k])))drafts[item.id]=d;}}localStorage.setItem(draftKey,JSON.stringify(drafts));}catch(error){draftStorage=false;}
 function storeDrafts(){if(!draftStorage)return false;try{const raw=JSON.stringify(drafts);if(raw.length>32768)throw Error('Draft cap');localStorage.setItem(draftKey,raw);return true;}catch(error){draftStorage=false;banner.textContent='Browser draft storage is unavailable. Keep this page open until Google Drive confirms your save.';return false;}}
 root.querySelector('#cut-summary').textContent=data.length+' candidates for listening. __HELD__ timing-flagged cuts remain on hold.';
 const questions=[['words_match','Does every spoken word match the text?'],['complete_edges','Are the first and last words complete?'],['clean_female_voice','Is this one clean female voice, without music or overlapping speech?']];
 for(const item of data){
  const draft=drafts[item.id],restored=draft&&draft.updated_at_ms>(item.review.saved_at_unix||0)*1000?draft:item.review;
  const card=document.createElement('section');card.style.cssText='background:#fffdf8;border:1px solid #dfe3d5;border-radius:12px;padding:18px;margin:18px 0';
  const title=document.createElement('h3');title.textContent=item.id+' · '+item.seconds.toFixed(2)+'s';title.style.cssText='font-size:15px;overflow-wrap:anywhere';card.append(title);
  const text=document.createElement('p');text.textContent=item.text;card.append(text);
  const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=item.audio;audio.style.width='100%';card.append(audio);
  const selects={};
  for(const [key,label] of questions){
   const row=document.createElement('label');row.style.cssText='display:block;margin:12px 0';row.append(document.createTextNode(label+' '));
   const select=document.createElement('select');select.style.cssText='font:inherit;padding:4px;border-radius:6px';
   for(const [value,caption] of [['unreviewed','Not checked'],['yes','Yes'],['no','No']]){const option=document.createElement('option');option.value=value;option.textContent=caption;select.append(option);}
   select.value=restored[key]||'unreviewed';selects[key]=select;row.append(select);card.append(row);
  }
  const notes=document.createElement('textarea');notes.maxLength=500;notes.placeholder='Optional notes (500 characters)';notes.value=restored.notes||'';notes.style.cssText='display:block;box-sizing:border-box;width:100%;min-height:68px;font:inherit';card.append(notes);
  const button=document.createElement('button');button.textContent='Save feedback';button.style.cssText='background:#e9ad23;color:#173f34;border:0;border-radius:24px;padding:10px 18px;margin-top:12px;font:600 16px system-ui;cursor:pointer';card.append(button);
  const status=document.createElement('p');status.setAttribute('role','status');status.style.fontSize='14px';card.append(status);
  let version=0;
  function keepDraft(){version++;const value={audio_sha256:item.audio_sha256,notes:notes.value,updated_at_ms:Date.now()};for(const [key,select] of Object.entries(selects))value[key]=select.value;drafts[item.id]=value;storeDrafts();}
  for(const select of Object.values(selects))select.addEventListener('change',()=>{keepDraft();status.textContent=draftStorage?'Draft kept in this browser. Save feedback to copy it to Drive.':'Unsaved changes. Keep this page open until Drive confirms your save.';});
  notes.addEventListener('input',keepDraft);
  if(restored===draft)status.textContent='Browser draft restored. Save feedback to copy it to Drive.';
  button.onclick=async()=>{
   keepDraft();const submittedVersion=version;button.disabled=true;status.textContent='Saving…';
   const payload={id:item.id,nonce,notes:notes.value};for(const [key,select] of Object.entries(selects))payload[key]=select.value;
   try{const response=await google.colab.kernel.invokeFunction('anyanwu.expanded_cuts.save',[payload],{});let result=response.data['application/json'];if(typeof result==='string')result=JSON.parse(result);
    if(result.saved&&result.durable&&version===submittedVersion){delete drafts[item.id];storeDrafts();}
    status.textContent=result.saved?(result.durable?'Saved to Google Drive. ':'Saved in this runtime; Drive backup is unavailable. ')+(result.status==='accepted_by_user_for_pilot'?'Accepted for the research pilot.':result.status==='held_by_user'?'Held based on your feedback.':'Review remains incomplete.')+' No training started.':('Not saved: '+result.error);
   }catch(error){status.textContent='Drive save is not connected. '+(draftStorage?'Your draft is kept in this browser. ':'Keep this page open to preserve your choices. ')+'Run the Restore review connection cell above, then try Save feedback again.';}finally{button.disabled=false;}
  };
  list.append(card);
 }
 let callbackVerified=false;
 try{const response=await google.colab.kernel.invokeFunction('anyanwu.expanded_cuts.check',[nonce],{});let result=response.data['application/json'];if(typeof result==='string')result=JSON.parse(result);callbackVerified=result.ready&&result.candidate_count===data.length&&!result.feedback_records_changed;}catch(error){}
 if(callbackVerified&&draftStorage)banner.textContent='Drive save connection is ready. Edits are also kept as browser drafts until saved.';
 else if(!callbackVerified)banner.textContent='Save connection is inactive. Keep your choices here, run Restore review connection above, then Save feedback again.';
 else banner.textContent='Drive save connection is ready. Browser draft storage is unavailable, so keep this page open until each save is confirmed.';
 window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:{phase:'expanded_listening_form_ready',audio_players:root.querySelectorAll('audio').length,review_fields:root.querySelectorAll('select').length,restored_drive_reviews:data.filter(item=>item.review.saved_at_unix).length,held_cuts:__HELD__,callback_verified:callbackVerified,browser_draft_storage:draftStorage,dataset_backup_verified:true,dataset_backup:backup,automatic_training:false}},'https://colab.research.google.com');
})();
</script>
'''
    page=template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c'))
    page=page.replace('__NONCE__',json.dumps(nonce)).replace('__HELD__',str(alignment['flagged_cuts'])).replace('__BACKUP__',json.dumps(backup))
    display(HTML(page))
    return {'phase':'expanded_listening_form_prepared','review_cases':len(cases),'held_cuts':alignment['flagged_cuts'],
            'candidate_seconds':round(sum(c['candidate_seconds'] for c in cases),6),'automatic_training':False,
            'drive_reviews_path':str(drive/'human-reviews.json'),'human_reviews_saved':len(state['records'])}
