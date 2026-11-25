"""Bounded whole-source listening review with Drive saves and browser drafts."""
from pathlib import Path
import base64
import hashlib
import json
import re
import secrets
import time


def validate_source_feedback(payload,allowed,nonce):
    if not isinstance(payload,dict) or payload.get('nonce')!=nonce:raise ValueError('Inactive review session')
    if payload.get('id') not in allowed or payload.get('audio_sha256')!=allowed[payload['id']]:raise ValueError('Unknown source/audio hash')
    choices=('words_match','complete_utterance','clean_female_voice')
    if any(payload.get(k) not in {'unreviewed','yes','no'} for k in choices):raise ValueError('Invalid review choice')
    if not isinstance(payload.get('notes'),str) or len(payload['notes'])>500:raise ValueError('Notes exceed500 characters')
    record={k:payload[k] for k in ('id','audio_sha256',*choices,'notes')}
    record['status']='accepted_for_research' if all(record[k]=='yes' for k in choices) else 'held_by_user' if any(record[k]=='no' for k in choices) else 'pending'
    return record


def restore_source_expansion(receipt):
    import zipfile
    archive=Path(receipt['drive_archive']);root=Path('/content/anyanwu-waxal-female5-expansion-v1')
    if archive.stat().st_size!=receipt['archive_bytes'] or archive.stat().st_size>16*1024**2 or hashlib.sha256(archive.read_bytes()).hexdigest()!=receipt['archive_sha256']:
        raise RuntimeError('Source backup integrity failure')
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        manifest=json.loads(package.read('manifest.json'))
        if len(manifest['files'])>60 or sum(v['bytes'] for v in manifest['files'].values())>16*1024**2:raise RuntimeError('Source restore size/count cap')
        allowed={'plan.json','data-result.json','SOURCE_CARD.txt','ATTRIBUTION.txt','source_expansion_review.py','waxal_expansion_data.py','link-refresh-receipt.json','acquisition-attempt-1-expired-link.json'}
        for name,meta in manifest['files'].items():
            if name not in allowed and not re.fullmatch(r'data/ibo_[0-9]+\.(mp3|wav)',name):raise RuntimeError('Unsafe source archive member')
            if meta['bytes']>1024**2:raise RuntimeError('Source member size cap')
            # Playback/review recovery needs only the WAVs and provenance, not source MP3s or executable files.
            if name not in {'plan.json','data-result.json'} and not name.endswith('.wav'):continue
            body=package.read(name)
            if len(body)!=meta['bytes'] or hashlib.sha256(body).hexdigest()!=meta['sha256']:raise RuntimeError('Source member hash mismatch')
            path=root/name;path.parent.mkdir(exist_ok=True)
            if path.exists() and path.read_bytes()!=body:raise RuntimeError('Different existing source file; no overwrite')
            if not path.exists():path.write_bytes(body)
    return root


def register_source_review_connection(recover_existing_form=False):
    from google.colab import output
    from IPython.display import JSON
    root=Path('/content/anyanwu-waxal-female5-expansion-v1')
    drive=Path('/content/drive/MyDrive/Anyanwu Voice Research/datasets/waxal-female5-expansion-v1')
    plan=json.loads((root/'plan.json').read_text());result=json.loads((root/'data-result.json').read_text())
    cases=result['clips'];allowed={c['id']:c['audio_sha256'] for c in cases}
    if not 1<=len(cases)<=24 or len(allowed)!=len(cases):raise RuntimeError('Unexpected source review count')
    nonce=None if recover_existing_form else secrets.token_hex(12)
    state={};saved=drive/'source-reviews.json'
    if saved.exists():
        if saved.stat().st_size>64*1024:raise RuntimeError('Saved source review cap')
        previous=json.loads(saved.read_text())
        if previous.get('experiment')!=plan['id'] or not isinstance(previous.get('records'),dict):raise RuntimeError('Saved source review experiment mismatch')
        state=previous['records']
        for key,record in state.items():
            if key!=record.get('id'):raise RuntimeError('Saved source review ID mismatch')
            validate_source_feedback({**record,'nonce':'snapshot'},allowed,'snapshot')
    def counts(records):
        statuses=[validate_source_feedback({**r,'nonce':'snapshot'},allowed,'snapshot')['status'] for r in records.values()]
        return {'saved_records':len(records),'accepted':statuses.count('accepted_for_research'),'held':statuses.count('held_by_user'),'incomplete':len(cases)-statuses.count('accepted_for_research')-statuses.count('held_by_user')}
    def save(payload):
        nonlocal nonce
        try:
            expected=nonce
            if expected is None:
                proposed=payload.get('nonce') if isinstance(payload,dict) else None
                if not isinstance(proposed,str) or not re.fullmatch(r'[0-9a-f]{24}',proposed):raise ValueError('Invalid existing-form session')
                expected=proposed
            record=validate_source_feedback(payload,allowed,expected)
            # Adopt an old form only after all ID/hash/choice/note checks pass.
            nonce=expected;record['saved_at_ms']=round(time.time()*1000)
            updated={**state,record['id']:record};encoded=json.dumps({'experiment':plan['id'],'records':updated,'automatic_training':False},ensure_ascii=False,indent=2)+'\n'
            if len(encoded.encode())>64*1024 or not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive unavailable; keep the browser draft')
            drive.mkdir(exist_ok=True,parents=True);stage=drive/'source-reviews.staging.json';stage.write_text(encoded)
            if stage.read_text()!=encoded:raise RuntimeError('Drive review staging verification failed')
            if saved.exists():(drive/'source-reviews.previous.json').write_bytes(saved.read_bytes())
            stage.replace(saved)
            if saved.read_text()!=encoded:raise RuntimeError('Final Drive review verification failed')
            state.clear();state.update(updated)
            return JSON({'saved':True,'durable':True,'id':record['id'],'status':record['status'],**counts(state),'training_started':False})
        except Exception as error:return JSON({'saved':False,'durable':False,'error':str(error)[:220],'training_started':False})
    def check(value):
        health=isinstance(value,dict) and value=={'read_only_health':True}
        valid=health or ((value==nonce) if nonce is not None else isinstance(value,str) and re.fullmatch(r'[0-9a-f]{24}',value) is not None)
        return JSON({'ready':bool(valid and Path('/content/drive/MyDrive').is_dir()),'source_count':len(cases),'feedback_changed':False,
                     'recovered_existing_form':recover_existing_form,'session_bound':nonce is not None,'health_only':health,**counts(state)})
    output.register_callback('anyanwu.waxal_expansion.save',save);output.register_callback('anyanwu.waxal_expansion.check',check)
    return nonce,state,counts


def show_source_expansion(receipt):
    from IPython.display import HTML,display
    root=Path('/content/anyanwu-waxal-female5-expansion-v1')
    result=json.loads((root/'data-result.json').read_text());cases=result['clips']
    nonce,state,counts=register_source_review_connection()
    data=[];total=0
    for c in cases:
        path=root/'data'/(c['id']+'.wav');audio=path.read_bytes();total+=len(audio)
        if hashlib.sha256(audio).hexdigest()!=c['audio_sha256'] or total>12*1024**2:raise RuntimeError('Source review audio hash/size failure')
        data.append({'id':c['id'],'text':c['text'],'seconds':c['seconds'],'audio_sha256':c['audio_sha256'],
          'audio':'data:audio/wav;base64,'+base64.b64encode(audio).decode(),'review':state.get(c['id'],{})})
    template=r'''
<div id="anyanwu-sources" style="background:#f7f5eb;color:#173f34;padding:24px;font:16px/1.6 system-ui;max-width:850px;border-radius:16px">
<h2>Review the new female Igbo recordings</h2>
<p>These are whole source recordings, not the generated model. Check the exact words, complete beginning and ending, and clean single female speech. Only accepted recordings can enter the next training plan.</p>
<p>You can use <b>Accept all three checks and save</b> after listening, or change individual checks and save. Mark a problem No and add a note. Saving never starts training.</p>
<p id="source-connection" role="status"></p><p id="source-counts"></p><div id="source-pager"></div><div id="source-list"></div>
<p style="font-size:14px">Google WAXAL / Media Trust Igbo TTS · CC BY-SA 4.0 · source speaker5, publisher-labelled Female. Original lossy previews decoded whole. Source text is a pronunciation target; historical/factual claims are not verified assistant knowledge. Four separate texts remain reserved for the next generated voice test.</p></div>
<script>(async()=>{
const data=__DATA__,nonce=__NONCE__,backup=__BACKUP__,initial=__COUNTS__,root=document.getElementById('anyanwu-sources'),list=root.querySelector('#source-list'),banner=root.querySelector('#source-connection'),summary=root.querySelector('#source-counts'),pager=root.querySelector('#source-pager');
const key='anyanwu:waxal-female5-source-drafts:v1',fields=['words_match','complete_utterance','clean_female_voice'],allowed=new Set(['unreviewed','yes','no']);let storage=true,drafts={},page=0,cards=[];
function showCounts(value){summary.textContent=value.accepted+' accepted · '+value.held+' held · '+value.incomplete+' still to finish ('+data.length+' recordings).';}showCounts(initial);
try{const raw=localStorage.getItem(key);if(raw&&raw.length<=65536){const parsed=JSON.parse(raw);for(const item of data){const d=parsed[item.id];if(d&&d.audio_sha256===item.audio_sha256&&typeof d.notes==='string'&&d.notes.length<=500&&Number.isFinite(d.updated_at_ms)&&fields.every(k=>allowed.has(d[k])))drafts[item.id]=d;}}localStorage.setItem(key,JSON.stringify(drafts));}catch(e){storage=false;}
function persist(){if(!storage)return;try{const raw=JSON.stringify(drafts);if(raw.length>65536)throw Error('Draft cap');localStorage.setItem(key,raw);}catch(e){storage=false;banner.textContent='Browser draft storage is unavailable. Keep this page open until Drive confirms saving.';}}
const previous=document.createElement('button'),next=document.createElement('button'),pageLabel=document.createElement('span');previous.textContent='Previous';next.textContent='Next';
for(const button of [previous,next])button.style.cssText='font:inherit;padding:8px 16px;border:1px solid #dfe3d5;border-radius:20px;background:#fffdf8;color:#173f34;margin:0 10px';pager.append(previous,pageLabel,next);
function showPage(){cards.forEach((card,i)=>card.style.display=Math.floor(i/5)===page?'block':'none');previous.disabled=page===0;next.disabled=page>=Math.ceil(data.length/5)-1;pageLabel.textContent='Page '+(page+1)+' of '+Math.ceil(data.length/5);}
previous.onclick=()=>{page=Math.max(0,page-1);showPage();pager.scrollIntoView?.({block:'start'});};next.onclick=()=>{page=Math.min(Math.ceil(data.length/5)-1,page+1);showPage();pager.scrollIntoView?.({block:'start'});};
for(const item of data){
 const draft=drafts[item.id],restored=draft&&draft.updated_at_ms>(item.review.saved_at_ms||0)?draft:item.review;
 const card=document.createElement('section');card.style.cssText='background:#fffdf8;border:1px solid #dfe3d5;border-radius:14px;padding:20px;margin:20px 0';
 const title=document.createElement('h3');title.textContent=item.id+' · '+item.seconds.toFixed(2)+'s';card.append(title);const text=document.createElement('p');text.textContent=item.text;card.append(text);
 const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=item.audio;audio.style.width='100%';card.append(audio);
 const labels={words_match:'Do all spoken words match the text?',complete_utterance:'Are the first and last words complete?',clean_female_voice:'Is this clean single female speech without music/overlap?'},selects={};
 for(const field of fields){const label=document.createElement('label');label.style.cssText='display:block;margin:10px 0';label.append(document.createTextNode(labels[field]+' '));const select=document.createElement('select');select.style.cssText='font:inherit;padding:5px';for(const [value,caption] of [['unreviewed','Not checked'],['yes','Yes'],['no','No']]){const option=document.createElement('option');option.value=value;option.textContent=caption;select.append(option);}select.value=restored[field]||'unreviewed';selects[field]=select;label.append(select);card.append(label);}
 const notes=document.createElement('textarea');notes.maxLength=500;notes.placeholder='Optional correction or audio problem (500 characters)';notes.value=restored.notes||'';notes.style.cssText='display:block;box-sizing:border-box;width:100%;min-height:65px;font:inherit';card.append(notes);
 const accept=document.createElement('button'),save=document.createElement('button');accept.textContent='Accept all three checks and save';save.textContent='Save current checks';
 for(const button of [accept,save])button.style.cssText='font:600 15px system-ui;padding:10px 16px;margin:12px 8px 0 0;border:1px solid #dfe3d5;border-radius:24px;color:#173f34;background:#fffdf8';accept.style.background='#e9ad23';card.append(accept,save);
 const status=document.createElement('p');status.setAttribute('role','status');card.append(status);let version=0;
 function keep(){version++;const d={audio_sha256:item.audio_sha256,notes:notes.value,updated_at_ms:Date.now()};for(const field of fields)d[field]=selects[field].value;drafts[item.id]=d;persist();}
 for(const select of Object.values(selects))select.onchange=()=>{keep();status.textContent=storage?'Draft kept here until saved to Drive.':'Unsaved; keep the page open.';};notes.oninput=keep;
 if(restored===draft)status.textContent='Browser draft restored.';else if(restored.saved_at_ms)status.textContent='Saved Drive review restored: '+restored.status;
 async function submit(){keep();const sent=version,payload={id:item.id,nonce,...drafts[item.id]};save.disabled=accept.disabled=true;status.textContent='Saving…';try{const response=await google.colab.kernel.invokeFunction('anyanwu.waxal_expansion.save',[payload],{});let r=response.data['application/json'];if(typeof r==='string')r=JSON.parse(r);if(r.saved&&r.durable){if(version===sent){delete drafts[item.id];persist();}showCounts(r);status.textContent='Saved to Google Drive · '+(r.status==='accepted_for_research'?'accepted for the research sample':r.status==='held_by_user'?'held for correction':'review incomplete')+'. No training started.';}else status.textContent='Not saved: '+r.error;}catch(e){banner.textContent='Save connection is inactive. Keep this page open and run Restore review connection above, then retry the same Save button.';status.textContent='Drive connection is inactive. Your browser draft is retained. Restore the review connection above, then save again. '+String(e?.message||e).slice(0,180);}finally{save.disabled=accept.disabled=false;}}
 save.onclick=submit;accept.onclick=()=>{for(const field of fields)selects[field].value='yes';return submit();};list.append(card);cards.push(card);
}showPage();
let ready=false;try{const response=await google.colab.kernel.invokeFunction('anyanwu.waxal_expansion.check',[nonce],{});let r=response.data['application/json'];if(typeof r==='string')r=JSON.parse(r);ready=r.ready&&r.source_count===data.length&&!r.feedback_changed;}catch(e){}
banner.textContent=ready?'Drive save connection is ready. '+(storage?'Drafts are kept until a confirmed save.':'Keep the page open until saved.'):'Save connection is inactive; retain your choices and restore the review connection.';
window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:{phase:'expanded_source_listening_ready',audio_players:root.querySelectorAll('audio').length,review_fields:root.querySelectorAll('select').length,pages:Math.ceil(data.length/5),callback_verified:ready,browser_draft_storage:storage,restored_drive_reviews:data.filter(x=>x.review.saved_at_ms).length,automatic_training:false,backup}},'https://colab.research.google.com');
})();</script>
'''
    page=template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')).replace('__NONCE__',json.dumps(nonce)).replace('__BACKUP__',json.dumps(receipt)).replace('__COUNTS__',json.dumps(counts(state)))
    display(HTML(page))
    return {'phase':'expanded_source_form_displayed','recordings':len(data),'human_review_pending':True,'automatic_training':False}
