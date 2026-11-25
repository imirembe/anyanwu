"""Audio-only recovery and explicit bounded human comparison feedback in Colab."""
from pathlib import Path
import base64
import hashlib
import json
import secrets
import time


def validate_comparison_feedback(payload,allowed,nonce):
    if not isinstance(payload,dict) or payload.get('nonce')!=nonce:raise ValueError('Inactive listening session')
    item=payload.get('id')
    if item not in allowed or payload.get('audio_hashes')!=allowed[item]:raise ValueError('Unknown comparison audio')
    if payload.get('preference') not in {'unreviewed','base','adapter','tie'}:raise ValueError('Invalid preference')
    for key in ('words_base','words_adapter'):
        if payload.get(key) not in {'unreviewed','yes','no'}:raise ValueError('Invalid word-match choice')
    for key in ('cadence_base','cadence_adapter'):
        if payload.get(key) not in {'unreviewed','1','2','3','4','5'}:raise ValueError('Invalid cadence score')
    notes=payload.get('notes')
    if not isinstance(notes,str) or len(notes)>500:raise ValueError('Notes must contain at most 500 characters')
    return {key:payload[key] for key in ('id','audio_hashes','preference','words_base','words_adapter','cadence_base','cadence_adapter','notes')}


def restore_comparison(receipt):
    import zipfile
    path=Path(receipt['drive_archive']);root=Path('/content/anyanwu-waxal-female5-comparison-v1')
    if path.stat().st_size!=receipt['archive_bytes'] or path.stat().st_size>16*1024**2 or hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['archive_sha256']:
        raise RuntimeError('Saved listening archive integrity failure')
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('archive-manifest.json'))
        if len(manifest['files'])>30 or sum(v['bytes'] for v in manifest['files'].values())>16*1024**2:raise RuntimeError('Recovery file cap')
        for name,meta in manifest['files'].items():
            if Path(name).name!=name or name.startswith('.') or meta['bytes']>3*1024**2:raise RuntimeError('Unsafe recovery member')
            content=archive.read(name)
            if len(content)!=meta['bytes'] or hashlib.sha256(content).hexdigest()!=meta['sha256']:raise RuntimeError('Recovery member hash mismatch')
            target=root/name
            if target.exists() and target.read_bytes()!=content:raise RuntimeError('Different existing listening file; no overwrite')
            if not target.exists():target.write_bytes(content)
    return root


def show_comparison(receipt):
    from google.colab import output
    from IPython.display import HTML,JSON,display
    root=Path('/content/anyanwu-waxal-female5-comparison-v1')
    drive=Path('/content/drive/MyDrive/Anyanwu Voice Research/auditions/waxal-female5-adapter-comparison-v1')
    result=json.loads((root/'generation-result.json').read_text());plan=json.loads((root/'plan.json').read_text())
    if len(result['clips'])!=6 or result['completed_wavs']!=6:raise RuntimeError('Incomplete generated comparison')
    pairs=[];total=0;allowed={}
    for text in plan['texts']:
        pair={'id':text['id'],'text':text['text'],'clips':{},'review':{}}
        for variant in ('base','adapter'):
            clip=next(c for c in result['clips'] if c['text_id']==text['id'] and c['variant']==variant)
            audio=(root/clip['audio_filename']).read_bytes();total+=len(audio)
            if hashlib.sha256(audio).hexdigest()!=clip['audio_sha256'] or total>12*1024**2:raise RuntimeError('Listening audio integrity/size failure')
            pair['clips'][variant]={'sha256':clip['audio_sha256'],'seconds':clip['seconds'],'warnings':clip['warnings'],'stop_reason':clip['stop_reason'],
                                   'audio':'data:audio/wav;base64,'+base64.b64encode(audio).decode()}
        allowed[text['id']]={v:pair['clips'][v]['sha256'] for v in ('base','adapter')};pairs.append(pair)
    saved=drive/'human-comparison.json';state={}
    if saved.exists():
        if saved.stat().st_size>16*1024:raise RuntimeError('Saved feedback cap')
        state=json.loads(saved.read_text()).get('records',{})
    nonce=secrets.token_hex(12)
    for pair in pairs:
        prior=state.get(pair['id'])
        if prior:
            validate_comparison_feedback({**prior,'nonce':nonce},allowed,nonce);pair['review']=prior
    def save(payload):
        try:
            review=validate_comparison_feedback(payload,allowed,nonce);review['saved_at_ms']=round(time.time()*1000)
            updated={**state,review['id']:review};data=json.dumps({'experiment':plan['id'],'records':updated,'automatic_training':False},ensure_ascii=False,indent=2)+'\n'
            if len(data.encode())>16*1024 or not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive is unavailable; browser draft remains')
            drive.mkdir(exist_ok=True,parents=True)
            stage=drive/'human-comparison.staging.json';stage.write_text(data)
            if stage.read_text()!=data:raise RuntimeError('Drive feedback verification failed')
            if saved.exists():(drive/'human-comparison.previous.json').write_bytes(saved.read_bytes())
            stage.replace(saved)
            if saved.read_text()!=data:raise RuntimeError('Final Drive feedback verification failed')
            state.clear();state.update(updated)
            return JSON({'saved':True,'durable':True,'id':review['id'],'automatic_training':False})
        except Exception as error:return JSON({'saved':False,'durable':False,'error':str(error)[:200]})
    def check(value):return JSON({'ready':value==nonce and Path('/content/drive/MyDrive').is_dir(),'pairs':3,'feedback_changed':False})
    output.register_callback('anyanwu.female_comparison.save',save)
    output.register_callback('anyanwu.female_comparison.check',check)
    template=r'''
<div id="anyanwu-comparison" style="background:#f7f5eb;color:#173f34;padding:24px;font:16px/1.6 system-ui;max-width:850px;border-radius:16px">
<h2>Listen to the female Igbo comparison</h2>
<p>Listen to both versions of each sentence. Compare the words, vowel sounds, tone, rhythm and naturalness. The text is the requested sentence, not a verified transcript of what the model said.</p>
<p>Original = published Flash base. Trained = the same model with our 60-update adapter. Both use the same female source reference and decoder. One take each; no pitch edits or extra training.</p>
<p id="connection" role="status"></p><div id="pairs"></div>
<p style="font-size:14px">Research audio, not a validated final Anyanwụ voice. Source text and reference: Media Trust / Google WAXAL, CC BY-SA 4.0. Flash checkpoint: Modularcomputing, CC BY-SA 4.0; CosyVoice3: FunAudioLLM, Apache 2.0. These texts were excluded from this adapter run; the base may have seen them.</p></div>
<script>(async()=>{
const data=__DATA__,nonce=__NONCE__,backup=__BACKUP__,root=document.getElementById('anyanwu-comparison'),banner=root.querySelector('#connection'),key='anyanwu:female-comparison-drafts:v1';
const fields={preference:['unreviewed','base','adapter','tie'],words_base:['unreviewed','yes','no'],words_adapter:['unreviewed','yes','no'],cadence_base:['unreviewed','1','2','3','4','5'],cadence_adapter:['unreviewed','1','2','3','4','5']};
let drafts={},storage=true;
const hashes=item=>Object.fromEntries(Object.entries(item.clips).map(([k,v])=>[k,v.sha256]));
const matches=(a,b)=>a&&a.base===b.base&&a.adapter===b.adapter;
try{const raw=localStorage.getItem(key);if(raw&&raw.length<=16384){const parsed=JSON.parse(raw);for(const item of data){const d=parsed[item.id];if(d&&matches(d.audio_hashes,hashes(item))&&typeof d.notes==='string'&&d.notes.length<=500&&Number.isFinite(d.updated_at_ms)&&Object.entries(fields).every(([k,v])=>v.includes(d[k])))drafts[item.id]=d;}}localStorage.setItem(key,JSON.stringify(drafts));}catch(e){storage=false;}
function persist(){try{if(!storage)return;const raw=JSON.stringify(drafts);if(raw.length>16384)throw Error('Draft cap');localStorage.setItem(key,raw);}catch(e){storage=false;banner.textContent='Browser drafts unavailable. Keep this page open until Drive confirms your save.';}}
for(const item of data){
 const draft=drafts[item.id],restored=draft&&draft.updated_at_ms>(item.review.saved_at_ms||0)?draft:item.review;
 const card=document.createElement('section');card.style.cssText='background:#fffdf8;border:1px solid #dfe3d5;border-radius:14px;padding:20px;margin:22px 0';
 const title=document.createElement('h3');title.textContent=item.text;card.append(title);
 for(const [v,label] of [['base','Original model'],['adapter','Trained adapter']]){const p=document.createElement('p');p.textContent=label+' · '+item.clips[v].seconds.toFixed(2)+'s';card.append(p);const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=item.clips[v].audio;audio.style.width='100%';card.append(audio);if(item.clips[v].warnings.length){const warning=document.createElement('p');warning.textContent=item.clips[v].warnings.join(' ');warning.style.color='#855900';card.append(warning);}}
 const labels={preference:'Which version sounds better?',words_base:'Original: do the spoken words match?',words_adapter:'Trained: do the spoken words match?',cadence_base:'Original: natural rhythm (1 poor, 5 natural)',cadence_adapter:'Trained: natural rhythm (1 poor, 5 natural)'};
 const captions={unreviewed:'Not checked',base:'Original',adapter:'Trained',tie:'About the same',yes:'Yes',no:'No'},selects={};
 for(const [field,choices] of Object.entries(fields)){const label=document.createElement('label');label.style.cssText='display:block;margin:12px 0';label.append(document.createTextNode(labels[field]+' '));const select=document.createElement('select');select.style.cssText='font:inherit;padding:5px';for(const value of choices){const option=document.createElement('option');option.value=value;option.textContent=captions[value]||value;select.append(option);}select.value=restored[field]||'unreviewed';selects[field]=select;label.append(select);card.append(label);}
 const notes=document.createElement('textarea');notes.maxLength=500;notes.placeholder='Optional pronunciation or cadence notes';notes.value=restored.notes||'';notes.style.cssText='display:block;box-sizing:border-box;width:100%;min-height:75px;font:inherit';card.append(notes);
 const button=document.createElement('button');button.textContent='Save feedback to Drive';button.style.cssText='background:#e9ad23;border:0;border-radius:24px;padding:12px 20px;margin-top:12px;font:600 16px system-ui;color:#173f34';card.append(button);
 const status=document.createElement('p');status.setAttribute('role','status');card.append(status);let version=0;
 function keep(){version++;const d={audio_hashes:hashes(item),notes:notes.value,updated_at_ms:Date.now()};for(const [k,v] of Object.entries(selects))d[k]=v.value;drafts[item.id]=d;persist();}
 for(const select of Object.values(selects))select.onchange=()=>{keep();status.textContent=storage?'Draft kept in this browser. Save to copy to Drive.':'Unsaved: keep this page open.';};notes.oninput=keep;
 if(restored===draft)status.textContent='Browser draft restored.';else if(restored.saved_at_ms)status.textContent='Saved Drive feedback restored.';
 button.onclick=async()=>{keep();const sent=version,payload={id:item.id,nonce,...drafts[item.id]};button.disabled=true;status.textContent='Saving…';try{const reply=await google.colab.kernel.invokeFunction('anyanwu.female_comparison.save',[payload],{});let result=reply.data['application/json'];if(typeof result==='string')result=JSON.parse(result);if(result.saved&&result.durable){if(version===sent){delete drafts[item.id];persist();}status.textContent='Saved to Google Drive. No training started.';}else status.textContent='Not saved: '+result.error;}catch(e){status.textContent='Save connection is inactive. Your browser draft is retained; use the audio-review recovery notebook, then save again.';}finally{button.disabled=false;}};
 root.querySelector('#pairs').append(card);
}
let ready=false;try{const reply=await google.colab.kernel.invokeFunction('anyanwu.female_comparison.check',[nonce],{});let r=reply.data['application/json'];if(typeof r==='string')r=JSON.parse(r);ready=r.ready&&r.pairs===3&&!r.feedback_changed;}catch(e){}
banner.textContent=ready?'Drive save connection is ready. '+(storage?'Drafts are kept until saved.':'Keep this page open until you save.'):'Save connection is inactive; keep your choices until the connection is restored.';
window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:{phase:'female_comparison_listening_ready',audio_players:root.querySelectorAll('audio').length,pairs:data.length,callback_verified:ready,browser_draft_storage:storage,restored_drive_reviews:data.filter(x=>x.review.saved_at_ms).length,automatic_training:false,backup}},'https://colab.research.google.com');
})();</script>
'''
    page=template.replace('__DATA__',json.dumps(pairs,ensure_ascii=False).replace('<','\\u003c')).replace('__NONCE__',json.dumps(nonce)).replace('__BACKUP__',json.dumps(receipt))
    display(HTML(page))
    return {'phase':'female_comparison_form_displayed','audio_players':6,'pairs':3,'durable_audio_backup':receipt['durable_backup_verified'],'human_listening_pending':True}
