"""Two whole-response transcript reviews, with explicit Drive saves and browser drafts."""
from pathlib import Path
import base64
import hashlib
import json
import re
import secrets
import time

REFERENCE_DRIVE = '/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research/references/conversational-igbo-v1'


def validate_transcript_review(payload, allowed, nonce):
    if not isinstance(payload, dict) or payload.get('nonce') != nonce:
        raise ValueError('Inactive review session')
    records = payload.get('records')
    if not isinstance(records, list) or len(records) != 2 or {r.get('id') for r in records if isinstance(r, dict)} != set(allowed):
        raise ValueError('Expected exactly the two selected recordings')
    checked = []
    for row in records:
        if not isinstance(row, dict) or (row.get('audio_sha256'), row.get('draft_sha256')) != allowed[row['id']]:
            raise ValueError('Source audio or draft differs')
        if any(row.get(k) not in ('unreviewed', 'yes', 'no') for k in ('words_match', 'clean_complete_speech')):
            raise ValueError('Invalid listening check')
        text, notes = row.get('transcript'), row.get('notes')
        if not isinstance(text, str) or len(text) > 6000 or not isinstance(notes, str) or len(notes) > 500:
            raise ValueError('Transcript/notes exceed their bounds')
        if row['words_match'] == 'yes' and not text.strip():
            raise ValueError('An empty transcript cannot be marked exact')
        checked.append({k: row[k] for k in ('id', 'audio_sha256', 'draft_sha256', 'transcript',
                                           'words_match', 'clean_complete_speech', 'notes')})
    return checked


def load_transcript_review():
    folder = Path(REFERENCE_DRIVE)
    path = folder/'transcription-v1/draft-transcripts.json'
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Reconnect research Google Drive; no model installation is needed')
    if path.is_symlink() or not 0 < path.stat().st_size <= 128*1024:
        raise RuntimeError('Draft output integrity/size failure')
    report = json.loads(path.read_bytes())
    rows = report.get('records', [])
    if (report.get('plan_id') != 'conversational-igbo-transcription-v1' or len(rows) != 2
            or [r['id'] for r in rows] != ['conversation-ref-03', 'conversation-ref-04']):
        raise RuntimeError('Unexpected transcript selection')
    for row in rows:
        wav = folder/(row['id']+'.wav')
        if (wav.is_symlink() or not 0 < wav.stat().st_size <= 2*1024**2
                or hashlib.sha256(wav.read_bytes()).hexdigest() != row['listening_sha256']
                or hashlib.sha256(row['draft_transcript'].encode()).hexdigest() != row['draft_sha256']):
            raise RuntimeError('Source audio/draft hash mismatch')
    return folder, report


def register_transcript_review(recover_existing_form=False, *, mount_root=None):
    from google.colab import output
    from IPython.display import JSON
    folder, report = load_transcript_review()
    mount_root = Path(mount_root) if mount_root is not None else Path('/content/drive/MyDrive')
    allowed = {r['id']: (r['listening_sha256'], r['draft_sha256']) for r in report['records']}
    nonce = None if recover_existing_form else secrets.token_hex(12)
    saved = folder/'transcription-v1/human-transcripts.json'
    known = saved.read_bytes() if saved.exists() else None
    state = {}
    if known is not None:
        if len(known) > 64*1024:
            raise RuntimeError('Saved transcript review size cap')
        value = json.loads(known)
        if value.get('experiment') != report['plan_id']:
            raise RuntimeError('Saved review experiment mismatch')
        records = validate_transcript_review({'nonce': 'read', 'records': list(value['records'].values())}, allowed, 'read')
        state = {r['id']: {**r, 'saved_at_ms': value['records'][r['id']]['saved_at_ms']} for r in records}

    def save(payload):
        nonlocal known, nonce
        try:
            proposed = payload.get('nonce') if isinstance(payload, dict) else None
            expected = nonce
            if expected is None:
                if not isinstance(proposed, str) or not re.fullmatch('[a-f0-9]{24}', proposed):
                    raise ValueError('Invalid existing-form session')
                expected = proposed
            records = validate_transcript_review(payload, allowed, expected)
            nonce = expected
            current = saved.read_bytes() if saved.exists() else None
            if current != known or not mount_root.is_dir():
                raise RuntimeError('Drive disconnected or review changed elsewhere; retain your browser draft')
            timestamp = round(time.time()*1000)
            updated = {r['id']: {**r, 'saved_at_ms': timestamp} for r in records}
            encoded = (json.dumps({'experiment': report['plan_id'], 'records': updated,
                                    'training_started': False, 'training_ready': False}, ensure_ascii=False, indent=2)+'\n').encode()
            if len(encoded) > 64*1024:
                raise ValueError('Saved transcript review size cap')
            stage = saved.with_name('human-transcripts.staging.json')
            stage.write_bytes(encoded)
            if stage.read_bytes() != encoded:
                raise RuntimeError('Drive staging verification failed')
            if known is not None:
                saved.with_name('human-transcripts.previous.json').write_bytes(known)
            stage.replace(saved)
            if saved.read_bytes() != encoded:
                raise RuntimeError('Final Drive save verification failed')
            known = encoded
            state.clear(); state.update(updated)
            return JSON({'saved': True, 'durable': True, 'saved_at_ms': timestamp,
                         'reviewed': sum(r['words_match'] == r['clean_complete_speech'] == 'yes' for r in records),
                         'training_started': False})
        except Exception as error:
            return JSON({'saved': False, 'durable': False, 'error': str(error)[:250], 'training_started': False})

    def check(value):
        valid = value == {'read_only_health': True} or value == nonce
        return JSON({'ready': bool(valid and mount_root.is_dir()),
                     'records': 2, 'saved_records': len(state), 'feedback_changed': False})

    output.register_callback('anyanwu.conversational_transcripts.save', save)
    output.register_callback('anyanwu.conversational_transcripts.check', check)
    return folder, report, nonce, state


def show_transcript_review():
    from IPython.display import HTML, display
    folder, report, nonce, state = register_transcript_review()
    items = []
    for row in report['records']:
        items.append({'id': row['id'], 'audio_sha256': row['listening_sha256'],
                      'draft_sha256': row['draft_sha256'], 'draft': row['draft_transcript'],
                      'seconds': row['seconds'], 'review': state.get(row['id'], {}),
                      'pauses': row['pause_analysis']['candidates'],
                      'audio': 'data:audio/wav;base64,'+base64.b64encode((folder/(row['id']+'.wav')).read_bytes()).decode()})
    template = r'''
<section id="anyanwu-conversation-transcripts" style="max-width:850px;padding:24px;border-radius:18px;background:#f7f5eb;color:#173f34;font:17px/1.6 system-ui">
<h2>Review the words · references 3 and 4</h2>
<p>Listen to each complete response and correct the text to exactly what is spoken, including dialect wording and any English. These are machine drafts; the topic prompts were not used as transcripts.</p>
<p id="ct-connection" role="status"></p><div id="ct-list"></div>
<button id="ct-save" style="font:600 16px system-ui;padding:12px 20px;border:0;border-radius:24px;background:#e9ad23;color:#173f34">Save both reviews to Drive</button>
<p id="ct-status" role="status"></p>
<p>Possible pauses are quiet intervals detected from audio, not confirmed phrase boundaries. Leave notes about phrasing, emphasis or unclear words. Full source timing is preserved.</p>
<p style="font-size:14px">Silencio Network, Igbo Speech Dataset, 2026. CC BY-NC 4.0. Research review only. Saving does not start training.</p></section>
<script>(async()=>{
const data=__DATA__,nonce=__NONCE__,root=document.getElementById('anyanwu-conversation-transcripts'),list=root.querySelector('#ct-list'),status=root.querySelector('#ct-status'),banner=root.querySelector('#ct-connection'),button=root.querySelector('#ct-save');
const key='anyanwu:conversational-transcripts:v1',choices=new Set(['unreviewed','yes','no']);let drafts={},storage=true,version=0;const cards=[];
function valid(d,item){return d&&d.audio_sha256===item.audio_sha256&&d.draft_sha256===item.draft_sha256&&typeof d.transcript==='string'&&d.transcript.length<=6000&&typeof d.notes==='string'&&d.notes.length<=500&&choices.has(d.words_match)&&choices.has(d.clean_complete_speech)&&Number.isFinite(d.updated_at_ms);}
try{const raw=localStorage.getItem(key);if(raw&&raw.length<=64000){const old=JSON.parse(raw);for(const item of data)if(valid(old[item.id],item))drafts[item.id]=old[item.id];}localStorage.setItem(key,JSON.stringify(drafts));}catch(e){storage=false;}
function persist(){if(!storage)return;try{const raw=JSON.stringify(drafts);if(raw.length>64000)throw Error('Draft limit');localStorage.setItem(key,raw);}catch(e){storage=false;banner.textContent='Browser draft storage is unavailable. Keep this page open until Drive confirms saving.';}}
for(const item of data){const saved=item.review,draft=drafts[item.id];const restored=draft&&draft.updated_at_ms>(saved.saved_at_ms||0)?draft:saved;
 const card=document.createElement('article');card.style.cssText='border:1px solid #dfe3d5;border-radius:16px;background:#fffdf8;padding:20px;margin:18px 0';
 const title=document.createElement('h3');title.textContent=item.id+' · '+item.seconds.toFixed(2)+'s';card.append(title);
 const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=item.audio;audio.style.width='100%';card.append(audio);
 const label=document.createElement('label');label.textContent='Corrected transcript';label.style.display='block';const text=document.createElement('textarea');text.maxLength=6000;text.value=restored.transcript??item.draft;text.style.cssText='width:100%;box-sizing:border-box;min-height:180px;font:inherit;line-height:1.7';label.append(text);card.append(label);
 const selects={};for(const [field,prompt] of [['words_match','Does the corrected text match every spoken word?'],['clean_complete_speech','Is the response complete and free of overlapping speech/music?']]){const row=document.createElement('label');row.style.cssText='display:block;margin:10px 0';row.append(document.createTextNode(prompt+' '));const select=document.createElement('select');select.style.font='inherit';for(const [value,caption] of [['unreviewed','Not checked'],['yes','Yes'],['no','No']]){const option=document.createElement('option');option.value=value;option.textContent=caption;select.append(option);}select.value=restored[field]||'unreviewed';selects[field]=select;row.append(select);card.append(row);}
 const notes=document.createElement('textarea');notes.maxLength=500;notes.value=restored.notes||'';notes.placeholder='Unclear words or phrasing observations (500 characters)';notes.style.cssText='width:100%;box-sizing:border-box;min-height:70px;font:inherit';card.append(notes);
 const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Machine pause candidates (need listening review)';details.append(summary);const pauses=document.createElement('p');pauses.textContent=item.pauses.map(p=>p.start_seconds.toFixed(2)+'–'+p.end_seconds.toFixed(2)+'s ('+p.position+')').join('; ')||'No quiet intervals met the threshold.';details.append(pauses);card.append(details);
 function value(){return {id:item.id,audio_sha256:item.audio_sha256,draft_sha256:item.draft_sha256,transcript:text.value,words_match:selects.words_match.value,clean_complete_speech:selects.clean_complete_speech.value,notes:notes.value,updated_at_ms:Date.now()};}
 function keep(){version++;drafts[item.id]=value();persist();status.textContent=storage?'Draft kept here until saved to Drive.':'Unsaved: keep this page open.';}
 text.oninput=notes.oninput=keep;for(const select of Object.values(selects))select.onchange=keep;cards.push({id:item.id,value});list.append(card);
}
button.onclick=async()=>{for(const card of cards)drafts[card.id]=card.value();version++;const sent=version,payload={nonce,records:cards.map(c=>c.value())};persist();button.disabled=true;status.textContent='Saving…';try{const response=await google.colab.kernel.invokeFunction('anyanwu.conversational_transcripts.save',[payload],{});let r=response.data['application/json'];if(typeof r==='string')r=JSON.parse(r);if(!r.saved||!r.durable)throw Error(r.error||'Save not confirmed');if(version===sent){drafts={};persist();}status.textContent='Saved to Google Drive · '+r.reviewed+' of 2 marked exact and complete. No training started.';}catch(e){status.textContent='Not saved. Your browser draft is retained. Reconnect the review and retry. '+String(e.message||e).slice(0,180);}finally{button.disabled=false;}};
let ready=false;try{const response=await google.colab.kernel.invokeFunction('anyanwu.conversational_transcripts.check',[nonce],{});let r=response.data['application/json'];if(typeof r==='string')r=JSON.parse(r);ready=r.ready&&r.records===2&&!r.feedback_changed;}catch(e){}
banner.textContent=ready?'Drive save connection ready. '+(storage?'Drafts are kept until a confirmed save.':'Keep this page open until saved.'):'Drive save connection is inactive. Keep your choices and reconnect the review.';
window.top.postMessage({type:'anyanwu-cloud-diagnostic',nonce:'anyanwu-20261122-setup-r2',payload:{phase:'conversational_asr_review_ready',players:root.querySelectorAll('audio').length,transcript_editors:2,callback_verified:ready,browser_drafts:storage,restored_drive_reviews:data.filter(i=>i.review.saved_at_ms).length,training_started:false}},'https://colab.research.google.com');
})();</script>'''
    def embedded(value):
        return json.dumps(value, ensure_ascii=False).replace('<', '\\u003c')
    display(HTML(template.replace('__DATA__', embedded(items)).replace('__NONCE__', embedded(nonce))))
