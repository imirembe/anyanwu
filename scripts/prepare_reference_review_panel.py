"""Combine existing clips and draft-protected feedback in a compact paged view."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'.build/VoiceResearch/auditions/conversational-reference-conditioning-v1'


def replace_once(source,before,after):
    if source.count(before)!=1:raise RuntimeError('Review template changed: '+before[:80])
    return source.replace(before,after)


def prepare():
    receipt=json.loads((FOLDER/'backup-result.json').read_text())
    if not receipt['durable_backup_verified'] or receipt['audio_clips']!=4:raise RuntimeError('Verified existing clips required')
    source=(FOLDER/'reference_review.py').read_text()
    source=replace_once(source,'<h2>Compare connected Igbo speech</h2>', '<h2 style="color:#173f34;margin:0 0 8px">Listen and give feedback</h2>')
    source=replace_once(source,'<p>Listen to both versions of each sentence. Compare the words, vowel sounds, tone, rhythm and naturalness. The text is the requested sentence, not a verified transcript of what the model said.</p>', '<p style="margin:0 0 8px">Listen, choose your ratings, then save each pair. Rhythm: 1 = poor, 5 = natural. Use Next for pair two.</p>')
    source=replace_once(source,'<p>Same preserved v1 model in both versions. Only the reference changes: reading versus the complete conversational recording you selected. Reference identity, style and duration change together. One take each; no new training or pitch edits.</p>', '<p style="margin:0 0 8px;font-size:14px">Same model, different references. The conversational clips are quieter; adjust playback volume comfortably.</p>')
    source=source.replace('padding:24px;font:16px/1.6','padding:16px;font:16px/1.4')
    source=source.replace('padding:20px;margin:22px 0','padding:14px;margin:12px 0;color:#173f34')
    source=replace_once(source," const title=document.createElement('h3');title.textContent=item.text;card.append(title);", " const title=document.createElement('h3');title.style.cssText='color:#173f34;font-size:17px;margin:0 0 8px';title.textContent=item.text;card.append(title);")
    source=replace_once(source,"p.textContent=label+' · '", "p.style.cssText='color:#173f34;margin:8px 0 4px';p.textContent=label+' · '")
    source=source.replace("'Which version sounds better?'","'Which sounds more natural?'")
    for a,b in [('Reading reference: do the spoken words match?','Reading: words match?'),('Conversational reference: do the spoken words match?','Conversational: words match?'),('Reading reference: natural rhythm (1 poor, 5 natural)','Reading: rhythm (1–5)'),('Conversational reference: natural rhythm (1 poor, 5 natural)','Conversational: rhythm (1–5)')]:source=source.replace(a,b)
    source=source.replace('display:block;margin:12px 0','display:block;margin:8px 0;color:#173f34')
    source=source.replace("select.style.cssText='font:inherit;padding:5px'", "select.style.cssText='font:inherit;padding:4px;color:#173f34;background:#fffdf8'")
    source=source.replace('Optional pronunciation or cadence notes','Optional notes: where does speech sound mechanical?')
    source=source.replace('min-height:75px;font:inherit','min-height:50px;font:inherit;color:#173f34;background:#fffdf8')
    source=replace_once(source,"button.textContent='Save feedback to Drive';", "button.textContent='Save feedback to Drive';button.setAttribute('data-save-feedback','true');")
    pager=r'''
const cards=[],pager=document.createElement('div'),previous=document.createElement('button'),pageLabel=document.createElement('span'),next=document.createElement('button');
pager.style.cssText='display:flex;gap:12px;align-items:center;margin:8px 0;color:#173f34';
previous.textContent='Previous';next.textContent='Next';
for(const b of [previous,next])b.style.cssText='font:inherit;padding:7px 14px;border-radius:20px;border:1px solid #c6d0be;background:#fffdf8;color:#173f34';
pager.append(previous,pageLabel,next);root.querySelector('#pairs').append(pager);let page=0;
function showPage(index){page=Math.max(0,Math.min(cards.length-1,index));for(let i=0;i<cards.length;i++){cards[i].style.display=i===page?'block':'none';if(i!==page)for(const audio of cards[i].querySelectorAll('audio'))if(typeof audio.pause==='function')audio.pause();}pageLabel.textContent='Pair '+(page+1)+' of '+cards.length;previous.disabled=page===0;next.disabled=page===cards.length-1;}
previous.onclick=()=>showPage(page-1);next.onclick=()=>showPage(page+1);
'''
    source=replace_once(source,"for(const item of data){\n const draft=",pager+"\nfor(const item of data){\n const draft=")
    source=replace_once(source," root.querySelector('#pairs').append(card);", " cards.push(card);root.querySelector('#pairs').append(card);")
    source=replace_once(source,'let ready=false;try{','showPage(0);\nlet ready=false;try{')
    source=source.replace("phase:'reference_conditioning_listening_ready'", "phase:'reference_conditioning_review_panel_ready'")
    source=replace_once(source,'audio_players:root.querySelectorAll(\'audio\').length,pairs:data.length,', "audio_players:root.querySelectorAll('audio').length,pairs:data.length,feedback_fields:root.querySelectorAll('select').length,visible_feedback_fields:[...root.querySelectorAll('select')].filter(e=>e.getBoundingClientRect().height>0).length,save_buttons:root.querySelectorAll('[data-save-feedback]').length,visible_save_buttons:[...root.querySelectorAll('[data-save-feedback]')].filter(e=>e.getBoundingClientRect().height>0).length,panel_height:Math.round(root.getBoundingClientRect().height),headings:[...root.querySelectorAll('h2,h3')].map(e=>({text:e.textContent,color:getComputedStyle(e).color})),")
    source=source.replace("'reference_conditioning_form_displayed'", "'reference_conditioning_review_panel_displayed'")
    tail='''
import importlib.util,platform
from IPython.display import Javascript,display
def publish_review_panel(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
 if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab only; no laptop media/models')
 if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Research Drive disconnected; use the existing CPU connection repair')
 saved=Path(RECEIPT['drive_archive']).parent/'human-comparison.json'
 if saved.exists() and saved.stat().st_size>16384:raise RuntimeError('Feedback file cap')
 before=saved.read_bytes() if saved.exists() else None
 restore_reference_listening(RECEIPT)
 show_reference_comparison(RECEIPT)
 after=saved.read_bytes() if saved.exists() else None
 publish_review_panel({'phase':'reference_conditioning_review_panel_loaded','existing_feedback_file_unchanged':before==after,'reviews_submitted_by_agent':False,'new_synthesis':False,'models_needed':False})
except Exception as error:
 publish_review_panel({'phase':'reference_conditioning_review_panel_stopped','error_type':type(error).__name__,'error':str(error)[:300],'reviews_submitted_by_agent':False,'new_synthesis':False})
 raise
'''
    cell='# Anyanwu — LISTEN AND GIVE FEEDBACK: existing clips only\n'+source+'\nRECEIPT='+repr(receipt)+'\n'+tail
    compile(source,'review-panel-module','exec');compile(cell,'review-panel-cell','exec')
    (FOLDER/'combined_review_panel.py').write_text(source);(FOLDER/'combined-review-cell.py').write_text(cell)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['# Anyanwụ · listen and give feedback\n','Existing saved clips only. Hosted CPU with research Drive connected; no synthesis, models or training. Saved feedback and valid browser drafts are restored.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':cell.splitlines(keepends=True)}]}
    (ROOT/'Voice Research/Anyanwu-Conversational-Reference-Review.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    result={'phase':'reference_conditioning_combined_review_prepared','source_sha256':hashlib.sha256(cell.encode()).hexdigest(),'new_synthesis':False,'models_needed':False,'feedback_schema_unchanged':True}
    (FOLDER/'combined-review-source-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
