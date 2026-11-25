// Exercise actual source-review JS without browser, network, audio or live saves.
import fs from 'node:fs';import vm from 'node:vm';import assert from 'node:assert/strict';
const {script}=JSON.parse(fs.readFileSync(0,'utf8'));const storage=new Map();
let failSave=true,blocked=false,defer=false,resolveSave,savedCalls=0,lastPayload;
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.style={};this.value='';this.textContent='';}
 append(...items){this.children.push(...items);}
 setAttribute(key,value){this[key]=value;}
 querySelectorAll(tag){return this.children.flatMap(c=>c instanceof Element?[...(c.tag===tag?[c]:[]),...c.querySelectorAll(tag)]:[]);}
}
async function render(){
 const root=new Element('root'),list=new Element('div'),banner=new Element('p'),summary=new Element('p'),pager=new Element('div');root.append(list,banner,summary,pager);
 root.querySelector=id=>({'#source-list':list,'#source-connection':banner,'#source-counts':summary,'#source-pager':pager}[id]);const notices=[];
 const context={document:{getElementById:()=>root,createElement:tag=>new Element(tag),createTextNode:text=>text},
 localStorage:{getItem:key=>{if(blocked)throw Error('blocked');return storage.get(key)||null;},setItem:(key,value)=>{if(blocked)throw Error('blocked');storage.set(key,value);}},window:{top:{postMessage:value=>notices.push(value)}},
 google:{colab:{kernel:{invokeFunction:async(name,args)=>{
  if(name.endsWith('.check'))return {data:{'application/json':{ready:true,source_count:14,feedback_changed:false}}};
  savedCalls++;lastPayload=args[0];if(failSave)throw Error('disconnected');if(defer)await new Promise(resolve=>{resolveSave=resolve;});
  return {data:{'application/json':{saved:true,durable:true,status:'accepted_for_research',accepted:1,held:0,incomplete:13}}};
 }}}}};
 await vm.runInNewContext(script,context);return {root,list,banner,pager,card:list.children[0],notices};
}
let view=await render();assert.equal(savedCalls,0);assert.equal(view.notices.at(-1).payload.audio_players,14);assert.equal(view.notices.at(-1).payload.pages,3);
assert.equal(view.list.children[0].style.display,'block');assert.equal(view.list.children[5].style.display,'none');view.pager.querySelectorAll('button')[1].onclick();assert.equal(view.list.children[5].style.display,'block');
const key='anyanwu:waxal-female5-source-drafts:v1';view.card.querySelectorAll('select')[0].value='no';view.card.querySelectorAll('select')[0].onchange();view.card.querySelectorAll('textarea')[0].value='Fixture: Ọ bụ m. ụ̄';view.card.querySelectorAll('textarea')[0].oninput();
await view.card.querySelectorAll('button')[1].onclick();assert.match(view.card.querySelectorAll('p').at(-1).textContent,/inactive/);
view=await render();assert.equal(view.card.querySelectorAll('select')[0].value,'no');assert.equal(view.card.querySelectorAll('textarea')[0].value,'Fixture: Ọ bụ m. ụ̄');
failSave=false;await view.card.querySelectorAll('button')[0].onclick();for(const field of ['words_match','complete_utterance','clean_female_voice'])assert.equal(lastPayload[field],'yes');assert.equal(JSON.parse(storage.get(key)).case0,undefined);
defer=true;const saving=view.card.querySelectorAll('button')[1].onclick();view.card.querySelectorAll('textarea')[0].value='Newer unsaved fixture';view.card.querySelectorAll('textarea')[0].oninput();resolveSave();await saving;assert.equal(JSON.parse(storage.get(key)).case0.notes,'Newer unsaved fixture');
blocked=true;view=await render();assert.equal(view.notices.at(-1).payload.browser_draft_storage,false);
process.stdout.write('Passed source pagination, no automatic saves, explicit three-check acceptance, failed-save/reload, Unicode, durable cleanup, concurrent edits and blocked storage fixtures.\n');
