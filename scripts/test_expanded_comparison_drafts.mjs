// Runs the actual comparison form script with fixture DOM/storage/callbacks only.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const {script}=JSON.parse(fs.readFileSync(0,'utf8'));
const storage=new Map();let failSave=true,blocked=false,deferred=false,resolveSave;
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.style={};this.value='';this.textContent='';}
 append(...children){this.children.push(...children);}
 setAttribute(name,value){this[name]=value;}
 querySelectorAll(tag){return this.children.flatMap(c=>c instanceof Element?[...(c.tag===tag?[c]:[]),...c.querySelectorAll(tag)]:[]);}
}
async function render(){
 const root=new Element('root'),pairs=new Element('div'),banner=new Element('p');root.append(pairs,banner);
 root.querySelector=id=>({'#pairs':pairs,'#connection':banner}[id]);const notices=[];
 const context={document:{getElementById:()=>root,createElement:tag=>new Element(tag),createTextNode:text=>text},
 localStorage:{getItem:key=>{if(blocked)throw Error('blocked');return storage.get(key)||null;},setItem:(key,value)=>{if(blocked)throw Error('blocked');storage.set(key,value);}},
 window:{top:{postMessage:value=>notices.push(value)}},google:{colab:{kernel:{invokeFunction:async(name,args)=>{
  if(name.endsWith('.check'))return {data:{'application/json':{ready:true,pairs:4,feedback_changed:false}}};
  if(failSave)throw Error('disconnected');if(deferred)await new Promise(resolve=>{resolveSave=resolve;});
  return {data:{'application/json':{saved:true,durable:true}}};
 }}}}};
 await vm.runInNewContext(script,context);return {root,pairs,banner,card:pairs.children[0],notices};
}
let view=await render();assert.equal(view.notices.at(-1).payload.audio_players,8);assert.equal(view.notices.at(-1).payload.callback_verified,true);
const key='anyanwu:expanded-comparison-drafts:v2';
view.card.querySelectorAll('select')[0].value='adapter';view.card.querySelectorAll('select')[0].onchange();
view.card.querySelectorAll('textarea')[0].value='Fixture: Ọ bụ m. ụ̄';view.card.querySelectorAll('textarea')[0].oninput();
await view.card.querySelectorAll('button')[0].onclick();assert.match(view.card.querySelectorAll('p').at(-1).textContent,/inactive/);
view=await render();assert.equal(view.card.querySelectorAll('select')[0].value,'adapter');assert.equal(view.card.querySelectorAll('textarea')[0].value,'Fixture: Ọ bụ m. ụ̄');
failSave=false;await view.card.querySelectorAll('button')[0].onclick();assert.equal(JSON.parse(storage.get(key)).case0,undefined);
deferred=true;const saving=view.card.querySelectorAll('button')[0].onclick();
view.card.querySelectorAll('textarea')[0].value='Newer unsaved fixture';view.card.querySelectorAll('textarea')[0].oninput();resolveSave();await saving;
assert.equal(JSON.parse(storage.get(key)).case0.notes,'Newer unsaved fixture');
blocked=true;view=await render();assert.equal(view.notices.at(-1).payload.browser_draft_storage,false);
process.stdout.write('Passed eight players, callback probe, failed-save/reload, Unicode, durable cleanup, concurrent edits and blocked storage fixtures.\n');
