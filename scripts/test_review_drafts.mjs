// Executes the actual embedded form script against a small DOM/storage fixture.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const {script}=JSON.parse(fs.readFileSync(0,'utf8'));
const storage=new Map();
let failSave=true,blockStorage=false,pendingResolve=null,deferSave=false;
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.events={};this.style={};this.value='';this.textContent='';}
 append(...items){this.children.push(...items);}
 addEventListener(name,handler){this.events[name]=handler;}
 setAttribute(name,value){this[name]=value;}
 querySelectorAll(tag){return this.children.flatMap(c=>c instanceof Element?[...(c.tag===tag?[c]:[]),...c.querySelectorAll(tag)]:[]);}
}
async function render(){
 const root=new Element('root'),list=new Element('div'),banner=new Element('p'),summary=new Element('p');
 root.append(list,banner,summary);root.querySelector=id=>({'#cut-list':list,'#connection-status':banner,'#cut-summary':summary}[id]);
 const notices=[];
 const context={document:{getElementById:()=>root,createElement:tag=>new Element(tag),createTextNode:text=>text},
  localStorage:{getItem:key=>{if(blockStorage)throw Error('blocked');return storage.get(key)||null;},setItem:(key,value)=>{if(blockStorage)throw Error('blocked');storage.set(key,value);}},
  window:{top:{postMessage:value=>notices.push(value)}},
  google:{colab:{kernel:{invokeFunction:async(name,args)=>{
   if(name.endsWith('.check'))return{data:{'application/json':{ready:true,candidate_count:7,feedback_records_changed:false}}};
   if(failSave)throw Error('disconnected');
   if(deferSave)await new Promise(resolve=>{pendingResolve=resolve;});
   return{data:{'application/json':{saved:true,durable:true,status:'pending'}}};
  }}}}};
 await vm.runInNewContext(script,context);
 return{root,list,banner,notices,card:list.children[0]};
}
let view=await render();
assert.equal(view.notices.at(-1).payload.audio_players,7);
const selects=view.card.querySelectorAll('select');
selects[0].value='yes';selects[0].events.change();
const note=view.card.querySelectorAll('textarea')[0];note.value='Fixture: Ọ bụ m. ụ̄';note.events.input();
await view.card.querySelectorAll('button')[0].onclick();
assert.match(view.card.querySelectorAll('p').at(-1).textContent,/not connected/);
assert.equal(JSON.parse(storage.get('anyanwu:expanded-cut-drafts:v1')).case0.words_match,'yes');
view=await render();
assert.equal(view.card.querySelectorAll('select')[0].value,'yes');
assert.equal(view.card.querySelectorAll('textarea')[0].value,'Fixture: Ọ bụ m. ụ̄');
failSave=false;
await view.card.querySelectorAll('button')[0].onclick();
assert.equal(JSON.parse(storage.get('anyanwu:expanded-cut-drafts:v1')).case0,undefined);
// An edit made while a save is in flight must remain as a new draft.
deferSave=true;
const promise=view.card.querySelectorAll('button')[0].onclick();
const edited=view.card.querySelectorAll('textarea')[0];edited.value='Newer unsaved fixture';edited.events.input();
pendingResolve();await promise;
assert.equal(JSON.parse(storage.get('anyanwu:expanded-cut-drafts:v1')).case0.notes,'Newer unsaved fixture');
blockStorage=true;view=await render();
assert.equal(view.notices.at(-1).payload.browser_draft_storage,false);
assert.match(view.banner.textContent,/storage is unavailable/);
process.stdout.write('Passed draft preservation across failed save/reload, Unicode, confirmed save cleanup, in-flight edits, and blocked storage.\n');
