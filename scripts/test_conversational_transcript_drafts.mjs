// Exercise the actual embedded transcript editor without touching real reviews.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const {script}=JSON.parse(fs.readFileSync(0,'utf8'));
const storage=new Map();let fail=true,blocked=false,deferred=false,resolveSave=null,saves=0;
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.style={};this.value='';this.textContent='';}
 append(...items){this.children.push(...items);}
 querySelectorAll(tag){return this.children.flatMap(c=>c instanceof Element?[...(c.tag===tag?[c]:[]),...c.querySelectorAll(tag)]:[]);}
}
async function render(){
 const root=new Element('root'),list=new Element('div'),banner=new Element('p'),status=new Element('p'),button=new Element('button'),notices=[];
 root.append(list,banner,status,button);root.querySelector=id=>({'#ct-list':list,'#ct-connection':banner,'#ct-status':status,'#ct-save':button}[id]);
 const context={document:{getElementById:()=>root,createElement:tag=>new Element(tag),createTextNode:value=>value},
  localStorage:{getItem:k=>{if(blocked)throw Error('blocked');return storage.get(k)||null;},setItem:(k,v)=>{if(blocked)throw Error('blocked');storage.set(k,v);}},
  window:{top:{postMessage:v=>notices.push(v)}},google:{colab:{kernel:{invokeFunction:async(name,args)=>{
   if(name.endsWith('.check'))return{data:{'application/json':{ready:true,records:2,feedback_changed:false}}};
   saves++;if(fail)throw Error('disconnected');if(deferred)await new Promise(resolve=>{resolveSave=resolve;});
   assert.equal(args[0].records.length,2);return{data:{'application/json':{saved:true,durable:true,reviewed:0}}};
  }}}}};
 await vm.runInNewContext(script,context);return{root,list,banner,status,button,notices};
}
let v=await render();assert.equal(saves,0);assert.equal(v.notices.at(-1).payload.players,2);
let text=v.list.children[0].querySelectorAll('textarea')[0];text.value='Ọ bụ m. Ị̀ dị mma?';text.oninput();
await v.button.onclick();assert.match(v.status.textContent,/Not saved/);
v=await render();text=v.list.children[0].querySelectorAll('textarea')[0];assert.equal(text.value,'Ọ bụ m. Ị̀ dị mma?');
fail=false;await v.button.onclick();assert.deepEqual(JSON.parse(storage.get('anyanwu:conversational-transcripts:v1')),{});
deferred=true;const pending=v.button.onclick();text.value='Newer unsaved ụ́';text.oninput();resolveSave();await pending;
assert.equal(JSON.parse(storage.get('anyanwu:conversational-transcripts:v1'))['conversation-ref-03'].transcript,'Newer unsaved ụ́');
blocked=true;v=await render();assert.equal(v.notices.at(-1).payload.browser_drafts,false);assert.match(v.banner.textContent,/Keep this page open/);
process.stdout.write('Passed actual JS: no auto-save, failed-save/reload, accents, confirmed-save cleanup, in-flight edits, blocked storage.\n');
