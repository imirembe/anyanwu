"""Prepare a short, read-only player panel from the saved comparison archive."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'.build/VoiceResearch/auditions/conversational-reference-conditioning-v1'


def prepare():
    receipt=json.loads((FOLDER/'backup-result.json').read_text())
    if not receipt['durable_backup_verified'] or receipt['audio_clips']!=4:
        raise RuntimeError('Verified four-clip archive required')
    cell='''# Anyanwu — LISTEN HERE: four saved comparison clips, no synthesis
from pathlib import Path
import base64,hashlib,html,importlib.util,json,platform,zipfile
from IPython.display import HTML,Javascript,display
def publish_audio_panel(value):
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
 if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Hosted Colab only; keep audio off laptop')
 if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Research Drive is disconnected; reconnect the existing CPU first')
 archive=Path(RECEIPT['drive_archive'])
 if archive.stat().st_size!=RECEIPT['archive_bytes'] or archive.stat().st_size>16*1024**2:raise RuntimeError('Saved archive size differs')
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=RECEIPT['archive_sha256']:raise RuntimeError('Saved archive hash differs')
 files=[];cards=[];audio_folder=archive.parent/'audio';audio_folder.mkdir(exist_ok=True)
 with zipfile.ZipFile(archive) as package:
  plan=json.loads(package.read('plan.json'));result=json.loads(package.read('generation-result.json'))
  if plan['id']!='anyanwu-conversational-reference-conditioning-v1' or result['completed_wavs']!=4 or len(result['clips'])!=4 or result['optimizer_updates_this_stage']!=0 or not result['source_checkpoints_unchanged']:raise RuntimeError('Expected original four-clip inference-only result')
  expected={(t['id'],v) for t in plan['texts'] for v in ('base','adapter')}
  if len(expected)!=4 or {(c['text_id'],c['variant']) for c in result['clips']}!=expected or len(set(plan['adapter_hashes'].values()))!=1:raise RuntimeError('Fixed-v1 pair manifest differs')
  for text in plan['texts']:
   parts=['<section style="background:#fffdf8;border:1px solid #dfe3d5;border-radius:14px;padding:14px;margin:12px 0"><h3 style="margin:0 0 8px;font-size:17px">'+html.escape(text['text'])+'</h3>']
   for variant,label,prefix in [('base','Reading reference · v1','reading-reference'),('adapter','Conversational reference · v1','conversational-reference')]:
    clip=next(c for c in result['clips'] if (c['text_id'],c['variant'])==(text['id'],variant))
    name=clip['audio_filename']
    if Path(name).name!=name or package.getinfo(name).file_size!=clip['bytes'] or not 0<clip['bytes']<=2*1024**2:raise RuntimeError('Unexpected WAV member path/size')
    raw=package.read(name);digest=hashlib.sha256(raw).hexdigest()
    if len(raw)!=clip['bytes'] or digest!=clip['audio_sha256']:raise RuntimeError('Saved audio checksum differs')
    target=audio_folder/(prefix+'-'+text['id']+'.wav')
    if target.exists() and (target.stat().st_size!=len(raw) or hashlib.sha256(target.read_bytes()).hexdigest()!=digest):raise RuntimeError('Different existing Drive audio preserved')
    if not target.exists():
     stage=target.with_suffix('.wav.staging');stage.write_bytes(raw)
     if hashlib.sha256(stage.read_bytes()).hexdigest()!=digest:raise RuntimeError('Drive audio staging checksum failed')
     stage.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('Drive audio copy checksum failed')
    files.append({'text_id':text['id'],'variant':variant,'label':label,'filename':target.name,'drive_path':str(target),'bytes':len(raw),'sha256':digest})
    parts.append('<p style="margin:8px 0 3px">'+html.escape(label)+' · '+str(clip['seconds'])+'s</p><audio controls preload="none" style="width:100%" src="data:audio/wav;base64,'+base64.b64encode(raw).decode()+'"></audio>')
   parts.append('</section>');cards.append(''.join(parts))
 info={'phase':'reference_conditioning_audio_files_prepared','files':files,'audio_bytes':sum(x['bytes'] for x in files),'drive_folder':str(audio_folder),'archive_sha256':RECEIPT['archive_sha256'],'audio_changed':False,'feedback_changed':False,'new_synthesis':False,'models_needed':False}
 if info['audio_bytes']>8*1024**2:raise RuntimeError('Four-player audio cap')
 index={'experiment':plan['id'],'files':files,'attribution':plan['attribution'],'source_archive_sha256':RECEIPT['archive_sha256'],'distribution':'Private noncommercial research; original source conditions retained.'}
 index_path=audio_folder/'audio-index.json';encoded=(json.dumps(index,ensure_ascii=False,indent=2)+'\\n').encode()
 if index_path.exists() and index_path.read_bytes()!=encoded:raise RuntimeError('Different Drive audio index preserved')
 if not index_path.exists():index_path.write_bytes(encoded)
 page='<div id="anyanwu-saved-audio" style="max-width:850px;background:#f7f5eb;color:#173f34;padding:16px;border-radius:16px;font:16px/1.45 system-ui"><h2 style="margin:0 0 8px">Listen here · four comparison clips</h2><p style="margin:0">Use the play buttons below. The conversational-reference clips are quieter; adjust playback volume comfortably.</p>'+''.join(cards)+'<p style="font-size:14px">These exact WAV files are also in your research Drive folder under auditions → conversational-reference-conditioning-v1 → audio. Your existing feedback form is unchanged.</p></div>'
 page+='<script>(()=>{const panel=document.getElementById("anyanwu-saved-audio"),message='+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':{**info,'phase':'reference_conditioning_audio_files_visible'}}).replace('<','\\u003c')+';message.payload.audio_players=panel.querySelectorAll("audio").length;message.payload.panel_height=Math.round(panel.getBoundingClientRect().height);message.payload.labels=[...panel.querySelectorAll("section p")].map(e=>e.textContent);window.top.postMessage(message,"https://colab.research.google.com");})();</script>'
 display(HTML(page))
except Exception as error:
 publish_audio_panel({'phase':'reference_conditioning_audio_panel_stopped','error_type':type(error).__name__,'error':str(error)[:300],'new_synthesis':False,'feedback_changed':False})
 raise
'''
    cell='RECEIPT='+repr(receipt)+'\n'+cell
    compile(cell,'audio-panel','exec')
    path=FOLDER/'audio-panel-cell.py';path.write_text(cell)
    result={'phase':'reference_conditioning_audio_panel_prepared','source_sha256':hashlib.sha256(cell.encode()).hexdigest(),'new_synthesis':False,'models_needed':False}
    (FOLDER/'audio-panel-source-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
