"""Package exact reviewed listening WAVs; no inference or audio changes."""
from pathlib import Path
import hashlib
import json
import uuid

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / '.build/VoiceResearch/auditions/conversational-reference-validation-v2'


def prepare():
    generation = json.loads((FOLDER / 'generation-result.json').read_text())
    if generation['completed_wavs'] != 12:
        raise RuntimeError('Twelve existing listening files required')
    clips = [{key: clip[key] for key in ('text_id', 'variant', 'seconds', 'requested_text',
        'listening_audio_filename', 'listening_audio_sha256', 'listening_bytes')} for clip in generation['clips']]
    export_id = str(uuid.uuid4())
    cell = '''# Anyanwu export exact reviewed audio: no models, generation or feedback changes
from pathlib import Path
import hashlib,importlib.util,json,platform,re,shutil,time,wave,zipfile
from IPython.display import Javascript,display
CLIPS=__CLIPS__
EXPORT_ID=__EXPORT_ID__
def publish_audio_downloads(value):
 value={**value,'export_id':EXPORT_ID,'new_synthesis':False,'audio_changed':False,'feedback_changed':False}
 display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
 print(json.dumps(value,ensure_ascii=False,indent=2))
def exact_copy(body,target):
 target=Path(target);digest=hashlib.sha256(body).hexdigest()
 if target.exists():
  if target.is_symlink() or target.read_bytes()!=body:raise RuntimeError('Different existing download preserved: '+target.name)
 else:
  stage=target.with_name(target.name+'.staging')
  if stage.exists():raise RuntimeError('Previous download staging preserved: '+target.name)
  stage.write_bytes(body)
  if hashlib.sha256(stage.read_bytes()).hexdigest()!=digest:raise RuntimeError('Download staging checksum failure')
  stage.replace(target)
 if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise RuntimeError('Final download checksum failure')
 return {'filename':target.name,'bytes':len(body),'sha256':digest,'drive_path':str(target)}
def make_audio_zip(items,audio_root):
 from io import BytesIO
 out=BytesIO();names=[]
 note=('Anyanwu: volume-matched voice research comparison.\\n'
       'These are the exact listening copies rated in the six-pair review. No new speech was generated.\\n'
       'Reading-reference and conversational-reference conditions use the same preserved v1 model.\\n'
       'Audio is AI-generated and is not a validated final product voice. Private noncommercial research.\\n'
       'Texts/read reference: Media Trust / Google WAXAL, CC BY-SA 4.0.\\n'
       'Conversational reference: Silencio Network, Igbo Speech Dataset, 2026, CC BY-NC 4.0.\\n'
       'Flash checkpoint: Modularcomputing, CC BY-SA 4.0. CosyVoice3: FunAudioLLM, Apache 2.0.\\n')
 manifest={'experiment':'anyanwu-conversational-reference-validation-v2','exact_reviewed_listening_files':True,'new_synthesis':False,'contains_models':False,'clips':items}
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
  for item in items:
   name=item['filename']
   if Path(name).name!=name or not re.fullmatch(r'(reading|conversational)-reference-ibo_[0-9]+[.]wav',name):raise RuntimeError('Unsafe audio ZIP member')
   body=(audio_root/name).read_bytes()
   if len(body)!=item['bytes'] or hashlib.sha256(body).hexdigest()!=item['sha256']:raise RuntimeError('Audio ZIP input differs')
   info=zipfile.ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o600<<16;archive.writestr(info,body);names.append(name)
  for name,body in [('README.txt',note.encode()),('manifest.json',(json.dumps(manifest,ensure_ascii=False,indent=2)+'\\n').encode())]:
   info=zipfile.ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o600<<16;archive.writestr(info,body);names.append(name)
 body=out.getvalue()
 if len(body)>8*1024**2:raise RuntimeError('Audio ZIP size cap')
 with zipfile.ZipFile(BytesIO(body)) as archive:
  if archive.testzip() is not None or set(archive.namelist())!=set(names) or len(archive.namelist())!=len(names):raise RuntimeError('Audio ZIP integrity failure')
  for item in items:
   if hashlib.sha256(archive.read(item['filename'])).hexdigest()!=item['sha256']:raise RuntimeError('Audio ZIP member checksum failure')
 return body
try:
 if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:raise RuntimeError('Existing hosted Colab only; no laptop audio transfer')
 if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Research Drive disconnected; no generation is needed')
 source_root=Path('/content/anyanwu-conversational-reference-validation-v2')
 target_root=Path('/content/drive/MyDrive/Anyanwu Voice Research/auditions/conversational-reference-validation-v2/audio')
 if len(CLIPS)!=12 or len({(c['text_id'],c['variant']) for c in CLIPS})!=12:raise RuntimeError('Exact twelve-clip export required')
 started=time.monotonic();initial_free=shutil.disk_usage('/content').free
 target_root.mkdir(parents=True,exist_ok=True);files=[];total=0
 for clip in CLIPS:
  if time.monotonic()-started>180 or initial_free-shutil.disk_usage('/content').free>32*1024**2:raise RuntimeError('Bounded audio export guard')
  original=clip['listening_audio_filename']
  if Path(original).name!=original or not re.fullmatch(r'listen-(base|adapter)-ibo_[0-9]+[.]wav',original):raise RuntimeError('Unsafe listening input')
  source=source_root/original
  if source.is_symlink() or not 0<source.stat().st_size==clip['listening_bytes']<=2*1024**2:raise RuntimeError('Listening input size/path guard')
  body=source.read_bytes();total+=len(body)
  if hashlib.sha256(body).hexdigest()!=clip['listening_audio_sha256'] or total>4*1024**2:raise RuntimeError('Reviewed audio checksum/total guard')
  with wave.open(str(source),'rb') as reader:
   if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate())!=(1,2,24000) or abs(reader.getnframes()/24000-clip['seconds'])>1e-9:raise RuntimeError('Reviewed WAV format/duration differs')
  label='reading' if clip['variant']=='base' else 'conversational'
  item=exact_copy(body,target_root/(label+'-reference-'+clip['text_id']+'.wav'))
  item.update(text_id=clip['text_id'],condition=label,seconds=clip['seconds'],requested_text=clip['requested_text']);files.append(item)
 weather=[item for item in files if item['text_id']=='ibo_1729']
 if len(weather)!=2:raise RuntimeError('Both weather versions required')
 bundles=[exact_copy(make_audio_zip(weather,target_root),target_root/'Anyanwu-weather-pair.zip'),exact_copy(make_audio_zip(files,target_root),target_root/'Anyanwu-all-six-pairs.zip')]
 index={'experiment':'anyanwu-conversational-reference-validation-v2','audio_files':files,'bundles':bundles,'all_source_listening_hashes_preserved':True,'new_synthesis':False}
 exact_copy((json.dumps(index,ensure_ascii=False,indent=2)+'\\n').encode(),target_root/'audio-index.json')
 publish_audio_downloads({'phase':'reference_validation_audio_downloads_ready','audio_files':files,'bundles':bundles,'audio_bytes':total,'drive_audio_folder':str(target_root),'all_source_listening_hashes_preserved':True})
except Exception as error:
 publish_audio_downloads({'phase':'reference_validation_audio_downloads_stopped','error_type':type(error).__name__,'error':str(error)[:350]})
 raise
'''.replace('__CLIPS__', repr(clips)).replace('__EXPORT_ID__', repr(export_id))
    compile(cell, 'validation-audio-downloads', 'exec')
    (FOLDER / 'audio-downloads-cell.py').write_text(cell)
    receipt = {'export_id': export_id, 'source_sha256': hashlib.sha256(cell.encode()).hexdigest(),
        'exact_listening_clips': 12, 'new_synthesis': False, 'feedback_changed': False}
    (FOLDER / 'audio-downloads-source-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
