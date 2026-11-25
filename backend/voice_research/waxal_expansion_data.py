"""Bounded whole-utterance expansion; data review only, no model loading."""
from pathlib import Path
import hashlib
import json
import platform
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import wave

ROOT=Path('/content/anyanwu-waxal-female5-expansion-v1')


def validate_expansion_plan(plan):
    clips=plan['clips'];reserved=plan['reserved_evaluation_texts'];blocked=set(plan['prior_ids'])
    ids=[c['id'] for c in clips];texts=[c['text'] for c in clips]
    if plan['id']!=ROOT.name or not 1<=len(clips)<=24 or len(ids)!=len(set(ids)):raise ValueError('Unexpected expansion sample')
    if set(ids)&(blocked|{c['id'] for c in reserved}) or set(texts)&({c['text'] for c in reserved}|set(plan['prior_texts'])):raise ValueError('Training/review/evaluation overlap')
    for clip in clips:
        if clip['speaker_id']!='5' or clip['gender']!='Female' or clip['truncated_cells'] or not 15<=len(clip['text'])<=180:raise ValueError('Source metadata check failed')
    if plan['limits']['optimizer_updates']!=0 or plan['limits']['laptop_media_or_models']:raise ValueError('Data-only scope changed')


def acquire_expansion(plan,urls,publish):
    if platform.system()!='Linux' or not Path('/content').is_dir():raise RuntimeError('Hosted Colab only; no laptop audio/models')
    validate_expansion_plan(plan)
    ROOT.mkdir(exist_ok=True);receipt=ROOT/'data-result.json'
    if receipt.exists():
        if json.loads((ROOT/'plan.json').read_text())!=plan:raise RuntimeError('Existing expansion plan changed')
        result=json.loads(receipt.read_text())
        for clip in result['clips']:
            for suffix,key in (('.wav','audio_sha256'),('.mp3','mp3_sha256')):
                if hashlib.sha256((ROOT/'data'/(clip['id']+suffix)).read_bytes()).hexdigest()!=clip[key]:raise RuntimeError('Saved whole utterance changed')
        return result
    if (ROOT/'acquisition-attempt.json').exists():raise RuntimeError('Earlier acquisition exists; inspect before technical repair')
    if shutil.disk_usage('/content').free<5*1024**3:raise RuntimeError('Five GiB cloud reserve required')
    (ROOT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');(ROOT/'data').mkdir(exist_ok=False)
    (ROOT/'acquisition-attempt.json').write_text(json.dumps({'phase':'started','clips':len(plan['clips']),'optimizer_updates':0}))
    count=0;received=0;redirects=0;started=time.monotonic();initial_free=shutil.disk_usage('/content').free
    hosts={'huggingface.co','datasets-server.huggingface.co','cdn-assets.huggingface.co','us.gcp.cdn.hf.co'}
    def checked(url):
        parsed=urllib.parse.urlsplit(url)
        if parsed.scheme!='https' or parsed.hostname not in hosts or parsed.username or parsed.password or parsed.port not in (None,443):raise RuntimeError('Unexpected public asset host')
        return url
    class Redirect(urllib.request.HTTPRedirectHandler):
        def http_error_302(self,req,fp,code,msg,headers):
            nonlocal count,redirects
            destination=checked(urllib.parse.urljoin(req.full_url,headers.get('Location','')));fp.close();count+=1;redirects+=1
            if redirects>3 or count>90:raise RuntimeError('Redirect/request cap')
            return self.parent.open(urllib.request.Request(destination,headers={'Accept-Encoding':'identity','User-Agent':'Anyanwu-Research/1'}),timeout=25)
        http_error_301=http_error_303=http_error_307=http_error_308=http_error_302
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),Redirect())
    def get(url,cap):
        nonlocal count,received,redirects
        checked(url)
        if count>=90 or time.monotonic()-started>600 or received>=32*1024**2 or initial_free-shutil.disk_usage('/content').free>64*1024**2:raise RuntimeError('Transfer/time/disk cap')
        count+=1;redirects=0
        request=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'Anyanwu-Research/1'})
        with opener.open(request,timeout=25) as response:
            length=response.headers.get('Content-Length')
            if length and (not length.isdigit() or int(length)>cap):raise RuntimeError('Source byte cap')
            body=response.read(min(cap,32*1024**2-received)+1);received+=len(body)
            if len(body)>cap or received>32*1024**2:raise RuntimeError('Response body cap')
            return body
    def head():
        value=json.loads(get('https://huggingface.co/api/datasets/google/WaxalNLP',1024**2))
        if value.get('sha')!=plan['revision'] or value.get('gated'):raise RuntimeError('Pinned public source revision changed')
    clips=[];held=[]
    try:
        head()
        for original in plan['clips']:
            clip={**original};url=urls[clip['id']]
            if f'/--/{plan["revision"]}/--/ibo_tts/train/{clip["row_idx"]}/audio/audio.mp3' not in urllib.parse.urlsplit(url).path:raise RuntimeError('Source asset revision/row mismatch')
            mp3=ROOT/'data'/(clip['id']+'.mp3');mp3.write_bytes(get(url,1024**2))
            duration=subprocess.run([shutil.which('ffprobe'),'-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(mp3)],capture_output=True,text=True,timeout=15)
            if duration.returncode:raise RuntimeError('Source preview format probe failed')
            seconds=float(duration.stdout)
            if not 1<=seconds<=15:
                held.append({'id':clip['id'],'reason':'Whole utterance exceeds1–15s small-model duration bounds','source_seconds':round(seconds,6)})
                publish({'phase':'whole_source_utterance_held','id':clip['id'],'held_count':len(held),'optimizer_updates':0});continue
            wav=ROOT/'data'/(clip['id']+'.wav')
            decoded=subprocess.run([shutil.which('ffmpeg'),'-nostdin','-hide_banner','-loglevel','error','-protocol_whitelist','file,pipe','-i',str(mp3),'-ac','1','-ar','16000','-c:a','pcm_s16le','-fs','500000',str(wav)],capture_output=True,timeout=35)
            if decoded.returncode:raise RuntimeError('Whole source decoding failed')
            with wave.open(str(wav),'rb') as reader:
                frames=reader.getnframes();decoded_seconds=frames/reader.getframerate()
                if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate())!=(1,2,16000) or abs(decoded_seconds-seconds)>.15:raise RuntimeError('Incomplete or unexpected decoded utterance')
            clip.update(cloud_audio_path=str(wav),audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),audio_bytes=wav.stat().st_size,seconds=round(decoded_seconds,6),sample_count=frames,
              mp3_sha256=hashlib.sha256(mp3.read_bytes()).hexdigest(),mp3_bytes=mp3.stat().st_size,review_status='pending',training_ready=False,
              representation='Whole lossy HF MP3 preview decoded to16k PCM; not lossless studio original')
            clips.append(clip);publish({'phase':'acquiring_expanded_female_utterances','completed':len(clips),'selected':len(plan['clips']),'technical_holds':len(held),'optimizer_updates':0})
        head()
        if not clips:raise RuntimeError('No complete source utterances within bounds')
        result={'phase':'expanded_female_source_sample_ready','clips':clips,'technical_holds':held,'selected_count':len(plan['clips']),'reviewable_count':len(clips),
          'audio_seconds':round(sum(c['seconds'] for c in clips),6),'audio_bytes':sum(c['audio_bytes'] for c in clips),'mp3_bytes':sum(c['mp3_bytes'] for c in clips),
          'response_body_bytes':received,'requests':count,'elapsed_seconds':round(time.monotonic()-started,2),'revision':plan['revision'],
          'hub_head_matched_before_after':True,'optimizer_updates':0,'training_started':False,'human_review_pending':True,'reserved_texts_not_downloaded':[c['id'] for c in plan['reserved_evaluation_texts']]}
        receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result
    except Exception as error:
        result={'phase':'expansion_acquisition_stopped','error':str(error)[:450],'completed':len(clips),'technical_holds':held,'requests':count,'response_body_bytes':received,'optimizer_updates':0}
        (ROOT/'acquisition-attempt.json').write_text(json.dumps(result,indent=2)+'\n');publish(result);raise
