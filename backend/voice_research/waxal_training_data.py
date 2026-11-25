"""Whole publisher-paired female Igbo samples; never BBC-derived cuts."""
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

WORK=Path('/content/anyanwu-waxal-female5-lora-v1')
REVISION='5f4d8ca24f2b9d168b2ee545f1febaaff4b40580'


def acquire_waxal_pairs(plan,urls,publish):
    if platform.system()!='Linux' or not Path('/content').is_dir():
        raise RuntimeError('Hosted cloud only; no laptop media')
    if len(plan['clips'])!=24 or len({c['id'] for c in plan['clips']})!=24:
        raise RuntimeError('Twenty-four unique publisher clips required')
    if any(c['speaker_id']!='5' or c['gender']!='Female' or c['truncated_cells'] or c['source']!='google/WaxalNLP' for c in plan['clips']):
        raise RuntimeError('Unexpected speaker, source or truncated text')
    receipt=WORK/'data-result.json'
    if receipt.exists():
        result=json.loads(receipt.read_text())
        for c in result['clips']:
            if hashlib.sha256(Path(c['cloud_audio_path']).read_bytes()).hexdigest()!=c['audio_sha256']:
                raise RuntimeError('Existing whole utterance failed integrity')
        return result
    if (WORK/'data').exists():
        raise RuntimeError('Partial acquisition exists; no undeclared repeat')
    folder=WORK/'data';folder.mkdir()
    count=0;received=0;started=time.monotonic()
    redirects=0
    hosts={'huggingface.co','datasets-server.huggingface.co','cdn-assets.huggingface.co','us.gcp.cdn.hf.co'}
    def checked(url):
        parsed=urllib.parse.urlsplit(url)
        if parsed.scheme!='https' or parsed.hostname not in hosts or parsed.username or parsed.password or parsed.port not in (None,443):
            raise RuntimeError('Unexpected public source host')
        return url
    class Redirect(urllib.request.HTTPRedirectHandler):
        def http_error_302(self,req,fp,code,msg,headers):
            nonlocal count,redirects
            destination=checked(urllib.parse.urljoin(req.full_url,headers.get('Location','')))
            fp.close();count+=1;redirects+=1
            if redirects>3 or count>90:raise RuntimeError('Source redirect/request cap')
            return self.parent.open(urllib.request.Request(destination,headers={'Accept-Encoding':'identity','User-Agent':'Anyanwu-Research/1'}),timeout=25)
        http_error_301=http_error_303=http_error_307=http_error_308=http_error_302
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),Redirect())
    def get(url,cap):
        nonlocal count,received,redirects
        checked(url)
        if count>=90 or time.monotonic()-started>480 or received>=32*1024**2:
            raise RuntimeError('Data transfer/request/time cap')
        request=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'Anyanwu-Research/1'})
        count+=1;redirects=0
        with opener.open(request,timeout=25) as response:
            content_length=response.headers.get('Content-Length')
            if content_length and (not content_length.isdigit() or int(content_length)>cap):
                raise RuntimeError('Source body exceeds declared cap')
            data=response.read(min(cap,32*1024**2-received)+1)
            received+=len(data)
            if len(data)>cap or received>32*1024**2:raise RuntimeError('Source transfer size cap')
            return data
    def head():
        data=json.loads(get('https://huggingface.co/api/datasets/google/WaxalNLP',1024**2))
        if data.get('sha')!=REVISION or data.get('gated'):raise RuntimeError('Pinned public dataset head changed')
    head()
    clips=[]
    for original in plan['clips']:
        row={**original};url=urls[row['id']]
        if f'/--/{REVISION}/--/ibo_tts/train/{row["row_idx"]}/audio/audio.mp3' not in urllib.parse.urlsplit(url).path:
            raise RuntimeError('Asset URL revision/config/row mismatch')
        mp3=folder/(row['id']+'.mp3')
        mp3.write_bytes(get(url,1024**2))
        wav=folder/(row['id']+'.wav')
        command=[shutil.which('ffmpeg'),'-nostdin','-hide_banner','-loglevel','error','-protocol_whitelist','file,pipe',
                 '-i',str(mp3),'-ac','1','-ar','16000','-c:a','pcm_s16le','-fs','500000',str(wav)]
        decoded=subprocess.run(command,capture_output=True,timeout=35)
        if decoded.returncode:raise RuntimeError('Whole preview decoding failed')
        with wave.open(str(wav),'rb') as reader:
            frames=reader.getnframes();seconds=frames/reader.getframerate()
            if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate())!=(1,2,16000) or not 1<=seconds<=15:
                raise RuntimeError('Whole utterance format/duration check failed')
        # -fs is a guard, never a training crop. Verify full decoded duration independently.
        duration=subprocess.run([shutil.which('ffprobe'),'-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(mp3)],capture_output=True,text=True,timeout=10)
        if duration.returncode or abs(float(duration.stdout)-seconds)>0.15:
            raise RuntimeError('Decoded preview appears incomplete; no truncated corpus installation')
        row.update(cloud_audio_path=str(wav),audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                   audio_bytes=wav.stat().st_size,seconds=round(seconds,6),sample_count=frames,
                   mp3_sha256=hashlib.sha256(mp3.read_bytes()).hexdigest(),mp3_bytes=mp3.stat().st_size,
                   label_status='Publisher-supplied full-utterance text; not independently human-verified',
                   representation='Lossy Hugging Face MP3 preview, decoded whole to16k PCM; not original lossless studio audio')
        clips.append(row)
        publish({'phase':'acquiring_paired_female_igbo','completed':len(clips),'total':24,'optimizer_updates':0})
    head()
    result={'phase':'paired_female_igbo_data_ready','clips':clips,'audio_seconds':round(sum(c['seconds'] for c in clips),6),
            'requests':count,'response_body_bytes':received,'hub_head_matched_before_after':True,'revision':REVISION,
            'source_speaker':'5','publisher_gender':'Female','rejected_bbc_pairs_used':0,'independent_human_validation':False}
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
