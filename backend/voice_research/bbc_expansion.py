"""Bounded new source windows for corpus expansion, all media on Colab only."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import wave

from backend.voice_research.bbc_probe import Transport, checked_url, cloud_guard
from backend.voice_research.bbc_samples import master_selection, playlist_window, ffmpeg
from backend.voice_research.caption_capture import contact_sheet

EXPANSION=Path('/content/anyanwu-bbc-expansion-v1')


def expansion_track(transport,url,staging,name,before,start,duration):
    playlist=transport.get(url).decode('utf-8-sig')
    segments=playlist_window(playlist,url,start=start,duration=duration)
    output=staging/(name+('.mp4' if segments[0]['init'] else '.ts'))
    with output.open('xb') as handle:
        urls=([segments[0]['init']] if segments[0]['init'] else [])+[s['url'] for s in segments]
        for segment in urls:
            remaining=12*1024**2-(transport.received-before)
            if remaining<=0:
                raise RuntimeError('Twelve MiB source transfer cap reached')
            handle.write(transport.get(segment,cap=min(3*1024**2,remaining)))
    return output,start-segments[0]['start'],len(segments)


def expand_bbc_samples(plan,publish):
    cloud_guard()
    from PIL import Image
    if not shutil.which('ffmpeg') or shutil.disk_usage('/content').free<5*1024**3:
        raise RuntimeError('Cloud ffmpeg and five GiB free disk required')
    receipt=EXPANSION/'expansion-result.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['plan']!=plan:
            raise RuntimeError('Existing expansion has a different plan; refusing overwrite')
        for row in saved['sources']:
            for item in row.get('artifacts',[]):
                path=EXPANSION/row['id']/item['filename']
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                    raise RuntimeError('Saved expansion integrity failure')
        return saved
    if EXPANSION.exists():
        raise RuntimeError('A partial expansion exists; inspect it before any declared repair')
    EXPANSION.mkdir()
    transport=Transport(budget=64*1024**2,requests=100,seconds=600)
    started=time.monotonic()
    rows=[]
    for source in plan['sources']:
        if time.monotonic()-started>600:
            raise RuntimeError('Ten-minute expansion deadline')
        start,duration=source['start_seconds'],source['duration_seconds']
        if start!=24 or duration!=20 or start+duration>source['source_duration_seconds']-1:
            raise RuntimeError('Unexpected or out-of-range expansion window')
        folder=EXPANSION/source['id']
        folder.mkdir()
        staging=folder/'staging'
        staging.mkdir()
        before=transport.received
        row={'id':source['id'],'source_url':source['url'],'video_id':source['video_id'],
             'nominal_source_start_seconds':start,'requested_seconds':duration,
             'source_group':source['id'],'training_ready':False,'split':None,
             'voice_status':'Female speech requested/reported by user; isolated speaker identity and voice characteristics not independently verified',
             'caption_status':'User reports BBC subtitle/speech agreement; copied text and exact cuts remain separate',
             'timing_note':'Nominal source times inherit stream/decoder offset uncertainty; frame display is not a word boundary',
             'artifacts':[]}
        try:
            metadata=json.loads(transport.get('https://open.live.bbc.co.uk/mediaselector/6/select/version/2.0/mediaset/pc/vpid/'+source['video_id']))
            if metadata.get('result'):
                raise RuntimeError('No accessible public BBC stream')
            urls=[c['href'] for m in metadata.get('media',[]) for c in m.get('connection',[])
                  if c.get('transferFormat')=='hls' and c.get('href','').startswith('https://')]
            if not urls:
                raise RuntimeError('No public HTTPS HLS rendition')
            master_url=checked_url(urls[0])
            master=transport.get(master_url).decode('utf-8-sig')
            video_url,audio_url,selected=master_selection(master,master_url)
            video,video_offset,count=expansion_track(transport,video_url,staging,'video',before,start,duration)
            audio,audio_offset=video,video_offset
            if audio_url:
                audio,audio_offset,_=expansion_track(transport,audio_url,staging,'audio',before,start,duration)
            row.update(rendition=selected,video_segments=count)
            wav=folder/'speech-20s.wav'
            ffmpeg(['-protocol_whitelist','file,pipe','-i',str(audio),'-ss',str(audio_offset),'-t',str(duration),
                    '-map','0:a:0','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le','-fs','700000',str(wav)])
            with wave.open(str(wav),'rb') as reader:
                row['audio_seconds']=reader.getnframes()/reader.getframerate()
                if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate())!=(1,2,16000) or not 19.7<=row['audio_seconds']<=20.1:
                    raise RuntimeError('Expanded WAV duration/format check failed')
            crops=[]
            for second in range(20):
                if time.monotonic()-started>600:
                    raise RuntimeError('Ten-minute expansion deadline')
                path=folder/f'caption-{start+second:02d}.png'
                ffmpeg(['-protocol_whitelist','file,pipe','-i',str(video),'-ss',str(video_offset+second),'-frames:v','1',
                        '-vf','crop=iw:ih*0.55:0:ih*0.45','-fs','500000',str(path)])
                with Image.open(path) as image:
                    image.verify()
                if not 0<path.stat().st_size<=500000:
                    raise RuntimeError('Caption image byte cap')
                crops.append((path,start+second))
            for group in range(5):
                contact_sheet(crops[group*4:(group+1)*4],folder/f'sheet-{group+1:02d}.jpg')
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix in ('.wav','.png','.jpg'):
                    row['artifacts'].append({'filename':path.name,'bytes':path.stat().st_size,
                                              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            row['status']='new_audio_and_caption_frames_ready_for_text_capture'
            row['unique_frame_hashes']=len({a['sha256'] for a in row['artifacts'] if a['filename'].endswith('.png')})
        except Exception as error:
            row['status']='expansion_source_stopped'
            row['error']=str(error)[:220] if isinstance(error,RuntimeError) else type(error).__name__
        finally:
            shutil.rmtree(staging)
        row['body_bytes']=transport.received-before
        rows.append(row)
        publish({'phase':'bbc_speech_expansion','completed_sources':len(rows),'last_status':row['status'],
                 'new_audio_seconds':sum(r.get('audio_seconds',0) for r in rows),'optimizer_updates_this_stage':0})
    report={'phase':'bbc_speech_expansion_complete','plan':plan,'sources':rows,
            'new_audio_seconds':sum(r.get('audio_seconds',0) for r in rows if r['status']=='new_audio_and_caption_frames_ready_for_text_capture'),
            'new_complete_sources':sum(r['status']=='new_audio_and_caption_frames_ready_for_text_capture' for r in rows),
            'body_bytes':transport.received,'requests':transport.requests,'elapsed_seconds':round(time.monotonic()-started,2),
            'optimizer_updates_this_stage':0,'local_media_or_weights':False,'full_videos_saved':False,
            'limitations':'Frames are source observations; no new caption/audio target or female voice quality has been approved automatically'}
    receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
