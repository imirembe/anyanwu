"""Offline caption-display repair and integrity receipt for saved cloud windows."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import wave

ROOT=Path('/content/anyanwu-bbc-expansion-v1')


def build_expansion_review():
    if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Cloud-only review; no laptop media or network fallback')
    from PIL import Image,ImageDraw,ImageOps
    original=json.loads((ROOT/'expansion-result.json').read_text())
    rows=[]
    for old in original['sources']:
        folder=ROOT/old['id']
        wav=folder/'speech-20s.wav'
        with wave.open(str(wav),'rb') as reader:
            if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate(),reader.getnframes())!=(1,2,16000,320000):
                raise RuntimeError('Expected 20-second mono16k PCM16 audio')
        audio={'path':str(wav),'bytes':wav.stat().st_size,'sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),'seconds':20}
        frames=[]
        for second in range(24,44):
            file=folder/f'caption-{second:02d}.png'
            if not 0<file.stat().st_size<=500000:
                raise RuntimeError('Missing/oversize saved caption frame')
            with Image.open(file) as image:
                image.verify()
            frames.append({'second':second,'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        pair_folder=folder/'review-pairs'
        pair_folder.mkdir(exist_ok=True)
        sheets=[]
        for index in range(10):
            tiles=[]
            for frame in frames[index*2:index*2+2]:
                with Image.open(frame['path']) as source:
                    image=ImageOps.grayscale(source).copy()
                tile=Image.new('L',(image.width,image.height+22),255)
                tile.paste(image,(0,22))
                ImageDraw.Draw(tile).text((6,4),'Nominal BBC source time '+str(frame['second'])+'s',fill=0)
                tiles.append(tile)
            sheet=Image.new('L',(tiles[0].width,sum(i.height for i in tiles)),255)
            y=0
            for tile in tiles:
                sheet.paste(tile,(0,y)); y+=tile.height
            output=pair_folder/f'sheet-{index+1:02d}.jpg'
            if not output.exists():
                for quality in (70,55,40,30,20,12,8):
                    sheet.save(output,format='JPEG',quality=quality,optimize=True)
                    if output.stat().st_size<=16500:
                        break
                else:
                    raise RuntimeError('Two-frame sheet still exceeds display limit; source frames retained')
            if not 0<output.stat().st_size<=16500:
                raise RuntimeError('Saved two-frame sheet size check failed')
            sheets.append({'index':index+1,'path':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size,
                           'nominal_source_seconds':[f['second'] for f in frames[index*2:index*2+2]]})
        rows.append({'id':old['id'],'source_url':old['source_url'],'source_group':old['source_group'],
                     'audio':audio,'frames':frames,'sheets':sheets,'training_ready':False,'split':None,
                     'voice_status':old['voice_status'],'text_status':'Caption images available; text not yet transcribed',
                     'alignment_status':'Not yet word-aligned; exact utterance cuts pending'})
    report={'phase':'expanded_speech_review_ready','sources':rows,'new_audio_seconds':100,'new_sources':5,'caption_frames':100,
            'caption_sheets':50,'additional_network_requests':0,'additional_optimizer_updates':0,'local_media':False,
            'body_bytes':original['body_bytes'],'requests':original['requests'],
            'repair':'Two frames/sheet instead of four; all original WAVs/frames retained without additional network',
            'source_receipt_sha256':hashlib.sha256((ROOT/'expansion-result.json').read_bytes()).hexdigest()}
    path=ROOT/'expansion-review-result.json'
    if path.exists() and json.loads(path.read_text())!=report:
        raise RuntimeError('Saved review receipt differs; refusing overwrite')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    summary={**report,'sources':[{k:v for k,v in row.items() if k!='frames'} | {'frame_count':len(row['frames'])} for row in rows]}
    return summary
