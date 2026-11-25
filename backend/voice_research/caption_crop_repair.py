"""Recover cut-off upper subtitle lines using the remaining capture budget."""
from pathlib import Path
import base64
import hashlib
import json
import shutil
import time

from backend.voice_research.bbc_probe import SOURCES, Transport, cloud_guard
from backend.voice_research.bbc_samples import master_selection, acquire_track, ffmpeg
from backend.voice_research.caption_capture import contact_sheet


def repair(publish):
    cloud_guard()
    from PIL import Image,ImageOps
    root=Path('/content/anyanwu-bbc-caption-crop-repair-v1')
    previous=json.loads(Path('/content/anyanwu-bbc-caption-capture-v2/capture.json').read_text())
    budget=32*1024**2-previous['combined_caption_stage_body_bytes']
    request_cap=80-previous['combined_caption_stage_requests']
    if root.exists():
        raise RuntimeError('The caption crop repair already exists; inspect rather than overwrite.')
    if budget<=0 or request_cap<18 or shutil.disk_usage('/content').free<5*1024**3:
        raise RuntimeError('Insufficient remaining caption capture budget or cloud disk.')
    root.mkdir()
    transport=Transport(budget=budget,requests=request_cap,seconds=300)
    rows=[]
    for source_id,times in [('bbc-igbo-01',[12,13,15,19]),('bbc-igbo-04',[12,17]),('bbc-igbo-05',[12,16])]:
        version=dict(SOURCES)[source_id]
        folder=root/source_id
        folder.mkdir()
        staging=folder/'staging'
        staging.mkdir()
        before=transport.received
        row={'id':source_id,'frames':[]}
        try:
            data=json.loads(transport.get(f'https://open.live.bbc.co.uk/mediaselector/6/select/version/2.0/mediaset/pc/vpid/{version}'))
            if data.get('result'):
                raise RuntimeError('BBC public media metadata unavailable.')
            urls=[c['href'] for m in data.get('media',[]) for c in m.get('connection',[]) if c.get('transferFormat')=='hls' and c.get('href','').startswith('https://')]
            if not urls: raise RuntimeError('No public BBC HLS stream available.')
            master=transport.get(urls[0]).decode('utf-8-sig')
            video_url,_,_=master_selection(master,urls[0])
            # This repair allocates at most 3 MiB/source from the remaining
            # overall 32 MiB/80-request caption-stage budget.
            video,offset,_=acquire_track(transport,video_url,staging,'video',before-3*1024**2)
            for index,second in enumerate(times,1):
                path=folder/f'caption-{second}.png'
                ffmpeg(['-protocol_whitelist','file,pipe','-i',str(video),'-ss',str(offset+second-12),'-frames:v','1','-vf','crop=iw:ih*0.48:0:ih*0.50','-fs','500000',str(path)])
                preview=folder/f'sheet-{index:02d}.jpg'
                contact_sheet([(path,second)],preview)
                row['frames'].append({'source_second':second,'png':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'preview':preview.name})
            row['status']='upper_caption_lines_captured'
        finally:
            shutil.rmtree(staging)
        row['body_bytes']=transport.received-before
        rows.append(row)
    # Read a tight, higher-quality crop of the one ambiguous word from source 02
    # using existing pixels, with no network request or invented accents.
    path=Path('/content/anyanwu-bbc-caption-capture-v2/bbc-igbo-02/caption-15.png')
    folder=root/'bbc-igbo-02'
    folder.mkdir()
    with Image.open(path) as image:
        image=ImageOps.grayscale(image).crop((35,50,330,145)).resize((590,190))
        preview=folder/'sheet-01.jpg'
        for quality in (80,60,40,25):
            image.save(preview,'JPEG',quality=quality,optimize=True)
            if preview.stat().st_size<=16500: break
    rows.append({'id':'bbc-igbo-02','frames':[{'source_second':15,'preview':'sheet-01.jpg','source_png_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}],'body_bytes':0,'status':'existing_caption_pixels_zoomed'})
    report={'phase':'caption_crop_repair_complete','sources':rows,'body_bytes':transport.received,'requests':transport.requests,
            'combined_caption_stage_body_bytes':previous['combined_caption_stage_body_bytes']+transport.received,
            'combined_caption_stage_requests':previous['combined_caption_stage_requests']+transport.requests,
            'overall_caption_stage_cap_bytes':32*1024**2,'trained':False,'source_pixels_cloud_only':True}
    (root/'repair.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for row in rows:
        for index,item in enumerate(row['frames'],1):
            image=(root/row['id']/item['preview']).read_bytes()
            if len(image)>16500: raise RuntimeError('Caption preview exceeded display cap.')
            publish({'phase':'caption_sheet','key':row['id']+f'/{index:02d}','jpeg_base64':base64.b64encode(image).decode()})
    return report
