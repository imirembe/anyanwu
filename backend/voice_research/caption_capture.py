"""Capture only caption pictures for the existing 12-second cloud pilot.

User has checked the original BBC subtitles. These frames support extracting
that text; user review does not certify OCR or precise utterance timestamps.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

from backend.voice_research.bbc_probe import SOURCES, Transport, cloud_guard
from backend.voice_research.bbc_samples import master_selection, acquire_track, ffmpeg


def contact_sheet(paths, output):
    from PIL import Image, ImageDraw, ImageOps
    tiles=[]
    for path,second in paths:
        with Image.open(path) as image:
            image=ImageOps.grayscale(image).copy()
        tile=Image.new("L",(image.width,image.height+22),255)
        tile.paste(image,(0,22))
        ImageDraw.Draw(tile).text((6,4),f"BBC source time {second:02d}s",fill=0)
        tiles.append(tile)
    sheet=Image.new("L",(max(t.width for t in tiles),sum(t.height for t in tiles)),255)
    y=0
    for tile in tiles:
        sheet.paste(tile,(0,y))
        y+=tile.height
    for quality in (55,40,25,15,8):
        sheet.save(output,format="JPEG",quality=quality,optimize=True)
        if output.stat().st_size<=16500:
            return
    raise RuntimeError("Caption contact sheet exceeded its display bound; original crops retained.")


def capture(publish):
    cloud_guard()
    from PIL import Image  # Existing cloud library only; no installs.
    if not shutil.which("ffmpeg") or shutil.disk_usage("/content").free<5*1024**3:
        raise RuntimeError("Cloud ffmpeg and five GiB free disk required.")
    root=Path("/content/anyanwu-bbc-caption-capture-v2")
    receipt=root/"capture.json"
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        for row in saved["sources"]:
            for artifact in row.get("artifacts",[]):
                path=root/row["id"]/artifact["filename"]
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=artifact["sha256"]:
                    raise RuntimeError("Existing caption frame failed integrity; no overwrite.")
        return saved
    if root.exists():
        raise RuntimeError("An incomplete caption capture already exists; inspect it before retrying.")
    root.mkdir()
    prior_file=Path("/content/anyanwu-bbc-caption-capture-v1/capture.json")
    prior=json.loads(prior_file.read_text()) if prior_file.exists() else {}
    prior_bytes=prior.get("body_bytes",0)
    prior_requests=prior.get("requests",0)
    if prior_bytes>=32*1024**2 or prior_requests>=80:
        raise RuntimeError("Caption-stage transfer budget was already spent.")
    transport=Transport(budget=32*1024**2-prior_bytes,requests=80-prior_requests,seconds=480)
    start=time.monotonic()
    rows=[]
    for source_id,version in SOURCES:
        folder=root/source_id
        folder.mkdir()
        staging=folder/"staging"
        staging.mkdir()
        before=transport.received
        row={"id":source_id,"video_id":version,"source_seconds":[12,24],"frame_interval_seconds":1,"training_ready":False,"artifacts":[]}
        try:
            data=json.loads(transport.get(f"https://open.live.bbc.co.uk/mediaselector/6/select/version/2.0/mediaset/pc/vpid/{version}"))
            if data.get("result"):
                raise RuntimeError("BBC did not expose an accessible public stream.")
            urls=[c["href"] for m in data.get("media",[]) for c in m.get("connection",[]) if c.get("transferFormat")=="hls" and c.get("href","").startswith("https://")]
            if not urls:
                raise RuntimeError("No BBC HTTPS HLS rendition available.")
            master_url=urls[0]
            master=transport.get(master_url).decode("utf-8-sig")
            video_url,_,selected=master_selection(master,master_url)
            old_bytes=next((r.get("response_body_bytes",0) for r in prior.get("sources",[]) if r["id"]==source_id),0)
            video,offset,segment_count=acquire_track(transport,video_url,staging,"video",before-old_bytes)
            row["rendition"]=selected
            row["segments"]=segment_count
            crops=[]
            for i in range(12):
                if time.monotonic()-start>600:
                    raise RuntimeError("Ten-minute caption capture deadline reached.")
                path=folder/f"caption-{12+i:02d}.png"
                # Keep the original subtitle pixels; display sheets may be JPEG.
                ffmpeg(["-protocol_whitelist","file,pipe","-i",str(video),"-ss",str(offset+i),"-frames:v","1","-vf","crop=iw:ih*0.28:0:ih*0.70","-fs","500000",str(path)])
                with Image.open(path) as image:
                    image.verify()
                if path.stat().st_size>500000:
                    raise RuntimeError("Caption frame exceeded its cap.")
                crops.append((path,12+i))
            for group in range(3):
                contact_sheet(crops[group*4:(group+1)*4],folder/f"sheet-{group+1:02d}.jpg")
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix in (".png",".jpg"):
                    row["artifacts"].append({"filename":path.name,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            row["status"]="caption_pictures_captured_text_not_yet_transcribed"
            row["seek_method"]="Output-side seek after decoding; avoids input keyframe-only snapshots"
            row["unique_caption_crops"]=len({a["sha256"] for a in row["artifacts"] if a["filename"].endswith(".png")})
        except Exception as error:
            row["status"]="capture_stopped"
            row["error"]=str(error)[:200] if isinstance(error,RuntimeError) else type(error).__name__
        finally:
            shutil.rmtree(staging)
        row["response_body_bytes"]=transport.received-before
        rows.append(row)
        publish({"phase":"caption_capture","completed":len(rows),"total":5,"last_status":row["status"],"trained":False})
    report={"phase":"bbc_caption_picture_capture_complete","sources":rows,"body_bytes":transport.received,"requests":transport.requests,"prior_caption_attempt_bytes":prior_bytes,"combined_caption_stage_body_bytes":prior_bytes+transport.received,"combined_caption_stage_requests":prior_requests+transport.requests,"cloud_only":True,"full_videos_saved":False,"trained":False,"optimizer_updates":0,
            "limits":{"body_bytes":32*1024**2,"per_source_bytes":6*1024**2,"cloud_output_staging_mib":64,"cloud_free_floor_gib":5,"minutes":10},
            "review_scope":"User confirmed BBC subtitle/speech match. Captured text and exact timings are not automatically certified by that review."}
    receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report
