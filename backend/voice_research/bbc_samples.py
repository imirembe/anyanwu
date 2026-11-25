"""Small public-stream samples, executed only on a hosted Colab CPU runtime.

No models, keys, training, geo override, or full-video fallback. Input URLs stay
in RAM; the report contains canonical source IDs, offsets and content hashes.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time
import urllib.parse
import wave

from backend.voice_research.bbc_probe import SOURCES, Transport, attributes, checked_url, cloud_guard

ROOT = Path("/content/anyanwu-bbc-igbo-pilot-v1/sample-v1")
START, DURATION = 12.0, 12.0


def attr(line):
    return {k: v.strip('"') for k, v in attributes(line).items()}


def master_selection(text, base):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "#EXTM3U":
        raise RuntimeError("Unexpected stream manifest; stopped.")
    if any(line.startswith("#EXT-X-SESSION-KEY:") and attr(line).get("METHOD") != "NONE" for line in lines):
        raise RuntimeError("Encrypted stream not supported; stopped.")
    variants, renditions = [], []
    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-MEDIA:"):
            renditions.append(attr(line))
        if line.startswith("#EXT-X-STREAM-INF:"):
            a = attr(line)
            if i+1 >= len(lines) or lines[i+1].startswith("#"):
                raise RuntimeError("Malformed HLS variant.")
            width = int(a.get("RESOLUTION", "0x0").split("x")[0])
            bandwidth = int(a.get("BANDWIDTH", 0))
            if width >= 360 and 0 < bandwidth <= 1_500_000:
                variants.append((bandwidth, checked_url(urllib.parse.urljoin(base, lines[i+1])), a))
    if not variants:
        raise RuntimeError("No bounded video rendition at least 360 pixels wide; no high-quality/full-video fallback.")
    bandwidth, video_url, a = min(variants, key=lambda v: v[0])
    audio = [r for r in renditions if r.get("TYPE") == "AUDIO" and r.get("GROUP-ID") == a.get("AUDIO") and r.get("URI")]
    audio.sort(key=lambda r: r.get("DEFAULT") != "YES")
    audio_url = checked_url(urllib.parse.urljoin(base, audio[0]["URI"])) if audio else None
    captions = [{k:r.get(k) for k in ("TYPE", "LANGUAGE", "NAME", "DEFAULT") if r.get(k)} for r in renditions if r.get("TYPE") in ("SUBTITLES", "CLOSED-CAPTIONS")]
    return video_url, audio_url, {"bandwidth": bandwidth, "resolution": a.get("RESOLUTION"), "codecs": a.get("CODECS"), "caption_renditions": captions}


def playlist_window(text, base, start=START, duration=DURATION):
    if not text.lstrip().startswith("#EXTM3U") or "#EXT-X-ENDLIST" not in text:
        raise RuntimeError("Only complete on-demand playlists are supported.")
    elapsed, next_duration, init_url, segments, discontinuity = 0.0, None, None, [], False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#EXT-X-KEY:") and attr(line).get("METHOD") != "NONE":
            raise RuntimeError("Encrypted stream not supported; stopped.")
        if line.startswith("#EXT-X-BYTERANGE") or (line.startswith("#EXT-X-MAP:") and "BYTERANGE" in attr(line)):
            raise RuntimeError("Byte-range stream not supported by this small pilot; stopped.")
        if line == "#EXT-X-DISCONTINUITY":
            discontinuity = True
        if line.startswith("#EXT-X-MAP:"):
            init_url = checked_url(urllib.parse.urljoin(base, attr(line)["URI"]))
        elif line.startswith("#EXTINF:"):
            next_duration = float(line.split(":", 1)[1].split(",")[0])
            if not 0 < next_duration <= 20:
                raise RuntimeError("Unexpected segment duration.")
        elif line and not line.startswith("#"):
            if next_duration is None:
                raise RuntimeError("Segment without a duration.")
            if elapsed+next_duration > start and elapsed < start+duration:
                if segments and (discontinuity or init_url != segments[0]["init"]):
                    raise RuntimeError("Sample crosses a stream discontinuity; stopped.")
                segments.append({"url": checked_url(urllib.parse.urljoin(base, line)), "start": elapsed, "duration": next_duration, "init": init_url})
            elapsed += next_duration
            next_duration, discontinuity = None, False
    if not segments or segments[-1]["start"]+segments[-1]["duration"] < start+duration-0.02:
        raise RuntimeError("Requested sample window is not covered.")
    if len(segments) > 8 or sum(s["duration"] for s in segments) > 32:
        raise RuntimeError("Too much segment overhead for this pilot.")
    return segments


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ffmpeg(arguments):
    result = subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", *arguments], capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError("Local cloud ffmpeg decoding failed; stopped without remote fallback.")


def acquire_track(transport, url, staging, name, source_start_bytes):
    playlist = transport.get(url).decode("utf-8-sig")
    segments = playlist_window(playlist, url)
    dest = staging/(name+(".mp4" if segments[0]["init"] else ".ts"))
    with dest.open("xb") as output:
        urls = ([segments[0]["init"]] if segments[0]["init"] else []) + [s["url"] for s in segments]
        for segment_url in urls:
            remaining = 6*1024**2-(transport.received-source_start_bytes)
            if remaining <= 0:
                raise RuntimeError("Six MiB source transfer cap reached.")
            output.write(transport.get(segment_url, cap=min(2*1024**2, remaining)))
    return dest, START-segments[0]["start"], len(segments)


def prepare_samples():
    cloud_guard()
    if shutil.disk_usage("/content").free < 5*1024**3:
        raise RuntimeError("At least five GiB free cloud disk required.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Cloud ffmpeg unavailable; no package installation attempted.")
    receipt = ROOT/"sample-report.json"
    if receipt.exists():
        report = json.loads(receipt.read_text())
        for row in report["sources"]:
            for asset in row.get("assets", []):
                path = ROOT/row["id"]/asset["filename"]
                if not path.is_file() or digest(path) != asset["sha256"]:
                    raise RuntimeError("Existing cloud sample failed integrity; no automatic overwrite.")
        return report
    if ROOT.exists():
        raise RuntimeError("A partial sample directory exists; preserve it for inspection before a new attempt.")
    ROOT.mkdir(parents=True)
    transport = Transport(budget=32*1024**2, requests=80, seconds=480)
    rows = []
    for source_id, version in SOURCES:
        row = {"id":source_id, "video_id":version, "source_start_seconds":START, "requested_seconds":DURATION, "training_ready":False, "language_alignment_speaker_review":"pending", "assets":[]}
        before = transport.received
        staging = ROOT/(source_id+"-staging")
        staging.mkdir()
        try:
            data = json.loads(transport.get(f"https://open.live.bbc.co.uk/mediaselector/6/select/version/2.0/mediaset/pc/vpid/{version}"))
            if data.get("result"):
                raise RuntimeError("BBC did not expose an accessible stream; stopped.")
            row["separate_caption_media"] = any(m.get("kind") == "captions" for m in data.get("media", []))
            urls = [c["href"] for m in data.get("media", []) for c in m.get("connection", []) if c.get("transferFormat") == "hls" and c.get("href", "").startswith("https://")]
            if not urls:
                raise RuntimeError("No public HTTPS HLS stream available.")
            master_url = checked_url(urls[0])
            master = transport.get(master_url).decode("utf-8-sig")
            video_url, audio_url, selected = master_selection(master, master_url)
            row["stream"] = selected
            video, video_offset, segment_count = acquire_track(transport, video_url, staging, "video", before)
            audio, audio_offset = video, video_offset
            if audio_url:
                audio, audio_offset, _ = acquire_track(transport, audio_url, staging, "audio", before)
            output = ROOT/source_id
            output.mkdir()
            wav = output/"speech-12s.wav"
            ffmpeg(["-protocol_whitelist","file,pipe","-ss",str(audio_offset),"-i",str(audio),"-t",str(DURATION),"-map","0:a:0","-vn","-ac","1","-ar","16000","-c:a","pcm_s16le", "-fs","500000",str(wav)])
            with wave.open(str(wav)) as reader:
                seconds = reader.getnframes()/reader.getframerate()
            row["audio_seconds"] = seconds
            row["duration_difference_seconds"] = round(seconds-DURATION, 6)
            if not DURATION-0.3 <= seconds <= DURATION+0.1:
                raise RuntimeError("Decoded audio duration does not cover the requested window.")
            row["video_segments"] = segment_count
            for i, offset in enumerate((1, 4, 7, 10)):
                frame = output/f"frame-{i+1}.jpg"
                ffmpeg(["-protocol_whitelist","file,pipe","-ss",str(video_offset+offset),"-i",str(video),"-frames:v","1","-vf","scale=480:-2","-q:v","3","-fs","250000",str(frame)])
                if not frame.is_file() or not 0 < frame.stat().st_size <= 250000:
                    raise RuntimeError("Frame output exceeded cap or failed.")
            for path in sorted(output.iterdir()):
                row["assets"].append({"filename":path.name,"bytes":path.stat().st_size,"sha256":digest(path)})
            row["frame_source_seconds"] = [START+n for n in (1,4,7,10)]
            row["status"] = "sample_ready_for_listening_and_caption_review"
        except Exception as error:
            row["status"] = "stopped"
            row["error"] = str(error)[:220] if isinstance(error, RuntimeError) else type(error).__name__
            # Keep any failed decoded output visibly separate from accepted assets.
            row["assets"] = []
        finally:
            # Only the staging directory created by this iteration is removed.
            shutil.rmtree(staging)
        row["response_body_bytes"] = transport.received-before
        rows.append(row)
    report = {"stage":"bbc_cloud_sample_v1","created_unix":time.time(),"sources":rows,
              "body_bytes":transport.received,"requests":transport.requests,"laptop_media_saved":False,
              "limits":{"body_bytes":32*1024**2,"per_source_body_bytes":6*1024**2,"cloud_staging_bytes":64*1024**2,"cloud_free_floor_gib":5,"minutes":8},
              "permission":"user_confirmed_training_permission","trained":False}
    receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report
