"""Small cloud-only spontaneous Igbo listening reference; no model work."""
from pathlib import Path
import base64
import csv
import hashlib
import html
import importlib.util
import io
import json
import platform
import shutil
import ssl
import subprocess
import time
import urllib.parse
import urllib.request
import wave

MIB = 1024**2
HOSTS = {"huggingface.co", "hf.co", "cdn-lfs.huggingface.co", "cdn-lfs.hf.co",
         "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co",
         "us.aws.cdn.hf.co", "eu.aws.cdn.hf.co", "us.gcp.cdn.hf.co"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_url(url):
    p = urllib.parse.urlsplit(url)
    if (p.scheme != "https" or p.hostname not in HOSTS or p.username or p.password
            or p.port not in (None, 443)):
        raise RuntimeError("Unexpected source host; no authentication or alternate download attempted")
    return url


class Transport:
    def __init__(self):
        self.received = 0
        self.requests = 0
        self.started = time.monotonic()
        self.redirects = 0
        owner = self

        class Redirects(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                owner.redirects += 1
                if owner.redirects > 3:
                    raise RuntimeError("Redirect limit reached")
                owner.request_guard()
                checked_url(newurl)
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), Redirects(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def request_guard(self):
        if self.requests >= 40 or time.monotonic() - self.started > 480:
            raise RuntimeError("Reference network request/time budget reached")
        self.requests += 1

    def fetch(self, url, limit):
        self.request_guard()
        self.redirects = 0
        cap = min(limit, 50*MIB - self.received)
        if cap <= 0:
            raise RuntimeError("50 MiB response budget reached")
        request = urllib.request.Request(checked_url(url), headers={
            "Accept-Encoding": "identity", "User-Agent": "Anyanwu-SpontaneousReference/1"})
        with self.opener.open(request, timeout=30) as response:
            checked_url(response.url)
            length = response.headers.get("Content-Length")
            if length and (not length.isdigit() or int(length) > cap):
                raise RuntimeError("Source exceeds declared file budget")
            chunks = []
            while True:
                if time.monotonic() - self.started > 480:
                    raise RuntimeError("Reference network deadline reached")
                chunk = response.read(min(65536, cap + 1 - sum(map(len, chunks))))
                self.received += len(chunk)
                if not chunk:
                    break
                chunks.append(chunk)
                if sum(map(len, chunks)) > cap:
                    raise RuntimeError("Source exceeded response budget")
            return b"".join(chunks)


def write_once(path, raw):
    if path.is_symlink():
        raise RuntimeError("Symlink refused")
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("Different existing reference file preserved: " + path.name)
        return
    with path.open("xb") as handle:
        handle.write(raw)
    if path.read_bytes() != raw:
        raise RuntimeError("Reference write verification failed")


def wav_info(path, listening=False):
    if path.is_symlink() or not 44 < path.stat().st_size <= (2 if listening else 8)*MIB:
        raise RuntimeError("WAV size/path check failed")
    with wave.open(str(path), "rb") as source:
        rate, channels, width, frames = (source.getframerate(), source.getnchannels(),
                                         source.getsampwidth(), source.getnframes())
        duration = frames / rate
        if source.getcomptype() != "NONE" or not 1 <= duration <= 45:
            raise RuntimeError("Expected complete, bounded PCM source")
        if listening and (rate, channels, width) != (24000, 1, 2):
            raise RuntimeError("Listening copy format mismatch")
        if not listening and (rate != 48000 or channels != 2 or width not in (2, 3, 4)):
            raise RuntimeError("Publisher source format changed")
        decoded = source.readframes(frames + 1)
        if len(decoded) != frames * channels * width:
            raise RuntimeError("Truncated source; no listening reference installed")
    return {"seconds": duration, "sample_rate": rate, "channels": channels,
            "sample_width": width, "bytes": path.stat().st_size,
            "sha256": sha(path.read_bytes())}


def verify_manifest(folder, plan):
    path = folder / "manifest.json"
    if path.is_symlink() or not 0 < path.stat().st_size <= 65536:
        raise RuntimeError("Invalid reference manifest")
    value = json.loads(path.read_bytes())
    if (value.get("plan_sha256") != sha(json.dumps(plan, sort_keys=True).encode())
            or len(value.get("clips", [])) != 6):
        raise RuntimeError("Existing reference selection differs; preserved")
    for clip, selected in zip(value["clips"], plan["clips"]):
        if (clip["id"] != selected["id"] or clip["filename"] != selected["id"] + ".wav"
                or clip["speaker_id"] != selected["speaker_id"]):
            raise RuntimeError("Reference identity mismatch")
        if wav_info(folder / clip["filename"], True) != clip["listening"]:
            raise RuntimeError("Reference listening copy integrity failed")
    return value


def run_reference(plan):
    if (platform.system() != "Linux" or not Path("/content").is_dir()
            or importlib.util.find_spec("google.colab") is None):
        raise RuntimeError("Hosted Colab only; never downloads audio to the laptop")
    from IPython.display import HTML, Javascript, display

    def publish(value):
        display(Javascript("window.top.postMessage(" + json.dumps({
            "type": "anyanwu-cloud-diagnostic", "nonce": "anyanwu-20261122-setup-r2",
            "payload": value}) + ', "https://colab.research.google.com");'))

    root = Path("/content/anyanwu-conversational-igbo-v1")
    drive = Path("/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research")
    backup = drive / "references/conversational-igbo-v1"
    net = Transport()
    try:
        if not drive.is_dir():
            publish({'phase': 'spontaneous_reference_waiting_for_drive', 'audio_fetched': False})
            from google.colab import drive as colab_drive
            colab_drive.mount('/content/drive', timeout_ms=300000)
            if not drive.is_dir():
                raise RuntimeError("Expected shared research Drive path unavailable; no audio fetched")
        if shutil.disk_usage("/content").free < 5*1024**3:
            raise RuntimeError("Cloud disk below 5 GiB reserve")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg unavailable; no install or model fallback")
        root.mkdir(exist_ok=True)
        if root.is_symlink() or backup.is_symlink():
            raise RuntimeError("Unexpected reference folder")
        def storage_guard():
            if (sum(p.stat().st_size for p in root.iterdir() if p.is_file()) > 128*MIB
                    or shutil.disk_usage('/content').free < 5*1024**3):
                raise RuntimeError('Reference cloud storage budget reached')
        storage_guard()
        if not (root / "manifest.json").exists() and (backup / "manifest.json").exists():
            restored = verify_manifest(backup, plan)
            for clip in restored["clips"]:
                write_once(root / clip["filename"], (backup / clip["filename"]).read_bytes())
            write_once(root / "manifest.json", (backup / "manifest.json").read_bytes())
        if (root / "manifest.json").exists():
            manifest = verify_manifest(root, plan)
        else:
            resolve = "https://huggingface.co/datasets/" + plan["repository"] + "/resolve/" + plan["revision"] + "/"
            card = net.fetch(resolve + "README.md", 256*1024)
            metadata = net.fetch(resolve + plan["metadata_path"], 256*1024)
            if sha(card) != plan["card_sha256"] or sha(metadata) != plan["metadata_sha256"]:
                raise RuntimeError("Pinned source metadata changed")
            rows = list(csv.DictReader(io.StringIO(metadata.decode("utf-8-sig"))))
            clips = []
            original_total = 0
            for selected in plan["clips"]:
                storage_guard()
                matches = [r for r in rows if r["speaker_id"] == selected["speaker_id"]
                           and r["script_id"] == selected["script_id"]]
                if len(matches) != 1:
                    raise RuntimeError("Source row identity is not unique")
                row = matches[0]
                if (row["gender"] != "female" or row["dialect"] != "Nigeria - Owerri"
                        or row["type_of_script"] != "free_speech" or row["transcript"] != "unknown"
                        or row["file_name"] != selected["file_name"]):
                    raise RuntimeError("Source selection differs from recorded metadata")
                if row["age_band"] not in ("18-24", "25-34", "35-44", "45-54", "55-64", "65+"):
                    raise RuntimeError("Adult source label required")
                path = "igbo_nigeria/free_speech/" + selected["file_name"]
                original = net.fetch(resolve + urllib.parse.quote(path, safe="/"), 8*MIB)
                original_total += len(original)
                if original_total > 48*MIB:
                    raise RuntimeError("Source audio budget reached")
                source_path = root / (selected["id"] + ".source.wav")
                write_once(source_path, original)
                source_info = wav_info(source_path)
                if abs(source_info["seconds"] - selected["publisher_seconds"]) > 1:
                    raise RuntimeError("Source duration differs from publisher metadata")
                listen_path = root / (selected["id"] + ".wav")
                if listen_path.exists():
                    raise RuntimeError("Partial prior output preserved; inspect before retry")
                subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                                "-i", str(source_path), "-map", "0:a:0", "-ac", "1", "-ar", "24000",
                                "-c:a", "pcm_s16le", str(listen_path)], check=True,
                               timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                listening = wav_info(listen_path, True)
                if abs(source_info["seconds"] - listening["seconds"]) > 1/24000:
                    raise RuntimeError("Listening copy lost source timing")
                clips.append({**selected, "filename": listen_path.name,
                              "source": source_info, "listening": listening,
                              "transcript": None, "training_ready": False})
            if sum(c["listening"]["bytes"] for c in clips) > 12*MIB:
                raise RuntimeError("Listening copy budget exceeded")
            manifest = {"phase": "spontaneous_reference_ready", "plan_sha256": sha(json.dumps(plan, sort_keys=True).encode()),
                        "repository": plan["repository"], "revision": plan["revision"],
                        "attribution": plan["attribution"], "license": "CC BY-NC 4.0",
                        "role": "Qualitative noncommercial source listening only; no model benchmark, training or synthesis",
                        "transcripts": "Absent. Elicitation topics are not spoken transcripts.",
                        "edits": "Downmix and resample only; complete timing retained, no silence/pitch/speed edits",
                        "clips": clips, "response_body_bytes": net.received, "requests": net.requests,
                        "optimizer_updates": 0, "models_downloaded": False}
            write_once(root / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
            write_once(root / "source-card.md", card)
        backup.mkdir(exist_ok=True, parents=True)
        storage_guard()
        for clip in manifest["clips"]:
            write_once(backup / clip["filename"], (root / clip["filename"]).read_bytes())
        write_once(backup / "manifest.json", (root / "manifest.json").read_bytes())
        write_once(backup / "plan.json", (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode())
        if (root / 'source-card.md').is_file():
            write_once(backup / 'source-card.md', (root / 'source-card.md').read_bytes())
        verify_manifest(backup, plan)
        cards = []
        for clip in manifest["clips"]:
            encoded = base64.b64encode((root / clip["filename"]).read_bytes()).decode()
            cards.append('<article style="border:1px solid #dfe2d5;border-radius:18px;padding:18px;margin:16px 0">'
                         + '<b>' + html.escape(clip["id"] + " · " + clip["topic"]) + '</b><p>'
                         + html.escape(clip["speaker_id"]) + ' · Female · Owerri label · '
                         + f'{clip["source"]["seconds"]:.1f} seconds</p>'
                         + '<audio controls preload="none" style="width:100%" src="data:audio/wav;base64,'
                         + encoded + '"></audio></article>')
        display(HTML('<section style="max-width:800px;padding:24px;color:#173f34;background:#f7f5eb;font:17px/1.6 system-ui">'
                     '<h2>Natural Igbo delivery · source listening</h2>'
                     '<p>Six whole spontaneous answers from three publisher-labelled female Owerri speakers. '
                     'These are human source recordings. Topics describe the prompts; verified transcripts are unavailable.</p>'
                     '<p>Listen for words joining smoothly, pauses between ideas, emphasis and changing pace. '
                     'Consider which delivery feels warm and useful for a spoken assistant. No ratings are submitted automatically.</p>'
                     + ''.join(cards) + '<p>' + html.escape(plan["attribution"]) + '</p>'
                     '<p>Research reference only. Full timing is retained; copies are downmixed/resampled. '
                     'Saved to the connected research Drive. No training started.</p></section>'))
        publish({"phase": "spontaneous_reference_listening_ready", "manifest": manifest,
                 "drive_backup": str(backup), "drive_backup_verified": True,
                 "players": 6, "new_optimizer_updates": 0, "laptop_media_downloaded": False})
    except Exception as error:
        publish({"phase": "spontaneous_reference_stopped", "error_type": type(error).__name__,
                 "error": str(error)[:700], "response_body_bytes": net.received,
                 "requests": net.requests, "optimizer_updates": 0})
        raise
