# Anyanwu female Igbo publisher preview — no model installation or generation
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import ssl
import urllib.error
import urllib.parse
import urllib.request

if (platform.system() != "Linux" or not Path("/content").is_dir()
        or importlib.util.find_spec("google.colab") is None):
    raise RuntimeError("Run this preview in the existing hosted Colab notebook, not on the laptop.")
from IPython.display import Audio, Javascript, display
REPO = "Modularcomputing/dynamic-accent-preview"
VOICE = "waxal_ibo_5_female"
PREVIEW_ROOT = Path("/content/anyanwu-voice-audition-v1/female-publisher-preview-v1")
SAMPLES = [("Short Igbo passage", "samples/flash/igbo_pure_female.mp3"),
           ("Longer Igbo passage", "samples/flash/igbo_para_female.mp3"),
           ("Igbo numbers and dates", "samples/flash/numbers_igbo.mp3")]
HOSTS = {"huggingface.co", "hf.co", "cdn-lfs.huggingface.co", "cdn-lfs.hf.co",
         "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co",
         "us.aws.cdn.hf.co", "eu.aws.cdn.hf.co", "us.gcp.cdn.hf.co"}
MIB = 1024**2


def checked(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != "https" or p.hostname not in HOSTS or p.username or p.password or p.port not in {None, 443}:
        raise RuntimeError(f"Unexpected source host: {p.hostname}; no file fetched.")
    return url


class Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 3
    max_repeats = 1
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        checked(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirects(),
                                    urllib.request.HTTPSHandler(context=ssl.create_default_context()))
received = 0
requests = 0


def fetch(url, limit):
    global received, requests
    if requests >= 8:
        raise RuntimeError("Preview request cap reached; no full-model fallback.")
    cap = min(limit, 3*MIB - received - 1)
    if cap < 1:
        raise RuntimeError("The 3 MiB preview transfer cap is exhausted.")
    requests += 1
    req = urllib.request.Request(checked(url), headers={"Accept-Encoding": "identity", "User-Agent": "Anyanwu-FemaleVoicePreview/1"})
    with opener.open(req, timeout=25) as response:
        length = response.headers.get("Content-Length")
        if length and (not length.isdigit() or int(length) > cap):
            raise RuntimeError("The preview file exceeds its size limit.")
        data = response.read(cap + 1)
        received += len(data)
        if len(data) > cap:
            raise RuntimeError("The preview file exceeded its size limit.")
        return data


def publish(payload):
    message = {"type": "anyanwu-cloud-diagnostic", "nonce": "anyanwu-20261122-setup-r2", "payload": payload}
    display(Javascript("window.top.postMessage(" + json.dumps(message) + ", \"https://colab.research.google.com\");"))


try:
    if shutil.disk_usage("/content").free < 5*1024**3:
        raise RuntimeError("The cloud disk is below its 5 GiB reserve.")
    manifest_path = PREVIEW_ROOT / "manifest.json"
    if manifest_path.exists():
        if manifest_path.stat().st_size > 65536:
            raise RuntimeError("The existing preview manifest is invalid; it was preserved.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != REPO or manifest.get("voice") != VOICE or len(manifest.get("clips", [])) != 3:
            raise RuntimeError("Existing preview metadata does not match this candidate.")
        for index, clip in enumerate(manifest["clips"], 1):
            if clip["filename"] != f"female-igbo-{index:02}.mp3":
                raise RuntimeError("Unexpected preview filename.")
            path = PREVIEW_ROOT / clip["filename"]
            if path.is_symlink() or not 0 < path.stat().st_size <= 512*1024 or hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
                raise RuntimeError("The saved preview failed its integrity check.")
    else:
        info = json.loads(fetch(f"https://huggingface.co/api/models/{REPO}?blobs=true", 512*1024))
        revision = info.get("sha", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or info.get("gated"):
            raise RuntimeError("A public exact model revision could not be verified.")
        files = {f["rfilename"] for f in info.get("siblings", [])}
        if not all(path in files for _, path in SAMPLES):
            raise RuntimeError("The expected publisher demo files are missing.")
        card = fetch(f"https://huggingface.co/{REPO}/raw/{revision}/README.md", 128*1024)
        if b"cc-by-sa-4.0" not in card.lower() or VOICE.encode() not in card:
            raise RuntimeError("The expected voice/licence declaration changed.")
        staged = []
        for index, (label, source_path) in enumerate(SAMPLES, 1):
            data = fetch(f"https://huggingface.co/{REPO}/resolve/{revision}/{source_path}", 512*1024)
            if not (data.startswith(b"ID3") or len(data) > 2 and data[0] == 255 and data[1] & 224 == 224):
                raise RuntimeError("A publisher preview was not an MP3.")
            staged.append(({"id": f"female-igbo-{index:02}", "label": label,
                            "filename": f"female-igbo-{index:02}.mp3", "source_path": source_path,
                            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, data))
        head = json.loads(fetch(f"https://huggingface.co/api/models/{REPO}", 512*1024))
        if head.get("sha") != revision:
            raise RuntimeError("The publisher revision changed; preview not installed.")
        manifest = {"status": "publisher_female_preview_pending_user_listening", "model": REPO,
                    "revision": revision, "voice": VOICE, "license_declared": "CC-BY-SA-4.0",
                    "source_card": f"https://huggingface.co/{REPO}/blob/{revision}/README.md",
                    "card_sha256": hashlib.sha256(card).hexdigest(),
                    "attribution": "Modular, Dynamic Accent Flash; source voice prompt from WAXAL / Media Trust Igbo speaker 5",
                    "role": "Unmodified publisher-generated female-voice demos, not freshly generated Anyanwu replies or a unique final voice",
                    "native_quality": "Not independently verified; awaiting user listening",
                    "training_or_voice_cloning_performed": False, "weights_downloaded": False,
                    "response_body_bytes": received, "requests": requests, "clips": [c for c, _ in staged]}
        PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
        for clip, data in staged:
            path = PREVIEW_ROOT / clip["filename"]
            if path.exists():
                raise RuntimeError("An existing clip was preserved; no overwrite.")
            path.write_bytes(data)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FEMALE IGBO CANDIDATE · Dynamic Accent Flash")
    print("Unmodified publisher demos. Judge the voice, Igbo pronunciation and rhythm; these are not newly generated Anyanwu replies.")
    print("Credit: Modular / WAXAL Media Trust · CC BY-SA 4.0")
    for clip in manifest["clips"]:
        print("\n" + clip["label"])
        display(Audio(filename=str(PREVIEW_ROOT / clip["filename"])))
    publish({"phase": "female_publisher_preview_ready", "manifest": manifest,
             "audio_bytes": sum(c["bytes"] for c in manifest["clips"]), "no_new_model_install": True})
except Exception as error:
    publish({"phase": "female_publisher_preview_failed", "error_type": type(error).__name__, "error": str(error)[:1000],
             "response_body_bytes": received, "requests": requests})
    raise
