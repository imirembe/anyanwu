"""Bounded WAXAL metadata and human-recording pilot; no model downloads."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import tempfile
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = ROOT / ".build/VoiceResearch"
REPOSITORY = "google/WaxalNLP"
CONFIG = "ibo_tts"
MIB = 1024**2
RESERVE = 20 * 1024**3
MAX_TRANSFER = 32 * MIB
MAX_METADATA = 5 * MIB
MAX_CLIPS = 6
MAX_AUDIO = 4 * MIB
MAX_REQUESTS = 24
HOSTS = {"huggingface.co", "hf.co", "datasets-server.huggingface.co", "cdn-lfs.huggingface.co",
         "cdn-lfs.hf.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co",
         "cdn-assets.huggingface.co"}


class PilotError(Exception):
    pass


def checked_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except (ValueError, TypeError):
        raise PilotError("Invalid source URL.") from None
    if (parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or parsed.password
            or port not in {None, 443} or parsed.fragment
            or any(parsed.path.lower().endswith(ext) for ext in (".parquet", ".arrow", ".tar.gz", ".zip", ".safetensors", ".pth", ".pt"))):
        raise PilotError("Only bounded metadata and individual audio on approved Hugging Face hosts are allowed.")
    return url


class Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 3
    max_repeats = 1

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        checked_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


class Client:
    def __init__(self, budget=MAX_TRANSFER):
        self.budget = budget
        self.received = 0
        self.metadata_received = 0
        self.requests = 0
        try:
            import certifi
        except ImportError:
            tls = ssl.create_default_context()
        else:
            tls = ssl.create_default_context(cafile=certifi.where())
        # No ambient proxy credentials or Hugging Face/API tokens are sent.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirects(),
                                                 urllib.request.HTTPSHandler(context=tls))

    def get(self, url: str, *, limit: int, audio=False) -> tuple[bytes, dict]:
        checked_url(url)
        if self.requests >= MAX_REQUESTS:
            raise PilotError("Pilot request limit reached; no large-file fallback is allowed.")
        available = self.budget - self.received
        if not audio:
            available = min(available, MAX_METADATA - self.metadata_received)
        # Reserve one detection byte for an unannounced oversized response.
        cap = min(limit, available - 1)
        if cap < 1:
            raise PilotError("The pilot transfer budget is exhausted.")
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "Anyanwu-VoiceResearch/1"})
        self.requests += 1
        try:
            response = self.opener.open(request, timeout=25)
        except urllib.error.HTTPError as error:
            error.close()
            raise PilotError(f"The source returned HTTP {error.code} for {urllib.parse.urlsplit(url).path}; no dataset archive will be downloaded.") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise PilotError("The bounded source request failed; no automatic large-file fallback is allowed.") from None
        with response:
            declared = response.headers.get("Content-Length")
            if declared and (not declared.isdigit() or int(declared) > cap):
                raise PilotError("A source response exceeds the pilot limit; it was not downloaded.")
            pieces = []
            size = 0
            while True:
                chunk = response.read(min(65536, cap - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                self.received += len(chunk)
                if not audio:
                    self.metadata_received += len(chunk)
                if size > cap:
                    raise PilotError("A source response exceeded the pilot byte limit.")
                pieces.append(chunk)
            return b"".join(pieces), dict(response.headers)

    def json(self, url, *, limit=MIB):
        body, _ = self.get(url, limit=limit)
        try:
            value = json.loads(body)
        except (ValueError, UnicodeError):
            raise PilotError("The source did not return bounded JSON.") from None
        if not isinstance(value, dict):
            raise PilotError("Unexpected dataset metadata shape.")
        return value


def ensure_space(extra=64 * MIB):
    if shutil.disk_usage(ROOT).free - extra < RESERVE:
        raise PilotError("The pilot would cross the required 20 GiB free-space reserve.")


def discover(client: Client):
    hub = client.json(f"https://huggingface.co/api/datasets/{REPOSITORY}")
    revision = hub.get("sha")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise PilotError("The Hub did not identify an exact dataset revision.")
    if hub.get("gated"):
        raise PilotError("This source requires access approval. No credentials or gated files were requested.")
    card, _ = client.get(f"https://huggingface.co/datasets/{REPOSITORY}/raw/{revision}/README.md", limit=MIB)
    card_text = card.decode("utf-8")
    if not all(value in card_text for value in ("Media Trust", "Igbo", "CC-BY-SA-4.0", "TTS")):
        raise PilotError("The expected Igbo TTS source/licence description changed.")
    config_names = [item.get("config_name") for item in hub.get("cardData", {}).get("configs", []) if isinstance(item, dict)]
    if CONFIG not in config_names:
        raise PilotError("The exact Igbo TTS configuration was not found in publisher metadata.")
    query = urllib.parse.urlencode({"dataset": REPOSITORY, "config": CONFIG, "split": "train", "offset": 0, "length": 48})
    endpoint = "rows"
    try:
        page = client.json("https://datasets-server.huggingface.co/rows?" + query)
    except PilotError as error:
        if not any(f"HTTP {code}" in str(error) for code in (501, 502, 503, 504)):
            raise
        endpoint = "first-rows"
        query = urllib.parse.urlencode({"dataset": REPOSITORY, "config": CONFIG, "split": "train"})
        page = client.json("https://datasets-server.huggingface.co/first-rows?" + query)
    rows = page.get("rows")
    if not isinstance(rows, list) or not 0 < len(rows) <= 100:
        raise PilotError("The bounded viewer did not expose eligible rows.")
    if page.get("dataset_git_revision") not in {None, revision}:
        raise PilotError("The viewer revision differs from the verified source snapshot.")
    summary = {"repository": REPOSITORY, "config": CONFIG, "hub_revision": revision,
               "card_sha256": hashlib.sha256(card).hexdigest(), "source_card_url": f"https://huggingface.co/datasets/{REPOSITORY}/blob/{revision}/README.md",
               "license_declared": "CC-BY-SA-4.0", "provider": "Media Trust / WAXAL",
               "viewer_endpoint": endpoint, "viewer_revision": page.get("dataset_git_revision"),
               "viewer_snapshot_note": "Viewer rows may not be pinned to the Hub revision; compare any supplied revision and verify head again.",
               "rows_returned": len(rows), "source_total_rows": page.get("num_rows_total"),
               "columns": sorted({key for entry in rows if isinstance(entry, dict) for key in entry.get("row", {})}),
               "speakers": sorted({str(entry.get("row", {}).get("speaker_id")) for entry in rows if isinstance(entry, dict)}),
               "metadata_bytes": client.metadata_received, "requests": client.requests}
    return summary, rows


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False, prefix=".writing-") as handle:
        pending = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    pending.replace(path)


def eligible_rows(rows, revision=None):
    """Publisher text stays unchanged; no AI-written transcript replacements."""
    result = []
    seen = set()
    for entry in rows:
        if not isinstance(entry, dict) or entry.get("truncated_cells"):
            continue
        row = entry.get("row", {})
        if not isinstance(row, dict):
            continue
        text, source_id = row.get("text"), row.get("id")
        index, speaker, locale = entry.get("row_idx"), row.get("speaker_id"), row.get("locale")
        if (not isinstance(text, str) or not 45 <= len(text) <= 420
                or not isinstance(source_id, str) or not re.fullmatch(r"ibo_[A-Za-z0-9_-]{1,80}", source_id)
                or source_id in seen or isinstance(index, bool) or not isinstance(index, int) or index < 0
                or not isinstance(speaker, (str, int)) or len(str(speaker)) > 80
                or not isinstance(locale, str) or locale not in {"ibo", "ig", "ig-NG", "ibo_NG", "ig_ng"}):
            continue
        audio = row.get("audio")
        if not isinstance(audio, list):
            continue
        for item in audio:
            if not isinstance(item, dict) or item.get("type") != "audio/mpeg" or not isinstance(item.get("src"), str):
                continue
            url = checked_url(item["src"])
            path = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
            location = re.fullmatch(r"/cached-assets/google/WaxalNLP/--/([0-9a-f]{40})/--/ibo_tts/train/([0-9]+)/audio/audio\.mp3", path)
            if (urllib.parse.urlsplit(url).hostname != "datasets-server.huggingface.co"
                    or location is None or int(location[2]) != index
                    or revision is not None and location[1] != revision):
                raise PilotError("The preview audio does not belong to the expected Igbo training subset.")
            result.append({"source_id": source_id, "source_row_index": index, "speaker_id": str(speaker),
                           "transcript": text, "locale": locale, "audio_asset_revision": location[1], "url": url})
            seen.add(source_id)
            break
    return result


def audio_duration(path: Path) -> float:
    """Use the OS decoder; no speech-model/dependency install or audio rewriting."""
    tool = shutil.which("afinfo")
    if not tool:
        raise PilotError("This pilot needs macOS afinfo to validate the downloaded MP3 duration.")
    try:
        result = subprocess.run([tool, str(path)], capture_output=True, text=True, timeout=10,
                                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    except (OSError, subprocess.TimeoutExpired):
        raise PilotError("The OS could not validate a preview audio clip.") from None
    match = re.search(r"estimated duration:\s*([0-9.]+)\s*sec", result.stdout)
    if result.returncode or not match:
        raise PilotError("The preview is not a decodable audio clip.")
    duration = float(match.group(1))
    if not math.isfinite(duration) or not 0 < duration <= 120:
        raise PilotError("The preview audio duration exceeds the bounded pilot.")
    return round(duration, 3)


def load_pilot(root=PILOT_ROOT):
    folder = root / "waxal-ibo-v1"
    manifest_path = folder / "manifest.json"
    if folder.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > MIB:
        raise PilotError("Prepare the listening pilot first: python3 scripts/voice_pilot.py prepare")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        clips = manifest["clips"]
        if not isinstance(clips, list) or len(clips) != MAX_CLIPS:
            raise ValueError()
        for number, clip in enumerate(clips, 1):
            if clip["id"] != f"clip-{number:02}" or clip["filename"] != f"clip-{number:02}.mp3":
                raise ValueError()
            audio = folder / clip["filename"]
            if (audio.is_symlink() or not audio.is_file() or not 0 < audio.stat().st_size <= MAX_AUDIO
                    or hashlib.sha256(audio.read_bytes()).hexdigest() != clip["sha256"]):
                raise ValueError()
    except (ValueError, TypeError, KeyError, OSError):
        raise PilotError("The local pilot failed its manifest/audio integrity checks; nothing was replaced.") from None
    return folder, manifest


def prepare(root=PILOT_ROOT, client=None):
    target = root / "waxal-ibo-v1"
    if target.exists():
        folder, manifest = load_pilot(root)
        print(f"Reusing the verified six-clip pilot at {folder}; no network request or download.")
        return manifest
    ensure_space()
    root.mkdir(parents=True, exist_ok=True)
    client = client or Client()
    receipt = {"status": "in_progress", "started_at": datetime.now(timezone.utc).isoformat(),
               "limits": {"response_body_bytes": MAX_TRANSFER, "metadata_bytes": MAX_METADATA,
                          "clips": MAX_CLIPS, "per_clip_bytes": MAX_AUDIO}, "clips_installed": 0}
    # A failed attempt leaves only its bounded receipt; no incomplete corpus is installed.
    receipt_path = root / "last-attempt.json"
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=root) as staging:
        stage = Path(staging)
        try:
            summary, rows = discover(client)
            candidates = eligible_rows(rows, summary["hub_revision"])
            if len(candidates) < MAX_CLIPS:
                raise PilotError("The bounded preview has too few eligible complete Igbo texts; no full-dataset fallback.")
            clips, speakers = [], {}
            for row in candidates:
                if len(clips) == MAX_CLIPS:
                    break
                ensure_space()
                data, _ = client.get(row["url"], limit=MAX_AUDIO, audio=True)
                if not (data.startswith(b"ID3") or len(data) >= 2 and data[0] == 255 and data[1] & 224 == 224):
                    raise PilotError("A preview audio response is not an MP3; no file was installed.")
                clip_id = f"clip-{len(clips)+1:02}"
                path = stage / (clip_id + ".mp3")
                path.write_bytes(data)
                duration = audio_duration(path)
                if not 3 <= duration <= 30:
                    path.unlink()
                    continue
                speaker = row["speaker_id"]
                speakers.setdefault(speaker, f"Speaker {chr(65+len(speakers))}")
                clips.append({key: value for key, value in row.items() if key != "url"} | {
                    "id": clip_id, "filename": path.name, "speaker_label": speakers[speaker],
                    "duration_seconds": duration, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "media_type": "audio/mpeg", "representation": "Hugging Face viewer MP3 preview, not a lossless source recording",
                    "source_row_url": f"https://huggingface.co/datasets/{REPOSITORY}/viewer/{CONFIG}/train?row={row['source_row_index']}",
                    "review_status": "pending_human_listening"})
                print(f"Prepared {clip_id}: {duration:.1f}s, {len(data)} bytes", flush=True)
            if len(clips) != MAX_CLIPS:
                raise PilotError("The bounded preview could not fill six suitable-length clips; no incomplete pilot was installed.")
            head = client.json(f"https://huggingface.co/api/datasets/{REPOSITORY}")
            if head.get("sha") != summary["hub_revision"]:
                raise PilotError("The source changed during acquisition; the pilot was not installed.")
            receipt.update(status="complete", response_body_bytes=client.received, metadata_bytes=client.metadata_received,
                           requests=client.requests, clips_installed=len(clips), stored_audio_bytes=sum(c["bytes"] for c in clips))
            manifest = {"schema": 1, "id": "waxal-ibo-v1", "purpose": "Human listening research; not training-ready or a generated Anyanwu voice",
                        "source": summary, "attribution": "Media Trust / WAXAL, google/WaxalNLP, Igbo TTS subset; CC BY-SA 4.0",
                        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                        "permissions_note": "Published TTS research source; individual consent records have not been independently audited. No speaker imitation is authorised by this pilot.",
                        "selection": "First six complete 45–420-character train texts whose viewer MP3 lasts 3–30 seconds; convenience sample, not dialect-balanced",
                        "speaker_count": len(speakers), "human_review": "pending", "training_ready": False,
                        "hub_head_matched_before_after": True, "audio_url_revisions_match_hub": True,
                        "acquisition": receipt, "clips": clips}
            atomic_json(stage / "manifest.json", manifest)
            ensure_space(0)
            stage.rename(target)
            atomic_json(receipt_path, receipt)
        except BaseException:
            receipt.update(status="failed", response_body_bytes=client.received, metadata_bytes=client.metadata_received,
                           requests=client.requests, clips_installed=0)
            atomic_json(receipt_path, receipt)
            raise
    print(json.dumps({"pilot": str(target), "clips": MAX_CLIPS, "speakers": len(speakers),
                      "audio_bytes": receipt["stored_audio_bytes"], "transfer_bytes": client.received,
                      "training_ready": False}, ensure_ascii=False, indent=2))
    return manifest


def inspect():
    ensure_space()
    client = Client(MAX_METADATA)
    summary, rows = discover(client)
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(PILOT_ROOT / "source-probe.json", summary)
    # No raw API page, demographic fields or signed media URLs are persisted.
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    first = rows[0].get("row", {})
    audio = first.get("audio")
    print(json.dumps({"first_row_id": first.get("id"), "text_characters": len(first.get("text", "")),
                      "locale": first.get("locale"),
                      "audio_locations": [{"host": urllib.parse.urlsplit(item.get("src", "")).hostname,
                                           "path": urllib.parse.unquote(urllib.parse.urlsplit(item.get("src", "")).path)}
                                          for item in audio if isinstance(item, dict)] if isinstance(audio, list) else [],
                      "audio_shape": type(audio).__name__, "audio_fields": list(audio[0]) if isinstance(audio, list) and audio else list(audio) if isinstance(audio, dict) else [],
                      "audio_types": [item.get("type") for item in audio if isinstance(item, dict)] if isinstance(audio, list) else []}, ensure_ascii=False))
