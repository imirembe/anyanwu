"""Loopback-only listening/review server. No microphone, AI API or training."""
from __future__ import annotations

from datetime import datetime, timezone
import hmac
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import urlsplit

from .pilot import MAX_AUDIO, PILOT_ROOT, ROOT, PilotError, atomic_json, load_pilot

ASSETS = Path(__file__).resolve().parent / "web"
FONTS = ROOT / "Sources/Anyanwu/Resources/Fonts"


class ReviewStore:
    def __init__(self, folder, manifest):
        self.path = folder / "reviews.json"
        self.manifest = manifest
        self.lock = threading.Lock()
        self.reviews = {}
        if self.path.exists():
            if self.path.is_symlink() or self.path.stat().st_size > 16384:
                raise PilotError("The listening review file is invalid; it was not overwritten.")
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if saved["pilot"] != manifest["id"] or not isinstance(saved["reviews"], dict):
                    raise ValueError()
                for clip_id, entry in saved["reviews"].items():
                    clean = self.validate(entry)
                    if clean["clip_id"] != clip_id:
                        raise ValueError()
                self.reviews = saved["reviews"]
            except (ValueError, TypeError, KeyError, OSError):
                raise PilotError("The listening review file failed validation; it was not overwritten.") from None

    def validate(self, entry):
        if not isinstance(entry, dict):
            raise ValueError("Invalid review.")
        clips = {clip["id"]: clip for clip in self.manifest["clips"]}
        clip = clips.get(entry.get("clip_id"))
        if clip is None or entry.get("audio_sha256") != clip["sha256"]:
            raise ValueError("Review does not match this audio clip.")
        scores = {}
        for key in ("intelligibility", "cadence", "pronunciation"):
            value = entry.get(key)
            if value is not None and (type(value) is not int or not 1 <= value <= 5):
                raise ValueError("Ratings must be 1–5 or unrated.")
            scores[key] = value
        match = entry.get("transcript_match")
        if match not in {None, "yes", "unsure", "no"}:
            raise ValueError("Invalid transcript rating.")
        notes = entry.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 500:
            raise ValueError("Keep notes within 500 characters.")
        return {"clip_id": clip["id"], "audio_sha256": clip["sha256"], **scores,
                "transcript_match": match, "notes": notes}

    def save(self, entry):
        clean = self.validate(entry)
        clean["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self.lock:
            updated = self.reviews | {clean["clip_id"]: clean}
            atomic_json(self.path, {"schema": 1, "pilot": self.manifest["id"],
                                   "review_type": "User listening judgments, not independent benchmark gold",
                                   "reviews": updated})
            self.reviews = updated
        return clean

    def snapshot(self):
        with self.lock:
            return dict(self.reviews)


def make_server(root=PILOT_ROOT, port=8790):
    folder, manifest = load_pilot(root)
    store = ReviewStore(folder, manifest)
    nonce = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        server_version = "AnyanwuVoiceResearch"

        def log_message(self, *_):
            pass  # No private listening notes, paths or requests logged.

        def valid_host(self):
            return (len(self.headers.get_all("Host", [])) == 1
                    and self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"
                    and self.headers.get("Sec-Fetch-Site", "none") in {"none", "same-origin"})

        def send(self, status, body=b"", media="text/plain; charset=utf-8", extra=None):
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def send_json(self, status, value):
            self.send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if not self.valid_host():
                return self.send(403)
            parsed = urlsplit(self.path)
            if parsed.query or parsed.fragment or len(self.path) > 200:
                return self.send(404)
            if self.path == "/":
                page = (ASSETS / "index.html").read_text(encoding="utf-8").replace("__NONCE__", nonce)
                return self.send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if self.path == "/api/pilot":
                return self.send_json(200, {"manifest": manifest, "reviews": store.snapshot()})
            files = {"/review.css": (ASSETS / "review.css", "text/css; charset=utf-8"),
                     "/review.mjs": (ASSETS / "review.mjs", "text/javascript; charset=utf-8"),
                     "/fonts/NotoSans.ttf": (FONTS / "NotoSans.ttf", "font/ttf"),
                     "/fonts/NotoSerif.ttf": (FONTS / "NotoSerif.ttf", "font/ttf")}
            if self.path in files:
                path, media = files[self.path]
                return self.send(200, path.read_bytes(), media)
            match = re.fullmatch(r"/audio/(clip-0[1-6])", self.path)
            if match:
                clip = next(c for c in manifest["clips"] if c["id"] == match.group(1))
                with (folder / clip["filename"]).open("rb") as handle:
                    data = handle.read(MAX_AUDIO + 1)
                if len(data) > MAX_AUDIO or hashlib.sha256(data).hexdigest() != clip["sha256"]:
                    return self.send(409, b"Audio changed; restart the verified pilot.")
                size = len(data)
                headers = {"Accept-Ranges": "bytes"}
                range_header = self.headers.get("Range")
                if range_header is not None:
                    span = re.fullmatch(r"bytes=([0-9]{1,10})-([0-9]{0,10})", range_header)
                    if not span:
                        return self.send(416, extra={"Content-Range": f"bytes */{size}"})
                    start, end = int(span[1]), min(int(span[2]) if span[2] else size - 1, size - 1)
                    if not 0 <= start <= end < size:
                        return self.send(416, extra={"Content-Range": f"bytes */{size}"})
                    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                    return self.send(206, data[start:end+1], "audio/mpeg", headers)
                return self.send(200, data, "audio/mpeg", headers)
            return self.send(404)

        do_HEAD = do_GET

        def do_POST(self):
            if (not self.valid_host() or self.headers.get("Origin") != f"http://127.0.0.1:{self.server.server_port}"
                    or not hmac.compare_digest(self.headers.get("X-Review-Nonce", "").encode(), nonce.encode())):
                return self.send(403)
            if self.path != "/api/review":
                return self.send(404)
            length = self.headers.get("Content-Length", "")
            if (not 1 <= len(length) <= 5 or not length.isascii() or not length.isdigit()
                    or not 0 < int(length) <= 8192 or self.headers.get("Transfer-Encoding")
                    or self.headers.get("Content-Type") != "application/json"):
                return self.send(413)
            self.connection.settimeout(5)
            try:
                body = self.rfile.read(int(length))
                if len(body) != int(length):
                    raise ValueError("Incomplete review.")
                saved = store.save(json.loads(body))
            except (ValueError, TypeError, KeyError):
                return self.send_json(400, {"error": "Review must match this clip and use valid ratings and notes of at most 500 characters."})
            except (OSError, TimeoutError):
                return self.send_json(503, {"error": "Review could not be saved. Your entries remain on screen."})
            return self.send_json(200, {"saved": saved})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def serve():
    try:
        server = make_server()
    except OSError:
        raise PilotError("Port 8790 could not be opened. No existing server was stopped.") from None
    print("Anyanwu listening pilot: http://127.0.0.1:8790", flush=True)
    print("Six human research recordings; no microphone or AI API. Control-C stops this server.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
