"""Run a real cloud voice turn; never download models or retain input audio."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import time
import urllib.request

from backend.server import MAX_AUDIO_BYTES, MAX_REQUEST_BYTES, load_environment


def fetch(url, token, *, payload=None):
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    request = urllib.request.Request(url, headers=headers, data=None if payload is None else json.dumps(payload).encode())
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(MAX_REQUEST_BYTES + 1)
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError("Service response exceeded its limit.")
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="A real test recording already on the cloud host (under 60 seconds).")
    parser.add_argument("--save-reply", type=Path, help="Optionally save generated MP3 for a fluent speaker to assess.")
    args = parser.parse_args()
    load_environment(Path(__file__).with_name(".env"))
    token = os.environ.get("ANYANWU_ACCESS_TOKEN", "")
    worker_token = os.environ.get("ANYANWU_WORKER_TOKEN", "")
    if not token or not worker_token:
        parser.error("Run the cloud configure script first.")
    mime = {".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".webm": "audio/webm"}.get(args.audio.suffix.lower())
    if not mime:
        parser.error("Choose m4a, mp3, wav, or webm.")
    with args.audio.open("rb") as file:
        audio = file.read(MAX_AUDIO_BYTES + 1)
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        parser.error("Audio must be nonempty and under 6 MB.")
    try:
        for port in (8001, 8002):
            if not fetch(f"http://127.0.0.1:{port}/readyz", worker_token)["ready"]:
                raise ValueError("A worker is not ready.")
        start = time.monotonic()
        result = fetch("http://127.0.0.1:8787/v1/turns", token, payload={
            "inputLanguages": ["ig", "en"], "responseLanguage": "ig", "interfaceLanguage": "en",
            "input": {"type": "audio", "audioBase64": base64.b64encode(audio).decode(), "mimeType": mime}})
        encoded = result.pop("audioBase64", None)
        if not result.get("transcript") or not encoded:
            raise ValueError("The speech turn did not produce both a transcript and audio.")
        reply = base64.b64decode(encoded, validate=True)
        if args.save_reply:
            args.save_reply.parent.mkdir(parents=True, exist_ok=True)
            # Do not silently overwrite a previous evaluation recording.
            with args.save_reply.open("xb") as file:
                file.write(reply)
        print(json.dumps({**result, "elapsedSeconds": round(time.monotonic() - start, 3),
                          "replyAudioBytes": len(reply), "savedReply": bool(args.save_reply)}, ensure_ascii=False, indent=2))
    except Exception as error:
        # HTTPError/URLError strings may contain endpoint details. Never print
        # credentials or provider bodies; inspect the authenticated health APIs.
        raise SystemExit("Speech smoke test failed: " + type(error).__name__) from None


if __name__ == "__main__":
    main()
