#!/usr/bin/env python3
"""Bounded BBC metadata preflight only. Never fetches video/audio/subtitles."""
from pathlib import Path
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / ".build/VoiceResearch/bbc-igbo-female-sources-v1"
CAP = 256 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("Metadata redirect refused; no alternate host or media fallback.")


def main():
    sources = json.loads((FOLDER / "sources.json").read_text(encoding="utf-8"))
    if sources["permission"]["status"] != "user_confirmed_training_permission":
        raise RuntimeError("Training permission has not been recorded.")
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(), urllib.request.HTTPSHandler(context=context))
    received = 0
    rows = []
    for source in sources["sources"]:
        pid = source["video_id"]
        if not isinstance(pid, str) or len(pid) != 8 or not pid.isalnum():
            raise RuntimeError("Unexpected source video identifier.")
        url = f"https://open.live.bbc.co.uk/mediaselector/6/select/version/2.0/mediaset/pc/vpid/{pid}"
        item = {"id": source["id"], "video_id": pid, "audio_video_downloaded": False}
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json", "Accept-Encoding": "identity", "User-Agent": "Anyanwu-SourceMetadata/1"})
            try:
                response = opener.open(request, timeout=15)
                status = response.status
            except urllib.error.HTTPError as error:
                response, status = error, error.code
            with response:
                length = response.headers.get("Content-Length")
                if length and (not length.isdigit() or int(length) > CAP):
                    raise RuntimeError("Metadata response exceeds 256 KiB; body not fetched.")
                body = response.read(CAP + 1)
                received += len(body)
                if len(body) > CAP:
                    raise RuntimeError("Metadata response exceeded its cap.")
            item["http_status"] = status
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("Unexpected metadata shape")
            item["provider_result"] = data.get("result")
            media = data.get("media", []) if status == 200 and not data.get("result") else []
            item["media"] = [{"kind": m.get("kind"), "encoding": m.get("encoding"),
                              "language": m.get("language"), "bitrate": m.get("bitrate"),
                              "width": m.get("width"), "height": m.get("height"),
                              "connections": [{"host": urllib.parse.urlsplit(c.get("href", "")).hostname,
                                               "format": c.get("transferFormat"), "protocol": c.get("protocol")}
                                              for c in m.get("connection", [])[:10] if isinstance(c, dict)]}
                             for m in media[:30] if isinstance(m, dict)]
        except (OSError, TimeoutError, ValueError, RuntimeError, urllib.error.URLError) as error:
            item["error"] = type(error).__name__ + ": " + (str(error)[:180] if isinstance(error, RuntimeError) else "Metadata unavailable; no media fallback attempted")
        rows.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    result = {"status": "metadata_preflight_complete", "provider": "BBC MediaSelector", "response_body_bytes": received,
              "request_count": len(rows), "response_body_cap_bytes": len(rows)*(CAP+1),
              "audio_video_subtitles_downloaded": False, "sources": rows}
    (FOLDER / "media-preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
