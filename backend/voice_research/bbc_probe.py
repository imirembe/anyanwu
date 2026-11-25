"""Cloud-only, bounded public BBC stream-manifest audit; no media segments."""
from pathlib import Path
import importlib.util
import json
import platform
import re
import time
import urllib.error
import urllib.parse
import urllib.request

SOURCES = (
    ("bbc-igbo-01", "p0plm33d"), ("bbc-igbo-02", "p0pl8yy9"),
    ("bbc-igbo-03", "p0nzt0g1"), ("bbc-igbo-04", "p0pn8my1"),
    ("bbc-igbo-05", "p0pm8gf1"),
)
HOSTS = frozenset({"open.live.bbc.co.uk", "vod-hls-uk-live.akamaized.net",
    "vod-hls-uk.live.cf.md.bbci.co.uk", "mm.bidi.bbc.co.uk",
    "vod-dash-uk-live.akamaized.net", "vod-dash-uk.live.cf.md.bbci.co.uk",
    "vod-hls-ww-live.akamaized.net", "vod-hls-ww.live.cf.md.bbci.co.uk"})


def cloud_guard():
    if platform.system() != "Linux" or not Path("/content").is_dir() or importlib.util.find_spec("google.colab") is None:
        raise RuntimeError("Use a Google Colab hosted runtime; media work is forbidden on the laptop.")


def checked_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise RuntimeError("Unapproved BBC media host or non-HTTPS URL; stopped.")
    return url


class Transport:
    def __init__(self, budget=2*1024**2, requests=20, seconds=120):
        self.budget, self.request_cap = budget, requests
        self.received, self.requests = 0, 0
        self.deadline = time.monotonic() + seconds
        owner = self

        class Redirects(urllib.request.HTTPRedirectHandler):
            def http_error_302(self, req, fp, code, msg, headers):
                target = headers.get("Location")
                if not target:
                    fp.close()
                    raise RuntimeError("BBC redirect has no destination.")
                newurl = urllib.parse.urljoin(req.full_url, target)
                try:
                    checked_url(newurl)
                finally:
                    fp.close()
                owner.redirects += 1
                owner.requests += 1
                if owner.redirects > 3 or owner.requests > owner.request_cap:
                    fp.close()
                    raise RuntimeError("BBC request/redirect cap reached.")
                # Do not let urllib consume an unbounded redirect response body.
                fp.close()
                follow = urllib.request.Request(newurl, headers={"User-Agent":"Anyanwu-ResearchPilot/1", "Accept-Encoding":"identity"})
                return self.parent.open(follow, timeout=min(15,max(1,owner.deadline-time.monotonic())))

            http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirects())

    def get(self, url, cap=256*1024):
        checked_url(url)
        if self.requests >= self.request_cap or self.received >= self.budget or time.monotonic() >= self.deadline:
            raise RuntimeError("BBC audit request/body/time budget reached.")
        self.requests += 1
        self.redirects = 0
        allowance = min(cap, self.budget-self.received)
        request = urllib.request.Request(url, headers={"User-Agent": "Anyanwu-ResearchPilot/1", "Accept-Encoding": "identity"})
        try:
            response = self.opener.open(request, timeout=min(15, max(1, self.deadline-time.monotonic())))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"BBC returned HTTP {error.code}; no geo, authentication or DRM bypass attempted.") from None
        with response:
            checked_url(response.url)
            length = response.headers.get("Content-Length")
            if length and (not length.isdigit() or int(length) > allowance):
                raise RuntimeError("BBC response exceeds declared body cap; body not read.")
            body = response.read(allowance+1)
            self.received += len(body)
            if len(body) > allowance:
                raise RuntimeError("BBC response exceeded body cap.")
        return body


def attributes(line):
    return dict(re.findall(r'([A-Z0-9-]+)=("[^"]*"|[^,]*)', line.split(":", 1)[1]))


def probe():
    cloud_guard()
    transport = Transport()
    rows, manifests = [], {}
    for source_id, version in SOURCES:
        row = {"id": source_id, "video_id": version, "media_downloaded": False}
        try:
            metadata = json.loads(transport.get(f"https://open.live.bbc.co.uk/mediaselector/6/select/version/2.0/mediaset/pc/vpid/{version}"))
            if metadata.get("result"):
                raise RuntimeError("BBC metadata returned " + str(metadata["result"])[:100])
            media = metadata.get("media", [])
            row["media_kinds"] = [m.get("kind") for m in media]
            urls = [c["href"] for m in media for c in m.get("connection", []) if c.get("transferFormat") == "hls" and c.get("href", "").startswith("https://")]
            row["available_hls_hosts"] = sorted({urllib.parse.urlsplit(u).hostname for u in urls})
            if not urls:
                raise RuntimeError("No public HTTPS HLS stream exposed.")
            url = checked_url(urls[0])
            master = transport.get(url).decode("utf-8-sig")
            if not master.startswith("#EXTM3U"):
                raise RuntimeError("Unexpected HLS manifest.")
            lines = master.splitlines()
            row["renditions"] = [attributes(line) for line in lines if line.startswith("#EXT-X-MEDIA:")]
            for rendition in row["renditions"]:
                rendition.pop("URI", None)
            row["variants"] = [attributes(line) for line in lines if line.startswith("#EXT-X-STREAM-INF:")]
            row["session_key_methods"] = [attributes(line).get("METHOD") for line in lines if line.startswith("#EXT-X-SESSION-KEY:")]
            row["manifest_tags"] = sorted({line.split(":", 1)[0] for line in lines if line.startswith("#EXT")})
            manifests[source_id] = {"master_url": url, "master": master}
            row["status"] = "master_manifest_checked"
        except Exception as error:
            # Never expose signed URLs or transport tracebacks in a saved report.
            row["status"] = "unavailable"
            row["error"] = str(error)[:200] if isinstance(error, RuntimeError) else type(error).__name__
        rows.append(row)
    report = {"stage": "bbc_cloud_manifest_probe", "sources": rows, "body_bytes": transport.received,
              "requests": transport.requests, "media_segments_downloaded": False}
    folder = Path("/content/anyanwu-bbc-igbo-pilot-v1")
    folder.mkdir(exist_ok=True)
    (folder/"manifest-probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report, manifests


if __name__ == "__main__":
    report, BBC_MANIFESTS = probe()
    print(json.dumps(report, ensure_ascii=False, indent=2))
