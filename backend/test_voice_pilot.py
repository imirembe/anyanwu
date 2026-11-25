"""Bounded acquisition, rollback, integrity and local review boundary checks."""
import hashlib
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from backend.voice_research import pilot
from backend.voice_research.review import ReviewStore, make_server


REVISION = "a" * 40
AUDIO = b"ID3" + b"audio-fixture" * 20
SUMMARY = {"hub_revision": REVISION, "license_declared": "CC-BY-SA-4.0"}


def temporary_directory():
    root = pilot.PILOT_ROOT / "tests"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def rows():
    return [{"row_idx": n, "truncated_cells": [], "row": {
        "id": f"ibo_{n}", "speaker_id": "1", "locale": "ibo",
        "text": "Ndewo. Ụtụtụ ọma. Ị chọrọ ka anyị kwurịta okwu taa? " + str(n),
        "audio": [{"type": "audio/mpeg", "src": f"https://datasets-server.huggingface.co/cached-assets/google/WaxalNLP/--/{REVISION}/--/ibo_tts/train/{n}/audio/audio.mp3?Expires=123"}]}}
        for n in range(8)]


class FakeClient:
    received = 0
    metadata_received = 0
    requests = 0

    def get(self, *_args, **_kwargs):
        self.received += len(AUDIO)
        self.requests += 1
        return AUDIO, {}

    def json(self, *_args, **_kwargs):
        self.requests += 1
        return {"sha": REVISION}


def fixture(root):
    with patch.object(pilot, "ensure_space"), patch.object(pilot, "discover", return_value=(SUMMARY, rows())), \
            patch.object(pilot, "audio_duration", return_value=5.5), patch("sys.stdout", io.StringIO()):
        return pilot.prepare(root, FakeClient())


class AcquisitionTests(unittest.TestCase):
    def test_allowlist_refuses_other_hosts_archives_and_bad_ports(self):
        for url in ("http://huggingface.co/x", "https://evil.example/audio.mp3",
                    "https://huggingface.co/x.parquet?download=1", "https://huggingface.co:bad/a",
                    "https://user:secret@huggingface.co/x", "https://huggingface.co/x#fragment"):
            with self.subTest(url=url), self.assertRaises(pilot.PilotError):
                pilot.checked_url(url)

    def test_declared_oversized_response_is_not_read(self):
        client = pilot.Client(100)
        response = Mock()
        response.headers = {"Content-Length": "101"}
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        client.opener.open = Mock(return_value=response)
        with self.assertRaises(pilot.PilotError):
            client.get("https://huggingface.co/api/datasets/x", limit=100)
        response.read.assert_not_called()
        self.assertEqual(client.received, 0)

    def test_chunked_body_and_shared_budget_stop_overflow(self):
        client = pilot.Client(100)
        class Response(io.BytesIO):
            headers = {}
        client.opener.open = Mock(side_effect=lambda *_a, **_k: Response(b"x" * 101))
        with self.assertRaises(pilot.PilotError):
            client.get("https://huggingface.co/api/datasets/x", limit=100)
        self.assertEqual(client.received, 100)
        with self.assertRaises(pilot.PilotError):
            client.get("https://huggingface.co/api/datasets/x", limit=100)
        self.assertEqual(client.opener.open.call_count, 1)

    def test_source_selection_preserves_unicode_and_excludes_truncated_text(self):
        candidates = rows()
        candidates[0]["truncated_cells"] = ["text"]
        candidates[1]["row"]["locale"] = "yor"
        result = pilot.eligible_rows(candidates)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0]["transcript"], candidates[2]["row"]["text"])
        candidates[2]["row"]["audio"][0]["src"] = "https://cdn-assets.huggingface.co/other/audio.mp3"
        with self.assertRaises(pilot.PilotError):
            pilot.eligible_rows(candidates)

    def test_success_is_hash_verified_and_reuses_without_network(self):
        with temporary_directory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            self.assertEqual(len(manifest["clips"]), 6)
            self.assertFalse(manifest["training_ready"])
            self.assertEqual(manifest["speaker_count"], 1)
            self.assertNotIn("Expires", json.dumps(manifest))
            with patch.object(pilot, "Client", side_effect=AssertionError("Network on reuse")), patch("sys.stdout", io.StringIO()):
                pilot.prepare(root)
            folder, _ = pilot.load_pilot(root)
            (folder / "clip-01.mp3").write_bytes(b"changed")
            with self.assertRaises(pilot.PilotError):
                pilot.load_pilot(root)

    def test_failed_download_leaves_receipt_without_partial_audio(self):
        with temporary_directory() as directory:
            root = Path(directory)
            client = FakeClient()
            with patch.object(pilot, "ensure_space"), patch.object(pilot, "discover", return_value=(SUMMARY, rows())), \
                    patch.object(pilot, "audio_duration", side_effect=pilot.PilotError("Invalid clip")):
                with self.assertRaises(pilot.PilotError):
                    pilot.prepare(root, client)
            self.assertFalse((root / "waxal-ibo-v1").exists())
            self.assertEqual(list(root.glob(".staging-*")), [])
            self.assertEqual(list(root.rglob("*.mp3")), [])
            self.assertEqual(json.loads((root / "last-attempt.json").read_text())["status"], "failed")

    def test_revision_change_prevents_installation(self):
        with temporary_directory() as directory:
            root = Path(directory)
            client = FakeClient()
            client.json = lambda *_a, **_k: {"sha": "b" * 40}
            with patch.object(pilot, "ensure_space"), patch.object(pilot, "discover", return_value=(SUMMARY, rows())), \
                    patch.object(pilot, "audio_duration", return_value=5.5), patch("sys.stdout", io.StringIO()):
                with self.assertRaises(pilot.PilotError):
                    pilot.prepare(root, client)
            self.assertFalse((root / "waxal-ibo-v1").exists())

    def test_disk_reserve_checked_before_network(self):
        with temporary_directory() as directory:
            with patch.object(pilot.shutil, "disk_usage", return_value=Mock(free=pilot.RESERVE)), \
                    patch.object(pilot, "Client", side_effect=AssertionError("Network before disk check")):
                with self.assertRaises(pilot.PilotError):
                    pilot.prepare(Path(directory))


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)
        self.folder = self.root / "waxal-ibo-v1"
        self.entry = {"clip_id": "clip-01", "audio_sha256": hashlib.sha256(AUDIO).hexdigest(),
                      "intelligibility": 4, "cadence": 3, "pronunciation": None,
                      "transcript_match": "unsure", "notes": "Ụda na ụdaolu"}

    def tearDown(self):
        self.temp.cleanup()

    def test_reviews_belong_to_exact_audio_and_preserve_other_entries(self):
        store = ReviewStore(self.folder, self.manifest)
        store.save(self.entry)
        store.save(self.entry | {"clip_id": "clip-02"})
        fresh = ReviewStore(self.folder, self.manifest)
        self.assertEqual(len(fresh.reviews), 2)
        self.assertEqual(fresh.reviews["clip-01"]["notes"], "Ụda na ụdaolu")
        for change in ({"audio_sha256": "wrong"}, {"clip_id": "../anything"},
                       {"cadence": True}, {"notes": "a" * 501}, {"transcript_match": "maybe"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                store.save(self.entry | change)

    def test_corrupt_review_file_is_preserved(self):
        path = self.folder / "reviews.json"
        path.write_text("not json")
        with self.assertRaises(pilot.PilotError):
            ReviewStore(self.folder, self.manifest)
        self.assertEqual(path.read_text(), "not json")

    def test_http_routes_ranges_origin_nonce_and_persistence(self):
        server = make_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_port
        def request(method, path, body=None, headers=None):
            connection = HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            result = response.status, dict(response.getheaders()), response.read()
            connection.close()
            return result
        try:
            status, headers, body = request("GET", "/")
            self.assertEqual(status, 200)
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            nonce = re.search(rb'name="review-nonce" content="([^"]+)"', body)[1].decode()
            self.assertEqual(request("GET", "/", headers={"Host": "evil.example"})[0], 403)
            self.assertEqual(request("GET", "/", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
            self.assertEqual(request("GET", "/../../AGENTS.md")[0], 404)
            self.assertEqual(request("GET", "/audio/clip-01?path=../../x")[0], 404)
            status, headers, data = request("GET", "/audio/clip-01", headers={"Range": "bytes=0-9"})
            self.assertEqual((status, data), (206, AUDIO[:10]))
            self.assertEqual(headers["Content-Range"], f"bytes 0-9/{len(AUDIO)}")
            self.assertEqual(request("GET", "/audio/clip-01", headers={"Range": "bytes=9999-"})[0], 416)
            body = json.dumps(self.entry, ensure_ascii=False).encode()
            headers = {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}", "X-Review-Nonce": nonce}
            self.assertEqual(request("POST", "/api/review", body, headers | {"X-Review-Nonce": "wrong"})[0], 403)
            self.assertEqual(request("POST", "/api/review", body, headers | {"Origin": "https://evil.example"})[0], 403)
            self.assertEqual(request("POST", "/api/review", body, headers)[0], 200)
            self.assertEqual(json.loads(request("GET", "/api/pilot")[2])["reviews"]["clip-01"]["notes"], self.entry["notes"])
            status, headers, data = request("HEAD", "/audio/clip-01")
            self.assertEqual((status, data, int(headers["Content-Length"])), (200, b"", len(AUDIO)))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
