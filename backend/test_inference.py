"""API/transport tests use explicit engines, never model downloads or paid APIs."""
from array import array
import asyncio
import base64
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock
import wave

from fastapi.testclient import TestClient

from backend.gateway import create_app as gateway_app
from backend.http_boundary import HTTPBoundary, model_call
from backend.inference.app import WorkerConfig, create_app
from backend.inference.audio import (decode_audio, audio_chunks, encode_audio, speech_chunks,
                                     MAX_AUDIO_BYTES, SAMPLE_RATE)
from backend.inference.engines import ASR_MODEL, TTS_MODEL, TTS_VOICE
from backend.server import APIError, Config, Providers

TOKEN = "a" * 40
AUTH = {"Authorization": "Bearer " + TOKEN}


class EngineFixture:
    def __init__(self):
        self.loaded = False
        self.calls = []
        self.error = None

    def load(self):
        self.loaded = True

    def transcribe(self, data):
        self.calls.append(("transcribe", data))
        if self.error:
            raise self.error
        return "Hello, kedu ka ị mere?"

    def speak(self, text, output):
        self.calls.append(("speak", text, output))
        if self.error:
            raise self.error
        return b"fixture-mp3", "audio/mpeg"


class WorkerTests(unittest.TestCase):
    def client(self, role="asr", engine=None, **kwargs):
        self.engine = engine or EngineFixture()
        return TestClient(create_app(WorkerConfig(role, TOKEN), engine=self.engine), **kwargs)

    @staticmethod
    def upload(client, **kwargs):
        return client.post("/v1/audio/transcriptions", headers=AUTH,
                           data={"model": ASR_MODEL, "response_format": "json"},
                           files={"file": ("untrusted-name.m4a", b"encoded-audio", "audio/mp4")}, **kwargs)

    def test_importing_api_does_not_load_weights_and_downloads_require_opt_in(self):
        with patch("backend.inference.engines.ASREngine.load") as load:
            app = create_app(WorkerConfig("asr", TOKEN))
            load.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "No weights were downloaded"):
                with TestClient(app):
                    pass
            load.assert_not_called()

    def test_health_tracks_actual_startup_and_reports_unmeasured_quality(self):
        with self.client() as client:
            response = client.get("/readyz", headers=AUTH)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(self.engine.loaded)
            self.assertTrue(response.json()["ready"])
            self.assertFalse(response.json()["qualityEvaluated"])
            self.assertFalse(response.json()["streaming"])
        self.assertEqual(client.get("/readyz", headers=AUTH).status_code, 503)

    def test_unauthorized_upload_is_rejected_before_model_or_body_parsing(self):
        with self.client() as client:
            response = client.post("/v1/audio/transcriptions", content=b"malformed", headers={"Content-Type": "x"})
            self.assertEqual(response.status_code, 401)
            self.assertEqual(self.engine.calls, [])

    def test_multipart_asr_preserves_the_models_actual_unicode_text(self):
        with self.client() as client:
            response = self.upload(client)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"text": "Hello, kedu ka ị mere?"})
            self.assertEqual(self.engine.calls, [("transcribe", b"encoded-audio")])
            self.assertEqual(response.headers["cache-control"], "no-store")

    def test_wrong_model_language_override_and_duplicate_fields_are_rejected(self):
        with self.client() as client:
            for data in ({"model": "other"}, {"model": ASR_MODEL, "language": "ig"},
                         {"model": ASR_MODEL, "response_format": "text"}):
                response = client.post("/v1/audio/transcriptions", headers=AUTH, data=data,
                                       files={"file": ("clip.wav", b"audio", "audio/wav")})
                self.assertEqual(response.status_code, 400)
            response = client.post("/v1/audio/transcriptions", headers=AUTH,
                                   files=[("model", (None, ASR_MODEL)), ("model", (None, ASR_MODEL)),
                                          ("file", ("clip.wav", b"audio", "audio/wav"))])
            self.assertEqual(response.status_code, 400)
            self.assertEqual(self.engine.calls, [])

    def test_upload_limit_and_declared_size_are_checked_before_model_work(self):
        with self.client() as client:
            response = client.post("/v1/audio/transcriptions", headers=AUTH, content=b"x" * (MAX_AUDIO_BYTES + 32_001))
            self.assertEqual(response.status_code, 413)
            response = client.post("/v1/audio/transcriptions", headers={**AUTH, "Content-Length": "7"}, content=b"x")
            self.assertEqual(response.status_code, 400)
            self.assertEqual(self.engine.calls, [])

    def test_worker_roles_are_isolated(self):
        with self.client("tts") as client:
            self.assertEqual(self.upload(client).status_code, 404)
        with self.client("asr") as client:
            self.assertEqual(client.post("/v1/audio/speech", headers=AUTH, json={}).status_code, 404)

    def test_speech_uses_igbo_voice_and_returns_binary_audio(self):
        with self.client("tts") as client:
            response = client.post("/v1/audio/speech", headers=AUTH, json={
                "model": TTS_MODEL, "voice": TTS_VOICE, "input": "Anyanwụ. Ị nọ ebe a?", "response_format": "mp3"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "audio/mpeg")
            self.assertEqual(response.content, b"fixture-mp3")
            self.assertEqual(self.engine.calls[0], ("speak", "Anyanwụ. Ị nọ ebe a?", "mp3"))

    def test_speech_rejects_other_voices_clone_parameters_and_oversized_text(self):
        body = {"model": TTS_MODEL, "voice": TTS_VOICE, "input": "Ndewo"}
        with self.client("tts") as client:
            for extra in ({"voice": "english"}, {"audio_prompt_path": "/untrusted/file"},
                          {"input": "x" * 601}, {"response_format": "pcm"}):
                response = client.post("/v1/audio/speech", headers=AUTH, json={**body, **extra})
                self.assertEqual(response.status_code, 400)
            self.assertEqual(self.engine.calls, [])

    def test_model_failures_are_sanitized_and_slot_is_released(self):
        engine = EngineFixture()
        engine.error = RuntimeError("private recording and secret")
        with self.client(engine=engine, raise_server_exceptions=False) as client:
            response = self.upload(client)
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("private", response.text)
            engine.error = APIError(422, "No audible speech was found.")
            self.assertEqual(self.upload(client).status_code, 422)
            engine.error = None
            self.assertEqual(self.upload(client).status_code, 200)

    def test_busy_worker_rejects_upload_but_health_stays_available(self):
        entered, release = threading.Event(), threading.Event()

        class SlowEngine(EngineFixture):
            def transcribe(self, data):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("fixture timeout")
                return super().transcribe(data)

        with self.client(engine=SlowEngine()) as client:
            result = []
            thread = threading.Thread(target=lambda: result.append(self.upload(client)))
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                busy = self.upload(client)
                self.assertEqual(busy.status_code, 503)
                self.assertEqual(busy.headers["retry-after"], "2")
                self.assertEqual(client.get("/health", headers=AUTH).status_code, 200)
            finally:
                release.set()
                thread.join(5)
            self.assertEqual(result[0].status_code, 200)
            self.assertEqual(len(self.engine.calls), 1)

    def test_invalid_worker_configuration_is_rejected(self):
        for role, token in (("other", TOKEN), ("asr", ""), ("tts", "ị" * 40)):
            with self.assertRaises(ValueError):
                WorkerConfig(role, token)


class AudioTests(unittest.TestCase):
    def test_decode_preserves_duration_and_chunks_below_upstream_limit(self):
        raw = array("f", [0.1] * (SAMPLE_RATE * 60)).tobytes()
        with patch("backend.inference.audio.ffmpeg", return_value=raw):
            chunks = list(audio_chunks(decode_audio(b"encoded")))
        self.assertEqual([len(chunk) // SAMPLE_RATE for chunk in chunks], [25, 25, 10])

    def test_decoder_rejects_overlong_silent_and_nonfinite_samples(self):
        for values in (array("f", [0.1] * (SAMPLE_RATE * 61)), array("f", [0] * 100), array("f", [float("nan")])):
            with self.subTest(length=len(values)), patch("backend.inference.audio.ffmpeg", return_value=values.tobytes()), self.assertRaises(APIError):
                decode_audio(b"encoded")

    def test_tts_chunks_preserve_dotted_letters_and_stacked_tones(self):
        text = "Ndewo. " + "ị\u0301 nọ ebe a? " * 20
        chunks = speech_chunks(text)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertIn("ị\u0301", " ".join(chunks))
        self.assertEqual(" ".join(chunks), text.strip())

    def test_output_is_valid_bounded_pcm_wav(self):
        data, mime = encode_audio(array("f", [-2, -0.1, 0.1, 2]), 24_000, "wav")
        self.assertEqual(mime, "audio/wav")
        with wave.open(io.BytesIO(data), "rb") as audio:
            self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()), (24_000, 1, 2, 4))
        with self.assertRaises(APIError):
            encode_audio(array("f", [float("inf")]), 24_000, "wav")

    def test_ffmpeg_uses_no_shell_and_cannot_follow_external_protocols(self):
        from backend.inference.audio import ffmpeg
        result = Mock(returncode=0, stdout=b"decoded")
        with patch("backend.inference.audio.subprocess.run", return_value=result) as run:
            ffmpeg(b"upload", ["-t", "61", "-f", "f32le"])
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-protocol_whitelist") + 1], "pipe")
        self.assertEqual(run.call_args.kwargs["timeout"], 20)
        self.assertNotIn("shell", run.call_args.kwargs)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for the real media check")
    def test_real_wav_mp3_and_iphone_m4a_decode_and_leave_no_temporary_input(self):
        Path(".build").mkdir(exist_ok=True)
        samples = array("f", (0.1 * math.sin(i * 2 * math.pi * 440 / 24_000) for i in range(24_000)))
        wav, _ = encode_audio(samples, 24_000, "wav")
        mp3, _ = encode_audio(samples, 24_000, "mp3")
        with tempfile.TemporaryDirectory(dir=".build", prefix="media-check-") as directory:
            path = Path(directory) / "iphone.m4a"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-c:a", "aac", str(path)],
                           input=wav, check=True, timeout=10)
            m4a = path.read_bytes()
            path.unlink()
            with patch.dict(os.environ, {"ANYANWU_AUDIO_TMP": directory}):
                for mime, encoded in (("wav", wav), ("mp3", mp3), ("m4a", m4a)):
                    with self.subTest(format=mime):
                        decoded = decode_audio(encoded)
                        self.assertTrue(15_000 <= len(decoded) <= 18_000)
                        self.assertEqual(list(Path(directory).iterdir()), [])
                with self.assertRaises(APIError):
                    decode_audio(b"#EXTM3U\n#EXT-X-TARGETDURATION:10\nmissing.wav\n")
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_aac_padding_does_not_reject_a_valid_maximum_length_recording(self):
        raw = array("f", [0.1] * (SAMPLE_RATE * 60 + 200)).tobytes()
        with patch("backend.inference.audio.ffmpeg", return_value=raw):
            self.assertEqual(len(decode_audio(b"encoded")), SAMPLE_RATE * 60)


class BoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunked_upload_is_capped_without_content_length(self):
        called = False

        async def downstream(scope, receive, send):
            nonlocal called
            called = True

        app = HTTPBoundary(downstream, token="", max_body=4, concurrency=1)
        chunks = iter([{"type": "http.request", "body": b"123", "more_body": True},
                       {"type": "http.request", "body": b"45", "more_body": False}])
        sent = []

        async def receive():
            return next(chunks)

        async def send(message):
            sent.append(message)

        await app({"type": "http", "method": "POST", "headers": []}, receive, send)
        self.assertFalse(called)
        self.assertEqual(sent[0]["status"], 413)
        self.assertEqual(app.active, 0)

    async def test_cancellation_waits_for_native_work_before_releasing_its_waiter(self):
        entered, release = threading.Event(), threading.Event()

        def native():
            entered.set()
            release.wait(5)
            return "done"

        task = asyncio.create_task(model_call(native))
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.01)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task


class GatewayTests(unittest.TestCase):
    def test_cloud_gateway_preserves_existing_voice_contract(self):
        engine = EngineFixture()

        class FixtureProviders:
            def transcribe(self, audio, mime):
                return engine.transcribe(audio)

            def speak(self, text):
                return engine.speak(text, "mp3")

        config = Config(access_token=TOKEN, asr_url="http://127.0.0.1:8001/v1/audio/transcriptions", asr_model=ASR_MODEL,
                        tts_url="http://127.0.0.1:8002/v1/audio/speech", tts_model=TTS_MODEL, tts_voice=TTS_VOICE)
        with TestClient(gateway_app(config, providers=FixtureProviders())) as client:
            response = client.post("/v1/turns", headers=AUTH, json={
                "interfaceLanguage": "en", "inputLanguages": ["ig", "en"], "responseLanguage": "ig",
                "input": {"type": "audio", "audioBase64": base64.b64encode(b"audio").decode(), "mimeType": "audio/mp4"}})
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(result["transcript"], "Hello, kedu ka ị mere?")
            self.assertEqual(result["responseLanguage"], "ig")
            self.assertEqual(result["mode"], "preview")
            self.assertIn("prepared Igbo reply", result["notice"])
            self.assertEqual(base64.b64decode(result["audioBase64"]), b"fixture-mp3")
            self.assertEqual(client.get("/health").status_code, 401)

    def test_existing_compatible_adapter_interoperates_with_both_worker_apis(self):
        # Run compatible request construction against both ASGI APIs in memory.
        # Only HTTP transport and the model engines are replaced here.
        clients = {role: self.enterContext(TestClient(create_app(WorkerConfig(role, TOKEN), engine=EngineFixture())))
                   for role in ("asr", "tts")}

        class ASGIProviders(Providers):
            def request(self, url, payload, key, content_type):
                role, path = ("asr", "/v1/audio/transcriptions") if "transcriptions" in url else ("tts", "/v1/audio/speech")
                response = clients[role].post(path, content=payload, headers={"Content-Type": content_type, "Authorization": "Bearer " + key})
                if response.status_code != 200:
                    raise AssertionError(response.text)
                return response.content, response.headers["content-type"].split(";", 1)[0]

        provider = ASGIProviders(Config(asr_url="http://localhost/v1/audio/transcriptions", asr_model=ASR_MODEL, asr_key=TOKEN,
                                        tts_url="http://localhost/v1/audio/speech", tts_model=TTS_MODEL, tts_key=TOKEN, tts_voice=TTS_VOICE))
        self.assertIn("ị", provider.transcribe(b"audio", "audio/mp4"))
        self.assertEqual(provider.speak("Ndewo!"), (b"fixture-mp3", "audio/mpeg"))


if __name__ == "__main__":
    unittest.main()
