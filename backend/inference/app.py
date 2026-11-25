"""Private ASGI worker with compatible ASR/TTS endpoints and no audio history."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os

from fastapi import FastAPI, Request
from starlette.datastructures import UploadFile
from starlette.responses import JSONResponse, Response

from backend.http_boundary import HTTPBoundary, model_call
from backend.inference.audio import MAX_AUDIO_BYTES, MAX_TTS_CHARACTERS
from backend.inference.engines import ASR_MODEL, TTS_MODEL, TTS_REVISION, TTS_VOICE, ASREngine, TTSEngine
from backend.server import APIError

log = logging.getLogger("anyanwu.inference")


@dataclass(frozen=True)
class WorkerConfig:
    role: str
    token: str
    allow_downloads: bool = False

    def __post_init__(self):
        if self.role not in {"asr", "tts"}:
            raise ValueError("Choose ANYANWU_WORKER_ROLE=asr or tts.")
        if len(self.token) < 32 or not self.token.isascii():
            raise ValueError("Set an ASCII ANYANWU_WORKER_TOKEN of at least 32 characters.")

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get("ANYANWU_WORKER_ROLE", ""),
                   os.environ.get("ANYANWU_WORKER_TOKEN", ""),
                   os.environ.get("ANYANWU_ALLOW_MODEL_DOWNLOAD", "0") == "1")


def create_app(config: WorkerConfig | None = None, *, engine=None):
    config = config or WorkerConfig.from_environment()
    supplied_engine = engine is not None
    engine = engine if supplied_engine else (ASREngine() if config.role == "asr" else TTSEngine())
    ready = False

    @asynccontextmanager
    async def lifespan(app):
        nonlocal ready
        if not supplied_engine and not config.allow_downloads:
            raise RuntimeError("Run this worker in the cloud with ANYANWU_ALLOW_MODEL_DOWNLOAD=1. No weights were downloaded.")
        await model_call(engine.load)
        ready = True
        try:
            yield
        finally:
            ready = False

    api = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @api.exception_handler(APIError)
    async def known_error(request, error):
        return JSONResponse({"error": str(error)}, status_code=error.status)

    @api.exception_handler(Exception)
    async def unexpected_error(request, error):
        # Never log a request body, transcript, token, or traceback containing it.
        log.error("Inference failed: %s", type(error).__name__)
        return JSONResponse({"error": "The speech model could not complete this request."}, status_code=500)

    @api.get("/health")
    @api.get("/readyz")
    async def health():
        return JSONResponse({"service": "anyanwu-" + config.role, "ready": ready,
                             "model": ASR_MODEL if config.role == "asr" else TTS_MODEL,
                             "revision": "v2 / omnilingual-asr 0.2.0" if config.role == "asr" else TTS_REVISION,
                             "streaming": False, "qualityEvaluated": False}, status_code=200 if ready else 503)

    @api.post("/v1/audio/transcriptions")
    async def transcribe(request: Request):
        if config.role != "asr":
            raise APIError(404, "Transcription is not available on this worker.")
        if not ready:
            raise APIError(503, "The model is not ready.")
        if not request.headers.get("content-type", "").startswith("multipart/form-data"):
            raise APIError(415, "Use multipart/form-data.")
        async with request.form(max_files=1, max_fields=3, max_part_size=16_000) as form:
            if set(form) - {"file", "model", "response_format"} or len(form.multi_items()) != len(form):
                raise APIError(400, "Unexpected or duplicate transcription fields.")
            if form.get("model") != ASR_MODEL or form.get("response_format", "json") != "json":
                raise APIError(400, "Choose the configured ASR model and JSON output.")
            upload = form.get("file")
            if not isinstance(upload, UploadFile) or upload.content_type not in {"audio/mp4", "audio/mpeg", "audio/wav", "audio/webm"}:
                raise APIError(415, "Upload a supported audio file.")
            data = await upload.read(MAX_AUDIO_BYTES + 1)
            if not data or len(data) > MAX_AUDIO_BYTES:
                raise APIError(413, "Audio must be nonempty and under 6 MB.")
            text = await model_call(engine.transcribe, data)
        return {"text": text}

    @api.post("/v1/audio/speech")
    async def speech(request: Request):
        if config.role != "tts":
            raise APIError(404, "Speech synthesis is not available on this worker.")
        if not ready:
            raise APIError(503, "The model is not ready.")
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise APIError(415, "Use application/json.")
        try:
            payload = await request.json()
        except ValueError:
            raise APIError(400, "Invalid JSON.") from None
        if not isinstance(payload, dict) or set(payload) - {"input", "model", "voice", "response_format"}:
            raise APIError(400, "Unexpected speech fields.")
        if payload.get("model") != TTS_MODEL or payload.get("voice") != TTS_VOICE:
            raise APIError(400, "Choose the configured Igbo model and built-in voice.")
        text = payload.get("input")
        output = payload.get("response_format", "mp3")
        if not isinstance(text, str) or not 0 < len(text.strip()) <= MAX_TTS_CHARACTERS:
            raise APIError(400, "Speech text must contain between 1 and 600 characters.")
        if output not in ("mp3", "wav"):
            raise APIError(400, "Choose mp3 or wav for speech output.")
        audio, mime = await model_call(engine.speak, text, output)
        if not audio or len(audio) > MAX_AUDIO_BYTES or mime not in {"audio/wav", "audio/mpeg"}:
            raise APIError(502, "The model returned unsupported or oversized audio.")
        return Response(audio, media_type=mime)

    return HTTPBoundary(api, token=config.token, max_body=MAX_AUDIO_BYTES + 32_000, concurrency=1)
