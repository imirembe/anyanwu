"""In-memory media conversion with bounded duration, output and runtime."""
from __future__ import annotations

from array import array
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unicodedata
import wave

from backend.server import APIError

SAMPLE_RATE = 16_000
MAX_SECONDS = 60
CHUNK_SECONDS = 25  # Under the upstream ASR pipeline's 40-second limit.
MAX_AUDIO_BYTES = 6_000_000
MAX_TTS_CHARACTERS = 600


def ffmpeg(data: bytes, arguments: list[str], *, seekable: bool = False) -> bytes:
    # AVAudioRecorder's M4A can put metadata after its samples. FFmpeg needs a
    # seekable input for that layout. Cloud /tmp is a bounded RAM filesystem;
    # TemporaryFile is unlinked automatically, including on decode failure.
    source = None
    extra = {}
    protocols, input_name = "pipe", "pipe:0"
    if seekable:
        directory = Path(os.environ.get("ANYANWU_AUDIO_TMP", ".build/audio-tmp"))
        directory.mkdir(parents=True, exist_ok=True)
        source = tempfile.TemporaryFile(dir=directory)
        source.write(data)
        source.seek(0)
        descriptor = source.fileno()
        prefix = "/proc/self/fd/" if sys.platform == "linux" else "/dev/fd/"
        input_name = "file:" + prefix + str(descriptor)
        protocols = "file,pipe"
        extra["pass_fds"] = (descriptor,)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
               "-protocol_whitelist", protocols]
    if seekable:
        # In particular, never interpret uploaded text as an HLS/concat playlist.
        command += ["-format_whitelist", "mov,mp3,wav,matroska,webm"]
    command += ["-i", input_name, *arguments, "pipe:1"]
    try:
        result = subprocess.run(command, input=b"" if seekable else data, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=20, check=False, **extra)
    except FileNotFoundError:
        raise APIError(503, "The audio decoder is unavailable.") from None
    except subprocess.TimeoutExpired:
        raise APIError(422, "The audio could not be decoded in time.") from None
    finally:
        if source is not None:
            source.close()
    if result.returncode or not result.stdout:
        raise APIError(422, "The recording is not valid, decodable audio.")
    return result.stdout


def decode_audio(data: bytes) -> array:
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise APIError(413, "Audio must be nonempty and under 6 MB.")
    # Decode a little beyond the limit to detect long files instead of silently
    # truncating them. Raw output is bounded to ~3.9 MB by format/rate/duration.
    raw = ffmpeg(data, ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-t", "61", "-f", "f32le"], seekable=True)
    # AAC adds codec padding (including to a valid 60-second iPhone clip).
    # Permit at most 250 ms of padding, then retain only the 60-second budget.
    if len(raw) % 4 or len(raw) > int(SAMPLE_RATE * (MAX_SECONDS + 0.25)) * 4:
        raise APIError(422, "Keep recordings at or below 60 seconds.")
    samples = array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    del samples[SAMPLE_RATE * MAX_SECONDS:]
    if not samples or not all(math.isfinite(value) for value in samples):
        raise APIError(422, "The recording contains invalid samples.")
    if max(abs(value) for value in samples) < 0.00001:
        raise APIError(422, "No audible speech was found. Please speak again.")
    return samples


def audio_chunks(samples: array):
    step = SAMPLE_RATE * CHUNK_SECONDS
    for offset in range(0, len(samples), step):
        yield samples[offset:offset + step]


def speech_chunks(text: str, limit: int = 120) -> list[str]:
    text = unicodedata.normalize("NFC", text).strip()
    if not text or len(text) > MAX_TTS_CHARACTERS:
        raise APIError(400, "Speech text must contain between 1 and 600 characters.")
    # Word boundaries preserve combining marks; never cut a code-point slice.
    chunks, current = [], ""
    for word in text.split():
        if len(word) > limit:
            raise APIError(400, "A word is too long to speak.")
        proposed = (current + " " + word).strip()
        if len(proposed) > limit:
            chunks.append(current)
            current = word
        else:
            current = proposed
        if current.endswith((".", "!", "?", ";")):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def encode_audio(samples: array, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    if not 8_000 <= sample_rate <= 48_000 or not samples or len(samples) > MAX_SECONDS * sample_rate:
        raise APIError(502, "The speech model returned invalid or oversized audio.")
    if not all(math.isfinite(value) for value in samples):
        raise APIError(502, "The speech model returned invalid audio.")
    pcm = array("h", (round(max(-1, min(1, value)) * 32767) for value in samples))
    if sys.byteorder != "little":
        pcm.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())
    data = output.getvalue()
    if response_format == "wav":
        return data, "audio/wav"
    if response_format != "mp3":
        raise APIError(400, "Choose mp3 or wav for speech output.")
    return ffmpeg(data, ["-vn", "-t", "60", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"]), "audio/mpeg"
