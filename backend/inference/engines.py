"""Actual model adapters. Heavy libraries are imported only at cloud startup."""
from __future__ import annotations

from array import array
from pathlib import Path
import unicodedata

from backend.server import APIError
from backend.inference.audio import audio_chunks, decode_audio, encode_audio, speech_chunks, MAX_SECONDS

ASR_MODEL = "omniASR_CTC_300M_v2"
TTS_MODEL = "PlotweaverAI/naija-tts"
TTS_REVISION = "6d3576c7f68143e10337ff6aab64e4b48f6c025f"
BASE_TTS_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
TTS_VOICE = "igbo-default"


class ASREngine:
    def load(self):
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        if not torch.cuda.is_available():
            raise RuntimeError("This cloud image requires an NVIDIA GPU; no automatic CPU fallback.")
        self.torch = torch
        self.pipeline = ASRInferencePipeline(model_card=ASR_MODEL, device="cuda", dtype=torch.float32)

    def transcribe(self, data: bytes) -> str:
        samples = decode_audio(data)
        texts = []
        with self.torch.inference_mode():
            # CTC is unconditioned: do not force English/mixed speech into an
            # Igbo-only decoder. Code-switching quality is still unmeasured.
            for chunk in audio_chunks(samples):
                result = self.pipeline.transcribe([
                    {"waveform": self.torch.tensor(chunk, dtype=self.torch.float32), "sample_rate": 16_000}
                ], batch_size=1)
                if len(result) != 1 or not isinstance(result[0], str):
                    raise APIError(502, "The transcription model returned invalid text.")
                texts.append(result[0].strip())
        text = unicodedata.normalize("NFC", " ".join(filter(None, texts)))
        if not text or len(text) > 8_000:
            raise APIError(422, "No clear transcript was returned. Please speak again.")
        return text


class TTSEngine:
    def load(self):
        import torch
        import chatterbox.mtl_tts as multilingual
        from huggingface_hub import hf_hub_download, snapshot_download

        if not torch.cuda.is_available():
            raise RuntimeError("This cloud image requires an NVIDIA GPU; no automatic CPU fallback.")
        self.torch = torch
        # Pin both the fine-tune and its auxiliary mapping. Do not download
        # training corpora, repository cache debris, or execute remote code.
        directory = Path(snapshot_download(repo_id=TTS_MODEL, revision=TTS_REVISION, allow_patterns=[
            "README.md", "ve.pt", "s3gen.pt", "conds.pt", "t3_mtl23ls_v2.safetensors",
            "grapheme_mtl_merged_expanded_v1.json",
        ]))
        mapping = hf_hub_download("ResembleAI/chatterbox", "Cangjie5_TC.json", revision=BASE_TTS_REVISION)
        target = directory / "Cangjie5_TC.json"
        if not target.exists():
            target.symlink_to(Path(mapping).resolve())
        multilingual.SUPPORTED_LANGUAGES["ig"] = "Igbo"
        self.model = multilingual.ChatterboxMultilingualTTS.from_local(
            str(directory), device="cuda")
        if self.model.conds is None:
            raise RuntimeError("The selected checkpoint has no built-in voice.")
        # Fail startup rather than silently losing dotted letters/tone marks.
        for sample in ("Anyanwụ", "ị ọ ụ ṅ", "à á ā ì í ù ú"):
            tokens = self.model.tokenizer.text_to_tokens(sample, language_id="ig")
            if (tokens == 1).any().item():  # The published tokenizer's [UNK] ID.
                raise RuntimeError("The TTS tokenizer cannot represent the Igbo validation sample.")

    def speak(self, text: str, response_format: str) -> tuple[bytes, str]:
        samples = array("f")
        with self.torch.inference_mode():
            for chunk in speech_chunks(text):
                wave = self.model.generate(chunk, language_id="ig")
                wave = wave.detach().cpu().float().reshape(-1)
                if wave.numel() + len(samples) > self.model.sr * MAX_SECONDS:
                    raise APIError(502, "The spoken reply exceeded 60 seconds. Please use a shorter reply.")
                samples.extend(wave.tolist())
        return encode_audio(samples, self.model.sr, response_format)
