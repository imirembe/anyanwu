"""Experimental Owerri pronunciation profiles, independent of training data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

DEFAULT_PROFILE = "owerri-marin"
VOICE_DIRECTION = (
    "Pronunciation target: Owere-Igbo as spoken in the Owerri area of Imo State, Nigeria. "
    "Use natural Igbo rhythm and lexical tones, clear vowel contrasts including ị, ọ and ụ, "
    "and relaxed conversational phrasing at a moderate, unhurried pace. Preserve Igbo names. "
    "Keep the sound understated and natural, without an exaggerated accent or a character impression. "
    "Do not invent dialect vocabulary or force a phonetic spelling you are unsure about. "
    "This is a pronunciation target, not a claim of native-speaker identity or verified dialect accuracy."
)
VOICE_DESIGN = (
    "An adult voice speaking the Igbo language, with Owerri/Owere-area pronunciation and intonation "
    "from Imo State, Nigeria as the target. A warm, clear mid-range voice with a natural, lightly "
    "textured timbre and an unhurried conversational pace. Preserve Igbo lexical tones and distinct "
    "vowels ị, ọ and ụ. Sound like someone in an ordinary conversation, with gentle breaths and "
    "complete word endings. Avoid an announcer style, exaggerated accent, caricature or imitation "
    "of a real person."
)
# Prepared UI copy only, not dialect-certified speech or a training/evaluation corpus.
COMPARISON_TEXT = (
    "Ndewo. Aha m bụ Anyanwụ. Kedu ka ị mere taa? Daalụ maka ịbịa. "
    "Biko, gwa m ihe ị chọrọ ka m nyere gị aka. Anyị nwere ike ikwu banyere njem, "
    "ihe nkiri, ma ọ bụ ihe ọzọ."
)
COMPARISON_INSTRUCTION = (
    "For this one voice comparison, immediately read the following Igbo passage exactly once, "
    "word for word, without an introduction, translation, added dialect words, or backend delegation. "
    "Use this session's selected voice and pronunciation direction. Then wait and resume normal "
    "Igbo conversation and web research when the caller next speaks. Passage: " + COMPARISON_TEXT
)


def design_payload() -> dict:
    return {"type": "prompt", "name": "Anyanwu Owere trial", "prompt": VOICE_DESIGN,
            "script_hint": COMPARISON_TEXT, "model": "auto"}


class VoiceProfiles:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.voice_id = None
        self.state = "ready"
        self.persisted = False
        if path:
            try:
                if not path.is_file() or path.stat().st_size > 8192:
                    return
                record = json.loads(path.read_text())
                if (isinstance(record, dict) and record.get("type") == "prompt" and
                        record.get("prompt") == VOICE_DESIGN and self.valid_id(record.get("id"))):
                    self.voice_id = record["id"]
                    self.persisted = True
            except (ValueError, OSError):
                pass

    @staticmethod
    def valid_id(value) -> bool:
        return isinstance(value, str) and bool(re.fullmatch(r"voice_[A-Za-z0-9_-]{1,160}", value))

    def catalog(self) -> list[dict]:
        choices = [
            {"id": "marin-original", "label": "Original Marin · baseline"},
            {"id": "owerri-marin", "label": "Marin · Owerri pronunciation target"},
        ]
        if self.voice_id:
            choices.append({"id": "owerri-designed", "label": "Designed voice · Owerri target (experimental)"})
        return choices

    def select(self, profile: str) -> tuple[object, str]:
        if profile == "marin-original":
            return "marin", ""
        if profile == "owerri-marin":
            return "marin", VOICE_DIRECTION
        if profile == "owerri-designed" and self.voice_id:
            return {"id": self.voice_id}, VOICE_DESIGN + " " + VOICE_DIRECTION
        raise ValueError("Choose an available voice from the comparison menu.")

    def install(self, voice_id: str) -> bool:
        if not self.valid_id(voice_id):
            raise ValueError("Invalid saved voice ID")
        # Keep it usable in this process even if saving public metadata fails.
        self.voice_id = voice_id
        self.state = "saved"
        if not self.path:
            return False
        record = {"id": voice_id, "type": "prompt", "prompt": VOICE_DESIGN, "script_hint": COMPARISON_TEXT}
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix="voice-design-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                os.chmod(temporary, 0o600)
                json.dump(record, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self.persisted = True
            return True
        except OSError:
            return False
        finally:
            if temporary and temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
