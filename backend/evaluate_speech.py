"""Evaluate actual cloud ASR against a bounded, human-transcribed JSONL set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import unicodedata

from backend.server import APIError, Config, Providers, load_environment, MAX_AUDIO_BYTES
from backend.inference.engines import ASR_MODEL


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    # Keep combining marks and underdots. Punctuation/spacing are not speech
    # errors in this report; accented and unaccented letters remain different.
    return " ".join("".join(" " if unicodedata.category(char)[0] in {"P", "Z"} else char
                            for char in text).split())


def distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        updated = [i]
        for j, right in enumerate(hypothesis, 1):
            updated.append(min(updated[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = updated
    return row[-1]


def evaluate(manifest: Path, provider: Providers, limit: int):
    groups = {language: {"samples": 0, "wordErrors": 0, "words": 0, "characterErrors": 0, "characters": 0,
                         "elapsedSeconds": 0.0} for language in ("ig", "en", "mixed")}
    ids = set()
    with manifest.open(encoding="utf-8") as source:
        for index in range(limit):
            line = source.readline(16_001)
            if not line:
                break
            if len(line) > 16_000:
                raise ValueError("A manifest row exceeds 16,000 characters.")
            item = json.loads(line)
            if (not isinstance(item, dict) or item.get("language") not in groups
                    or not isinstance(item.get("id"), str) or not 0 < len(item["id"]) <= 100
                    or item["id"] in ids or not isinstance(item.get("audio"), str)
                    or not isinstance(item.get("text"), str) or not 0 < len(item["text"]) <= 2_000):
                raise ValueError("Invalid or duplicate evaluation row.")
            ids.add(item["id"])
            path = (manifest.parent / item["audio"]).resolve()
            # Keep evaluation files with their manifest; refuse arbitrary paths.
            if not path.is_relative_to(manifest.parent.resolve()):
                raise ValueError("Evaluation audio must stay in the manifest directory.")
            mime = {".wav": "audio/wav", ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".webm": "audio/webm"}.get(path.suffix.lower())
            if not mime:
                raise ValueError("Unsupported evaluation audio format.")
            with path.open("rb") as file:
                audio = file.read(MAX_AUDIO_BYTES + 1)
            if not audio or len(audio) > MAX_AUDIO_BYTES:
                raise ValueError("An evaluation clip exceeds the audio limit.")
            start = time.monotonic()
            hypothesis = provider.transcribe(audio, mime)
            reference = normalize(item["text"])
            hypothesis = normalize(hypothesis)
            if not reference:
                raise ValueError("An evaluation reference is empty after normalization.")
            group = groups[item["language"]]
            group["samples"] += 1
            group["elapsedSeconds"] += time.monotonic() - start
            group["wordErrors"] += distance(reference.split(), hypothesis.split())
            group["words"] += len(reference.split())
            group["characterErrors"] += distance(reference, hypothesis)
            group["characters"] += len(reference)
    for group in groups.values():
        group["WER"] = group["wordErrors"] / group["words"] if group["words"] else None
        group["CER"] = group["characterErrors"] / group["characters"] if group["characters"] else None
        group["elapsedSeconds"] = round(group["elapsedSeconds"], 3)
    return {"model": ASR_MODEL, "normalization": "NFC, casefold, punctuation/spacing normalized; diacritics preserved",
            "groups": groups, "samples": len(ids), "limit": limit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(".build/evaluations/asr.json"))
    args = parser.parse_args()
    if not 1 <= args.limit <= 250:
        parser.error("Choose 1–250 evaluation samples per run.")
    load_environment(Path(__file__).parent / "cloud" / ".env")
    token = os.environ.get("ANYANWU_WORKER_TOKEN", "")
    if not token:
        parser.error("Run this on the cloud host with its worker token configured.")
    provider = Providers(Config(asr_url="http://127.0.0.1:8001/v1/audio/transcriptions", asr_model=ASR_MODEL, asr_key=token))
    try:
        report = evaluate(args.manifest, provider, args.limit)
        if not report["samples"]:
            raise ValueError("No evaluation samples were provided.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print(f"Evaluated {report['samples']} recordings. Aggregate report: {args.output}")
    except (APIError, OSError, ValueError, TypeError) as error:
        raise SystemExit("Evaluation failed: " + type(error).__name__ + ". No successful report was written.") from None


if __name__ == "__main__":
    main()
