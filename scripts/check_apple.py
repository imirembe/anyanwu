#!/usr/bin/env python3
"""Test Apple speech infrastructure with a 91 KB tokenizer, never model weights."""
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".build/AppleChecks"
OUTPUT.mkdir(parents=True, exist_ok=True)
tokenizer = OUTPUT / "tokenizer.model"
if not tokenizer.exists():
    subprocess.run(["curl", "-fLsS", "--max-filesize", "100000", "--max-time", "45",
                    "https://huggingface.co/aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s/resolve/ce99f14bbc768d8dc6ffd3235a21cd9751ceb139/tokenizer.model",
                    "-o", str(tokenizer)], cwd=ROOT, check=True)
sources = [ROOT / "Sources/Anyanwu/Models.swift"]
sources += [ROOT / "Sources/Anyanwu/Services" / name for name in (
    "AppleSpeechTypes.swift", "SpeechPreparationPolicy.swift", "SentencePieceLexicon.swift", "ModelFileDownload.swift", "AppleSpeechAssets.swift", "AppleSpeechRecognizer.swift",
    "AssistantService.swift", "BoundedHTTP.swift", "SecureToken.swift")]
sources += [ROOT / "Tests/AppleChecks/AppleChecks.swift"]
subprocess.run(["xcrun", "swiftc", "-swift-version", "5", "-parse-as-library", "-module-cache-path", str(OUTPUT / "ModuleCache"),
                *map(str, sources), "-o", str(OUTPUT / "checks")], cwd=ROOT, check=True)
subprocess.run([str(OUTPUT / "checks"), str(OUTPUT / ("fixtures-" + uuid.uuid4().hex)), str(tokenizer)], cwd=ROOT, check=True)
