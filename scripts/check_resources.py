#!/usr/bin/env python3
"""Run real SQLite/audio/context/HTTP checks on macOS, without a simulator."""
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".build/ResourceChecks"
OUTPUT.mkdir(parents=True, exist_ok=True)
sources = [ROOT / "Sources/Anyanwu/Models.swift"]
sources += [ROOT / "Sources/Anyanwu/AssistantStore.swift"]
sources += [ROOT / "Sources/Anyanwu/Services" / name for name in (
    "LocalRepository.swift", "LegacyConversationReader.swift", "ConversationDatabase.swift",
    "ConversationArchive.swift", "AssistantService.swift", "BoundedHTTP.swift", "SecureToken.swift", "AppleSpeechTypes.swift")]
sources += sorted((ROOT / "Tests/ResourceChecks").glob("*.swift"))
subprocess.run(["xcrun", "swiftc", "-swift-version", "5", "-parse-as-library",
                "-module-cache-path", str(OUTPUT / "ModuleCache"), *map(str, sources),
                "-o", str(OUTPUT / "checks")], cwd=ROOT, check=True)
subprocess.run([str(OUTPUT / "checks"), str(OUTPUT / ("fixtures-" + uuid.uuid4().hex))], cwd=ROOT, check=True)
