#!/usr/bin/env python3
"""Run Anyanwụ's voice trial, reusing its OpenAI API key from macOS Keychain."""
from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from voice_credentials import CredentialError, forget_key, load_key

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".build" / "OpenAIVoiceTest" / "venv"


def dependencies_ready() -> bool:
    for line in (ROOT / "backend/requirements-voice-test.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        package, version = line.split("==", 1)
        try:
            if metadata.version(package) != version:
                return False
        except metadata.PackageNotFoundError:
            return False
    return True


def prepare_environment(arguments: list[str]) -> None:
    python = VENV / "bin" / "python"
    if Path(sys.prefix).resolve() != VENV.resolve():
        if not python.exists():
            import shutil
            if shutil.disk_usage(ROOT).free < 20 * 1024**3 + 100 * 1024**2:
                raise CredentialError("Keep at least 20 GiB free before installing the small test environment.")
            subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True,
                           env={k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"})
        # Re-exec before reading the key. Flags survive; no secret is in argv.
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *arguments])
        return
    if not dependencies_ready():
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                        "-r", str(ROOT / "backend/requirements-voice-test.txt")], check=True,
                       env={k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Inspect the page without a key or API calls")
    parser.add_argument("--port", type=int, default=8788)
    keys = parser.add_mutually_exclusive_group()
    keys.add_argument("--replace-key", action="store_true", help="Privately replace Anyanwụ's saved Keychain key")
    keys.add_argument("--forget-key", action="store_true", help="Remove only Anyanwụ's saved Keychain key and exit")
    keys.add_argument("--no-save-key", action="store_true", help="Use a temporary key without reading or writing Keychain")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port from 1024 to 65535.")
    if args.preview and (args.replace_key or args.forget_key or args.no_save_key):
        parser.error("Preview does not access credentials; omit the key-management flag.")
    key = ""
    if not args.preview:
        try:
            prepare_environment(sys.argv[1:])
            if args.forget_key:
                print("Removed Anyanwụ's stored API key." if forget_key() else "Anyanwụ has no stored API key.")
                return
            key, source = load_key(environment=os.environ, replace=args.replace_key, temporary=args.no_save_key)
            print({"saved": "Saved to macOS Keychain. Future launches will reuse this key.",
                   "keychain": "Using Anyanwụ's saved API key from macOS Keychain.",
                   "environment": "Using OPENAI_API_KEY for this run; the saved Keychain key is unchanged.",
                   "temporary": "Using a temporary API key for this run."}[source])
        except CredentialError as exc:
            parser.error(str(exc))
    sys.path.insert(0, str(ROOT / "backend"))
    from openai_voice_test import serve
    try:
        serve(key, args.port)
    except OSError as exc:
        if exc.errno == 48:
            parser.error(f"Port {args.port} is in use. Stop this project's previous voice-test command or choose --port 8789.")
        raise


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nVoice test stopped.")
