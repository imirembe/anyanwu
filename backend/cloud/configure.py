"""Initialize secrets on a Linux GPU host; never print or overwrite secrets."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import secrets
import subprocess


def main():
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit("Run this script on an x86-64 Linux GPU cloud host. Nothing was downloaded or changed.")
    try:
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       check=True, timeout=10, stdout=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.SubprocessError):
        raise SystemExit("An NVIDIA GPU with working drivers is required. No configuration was written.") from None
    path = Path(__file__).with_name(".env")
    content = ("# Cloud-only configuration. Keep this file private.\n"
               "ANYANWU_ACCESS_TOKEN=" + secrets.token_urlsafe(32) + "\n"
               "ANYANWU_WORKER_TOKEN=" + secrets.token_urlsafe(32) + "\n"
               "ANYANWU_ALLOW_MODEL_DOWNLOAD=1\n")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise SystemExit("Cloud .env already exists; its tokens were preserved.") from None
    with os.fdopen(descriptor, "w") as file:
        file.write(content)
    print("Created private backend/cloud/.env. Tokens were not printed. Model downloads are enabled on this host.")


if __name__ == "__main__":
    main()
