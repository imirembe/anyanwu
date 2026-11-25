"""Storage and ownership rules shared by setup, preparation and training."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

PROJECT = Path(__file__).resolve().parents[2]
LAB = PROJECT / ".build/IgboLab"
GIB = 1024 ** 3
MIB = 1024 ** 2
WORKSPACE_LIMIT = 2 * GIB
FREE_RESERVE = 20 * GIB
MLX_MEMORY_GUIDELINE = 2 * GIB
MLX_CACHE_LIMIT = 128 * MIB
CORPUS_LIMIT = 32 * MIB
DOCUMENT_LIMIT = 32_768
DOCUMENT_COUNT_LIMIT = 20_000
TOKEN_LIMIT = 8_000_000
MAX_PARAMETERS = 5_000_000


class LabError(RuntimeError):
    pass


def owned(path):
    path = Path(path)
    if LAB.is_symlink() or not LAB.resolve().is_relative_to(PROJECT.resolve()):
        raise LabError("The training workspace must stay inside this project.")
    if not path.resolve().is_relative_to(LAB.resolve()):
        raise LabError("Refusing a path outside .build/IgboLab.")
    # Reject symlinks in managed output paths, including links back into the lab.
    current = path
    while current != LAB:
        if current.is_symlink():
            raise LabError("Refusing a symlink in a managed output path.")
        if current.parent == current:
            raise LabError("Invalid managed output path.")
        current = current.parent
    return path


def project_input(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT.resolve()) or not path.is_file():
        raise LabError("Keep input files inside the Anyanwụ project directory.")
    return path


def directory_bytes(path=None):
    path = LAB if path is None else path
    total = 0
    if not Path(path).exists():
        return total
    for parent, directories, files in os.walk(path, followlinks=False):
        directories[:] = [name for name in directories if not Path(parent, name).is_symlink()]
        for name in files:
            try:
                # lstat counts the link itself, not external Python installations.
                total += Path(parent, name).lstat().st_size
            except FileNotFoundError:
                pass
    return total


def ensure_capacity(extra=0):
    owned(LAB)
    used = directory_bytes()
    free = shutil.disk_usage(PROJECT).free
    if extra < 0 or used + extra > WORKSPACE_LIMIT:
        raise LabError("The 2 GiB training-workspace budget would be exceeded.")
    if free - extra < FREE_RESERVE:
        raise LabError("Stopped to preserve at least 20 GiB free on this Mac.")
    return used, free


def write_json(path, value):
    path = owned(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    ensure_capacity(len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive():
    owned(LAB)
    LAB.mkdir(parents=True, exist_ok=True)
    with owned(LAB / ".lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LabError("Another Igbo Lab setup or training process is active.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def remove_generated(path):
    path = owned(path)
    if path == LAB or path.name not in {"corpus-staging", "checkpoint-staging", "import-staging", "assembly-staging", "retired-checkpoint", "previous", "tmp"}:
        raise LabError("This path is not on the generated-file cleanup allowlist.")
    if path.is_symlink():
        raise LabError("Refusing to clean a symlink.")
    if path.exists():
        shutil.rmtree(path)
