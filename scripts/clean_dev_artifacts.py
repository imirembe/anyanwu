#!/usr/bin/env python3
"""Retire only known, obsolete Anyanwụ outputs. Dry run unless --apply is given."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1] / ".build"
# Never add the current DerivedData.nosync, app data, source, or artwork here.
OBSOLETE = (
    "DerivedData", "SwiftModuleCache", "MinimalHomeLanguageTests.xcresult",
    "NdichieTests.xcresult", "NdichieIsolatedTests.xcresult", "VerifiedTests.xcresult",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="remove the listed generated outputs")
    args = parser.parse_args()
    if ROOT.is_symlink():
        raise SystemExit("Refusing a symlinked build directory")
    total = 0
    for name in OBSOLETE:
        path = ROOT / name
        if not path.exists():
            continue
        if path.is_symlink() or path.resolve().parent != ROOT.resolve():
            raise SystemExit("Refusing a path outside this project's build directory")
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) if path.is_dir() else path.stat().st_size
        print(f"{'Removing' if args.apply else 'Would remove'} {name}: {size / 1024 ** 2:.1f} MiB")
        if args.apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        total += size
    print(f"{'Removed' if args.apply else 'Eligible'}: {total / 1024 ** 2:.1f} MiB")


if __name__ == "__main__":
    main()
