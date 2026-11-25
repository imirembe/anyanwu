#!/usr/bin/env python3
"""Prepare and review a small, bounded Igbo voice-research listening pilot."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.voice_research.pilot import PilotError, inspect, prepare


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["inspect"]:
            inspect()
        elif sys.argv[1:] == ["prepare"]:
            prepare()
        elif sys.argv[1:] == ["review"]:
            from backend.voice_research.review import serve
            serve()
        elif sys.argv[1:] == ["export-audition"]:
            from backend.voice_research.audition import export
            export()
        else:
            raise PilotError("Use: python3 scripts/voice_pilot.py inspect | prepare | review | export-audition")
    except (PilotError, KeyboardInterrupt) as error:
        print(f"Voice pilot: {error}", file=sys.stderr)
        sys.exit(1)
