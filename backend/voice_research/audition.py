"""Summarise real listening judgments and export a small cloud audition notebook."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from .pilot import PILOT_ROOT, PilotError, atomic_json, ensure_space, load_pilot
from .review import ReviewStore

HERE = Path(__file__).resolve().parent
AUDITION_ID = "naija-tts-six-sentences-v1"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def export(root=PILOT_ROOT):
    folder, manifest = load_pilot(root)
    reviews = ReviewStore(folder, manifest).snapshot()
    complete = [clip for clip in manifest["clips"] if (review := reviews.get(clip["id"]))
                and all(review.get(key) is not None for key in ("intelligibility", "cadence", "pronunciation", "transcript_match"))]
    if len(complete) != 6:
        raise PilotError("All six clips need saved ratings and a transcript-match decision before this export.")
    review_bytes = (folder / "reviews.json").read_bytes()
    digest = hashlib.sha256(review_bytes).hexdigest()
    target = root / "auditions" / AUDITION_ID
    if target.exists():
        existing = json.loads((target / "review-summary.json").read_text(encoding="utf-8"))
        if existing["reviews_sha256"] != digest:
            raise PilotError("The saved feedback changed after this export; preserve the first snapshot and prepare a new audition revision.")
        print(f"Reusing the prepared notebook: {target / 'Anyanwu-voice-audition.ipynb'}")
        return target
    ensure_space(1024**2)
    matches = {value: sum(r["transcript_match"] == value for r in reviews.values()) for value in ("yes", "unsure", "no")}
    if matches["yes"] != 6:
        raise PilotError("Resolve the non-matching/uncertain source transcripts before exporting a same-sentence audition.")
    averages = {key: round(sum(reviews[c["id"]][key] for c in complete) / 6, 2)
                for key in ("intelligibility", "cadence", "pronunciation")}
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "pilot": manifest["id"],
               "reviews_sha256": digest, "reviewer": "User listening judgments; reviewer background not independently recorded",
               "reviewed_clips": 6, "speaker_count": manifest["speaker_count"], "averages_out_of_5": averages,
               "transcript_matches": matches, "notes_present": sum(bool(r["notes"].strip()) for r in reviews.values()),
               "interpretation": "Positive source-reference judgments apply to these six human recordings, not a model or every Igbo dialect.",
               "training_ready": False, "source_audio_used_for_voice_conditioning": False}
    package = {"id": AUDITION_ID, "source_attribution": manifest["attribution"],
               "source_revision": manifest["source"]["hub_revision"],
               "cases": [{"id": clip["id"], "text": clip["transcript"], "source_row_index": clip["source_row_index"]}
                         for clip in complete]}
    runner = (HERE / "cloud_audition.py").read_text(encoding="utf-8")
    setup = (HERE / "colab_setup.py").read_text(encoding="utf-8")
    run = ("import json\n"
           f"package = json.loads({json.dumps(json.dumps(package, ensure_ascii=False))})\n"
           f"runner_source = {runner!r}\n"
           "(WORK / 'cases.json').write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding='utf-8')\n"
           "(WORK / 'run.py').write_text(runner_source, encoding='utf-8')\n"
           "bounded([str(PYTHON), str(WORK / 'run.py'), str(WORK / 'cases.json'), str(WORK / 'outputs')])\n")
    listen = ("import json\nfrom IPython.display import Audio, display\n"
              "result = json.loads((WORK / 'outputs/manifest.json').read_text(encoding='utf-8'))\n"
              "assert result['status'] == 'generated_pending_listening' and len(result['clips']) == 6\n"
              "for clip in result['clips']:\n"
              "    print(clip['id'], '· AI-generated built-in voice ·', clip['text'])\n"
              "    display(Audio(filename=str(WORK / 'outputs' / clip['filename'])))\n")
    optional = ("# Optional: set True only if you want the small result ZIP on your laptop.\n"
                "DOWNLOAD_RESULTS = False\n"
                "if DOWNLOAD_RESULTS:\n"
                "    import zipfile\n    from google.colab import files\n"
                "    archive = WORK / 'Anyanwu-voice-audition.zip'\n"
                "    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:\n"
                "        for path in sorted((WORK / 'outputs').iterdir()):\n"
                "            if path.is_file(): z.write(path, path.name)\n"
                "        z.write(WORK / 'environment.txt', 'environment.txt')\n"
                "    assert archive.stat().st_size <= 25 * 1024**2\n"
                "    files.download(str(archive))\n")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"colab": {"name": "Anyanwu-voice-audition.ipynb"}, "accelerator": "GPU",
                             "anyanwu_setup_revision": 2,
                             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [cell("markdown", "# Anyanwụ: first generated Igbo comparison\n\n"
                    "Open in [Google Colab](https://colab.research.google.com/), choose a hosted GPU runtime (a free T4 if offered), then run cells in order. Never select a local runtime. Free GPU availability varies; this notebook buys nothing.\n\n"
                    "Generates the same six published sentences you rated, once each, using pinned PlotweaverAI/naija-tts and its built-in voice. No human audio, feedback notes, private conversations, API key, voice cloning or training is uploaded. These already-heard sentences are a reference comparison, not a held-out benchmark or proof of a unique Anyanwụ voice.\n\n"
                    "Setup/model assets stay in the cloud VM: up to 20 GiB cloud-disk increase, 4 GiB selected model files, 30 minutes of child work and six output WAVs ≤30 seconds/4 MiB each, ≤24 MiB total. These are guard limits, not measured requirements. GPU compatibility/inference are not yet tested; stop on errors. Disconnect and delete the runtime after listening.\n\n"
                    "The original reference is [Media Trust / WAXAL Igbo TTS](https://huggingface.co/datasets/google/WaxalNLP), CC BY-SA 4.0. The candidate's publisher lists MIT; upstream voice/training rights are not independently audited. Preserve the candidate watermark.\n"),
                    cell("code", setup), cell("markdown", "## Generate once\nNo reference-audio conditioning or alternate-take selection. A failure stops the run."),
                    cell("code", run), cell("markdown", "## Listen\nCompare these generated clips with the human recordings at http://127.0.0.1:8790 on your Mac. Judge clarity, Igbo sounds/tones, rhythm, missing/repeated words and voice consistency. Record your feedback in chat; the original human-clip ratings stay unchanged."),
                    cell("code", listen), cell("markdown", "## Optional small export\nDownloads remain disabled by default. If needed, enable the cell below and save the ZIP inside the Anyanwụ project. No weights are included. Finally use **Runtime → Disconnect and delete runtime**."),
                    cell("code", optional)]}
    for index, item in enumerate(notebook["cells"]):
        item["id"] = f"anyanwu-{index+1:02}"
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"notebook-cell-{index}", "exec")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".audition-staging-", dir=target.parent) as temporary:
        stage = Path(temporary)
        atomic_json(stage / "review-summary.json", summary)
        atomic_json(stage / "cases.json", package)
        atomic_json(stage / "Anyanwu-voice-audition.ipynb", notebook)
        stage.rename(target)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Prepared notebook: {target / 'Anyanwu-voice-audition.ipynb'}")
    return target
