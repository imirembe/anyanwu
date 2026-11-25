#!/usr/bin/env python3
"""Run Anyanwụ's small, storage-bounded local Igbo voice-assistant research lab."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from research.igbo_lab.resources import (LAB, GIB, MIB, WORKSPACE_LIMIT, FREE_RESERVE,
                                       LabError, directory_bytes, ensure_capacity,
                                       exclusive, owned, remove_generated, write_json)

VENV = LAB / "venv"
PYTHON = VENV / "bin/python"
DEMO = PROJECT / "research/igbo_lab/data/demo.jsonl"
VOICE_EVAL = PROJECT / "research/igbo_lab/data/voice_eval_v1.jsonl"


def environment():
    env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    temporary = owned(LAB / "tmp"); temporary.mkdir(parents=True, exist_ok=True)
    env.update({"TMPDIR": str(temporary), "PIP_CONFIG_FILE": os.devnull,
                "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def setup():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise LabError("This starter uses native Apple-silicon MLX on macOS.")
    ensure_capacity(512 * MIB)
    owned(VENV)
    if not PYTHON.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True, env=environment())
    subprocess.run([str(PYTHON), "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:",
                    "--no-deps", "--require-virtualenv", "--index-url", "https://pypi.org/simple",
                    "-r", str(PROJECT / "research/igbo_lab/requirements.txt")],
                   check=True, cwd=PROJECT, env=environment())
    ensure_capacity()
    write_json(LAB / "setup.json", {"platform": platform.platform(), "machine": platform.machine(),
                                   "framework": "MLX", "pretrained_weights_downloaded": False,
                                   "workspace_limit_bytes": WORKSPACE_LIMIT, "free_reserve_bytes": FREE_RESERVE})
    print("Local MLX environment ready. No datasets or pretrained model weights downloaded.")


def status():
    owned(LAB)
    used, free = directory_bytes(), shutil.disk_usage(PROJECT).free
    report = {"workspace": str(LAB), "workspace_MiB": round(used / MIB, 1),
              "workspace_limit_GiB": WORKSPACE_LIMIT / GIB, "disk_free_GiB": round(free / GIB, 2),
              "minimum_free_GiB": FREE_RESERVE / GIB, "environment_ready": (LAB / "setup.json").exists(),
              "purpose": "Experimental reply and request-understanding models for the Igbo voice assistant; no audio is retained.",
              "corpus": None, "checkpoint": None}
    for key, path in (("corpus", LAB / "corpus/manifest.json"), ("checkpoint", LAB / "checkpoints/current/checkpoint.json")):
        owned(path)
        if path.exists() and path.stat().st_size < MIB:
            report[key] = json.loads(path.read_text())
    latest = owned(LAB / "latest_experiment.json")
    if latest.exists() and latest.stat().st_size < MIB:
        report["latest_experiment"] = json.loads(latest.read_text())
    latest_intent=owned(LAB/"latest_intent_experiment.json")
    if latest_intent.exists() and latest_intent.stat().st_size<MIB:
        report["latest_intent_experiment"]=json.loads(latest_intent.read_text())
    latest_request=owned(LAB/"latest_request_experiment.json")
    if latest_request.exists() and latest_request.stat().st_size<MIB:
        report["latest_request_experiment"]=json.loads(latest_request.read_text())
    latest_data=owned(LAB/"latest_request_data.json")
    if latest_data.exists() and latest_data.stat().st_size<MIB:
        report["latest_request_data"]=json.loads(latest_data.read_text())
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("status", help="show storage, corpus and checkpoint information")
    commands.add_parser("setup", help="install only pinned MLX/Metal/SentencePiece wheels locally")
    prepare = commands.add_parser("prepare", help="prepare a small JSONL corpus inside this project")
    prepare.add_argument("input", nargs="?", default=str(DEMO))
    prepare.add_argument("--vocabulary", type=int, default=512)
    prepare.add_argument("--rebuild-tokenizer", action="store_true", help="requires fresh model training afterward")
    train = commands.add_parser("train", help="train from scratch, or resume this lab's own checkpoint")
    train.add_argument("--steps", type=int, default=120)
    train.add_argument("--seconds", type=int, default=180)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--cpu", action="store_true")
    train.add_argument("--batch-size", type=int, default=2)
    reply = commands.add_parser("reply", help="test the research model with a speech transcript")
    reply.add_argument("transcript")
    reply.add_argument("--maximum", type=int, default=64)
    reply.add_argument("--cpu", action="store_true")
    reply.add_argument("--checkpoint", choices=["current", "previous"], default="current")
    demo = commands.add_parser("demo", help="prepare tiny synthetic fixtures and run a short experiment")
    demo.add_argument("--steps", type=int, default=120)
    demo.add_argument("--seconds", type=int, default=180)
    demo.add_argument("--cpu", action="store_true")
    commands.add_parser("clean", help="remove only abandoned temporary/staging outputs, never checkpoints or source data")
    commands.add_parser("check", help="verify corpus isolation, Unicode, budgets and causal prediction")
    importing = commands.add_parser("import-text", help="import a bounded text-only Hub sample for review")
    importing.add_argument("--source", choices=["public-igbo", "africanvoice", "aya-igbo", "aya-originals", "injongo-igbo", "injongo-igbo-v2", "injongo-igbo-other", "clinc-english", "orinode-mixed", "clinc-english-v3", "clinc-english-other", "injongo-igbo-v3", "injongo-igbo-other-v3", "clinc-original-oos"], default="public-igbo")
    importing.add_argument("--rows", type=int, default=100)
    importing.add_argument("--offset", type=int, default=0, help="start a separate immutable row batch; transcript previews support only zero")
    importing.add_argument("--token-prompt", action="store_true", help="privately prompt for a read-only HF token; never save it")
    reviewing = commands.add_parser("review-sample", help="apply traceable Codex/human review decisions")
    reviewing.add_argument("decisions")
    reviewing.add_argument("--source", choices=["public-igbo", "africanvoice", "aya-igbo", "aya-originals", "injongo-igbo", "injongo-igbo-v2", "injongo-igbo-other", "clinc-english", "orinode-mixed", "clinc-english-v3", "clinc-english-other", "injongo-igbo-v3", "injongo-igbo-other-v3", "clinc-original-oos"], default="public-igbo")
    reviewing.add_argument("--cases", default=str(VOICE_EVAL))
    reviewing.add_argument("--offset", type=int, default=0)
    assembling = commands.add_parser("assemble-reviewed", help="combine accepted batches without replacing the active corpus or training")
    assembling.add_argument("samples", nargs="+", help="source or source@offset")
    assembling.add_argument("--cases", default=str(VOICE_EVAL))
    experiment = commands.add_parser("run-experiment", help="prepare one reviewed assembly and run one fixed 120-step baseline comparison")
    experiment.add_argument("assembly")
    experiment.add_argument("--cases", default=str(VOICE_EVAL))
    prepared = commands.add_parser("prepare-reviewed", help="prepare only the reviewed sample with frozen evaluation metadata")
    prepared.add_argument("--source", choices=["public-igbo", "africanvoice", "aya-igbo", "aya-originals", "injongo-igbo", "injongo-igbo-v2", "injongo-igbo-other", "clinc-english", "orinode-mixed", "clinc-english-v3", "clinc-english-other", "injongo-igbo-v3", "injongo-igbo-other-v3", "clinc-original-oos"], default="public-igbo")
    prepared.add_argument("--cases", default=str(VOICE_EVAL))
    prepared.add_argument("--offset", type=int, default=0)
    evaluation = commands.add_parser("evaluate-transcripts", help="evaluate a saved model against separate voice-style transcript prompts")
    evaluation.add_argument("--label", choices=["baseline", "candidate"], required=True)
    evaluation.add_argument("--cases", default=str(VOICE_EVAL))
    evaluation.add_argument("--replace-baseline", action="store_true", help="explicitly begin a new comparison baseline")
    evaluation.add_argument("--report-directory", help="owned lab directory for this comparison; default is the original pilot")
    comparing = commands.add_parser("compare", help="compare baseline/candidate using the same tokenizer and frozen cases")
    comparing.add_argument("--report-directory", help="owned lab comparison directory; default is the original pilot")
    intent_run = commands.add_parser("intent-experiment", help="fit/calibrate/test one small in-memory Igbo request classifier")
    intent_run.add_argument("--name", default="intent-prototype-v1")
    intent_run.add_argument("--requests", type=Path, default=LAB/"curation/assemblies/90cbf2af9a0dfd46")
    intent_run.add_argument("--background", type=Path, default=LAB/"curation/assemblies/55d5ff6b7e546915")
    intent_run.add_argument("--background-review", type=Path, default=PROJECT/"research/igbo_lab/data/intent_background_review_v1.jsonl")
    intent_lookup = commands.add_parser("classify-intent", help="recognise a transcript using the experimental in-memory index; execute no actions")
    intent_lookup.add_argument("transcript")
    intent_lookup.add_argument("--experiment", default="intent-prototype-v1")
    request_run = commands.add_parser("request-experiment", help="prepare or evaluate the separate version-2 intent/details experiment")
    request_run.add_argument("--phase", choices=["prepare","evaluate"], default="prepare")
    request_run.add_argument("--name", default="request-understanding-v2")
    request_run.add_argument("--snapshot", type=Path, default=LAB/"request_data_snapshots/88f830477b4f9807")
    request_run.add_argument("--detail-review", type=Path, default=PROJECT/"research/igbo_lab/data/request_details_v2_review.jsonl")
    understanding = commands.add_parser("understand-request", help="return an offline intent candidate and tentative raw details; execute no action")
    understanding.add_argument("transcript")
    understanding.add_argument("--experiment", default="request-understanding-v2")
    args = parser.parse_args()
    if not args.command or args.command == "status":
        status(); return
    if args.command not in {"setup", "clean", "import-text", "review-sample", "assemble-reviewed", "compare", "request-experiment", "understand-request"} and Path(sys.prefix).resolve() != VENV.resolve():
        if not PYTHON.exists():
            raise LabError("Run python3 scripts/igbo_lab.py setup first.")
        # Replace the wrapper so Ctrl-C/SIGTERM reaches the training process,
        # whose handler saves a checkpoint at the next safe step boundary.
        os.execve(str(PYTHON), [str(PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]], environment())
    with exclusive():
        if args.command == "setup":
            setup(); return
        if args.command == "clean":
            for path in (LAB / "tmp", LAB / "corpus-staging", LAB / "checkpoints/checkpoint-staging", LAB / "imports/import-staging", LAB / "curation/assembly-staging"):
                remove_generated(path)
            print("Removed abandoned lab temporary files; corpus, checkpoints and original inputs preserved.")
            return
        if args.command == "import-text":
            from research.igbo_lab.importer import import_text
            import_text(args.source, args.rows, args.token_prompt, offset=args.offset)
            return
        if args.command == "review-sample":
            from research.igbo_lab.review import compile_review
            compile_review(args.source, args.decisions, args.cases, args.offset)
            return
        if args.command == "assemble-reviewed":
            from research.igbo_lab.assembly import assemble_reviewed
            assemble_reviewed(args.samples, args.cases)
            return
        if args.command == "compare":
            from research.igbo_lab.evaluation import compare_evaluations
            compare_evaluations(args.report_directory)
            return
        if args.command == "intent-experiment":
            from research.igbo_lab.intent import run_intent_experiment
            run_intent_experiment(args.requests,args.background,args.background_review,args.name)
            return
        if args.command == "classify-intent":
            from research.igbo_lab.intent import classify_intent
            classify_intent(args.transcript,args.experiment)
            return
        if args.command == "request-experiment":
            from research.igbo_lab.request_understanding import prepare_experiment, evaluate_experiment
            if args.phase == "prepare": prepare_experiment(args.snapshot,args.detail_review,args.name)
            else: evaluate_experiment(args.name)
            return
        if args.command == "understand-request":
            from research.igbo_lab.request_understanding import understand_request
            understand_request(args.transcript,args.experiment)
            return
        try:
            import mlx.core
            import sentencepiece
        except ImportError as error:
            raise LabError("Run python3 scripts/igbo_lab.py setup to install the small local dependencies.") from error
        if args.command == "check":
            import unittest
            from research.igbo_lab import test_lab, test_pipeline, test_assembly, test_experiments, test_request_data, test_intent, test_request_understanding
            suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module) for module in (test_lab, test_pipeline, test_assembly, test_experiments, test_request_data, test_intent, test_request_understanding)])
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful():
                raise LabError("Local training checks failed.")
            return
        if args.command == "evaluate-transcripts":
            from research.igbo_lab.evaluation import evaluate_checkpoint
            evaluate_checkpoint(args.cases, args.label, args.replace_baseline, args.report_directory)
            return
        if args.command == "run-experiment":
            from research.igbo_lab.experiments import run_experiment
            run_experiment(args.assembly, args.cases)
            return
        if args.command == "prepare-reviewed":
            from research.igbo_lab.review import prepare_reviewed
            prepare_reviewed(args.cases, args.source, args.offset)
            return
        if args.command in {"prepare", "demo"}:
            from research.igbo_lab.corpus import prepare as prepare_corpus
            if args.command == "demo" and (LAB / "corpus/manifest.json").exists():
                if not json.loads((LAB / "corpus/manifest.json").read_text()).get("demo_only", False):
                    raise LabError("Demo refuses to replace a real corpus. Use train to continue your experiment.")
            prepare_corpus(args.input if args.command == "prepare" else DEMO,
                           args.vocabulary if args.command == "prepare" else 512,
                           args.rebuild_tokenizer if args.command == "prepare" else False)
        if args.command in {"train", "demo"}:
            from research.igbo_lab.runner import train as train_model
            train_model(steps=args.steps, seconds=args.seconds, cpu=args.cpu,
                        resume=args.resume if args.command == "train" else False,
                        batch_size=args.batch_size if args.command == "train" else 2)
        if args.command == "reply":
            from research.igbo_lab.runner import reply as generate_reply
            print(json.dumps(generate_reply(args.transcript, args.maximum, args.cpu, args.checkpoint), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (LabError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Igbo Lab: {error}", file=sys.stderr)
        raise SystemExit(1)
