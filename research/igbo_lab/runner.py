"""Training and inference use local, explicitly prepared data; no network calls."""
from contextlib import ExitStack
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import random
import re
import resource
import shutil
import signal
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import sentencepiece as spm

from .model import IgboLanguageModel, ModelConfig
from .resources import (LAB, GIB, MIB, MLX_MEMORY_GUIDELINE, MLX_CACHE_LIMIT,
                        TOKEN_LIMIT, MAX_PARAMETERS, LabError, ensure_capacity,
                        owned, remove_generated, write_json)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def configure_runtime(cpu=False):
    mx.set_default_device(mx.cpu if cpu else mx.gpu)
    mx.set_memory_limit(MLX_MEMORY_GUIDELINE)
    mx.set_cache_limit(MLX_CACHE_LIMIT)
    mx.random.seed(42)


def read_manifest():
    path = owned(LAB / "corpus/manifest.json")
    if not path.exists() or path.stat().st_size > MIB:
        raise LabError("Prepare a corpus first.")
    manifest = json.loads(path.read_text())
    if digest(LAB / "corpus/tokenizer.model") != manifest["tokenizer_sha256"]:
        raise LabError("Tokenizer changed after preparation. Prepare the corpus again.")
    if not manifest["demo_only"] and "review_summary" not in manifest:
        raise LabError("A real imported sample needs explicit review. Use review-sample, then prepare-reviewed before training.")
    if "frozen_evaluation_sha256" in manifest:
        baseline_path = owned(LAB / "evaluations/baseline.json")
        if not baseline_path.exists():
            raise LabError("Evaluate the baseline on the frozen transcript cases before training this candidate.")
        baseline = json.loads(baseline_path.read_text())
        if baseline["evaluation_sha256"] != manifest["frozen_evaluation_sha256"] or baseline["tokenizer_sha256"] != manifest["tokenizer_sha256"]:
            raise LabError("The baseline evaluation must match the frozen cases and tokenizer.")
    if experiment_id := manifest.get("controlled_experiment"):
        if not re.fullmatch(r"expanded-[a-f0-9]{16}-v1", experiment_id):
            raise LabError("Invalid controlled-experiment identifier.")
        state = json.loads(owned(LAB / "experiments" / experiment_id / "run.json").read_text())
        if state.get("status") != "training" or state.get("owner_pid") != os.getpid():
            raise LabError("This prepared corpus belongs to a single controlled experiment. Inspect its report; do not repeat training on the frozen benchmark.")
        if digest(owned(LAB / "experiments" / experiment_id / "plan.json")) != manifest.get("controlled_plan_sha256"):
            raise LabError("The registered experiment plan changed after preparation.")
        for name, expected in manifest.get("token_files_sha256", {}).items():
            if name not in {"train.tokens", "validation.tokens", "test.tokens"} or digest(owned(LAB / "corpus" / name)) != expected:
                raise LabError("Prepared token files changed after the experiment was registered.")
    return manifest


class TokenFile:
    def __init__(self, path):
        self.path = owned(path)

    def __enter__(self):
        size = self.path.stat().st_size
        if not 0 < size <= TOKEN_LIMIT * 2 or size % 2:
            raise LabError("Invalid or oversized token file.")
        self.file = self.path.open("rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.tokens = memoryview(self.mapping).cast("H")
        return self

    def batch(self, size, context, rng, allow_short=False):
        if len(self.tokens) <= context:
            if not allow_short or len(self.tokens) < 2:
                raise LabError(f"The {self.path.stem} split needs more than {context} tokens.")
            context = len(self.tokens) - 1
        starts = [rng.randrange(len(self.tokens) - context) for _ in range(size)]
        return mx.array([list(self.tokens[start:start + context + 1]) for start in starts], dtype=mx.int32)

    def __exit__(self, *_):
        self.tokens.release(); self.mapping.close(); self.file.close()


def loss_function(model, batch):
    return nn.losses.cross_entropy(model(batch[:, :-1]), batch[:, 1:], reduction="mean")


def evaluate(model, data, context, batches=4):
    rng = random.Random(91)
    values = [loss_function(model, data.batch(2, context, rng, allow_short=True)).item() for _ in range(batches)]
    return sum(values) / len(values)


def current_checkpoint(slot="current"):
    if slot not in {"current", "previous"}:
        raise LabError("Choose the current or previous checkpoint slot.")
    root = owned(LAB / "checkpoints")
    retired = owned(root / "retired-checkpoint")
    current = owned(root / "current")
    if not current.exists() and (retired / "checkpoint.json").is_file():
        os.replace(retired, current)
    for name in (("current", "previous") if slot == "current" else ("previous",)):
        candidate = owned(root / name)
        if (candidate / "checkpoint.json").is_file():
            return candidate
    raise LabError("No local checkpoint exists. Run a short training experiment first.")


def load_checkpoint(slot="current"):
    checkpoint = current_checkpoint(slot)
    meta_path = checkpoint / "checkpoint.json"
    if meta_path.stat().st_size > MIB:
        raise LabError("Invalid checkpoint metadata.")
    meta = json.loads(meta_path.read_text())
    config = ModelConfig(**meta["config"]); config.validate()
    for name in ("model.safetensors", "optimizer.safetensors", "tokenizer.model"):
        path = owned(checkpoint / name)
        if path.stat().st_size > 128 * MIB or digest(path) != meta["files"][name]:
            raise LabError("Checkpoint size or integrity verification failed.")
    model = IgboLanguageModel(config)
    model.load_weights(str(checkpoint / "model.safetensors"))
    mx.eval(model.parameters())
    return checkpoint, meta, model


def save_checkpoint(model, optimizer, manifest, step, metrics, reason):
    # At most current + previous are retained, with one bounded atomic staging copy.
    parameters = sum(value.size for _, value in tree_flatten(model.parameters()))
    ensure_capacity(parameters * 12 + 8 * MIB)
    root = owned(LAB / "checkpoints"); root.mkdir(exist_ok=True)
    stage = owned(root / "checkpoint-staging")
    remove_generated(stage); stage.mkdir()
    try:
        model.save_weights(str(stage / "model.safetensors"))
        optimizer_state = {key: value if isinstance(value, mx.array) else mx.array(value)
                           for key, value in tree_flatten(optimizer.state)}
        mx.save_safetensors(str(stage / "optimizer.safetensors"), optimizer_state)
        shutil.copyfile(LAB / "corpus/tokenizer.model", stage / "tokenizer.model")
        meta = {
            "format": 1, "kind": "experimental-igbo-reply-model", "initialized": "random/from-scratch",
            "production_ready": False, "demo_only": manifest["demo_only"],
            "config": model.configuration(), "parameters": parameters, "step": step,
            "tokenizer_sha256": manifest["tokenizer_sha256"], "corpus_sha256": manifest["corpus_sha256"],
            "data_review": manifest.get("review_summary", {"type": "synthetic smoke fixtures"}),
            "frozen_evaluation_sha256": manifest.get("frozen_evaluation_sha256"),
            "controlled_experiment": manifest.get("controlled_experiment"), "assembly_sha256": manifest.get("assembly_sha256"),
            "finish_reason": reason, "metrics": metrics,
            "files": {name: digest(stage / name) for name in ("model.safetensors", "optimizer.safetensors", "tokenizer.model")},
        }
        write_json(stage / "checkpoint.json", meta)
        current, previous = owned(root / "current"), owned(root / "previous")
        baseline_path = owned(LAB / "evaluations/baseline.json")
        pinned = False
        if baseline_path.exists() and (previous / "checkpoint.json").exists():
            baseline = json.loads(baseline_path.read_text())
            previous_meta = json.loads((previous / "checkpoint.json").read_text())
            pinned = previous_meta["files"]["model.safetensors"] == baseline["model_sha256"]
        if pinned:
            retired = owned(root / "retired-checkpoint")
            if not current.exists() and (retired / "checkpoint.json").exists():
                os.replace(retired, current)
            remove_generated(retired)
            if current.exists():
                os.replace(current, retired)
            try:
                os.replace(stage, current)
            except BaseException:
                if retired.exists() and not current.exists():
                    os.replace(retired, current)
                raise
            remove_generated(retired)
            return meta
        remove_generated(previous)
        if current.exists():
            os.replace(current, previous)
        try:
            os.replace(stage, current)
        except BaseException:
            if previous.exists() and not current.exists():
                os.replace(previous, current)
            raise
        return meta
    finally:
        remove_generated(stage)


def train(steps=120, seconds=180, resume=False, cpu=False, batch_size=2, resume_slot="current"):
    if not 1 <= steps <= 2000 or not 5 <= seconds <= 600 or not 1 <= batch_size <= 4:
        raise LabError("Use 1–2000 steps, 5–600 seconds, and batch size 1–4 for this laptop pilot.")
    ensure_capacity(96 * MIB)
    configure_runtime(cpu)
    manifest = read_manifest()
    if experiment_id := manifest.get("controlled_experiment"):
        plan = json.loads(owned(LAB / "experiments" / experiment_id / "plan.json").read_text())
        if {"steps":steps,"seconds":seconds,"batch_size":batch_size,"resume_slot":resume_slot,"resume":resume,"cpu":cpu} != plan["training"]:
            raise LabError("Training settings differ from the registered experiment plan.")
    offset = 0
    if resume:
        checkpoint, previous, model = load_checkpoint(resume_slot)
        if previous["tokenizer_sha256"] != manifest["tokenizer_sha256"]:
            raise LabError("A rebuilt tokenizer requires fresh model training; omit --resume.")
        offset = previous["step"]
    else:
        model = IgboLanguageModel(ModelConfig(vocabulary=manifest["vocabulary_size"]))
        mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=0.0005, weight_decay=0.01)
    if resume:
        optimizer.state = tree_unflatten(list(mx.load(str(checkpoint / "optimizer.safetensors")).items()))
    parameters = sum(value.size for _, value in tree_flatten(model.parameters()))
    if parameters > MAX_PARAMETERS:
        raise LabError("Parameter budget exceeded.")
    print(f"Training {parameters:,} parameters from {'the saved checkpoint' if resume else 'random initialization'}.", flush=True)
    print("EXPERIMENTAL: demo fixtures are not evidence of Igbo conversation quality." if manifest["demo_only"]
          else "EXPERIMENTAL: evaluate with fluent speakers before voice-assistant use.", flush=True)
    gradient_function = nn.value_and_grad(model, loss_function)
    rng = random.Random(42 + offset)
    stop = {"requested": False}
    original_signals = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        original_signals[number] = signal.signal(number, lambda *_: stop.update(requested=True))
    started = time.monotonic(); reason = "step limit"; completed = 0; latest_loss = None
    metrics = {"parameters": parameters, "demo_only": manifest["demo_only"],
               "parent_model_sha256": previous["files"]["model.safetensors"] if resume else None,
               "parent_checkpoint_slot": resume_slot if resume else None}
    try:
        with ExitStack() as stack:
            train_data = stack.enter_context(TokenFile(LAB / "corpus/train.tokens"))
            validation = stack.enter_context(TokenFile(LAB / "corpus/validation.tokens"))
            test = stack.enter_context(TokenFile(LAB / "corpus/test.tokens"))
            initial_validation = evaluate(model, validation, model.config.context)
            initial_test = evaluate(model, test, model.config.context)
            metrics["validation_context_tokens"] = min(model.config.context, len(validation.tokens) - 1)
            metrics["test_context_tokens"] = min(model.config.context, len(test.tokens) - 1)
            metrics["initial_validation_loss"] = initial_validation
            metrics["initial_test_loss"] = initial_test
            metrics["holdout_method"] = "Fixed-seed sampled causal windows, four batches of two per split; not full-corpus or audio accuracy."
            print(f"Initial validation loss: {initial_validation:.4f}", flush=True)
            for index in range(steps):
                if stop["requested"] or time.monotonic() - started >= seconds:
                    reason = "interrupted" if stop["requested"] else "time limit"
                    break
                batch = train_data.batch(batch_size, model.config.context, rng)
                loss, gradients = gradient_function(model, batch)
                norm = mx.sqrt(sum(mx.sum(value * value) for _, value in tree_flatten(gradients)))
                multiplier = mx.minimum(1.0, 1.0 / (norm + 1e-6))
                optimizer.update(model, tree_map(lambda value: value * multiplier, gradients))
                mx.eval(loss, model.parameters(), optimizer.state)
                latest_loss = loss.item(); completed = index + 1
                if not math.isfinite(latest_loss):
                    raise LabError("Nonfinite training loss; the existing saved checkpoint was preserved.")
                if completed == 1 or completed % 10 == 0:
                    ensure_capacity()
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    peak = mx.get_peak_memory()
                    print(f"Step {offset + completed}: loss={latest_loss:.4f}, MLX peak={peak / MIB:.1f} MiB, process peak={rss / MIB:.1f} MiB", flush=True)
                    if rss > 3 * GIB or peak > MLX_MEMORY_GUIDELINE:
                        reason = "observed memory budget"
                        break
                if completed % 50 == 0:
                    save_checkpoint(model, optimizer, manifest, offset + completed, {**metrics, "train_loss": latest_loss}, "periodic checkpoint")
            metrics.update({"train_loss": latest_loss, "validation_loss": evaluate(model, validation, model.config.context),
                            "test_loss": evaluate(model, test, model.config.context), "completed_steps": completed,
                            "elapsed_seconds": round(time.monotonic() - started, 2),
                            "mlx_peak_bytes": mx.get_peak_memory(), "process_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            meta = save_checkpoint(model, optimizer, manifest, offset + completed, metrics, reason)
            write_json(LAB / "last_training.json", meta)
            print(json.dumps({"step": meta["step"], "finish_reason": reason, **metrics}, indent=2), flush=True)
            return meta
    finally:
        for number, handler in original_signals.items():
            signal.signal(number, handler)
        mx.clear_cache()


def reply(transcript, maximum=64, cpu=False, checkpoint_slot="current"):
    from .corpus import MARKERS, normalized
    if not 1 <= maximum <= 128:
        raise LabError("Generate at most 128 new tokens in this pilot.")
    transcript = normalized(transcript)
    if len(transcript) > 512 or any(marker in transcript for marker in MARKERS):
        raise LabError("Use a transcript under 512 characters, without internal role markers.")
    ensure_capacity(); configure_runtime(cpu)
    checkpoint, meta, model = load_checkpoint(checkpoint_slot)
    tokenizer = spm.SentencePieceProcessor(model_file=str(checkpoint / "tokenizer.model"))
    ids = tokenizer.encode("<|user|>" + transcript + "<|assistant|>")
    generated = []
    stops = {tokenizer.eos_id(), *(tokenizer.piece_to_id(marker) for marker in MARKERS)}
    for _ in range(maximum):
        logits = model(mx.array([ids[-model.config.context:]], dtype=mx.int32))[0, -1]
        token = int(mx.argmax(logits).item())
        if token in stops:
            break
        generated.append(token); ids.append(token)
    return {"experimental": True, "demo_only": meta["demo_only"], "checkpoint_slot": checkpoint_slot, "checkpoint_step": meta["step"],
            "reply": tokenizer.decode(generated), "note": "Research output; not connected to Anyanwụ's live voice replies."}
