"""Frozen transcript-response comparisons, separate from ASR/TTS audio quality."""
import hashlib
import json
import math

from .resources import LAB, MIB, LabError, ensure_capacity, owned, project_input, write_json
from .review import load_evaluation, prompt_key


def evaluate_checkpoint(cases_path, label="baseline", replace_baseline=False, report_directory=None):
    folder = owned(report_directory if report_directory is not None else LAB / "evaluations")
    if label not in {"baseline", "candidate"}:
        raise LabError("Use baseline or candidate; the pilot retains only these two evaluation reports.")
    if label == "baseline" and owned(folder / "baseline.json").exists() and not replace_baseline:
        raise LabError("The comparison baseline is frozen. Use --replace-baseline only when intentionally starting a new comparison.")
    import mlx.core as mx
    import mlx.nn as nn
    import sentencepiece as spm
    from .runner import configure_runtime, load_checkpoint
    ensure_capacity(2 * MIB)
    cases_path = project_input(cases_path)
    cases = load_evaluation(cases_path)
    configure_runtime()
    checkpoint, metadata, model = load_checkpoint()
    tokenizer = spm.SentencePieceProcessor(model_file=str(checkpoint / "tokenizer.model"))
    marker = tokenizer.piece_to_id("<|assistant|>")
    if marker == tokenizer.unk_id():
        raise LabError("The checkpoint lacks the assistant conversation marker.")
    results = []
    for case in cases:
        full = tokenizer.encode(f"<|user|>{case['prompt']}<|assistant|>{case['expected']}<|end|>")
        if len(full) - 1 > model.config.context:
            raise LabError("An evaluation turn exceeds this model's context. Shorten the frozen benchmark before comparing both models.")
        boundary = full.index(marker) + 1
        logits = model(mx.array([full[:-1]], dtype=mx.int32))[0]
        losses = nn.losses.cross_entropy(logits, mx.array(full[1:], dtype=mx.int32), reduction="none")
        # Score only the reference reply + end marker, never the user's prompt.
        answer_loss = mx.mean(losses[boundary - 1:]).item()
        if not math.isfinite(answer_loss):
            raise LabError("Nonfinite evaluation loss.")
        prefix = tokenizer.encode(f"<|user|>{case['prompt']}<|assistant|>")
        generated = []
        stop_ids = {tokenizer.eos_id(), *(tokenizer.piece_to_id(token) for token in ("<|user|>", "<|assistant|>", "<|end|>"))}
        reason = "token limit"
        for _ in range(48):
            token = int(mx.argmax(model(mx.array([prefix[-model.config.context:]], dtype=mx.int32))[0, -1]).item())
            if token in stop_ids:
                reason = "end/role token"; break
            generated.append(token); prefix.append(token)
        results.append({"id": case["id"], "language": case["language"], "prompt": case["prompt"],
                        "reference": case["expected"], "generated": tokenizer.decode(generated),
                        "reference_reply_loss": answer_loss, "generation_finish_reason": reason})
        print(f"Evaluated {case['id']}: reference-reply loss {answer_loss:.4f}", flush=True)
    means = {language: sum(row["reference_reply_loss"] for row in results if row["language"] == language)
             / sum(row["language"] == language for row in results) for language in ("en", "ig", "mixed")}
    report = {"format": 1, "label": label, "modality": "text transcripts and replies; no audio benchmark",
              "evaluation_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
              "tokenizer_sha256": metadata["tokenizer_sha256"], "model_sha256": metadata["files"]["model.safetensors"],
              "model_config": metadata["config"], "checkpoint_step": metadata["step"],
              "model_demo_only": metadata["demo_only"], "evaluation_review": "Codex/AI; not native-speaker validated",
              "cases": results, "mean_reference_reply_loss": sum(row["reference_reply_loss"] for row in results) / len(results),
              "mean_by_input_language": means, "production_ready": False,
              "limitation": "Reply likelihood alone does not establish naturalness, factuality, instruction following or spoken performance. Inspect generated replies."}
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"{label}.json", report)
    print(json.dumps({key:value for key,value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    mx.clear_cache()
    return report


def compare_evaluations(report_directory=None):
    folder = owned(report_directory if report_directory is not None else LAB / "evaluations")
    reports = []
    for label in ("baseline", "candidate"):
        path = owned(folder / f"{label}.json")
        if not path.exists() or path.stat().st_size > 2 * MIB:
            raise LabError("Evaluate both baseline and candidate checkpoints first.")
        reports.append(json.loads(path.read_text()))
    baseline, candidate = reports
    for key in ("evaluation_sha256", "tokenizer_sha256", "model_config"):
        if baseline[key] != candidate[key]:
            raise LabError("A comparison requires the same frozen cases, tokenizer and model configuration.")
    difference = candidate["mean_reference_reply_loss"] - baseline["mean_reference_reply_loss"]
    report = {"evaluation_sha256": baseline["evaluation_sha256"],
              "baseline_model_sha256": baseline["model_sha256"], "candidate_model_sha256": candidate["model_sha256"],
              "baseline_reply_loss": baseline["mean_reference_reply_loss"], "candidate_reply_loss": candidate["mean_reference_reply_loss"],
              "candidate_minus_baseline": difference, "lower_reply_loss": difference < 0,
              "language_comparison": {language: {"baseline": baseline["mean_by_input_language"][language],
                                                   "candidate": candidate["mean_by_input_language"][language]}
                                      for language in ("en", "ig", "mixed")},
              "review_type": "AI-authored reference set; generation-quality review still required",
              "audio_evaluated": False, "production_ready": False,
              "meaning": "Lower loss means the reference reply tokens were more likely; it does not prove a better voice assistant."}
    write_json(folder / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
