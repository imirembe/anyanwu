"""One registered, bounded training run per immutable reviewed assembly."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from .assembly import assemble_reviewed
from .resources import LAB, MIB, LabError, ensure_capacity, owned, project_input, write_json
from .review import load_evaluation, load_jsonl


TRAINING = {"steps":120, "seconds":180, "batch_size":2, "resume_slot":"previous", "resume":True, "cpu":False}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded_json(path, limit=2*MIB):
    path = owned(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise LabError("A required bounded experiment artifact is missing or oversized.")
    return json.loads(path.read_text())


def assembly_identity(path):
    folder = owned(Path(path).resolve())
    root = owned(LAB / "curation/assemblies").resolve()
    manifest = bounded_json(folder / "manifest.json")
    identity = manifest.get("assembly_sha256", "")
    if not re.fullmatch(r"[a-f0-9]{64}", identity) or folder.parent != root or folder.name != identity[:16]:
        raise LabError("Choose an immutable assembly from this lab's curations directory.")
    records = owned(folder / "records.jsonl")
    if records.stat().st_size > 5*MIB or sha256(records) != manifest.get("records_sha256"):
        raise LabError("The assembled records changed; no experiment was started.")
    return folder, manifest


def preflight(path, cases_path):
    folder, assembly = assembly_identity(path)
    if not any(r.get("text","").startswith("<|user|>") and "<|assistant|>" in r["text"] for r in load_jsonl(folder/"records.jsonl")):
        raise LabError("This corpus has no assistant reply targets. Request-only/language text needs a separate intent or language-model experiment plan.")
    cases_path = project_input(cases_path)
    load_evaluation(cases_path)
    if sha256(cases_path) != assembly["evaluation_sha256"]:
        raise LabError("The experiment must preserve the assembly's frozen evaluation cases.")
    rebuilt = assemble_reviewed([item["sample"] for item in assembly["parents"]], cases_path)
    if rebuilt != assembly or not assembly.get("training_ready"):
        raise LabError("The assembly is not structurally ready for this experiment.")
    baseline = bounded_json(LAB / "evaluations/baseline.json")
    checkpoint = bounded_json(LAB / "checkpoints/previous/checkpoint.json")
    for name in ("model.safetensors", "optimizer.safetensors", "tokenizer.model"):
        file = owned(LAB / "checkpoints/previous" / name)
        if file.stat().st_size > 128*MIB or sha256(file) != checkpoint["files"][name]:
            raise LabError("The preserved baseline checkpoint failed its integrity check.")
    if (baseline["model_sha256"] != checkpoint["files"]["model.safetensors"]
            or baseline["evaluation_sha256"] != sha256(cases_path)
            or baseline["tokenizer_sha256"] != checkpoint["tokenizer_sha256"]
            or baseline["model_config"] != checkpoint["config"]
            or sha256(owned(LAB / "corpus/tokenizer.model")) != checkpoint["tokenizer_sha256"]):
        raise LabError("Baseline model, tokenizer, architecture and frozen cases must match before training.")
    return folder, assembly, baseline, checkpoint


def run_experiment(path, cases_path):
    folder, assembly = assembly_identity(path)
    cases_path = project_input(cases_path)
    if sha256(cases_path) != assembly["evaluation_sha256"]:
        raise LabError("The experiment must preserve the assembly's frozen evaluation cases.")
    experiment_id = "expanded-" + assembly["assembly_sha256"][:16] + "-v1"
    destination = owned(LAB / "experiments" / experiment_id)
    if destination.exists():
        state = bounded_json(destination / "run.json")
        if state.get("status") == "complete":
            result = bounded_json(destination / "result.json")
            print(json.dumps({"reused_completed_experiment":True, "experiment":experiment_id, "result":result}, ensure_ascii=False, indent=2))
            return result
        raise LabError("This experiment already started. Inspect its saved run status; training is not repeated automatically.")
    folder, assembly, baseline, checkpoint = preflight(folder, cases_path)
    ensure_capacity(100*MIB)
    destination.mkdir(parents=True)
    plan = {"format":1, "experiment_id":experiment_id, "registered_at":datetime.now(timezone.utc).isoformat(),
            "assembly_sha256":assembly["assembly_sha256"], "records_sha256":assembly["records_sha256"],
            "evaluation_sha256":assembly["evaluation_sha256"], "baseline_model_sha256":baseline["model_sha256"],
            "baseline_report_sha256":sha256(LAB / "evaluations/baseline.json"),
            "baseline_step":checkpoint["step"], "model_config":checkpoint["config"], "tokenizer_sha256":checkpoint["tokenizer_sha256"],
            "training":dict(TRAINING), "optimizer":{"name":"AdamW", "learning_rate":0.0005, "weight_decay":0.01, "gradient_clip_norm":1.0},
            "mlx_seed":42, "sampling_seed":42+checkpoint["step"],
            "holdout_policy":"Fixed-seed sampled-window losses before/after; frozen transcript replies scored once after training. No tuning/retries based on these scores.",
            "implementation_sha256":{name:sha256(Path(__file__).parent/name) for name in ("experiments.py","runner.py","model.py","corpus.py","evaluation.py")},
            "checkpoint_retention":"Preserved baseline in previous; this run replaces current. No third completed checkpoint.",
            "audio_evaluated":False, "production_ready":False}
    write_json(destination / "plan.json", plan)
    before = {"corpus_manifest":bounded_json(LAB / "corpus/manifest.json"),
              "current_checkpoint_metadata":bounded_json(LAB / "checkpoints/current/checkpoint.json")}
    for name in ("baseline", "candidate", "comparison", "generation_review"):
        original = owned(LAB / "evaluations" / (name+".json"))
        if original.exists():
            before[name] = bounded_json(original)
    write_json(destination / "before.json", before)
    state = {"experiment_id":experiment_id, "status":"preparing", "owner_pid":os.getpid()}
    write_json(destination / "run.json", state)
    try:
        from .corpus import prepare
        from .runner import train
        from .evaluation import evaluate_checkpoint, compare_evaluations
        prepared = prepare(folder / "records.jsonl")
        if prepared["tokenizer_sha256"] != plan["tokenizer_sha256"]:
            raise LabError("Token IDs changed during preparation; training was refused.")
        prepared.update({"assembly_sha256":assembly["assembly_sha256"], "controlled_experiment":experiment_id,
                         "controlled_plan_sha256":sha256(destination / "plan.json"),
                         "frozen_evaluation_sha256":assembly["evaluation_sha256"],
                         "token_files_sha256":{name:sha256(LAB / "corpus" / name) for name in ("train.tokens","validation.tokens","test.tokens")},
                         "review_note":"AI-reviewed/edited text; no native-speaker or audio validation."})
        write_json(LAB / "corpus/manifest.json", prepared)
        write_json(destination / "prepared_corpus.json", prepared)
        state["status"] = "training"; write_json(destination / "run.json", state)
        trained = train(**TRAINING)
        if trained["metrics"]["parent_model_sha256"] != plan["baseline_model_sha256"]:
            raise LabError("The run did not start from its registered baseline.")
        for name, expected in checkpoint["files"].items():
            if sha256(owned(LAB / "checkpoints/previous" / name)) != expected:
                raise LabError("The preserved baseline changed during checkpoint rotation.")
        state["status"] = "evaluating"; write_json(destination / "run.json", state)
        reports = owned(destination / "evaluations"); reports.mkdir()
        write_json(reports / "baseline.json", baseline)  # Reuse identical model/cases/tokenizer scores; do not retune them.
        candidate = evaluate_checkpoint(cases_path, "candidate", report_directory=reports)
        if candidate["model_sha256"] != trained["files"]["model.safetensors"]:
            raise LabError("The evaluated candidate does not match this training run.")
        comparison = compare_evaluations(reports)
        result = {"experiment_id":experiment_id, "plan_sha256":sha256(destination / "plan.json"),
                  "assembly_sha256":assembly["assembly_sha256"], "training":trained,
                  "comparison":comparison, "generation_review_status":"pending AI review of all saved outputs",
                  "production_ready":False, "audio_evaluated":False,
                  "stop_policy":"One registered run completed; do not add steps or tune against the frozen benchmark."}
        write_json(destination / "result.json", result)
        state.update({"status":"complete", "completed_at":datetime.now(timezone.utc).isoformat()})
        write_json(destination / "run.json", state)
        write_json(LAB / "latest_experiment.json", {"experiment_id":experiment_id, "status":"complete", "directory":str(destination),
                                                   "reports_directory":str(reports), "model_sha256":candidate["model_sha256"]})
        print(json.dumps({"experiment_id":experiment_id,"steps":trained["metrics"]["completed_steps"],
                          "baseline_reply_loss":comparison["baseline_reply_loss"],"candidate_reply_loss":comparison["candidate_reply_loss"],
                          "report":str(destination / "result.json"),"production_ready":False},indent=2))
        return result
    except BaseException as error:
        state.update({"status":"failed", "error_type":type(error).__name__, "error":str(error)[:1000]})
        try:
            write_json(destination / "run.json", state)
        except OSError:
            pass
        raise
