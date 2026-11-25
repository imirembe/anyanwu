"""Combine reviewed batches into an immutable, inspectable text corpus.

This does not prepare tokens, train, or replace the measured corpus/checkpoints.
"""
import hashlib
import json
import os
import re

from .corpus import group_split
from .importer import sample_directory
from .review import curation_directory, load_evaluation, load_jsonl, prompt_key
from .resources import LAB, MIB, LabError, ensure_capacity, owned, project_input, remove_generated, write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_sample(value):
    match = re.fullmatch(r"([a-z][a-z0-9-]*)(?:@([0-9]{1,7}))?", value)
    if not match:
        raise LabError("Use source or source@offset, for example aya-igbo@2000.")
    source, offset = match[1], int(match[2] or 0)
    sample_directory(source, offset)
    return source, offset


def record_prompt(record):
    text = record["text"]
    if text.startswith("<|user|>") and "<|assistant|>" in text:
        return prompt_key(text[len("<|user|>"):].split("<|assistant|>", 1)[0])
    return None


def assemble_reviewed(samples, evaluation_path):
    if not 1 <= len(samples) <= 12:
        raise LabError("Assemble 1–12 explicitly selected reviewed batches.")
    selectors = [parse_sample(value) for value in samples]
    if len(set(selectors)) != len(selectors):
        raise LabError("Choose each reviewed batch once.")
    cases_path = project_input(evaluation_path)
    protected = {prompt_key(case["prompt"]) for case in load_evaluation(cases_path)}
    cases_hash = sha256(cases_path)
    parents, records, exclusions = [], [], []
    for source, offset in selectors:
        folder = curation_directory(source, offset)
        receipt_path = owned(folder / "review_report.json")
        if not receipt_path.is_file() or receipt_path.stat().st_size > MIB:
            raise LabError("Every selected batch needs a bounded, completed review receipt.")
        report = json.loads(receipt_path.read_text())
        path = owned(folder / "reviewed.jsonl")
        if report.get("source") != source or report.get("batch_start_offset", 0) != offset:
            raise LabError("Review receipt source/batch does not match its selector.")
        if report.get("evaluation_sha256") != cases_hash:
            raise LabError("All batches must use the same frozen evaluation cases.")
        if not path.is_file() or path.stat().st_size > 5*MIB or sha256(path) != report.get("reviewed_corpus_sha256"):
            raise LabError("A reviewed batch changed after its review receipt was written.")
        raw_folder = sample_directory(source, offset)
        raw_manifest = json.loads(owned(raw_folder / "manifest.json").read_text())
        raw_path = owned(raw_folder / "candidates.jsonl")
        if raw_path.stat().st_size > 5*MIB or sha256(raw_path) != raw_manifest.get("candidates_sha256"):
            raise LabError("An original source sample changed after import.")
        raw_by_id = {row["id"]:row for row in load_jsonl(raw_path)}
        selected = load_jsonl(path)
        for record in selected:
            review = record.get("review", {})
            original = raw_by_id.get(record.get("id"), {})
            if (review.get("decision") not in {"accept", "edit"} or review.get("type") not in {"AI", "human"}
                    or review.get("confidence") not in {"high", "medium"} or not review.get("reviewer") or not review.get("reason")
                    or record.get("source_content_sha256") != original.get("content_sha256") or not original):
                raise LabError("Every assembled record must retain explicit accepted review and original-content provenance.")
            split = record.get("split", group_split(record["source"] + "/" + record["group"]))
            if split not in {"train", "validation", "test"}:
                raise LabError("An assembled record has an invalid split.")
            records.append({**record, "split":split})
        parents.append({"sample":source + (f"@{offset}" if offset else ""),
                        "review_receipt_sha256":sha256(receipt_path), "reviewed_corpus_sha256":sha256(path),
                        "raw_candidates_sha256":sha256(raw_path), "source_revision":raw_manifest.get("revision"),
                        "restrictions":raw_manifest.get("restrictions", "See preserved source card."),
                        "accepted_records":len(selected)})
    if len(records) > 1000:
        raise LabError("Keep this assembled pilot under 1,000 reviewed records.")

    # Keep each existing source group in its original split. If the same prompt
    # appears across splits, exclude lower-priority groups instead of moving a
    # held-out example into training. Entire speaker/source groups stay together.
    priority = {"test":0, "validation":1, "train":2}
    groups, prompt_groups, blocked = {}, {}, set()
    for record in records:
        group = record["source"] + "/" + record["group"]
        split = record["split"]
        if group in groups and groups[group] != split:
            raise LabError("A source/speaker group spans training and evaluation splits.")
        groups[group] = split
        prompt = record_prompt(record)
        if prompt in protected:
            blocked.add(group)
        if prompt:
            prompt_groups.setdefault(prompt, set()).add(group)
    for members in prompt_groups.values():
        best = min(priority[groups[group]] for group in members)
        blocked.update(group for group in members if priority[groups[group]] > best)
    seen_text, seen_prompts, seen_ids, kept = set(), set(), set(), []
    for record in sorted(records, key=lambda row:priority[row["split"]]):
        group = record["source"] + "/" + record["group"]
        prompt = record_prompt(record)
        text_hash = hashlib.sha256(record["text"].encode()).hexdigest()
        identifier = (record["source"], record["id"])
        reason = ("Protected prompt or prompt overlap across source splits; whole source group excluded." if group in blocked else
                  "Duplicate reviewed text, prompt or source record." if text_hash in seen_text or identifier in seen_ids or (prompt and prompt in seen_prompts) else None)
        if reason:
            exclusions.append({"id":record["id"], "source":record["source"], "reason":reason})
            continue
        seen_text.add(text_hash); seen_ids.add(identifier)
        if prompt:
            seen_prompts.add(prompt)
        kept.append(record)
    counts = {split:sum(row["split"] == split for row in kept) for split in priority}
    issues = []
    if len(kept) < 12:
        issues.append("Fewer than 12 reviewed records remain.")
    if any(not value for value in counts.values()):
        issues.append("Supply accepted examples in every train/validation/test split.")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept).encode()
    if len(payload) > 5*MIB:
        raise LabError("The assembled text exceeds this pilot's 5 MiB output cap.")
    records_hash = hashlib.sha256(payload).hexdigest()
    identity = hashlib.sha256(json.dumps({"parents":parents,"cases":cases_hash,"records":records_hash},sort_keys=True).encode()).hexdigest()
    destination = owned(LAB / "curation/assemblies" / identity[:16])
    manifest = {"format":1,"assembly_sha256":identity,"records_sha256":records_hash,
                "evaluation_sha256":cases_hash,"parents":parents,"records":len(kept),"splits":counts,
                "excluded_records":exclusions,"licenses_declared":sorted({row["license"] for row in kept}),
                "ai_reviewed":sum(row["review"]["type"] == "AI" for row in kept),
                "human_validated":bool(kept) and all(row["review"]["type"] == "human" for row in kept),
                "training_ready":not issues,"training_readiness_issues":issues,
                "active_corpus_replaced":False,"training_performed":False,"audio_evaluated":False,
                "limitation":"Structural preparation checks only. Preserve per-source attribution and restrictions. AI review does not establish fluent Igbo or voice-assistant quality."}
    ensure_capacity(len(payload) + 2*MIB)
    if destination.exists():
        if sha256(owned(destination / "records.jsonl")) != records_hash or json.loads(owned(destination / "manifest.json").read_text()) != manifest:
            raise LabError("An existing immutable assembly changed; refusing to overwrite it.")
    else:
        stage = owned(LAB / "curation/assembly-staging")
        remove_generated(stage); stage.mkdir(parents=True)
        try:
            (stage / "records.jsonl").write_bytes(payload)
            write_json(stage / "manifest.json", manifest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, destination)
        finally:
            remove_generated(stage)
    print(json.dumps({"assembly":str(destination),"records":len(kept),"splits":counts,"excluded":len(exclusions),
                      "training_ready":not issues,"training_performed":False},ensure_ascii=False,indent=2))
    return manifest
