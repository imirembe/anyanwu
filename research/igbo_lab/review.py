"""Apply explicit review decisions with a traceable source-to-training record."""
import hashlib
import json
import re
import unicodedata

from .corpus import MARKERS, normalized
from .importer import SOURCES, sample_directory
from .resources import LAB, MIB, LabError, ensure_capacity, owned, project_input, write_json


def prompt_key(text):
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def load_jsonl(path, limit=5 * MIB):
    path = project_input(path)
    if path.stat().st_size > limit:
        raise LabError("This review input exceeds its bounded text-file size.")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LabError("Review inputs must contain JSON objects.")
                records.append(value)
                if len(records) > 1000:
                    raise LabError("Review at most 1,000 records in this pilot.")
    return records


def load_evaluation(path):
    cases = load_jsonl(path, MIB)
    if not 6 <= len(cases) <= 100:
        raise LabError("Use a separate evaluation set of 6–100 conversation prompts.")
    ids, prompts = set(), set()
    for case in cases:
        for field in ("id", "prompt", "expected", "group", "language", "reviewer", "review_type"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise LabError(f"Evaluation cases need a nonempty {field}.")
        if case["id"] in ids or prompt_key(case["prompt"]) in prompts:
            raise LabError("Evaluation IDs/prompts must be unique.")
        if case["language"] not in {"en", "ig", "mixed"} or case["review_type"] not in {"AI", "human"}:
            raise LabError("Invalid evaluation language or reviewer type.")
        if len(case["prompt"]) > 512 or len(case["expected"]) > 512 or any(marker in case["prompt"] + case["expected"] for marker in MARKERS):
            raise LabError("Keep evaluation turns short and free of internal role markers.")
        ids.add(case["id"]); prompts.add(prompt_key(case["prompt"]))
    if {case["language"] for case in cases} != {"en", "ig", "mixed"}:
        raise LabError("The voice-assistant evaluation must include English, Igbo and mixed transcripts.")
    return cases


def curation_directory(source, offset=0):
    if source not in SOURCES:
        raise LabError("Unknown imported sample.")
    # Preserve the measured public pilot's existing paths and corpus.
    sample_directory(source, offset)  # Validate the batch selector as well.
    base = owned(LAB / "curation" if source == "public-igbo" else LAB / "curation" / source)
    return owned(base / "batches" / f"offset-{offset:07d}") if offset else base


def training_readiness(records, source):
    issues = []
    if len(records) < 12:
        issues.append("Fewer than 12 reviewed records survived; this remains a review sample.")
    if source == "africanvoice" or SOURCES[source]["kind"] in {"intent","mixed-text"}:
        for split in ("train", "validation", "test"):
            if not any(record.get("split") == split for record in records):
                issues.append(f"No accepted records remain in the official {split} split.")
        groups = {}
        for record in records:
            group, split = record["group"], record.get("split")
            if group in groups and groups[group] != split:
                issues.append("An accepted speaker group spans official splits." if source == "africanvoice" else "An accepted utterance group spans official splits.")
                break
            groups[group] = split
    return issues


def compile_review(source, decisions_path, evaluation_path, offset=0):
    if source not in SOURCES:
        raise LabError("Unknown imported sample.")
    directory = sample_directory(source, offset)
    raw_path = owned(directory / "candidates.jsonl")
    manifest = json.loads(owned(directory / "manifest.json").read_text())
    if raw_path.stat().st_size > 5 * MIB or hashlib.sha256(raw_path.read_bytes()).hexdigest() != manifest["candidates_sha256"]:
        raise LabError("The raw source sample changed; its review no longer matches.")
    candidates = {record["id"]: record for record in load_jsonl(raw_path)}
    decision_path = project_input(decisions_path)
    decisions = load_jsonl(decision_path)
    cases = load_evaluation(evaluation_path)
    heldout_prompts = {prompt_key(case["prompt"]) for case in cases}
    heldout_groups = {record["group"] for record in candidates.values()
                      if "input" in record["content"] and prompt_key(record["content"]["input"]) in heldout_prompts}
    seen, accepted, rejected, edits = set(), [], [], 0
    for decision in decisions:
        identifier = decision.get("id")
        if identifier not in candidates or identifier in seen:
            raise LabError("Review IDs must identify unique records in this exact source sample.")
        seen.add(identifier)
        candidate = candidates[identifier]
        if candidate["content_sha256"] != decision.get("content_sha256"):
            raise LabError("A review decision targets different original content.")
        if decision.get("decision") not in {"accept", "edit", "reject"} or decision.get("review_type") not in {"AI", "human"}:
            raise LabError("Every decision needs accept/edit/reject and an explicit AI/human review type.")
        if not decision.get("reviewer") or not decision.get("reason") or decision.get("confidence") not in {"high", "medium", "low"}:
            raise LabError("Record reviewer, reason and confidence for every review decision.")
        if decision["decision"] == "reject" or decision["confidence"] == "low":
            rejected.append({"id": identifier, "reason": decision["reason"]})
            continue
        content = candidate["content"]
        use = decision.get("content_use", "assistant-pair" if "input" in content else "language-reference")
        if use not in {"assistant-pair", "language-reference"} or ("input" not in content and use == "assistant-pair"):
            raise LabError("Choose assistant-pair for a question/reply or language-reference for plain source text.")
        family = decision.get("family_group")
        if family is not None:
            if ("input" not in content or "split" in candidate or not isinstance(family, str)
                    or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", family)
                    or not isinstance(decision.get("family_reason"), str) or not decision["family_reason"].strip()):
                raise LabError("Semantic families need a short label and reason; never replace an official speaker/split group.")
        if "input" in content:
            user = normalized(decision.get("input", content["input"]))
            answer = normalized(decision.get("output", content["output"]))
            changed = user != content["input"] or answer != content["output"]
            if candidate["group"] in heldout_groups or prompt_key(user) in heldout_prompts:
                rejected.append({"id": identifier, "reason": "Evaluation prompt overlap; excluded before training."})
                continue
            if any(marker in user + answer for marker in MARKERS):
                raise LabError("Source text cannot contain internal conversation role markers.")
            if use == "language-reference":
                if decision["decision"] != "edit" or "text" not in decision or "input" in decision or "output" in decision:
                    raise LabError("Formatting a published pair as reference text needs an explicit edit/text, without replacement input/output.")
                text = normalized(decision["text"])
                changed = True  # Role removal/formatting is a disclosed derivative, even when words are unchanged.
            else:
                if "text" in decision:
                    raise LabError("Use input/output edits for an assistant pair, not an ignored text field.")
                text = f"<|user|>{user}<|assistant|>{answer}<|end|>"
        else:
            text = normalized(decision.get("text", content["text"]))
            changed = text != content["text"]
            if SOURCES[source]["kind"] in {"intent","mixed-text"} and (prompt_key(content["text"]) in heldout_prompts or prompt_key(text) in heldout_prompts):
                rejected.append({"id":identifier,"reason":"Evaluation prompt overlap; excluded before training."})
                continue
        if decision["decision"] == "accept" and changed:
            raise LabError("Changed text must be marked edit, not accept.")
        if use == "language-reference" and any(marker in text for marker in MARKERS):
            raise LabError("Plain language references cannot contain internal conversation role markers.")
        edits += int(changed)
        group = hashlib.sha256((candidate["source"] + "/review-family/" + family).encode()).hexdigest()[:24] if family else candidate["group"]
        accepted.append({"id": identifier, "group": group, "source": candidate["source"],
                         "license": candidate["license"], "text": text, "demo_only": False,
                         "source_revision": candidate["source_revision"], "source_row_index": candidate["source_row_index"],
                         "source_url": candidate["source_url"],
                         "source_revision_note": candidate.get("source_revision_note", "Hub head seen before/after; indexed row revision not independently guaranteed."),
                         "origin": candidate["origin"], "source_content_sha256": candidate["content_sha256"],
                         "review": {"type": decision["review_type"], "reviewer": decision["reviewer"],
                                    "decision": decision["decision"], "reason": decision["reason"],
                                    "confidence": decision["confidence"], "human_validated": decision["review_type"] == "human"}})
        if source == "africanvoice":
            accepted[-1]["text_derivation"] = ("AI-edited speech text; not an audio-verified ASR target."
                                                if decision["decision"] == "edit" else "Source transcript; not audio-verified.")
        elif SOURCES[source]["kind"] == "intent":
            accepted[-1]["text_derivation"] = ("AI-edited published user request; original intent label retained; no assistant reply target or audio verification."
                                                if decision["decision"] == "edit" else "Published user request with source intent label; no assistant reply target or audio verification.")
        elif SOURCES[source]["kind"] == "mixed-text":
            accepted[-1]["text_derivation"] = "Published scripted Igbo-English customer-service text; AI-reviewed/format-edited, not spontaneous or audio-verified speech and no assistant reply target."
        elif "input" in content and use == "language-reference":
            accepted[-1]["text_derivation"] = "AI-formatted published language reference; not a user/assistant conversation or native-speaker validated."
        elif decision["decision"] == "edit":
            accepted[-1]["text_derivation"] = "AI-edited prompt/response; not native-speaker validated."
        if "content_use" in decision:
            accepted[-1]["content_use"] = use
        if family:
            accepted[-1]["source_group"] = candidate["group"]
            accepted[-1]["review"].update({"family_group":family, "family_reason":decision["family_reason"]})
        for field in ("supporting_sources", "reused_decision"):
            if field in decision:
                accepted[-1]["review"][field] = decision[field]
        for field in ("source_language_code", "source_annotation_type", "source_intent", "source_example_id", "source_config", "source_file", "source_language_note"):
            if field in candidate:
                accepted[-1][field] = candidate[field]
        if "source_slot_status" in candidate:
            for field in ("source_slot_status","source_slots","source_slot_text_sha256","source_slot_offset_unit","source_slot_alignment_valid","source_slot_semantics_reviewed","source_spans_sha256"):
                if field in candidate:accepted[-1][field]=candidate[field]
            accepted[-1]["slots_usable_with_reviewed_text"]=candidate["source_slot_alignment_valid"] and not changed
            accepted[-1]["slot_reference"]="Original imported content.text. Edits invalidate direct offsets into reviewed text; semantic labels have not been independently verified."
        for field in ("source_intent_id","source_example_id_kind"):
            if field in candidate:accepted[-1][field]=candidate[field]
        for field in ("request_scope","language_observation","needs_clarification"):
            if field in decision:accepted[-1]["review"][field]=decision[field]
        if "split" in candidate:
            accepted[-1]["split"] = candidate["split"]
    readiness = training_readiness(accepted, source)
    destination = curation_directory(source, offset); destination.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in accepted).encode()
    ensure_capacity(len(payload) + MIB)
    output = owned(destination / "reviewed.jsonl")
    temporary = owned(destination / "reviewed.tmp")
    try:
        temporary.write_bytes(payload); temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"source": source, "batch_start_offset": offset, "candidate_records": len(candidates), "reviewed_records": len(seen),
              "accepted_records": len(accepted), "edited_records": edits, "rejected_records": rejected,
              "pending_records": len(candidates) - len(seen),
              "accepted_counts_by_use": {use:sum(record.get("content_use", "assistant-pair" if record["text"].startswith("<|user|>") else "language-reference") == use for record in accepted)
                                         for use in ("assistant-pair", "language-reference")},
              "ai_reviewed": sum(record["review"]["type"] == "AI" for record in accepted),
              "human_validated": bool(accepted) and all(record["review"]["human_validated"] for record in accepted),
              "training_ready": not readiness, "training_readiness_issues": readiness,
              "accepted_counts_by_split": ({split:sum(record.get("split") == split for record in accepted) for split in ("train", "validation", "test")}
                                            if source == "africanvoice" or SOURCES[source]["kind"] in {"intent","mixed-text"} else {}),
              "review_file_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
              "reviewed_corpus_sha256": hashlib.sha256(payload).hexdigest(),
              "evaluation_sha256": hashlib.sha256(project_input(evaluation_path).read_bytes()).hexdigest(),
              "evaluation_cases": len(cases), "training_prompts_overlap_evaluation": False,
              "limitation": "AI review is not native-speaker validation. Source origin/permissions remain as declared; edited examples retain provenance."}
    write_json(destination / "review_report.json", report)
    print(json.dumps({key:value for key,value in report.items() if key != "rejected_records"}, ensure_ascii=False, indent=2))
    return report


def prepare_reviewed(evaluation_path, source="public-igbo", offset=0):
    from .corpus import prepare
    if SOURCES[source]["kind"] in {"intent","mixed-text"}:
        raise LabError("Published requests are not assistant reply targets. Keep this corpus separate and declare an intent-understanding experiment plan before training.")
    folder = curation_directory(source, offset)
    report = json.loads(owned(folder / "review_report.json").read_text())
    if report.get("source") != source or report.get("batch_start_offset", 0) != offset:
        raise LabError("The review receipt does not match the selected source.")
    path = owned(folder / "reviewed.jsonl")
    if path.stat().st_size > 5 * MIB or hashlib.sha256(path.read_bytes()).hexdigest() != report["reviewed_corpus_sha256"]:
        raise LabError("The reviewed corpus changed. Repeat its explicit review compilation.")
    issues = training_readiness(load_jsonl(path), source)
    if issues:
        raise LabError("The reviewed sample is not ready for training: " + " ".join(issues))
    cases_path = project_input(evaluation_path)
    if hashlib.sha256(cases_path.read_bytes()).hexdigest() != report["evaluation_sha256"]:
        raise LabError("The frozen evaluation set changed after review. Recheck overlap before preparing.")
    load_evaluation(cases_path)
    # Retain the existing tokenizer for this first baseline/candidate comparison.
    manifest = prepare(path)
    manifest.update({"review_summary": {key:report[key] for key in ("accepted_records", "edited_records", "ai_reviewed", "human_validated")},
                     "reviewed_corpus_sha256": report["reviewed_corpus_sha256"],
                     "frozen_evaluation_sha256": report["evaluation_sha256"],
                     "review_note": "AI-reviewed as requested; not native-speaker or real-audio validation."})
    write_json(LAB / "corpus/manifest.json", manifest)
    return manifest
