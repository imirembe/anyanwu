"""Bounded JSONL ingestion; source-group holdouts and Unicode-preserving tokens."""
from array import array
import hashlib
import json
import os
from pathlib import Path
import shutil
import unicodedata

from .resources import (LAB, CORPUS_LIMIT, DOCUMENT_LIMIT, DOCUMENT_COUNT_LIMIT,
                        TOKEN_LIMIT, LabError, ensure_capacity, owned, project_input,
                        remove_generated, write_json)

SPLITS = {"train", "validation", "test"}
MARKERS = ["<|user|>", "<|assistant|>", "<|end|>"]


def normalized(text):
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise LabError("Each record needs nonempty text without NUL characters.")
    text = unicodedata.normalize("NFC", text).strip()
    if len(text.encode("utf-8")) > DOCUMENT_LIMIT:
        raise LabError("Split long documents into records under 32 KiB, keeping their group unchanged.")
    return text


def group_split(group):
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def read_records(path):
    path = project_input(path)
    if path.stat().st_size > CORPUS_LIMIT:
        raise LabError("This pilot accepts at most 32 MiB of source JSONL.")
    records, seen, groups, total = [], set(), {}, 0
    with path.open("rb") as handle:
        while line := handle.readline(DOCUMENT_LIMIT * 8 + 1):
            if len(line) > DOCUMENT_LIMIT * 8:
                raise LabError("A JSONL line exceeds the bounded record size.")
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeError) as error:
                raise LabError("Input must be UTF-8 JSONL, one document per line.") from error
            if not isinstance(record, dict):
                raise LabError("Each JSONL record must be an object.")
            for key in ("id", "group", "source", "license"):
                if not isinstance(record.get(key), str) or not record[key].strip() or len(record[key]) > 512:
                    raise LabError(f"Every record needs a short {key!r} field for provenance and splitting.")
            text = normalized(record.get("text"))
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            split = record.get("split", group_split(record["source"] + "/" + record["group"]))
            if split not in SPLITS:
                raise LabError("split must be train, validation or test.")
            group = record["source"] + "/" + record["group"]
            if group in groups and groups[group] != split:
                raise LabError("A source group/book cannot span training and evaluation splits.")
            groups[group] = split
            total += len(text.encode())
            if total > CORPUS_LIMIT or len(records) >= DOCUMENT_COUNT_LIMIT:
                raise LabError("The corpus exceeds this pilot's text/document budget.")
            records.append({**record, "text": text, "split": split, "sha256": digest})
    if not records or any(not any(r["split"] == split for r in records) for split in SPLITS):
        raise LabError("Supply separate source groups for train, validation and test.")
    return records


def prepare(path, vocab_size=512, rebuild_tokenizer=False):
    import sentencepiece as spm
    if not 512 <= vocab_size <= 8192:
        raise LabError("Vocabulary size must be between 512 and 8192 for this pilot.")
    records = read_records(path)
    ensure_capacity(CORPUS_LIMIT * 5)
    stage = owned(LAB / "corpus-staging")
    current = owned(LAB / "corpus")
    remove_generated(stage)
    stage.mkdir()
    try:
        for split in sorted(SPLITS):
            with (stage / f"{split}.txt").open("w", encoding="utf-8") as handle:
                for record in records:
                    if record["split"] == split:
                        handle.write(record["text"].replace("\n", " ") + "\n")
        reused = (current / "tokenizer.model").exists() and not rebuild_tokenizer
        if reused:
            shutil.copyfile(current / "tokenizer.model", stage / "tokenizer.model")
        else:
            spm.SentencePieceTrainer.train(
                input=str(stage / "train.txt"), model_prefix=str(stage / "tokenizer"),
                vocab_size=vocab_size, model_type="unigram", character_coverage=1.0,
                normalization_rule_name="identity", byte_fallback=True,
                remove_extra_whitespaces=False, hard_vocab_limit=False,
                user_defined_symbols=MARKERS, max_sentence_length=DOCUMENT_LIMIT,
                num_threads=2, minloglevel=2, shuffle_input_sentence=False,
            )
        processor = spm.SentencePieceProcessor(model_file=str(stage / "tokenizer.model"))
        probes = ["ị ọ ụ ṅ", "Ị́ Ị̀ Ọ́ Ọ̀ Ụ́ Ụ̀ Ṅ́", "Ndewo, kedu ka ị mere?", "NewName123"]
        for text in probes:
            text = unicodedata.normalize("NFC", text)
            if processor.decode(processor.encode(text)) != text:
                raise LabError("The tokenizer did not preserve an Igbo/Unicode round-trip probe.")
        counts = {}
        for split in sorted(SPLITS):
            tokens = array("H")
            for record in records:
                if record["split"] == split:
                    encoded = processor.encode(record["text"]) + [processor.eos_id()]
                    if len(tokens) + len(encoded) > TOKEN_LIMIT:
                        raise LabError("Tokenized data exceeds the pilot limit; use a smaller sample.")
                    tokens.extend(encoded)
            with (stage / f"{split}.tokens").open("wb") as handle:
                tokens.tofile(handle)
            counts[split] = {"documents": sum(r["split"] == split for r in records), "tokens": len(tokens)}
        with (stage / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest = {
            "format": 1, "purpose": "reply-model research for the voice assistant",
            "demo_only": all(r.get("demo_only") is True for r in records),
            "tokenizer_sha256": hashlib.sha256((stage / "tokenizer.model").read_bytes()).hexdigest(),
            "tokenizer_reused": reused, "vocabulary_size": processor.vocab_size(),
            "corpus_sha256": hashlib.sha256((stage / "records.jsonl").read_bytes()).hexdigest(),
            "splits": counts, "licenses_declared": sorted({r["license"] for r in records}),
            "provenance": "Declared by each source record; permissions require review before importing real material.",
        }
        reviews = [record.get("review", {}) for record in records]
        if all(isinstance(item, dict) and item.get("type") in {"AI", "human"} and item.get("reviewer")
               and item.get("decision") in {"accept", "edit"} and item.get("confidence") in {"high", "medium"}
               and item.get("reason") for item in reviews):
            manifest["review_summary"] = {
                "accepted_records": len(records), "edited_records": sum(item["decision"] == "edit" for item in reviews),
                "ai_reviewed": sum(item["type"] == "AI" for item in reviews),
                "human_validated": all(item["type"] == "human" for item in reviews),
            }
        write_json(stage / "manifest.json", manifest)
        # Rename the complete preparation into place; preserve it on a failed build.
        previous = owned(LAB / "previous")
        remove_generated(previous)
        if current.exists():
            os.replace(current, previous)
        try:
            os.replace(stage, current)
        except BaseException:
            if previous.exists() and not current.exists():
                os.replace(previous, current)
            raise
        remove_generated(previous)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest
    finally:
        remove_generated(stage)
