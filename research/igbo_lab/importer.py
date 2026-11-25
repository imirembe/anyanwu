"""Small, text-only Hub samples through the official dataset viewer API.

No datasets/audio libraries, Parquet files, media URLs or model weights are
downloaded. All requests share one 5 MiB response budget, including metadata.
"""
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
from urllib.parse import urlencode, urlparse

from .resources import LAB, MIB, LabError, ensure_capacity, owned, write_json

MAX_TRANSFER = 5 * MIB
MAX_PAGE = 512 * 1024
MAX_PREVIEW = MIB
SOURCES = {
    "public-igbo": {
        "dataset": "0xnu/igbo", "config": "default", "splits": ["train"],
        "license": "apache-2.0", "kind": "pair",
        "origin": "Publisher collection mixes synthetic and other examples; individual human authorship is unverified.",
        "restrictions": "Record publisher attribution. Source quality needs review; do not assume generated facts or grammar are correct.",
    },
    "africanvoice": {
        "dataset": "Africanvoice/igbo_african_voice", "config": "ibo", "splits": ["train", "dev", "dev_test"],
        "license": "cc-by-4.0", "kind": "transcript",
        "origin": "Publisher describes human speech transcriptions; sample not independently verified.",
        "restrictions": "Text/ASR research only in this importer. Publisher prohibits TTS/voice cloning/adaptation, biometrics and imitation.",
    },
    "aya-igbo": {
        "dataset": "CohereLabs/aya_dataset", "config": "default", "splits": ["train"],
        "license": "apache-2.0", "kind": "instruction",
        "origin": "Publisher describes human annotations; only Igbo original-annotations are selected. Individual authorship and language quality are not independently verified.",
        "restrictions": "Preserve Aya contributor attribution and source card. Exclude re-annotations, other languages and demographics. Do not treat this text sample as audio data.",
    },
}
SOURCES["aya-originals"] = {
    **SOURCES["aya-igbo"],
    "viewer_endpoint": "filter", "page_rows": 50,
    "viewer_where": '"language_code"=\'ibo\' AND "annotation_type"=\'original-annotations\'',
}
SOURCES["injongo-igbo"] = {
    "dataset":"masakhane/InjongoIntent", "config":"ibo", "splits":["train","validation","test"],
    "license":"apache-2.0", "kind":"intent",
    "intents":["alarm","calendar_update","make_call","play_music","shopping_list_update","text","time","weather"],
    "origin":"Publisher describes native-speaker elicited requests inspired by CLINC intent examples. No paired assistant answers; individual authorship and language quality not independently verified.",
    "restrictions":"Request-understanding text only; no audio or assistant replies supplied. Preserve author attribution. Hub declares Apache-2.0; the ACL paper declares dataset CC BY 4.0 and cites CLINC. Preserve both statements rather than infer an exclusive license.",
}
SOURCES["injongo-igbo-v2"] = {**SOURCES["injongo-igbo"], "retain_slots":True,
    "within_class_start":{"train":14,"validation":2,"test":4}, "required_revision":"fe4be3882a1614161dfe231ec793197bb74f4b44"}
SOURCES["injongo-igbo-other"] = {**SOURCES["injongo-igbo"], "retain_slots":True,
    "intents":["balance","book_flight","book_hotel","pin_change","recipe","restaurant_reservation","transfer","translate"],
    "task_role":"outside-eight-supported-intents", "required_revision":"fe4be3882a1614161dfe231ec793197bb74f4b44"}
SOURCES["clinc-english"] = {
    "dataset":"clinc/clinc_oos", "config":"plus", "splits":["train","validation","test"],
    "license":"cc-by-3.0", "kind":"intent", "adapter":"clinc", "language_code":"eng",
    "intents":list(SOURCES["injongo-igbo"]["intents"]),
    "origin":"Publisher describes crowd-authored English intent requests. Selected original English text, not newly translated/synthetic dialogue; individual authorship not independently verified.",
    "restrictions":"Request-understanding text only; no assistant answers, audio or slot annotations supplied. Preserve Larson et al. (2019) attribution and publisher source card under declared CC BY 3.0.",
}
SOURCES["orinode-mixed"] = {
    "dataset":"Orinode/naija-customer-call-code-switch","config":"default","splits":["train"],"kind":"mixed-text",
    "license":"cc-by-4.0","page_rows":32,"source_file":"igbo_english_5000.txt",
    "required_revision":"ca25901b832af1c3ede4db96aad060f18aa2c6f2",
    "origin":"Publisher describes hand-written naturalistic Igbo-English customer-service text with fictional details. Individual authorship/mixing not independently verified; scripted text, not recorded spontaneous speech.",
    "restrictions":"Preserve Orinode Ltd (2026) attribution, CC BY 4.0, original source text and review decisions. Only the first 32 lines of the pinned Igbo file; keep the entire file group in training, with no independent mixed-speech test claim.",
}
SOURCES["clinc-english-v3"] = {**SOURCES["clinc-english"],
    "within_class_start":{"train":8,"validation":2,"test":2},
    "required_revision":"155b9c710419136e17307b80d0a13e68cd46b4ec",
    "class_block_positions":{"alarm":73,"calendar_update":132,"make_call":72,"play_music":131,
                             "shopping_list_update":20,"text":127,"time":14,"weather":108}}
SOURCES["clinc-english-other"] = {**SOURCES["clinc-english"],
    "intents":list(SOURCES["injongo-igbo-other"]["intents"]),
    "task_role":"outside-eight-supported-intents",
    "class_lookup_rows":1,
    "required_revision":"155b9c710419136e17307b80d0a13e68cd46b4ec"}
SOURCES["injongo-igbo-v3"] = {**SOURCES["injongo-igbo"],
    "within_class_start":{"train":28,"validation":4,"test":8},
    "required_revision":"fe4be3882a1614161dfe231ec793197bb74f4b44"}
SOURCES["injongo-igbo-other-v3"] = {**SOURCES["injongo-igbo-other"],"retain_slots":False,
    "within_class_start":{"train":7,"validation":1,"test":2}}
SOURCES["clinc-original-oos"] = {**SOURCES["clinc-english"],"original_oos":True,"intents":["oos"],
    "required_revision":"155b9c710419136e17307b80d0a13e68cd46b4ec",
    "origin":"Publisher crowd-authored original English out-of-scope partition. Outside CLINC's 150 intents; each query still requires AI review against Anyanwụ's eight tasks.",
    "task_role":"original-source-oos; local unknown mapping requires explicit review"}


def slot_metadata(row):
    """Validate byte boundaries only; do not claim semantic slot correctness."""
    spans=row.get("spans")
    original=row["text"].encode("utf-8")
    result={"source_slot_text_sha256":hashlib.sha256(original).hexdigest(),"source_slot_offset_unit":"utf8-byte",
            "source_slots":[],"source_slot_semantics_reviewed":False}
    if not isinstance(spans,list) or len(spans)>32:
        return {**result,"source_slot_status":"missing-or-invalid-shape","source_slot_alignment_valid":False}
    result["source_spans_sha256"]=hashlib.sha256(json.dumps(spans,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    clean=[]
    for span in spans:
        if not isinstance(span,dict):break
        start,end,label=span.get("start_byte"),span.get("limit_byte"),span.get("label")
        if (type(start) is not int or type(end) is not int or not 0<=start<end<=len(original)
                or not isinstance(label,str) or not re.fullmatch(r"[A-Z][A-Z_]{0,63}",label)):break
        try:value=original[start:end].decode("utf-8")
        except UnicodeDecodeError:break
        clean.append({"start_byte":start,"limit_byte":end,"label":label,"value":value})
    if len(clean)!=len(spans):
        return {**result,"source_slot_status":"invalid-byte-boundary-or-label","source_slot_alignment_valid":False}
    return {**result,"source_slots":clean,"source_slot_status":"valid" if clean else "empty","source_slot_alignment_valid":True}


class HubAccessRequired(LabError):
    pass


class HubHTTPError(LabError):
    def __init__(self, path, status):
        self.status = status
        super().__init__(f"The dataset API {path} returned HTTP {status}; no full-dataset fallback is attempted.")


class BoundedHubClient:
    def __init__(self, token=None, budget=MAX_TRANSFER):
        if not 0 < budget <= MAX_TRANSFER:
            raise LabError("The importer accepts at most 5 MiB of HTTP response bodies.")
        if token and not re.fullmatch(r"hf_[A-Za-z0-9]{10,256}", token):
            raise LabError("Use a Hugging Face read-only token, entered privately in the terminal.")
        self.token = token
        self.remaining = budget
        self.received = 0
        self.requests = 0

    def get(self, url, *, json_response=True, _retry=True):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {"huggingface.co", "datasets-server.huggingface.co"} or parsed.username or parsed.password:
            raise LabError("The importer only contacts the two official Hugging Face API hosts over HTTPS.")
        if self.remaining < 1024:
            raise LabError("The import's shared response budget has been exhausted.")
        # HF previews may be larger than row slices, but remain inside the
        # same shared budget. Never raise the allowance for files/media.
        page_limit = MAX_PREVIEW if parsed.hostname == "datasets-server.huggingface.co" and parsed.path == "/first-rows" else MAX_PAGE
        limit = min(page_limit, self.remaining)
        request_seconds = 55 if parsed.hostname == "datasets-server.huggingface.co" and parsed.path == "/filter" else 25
        command = ["curl", "--silent", "--show-error", "--proto", "=https", "--connect-timeout", "8",
                   "--max-time", str(request_seconds), "--max-filesize", str(limit), "--header", "Accept-Encoding: identity",
                   "--header", "Accept: application/json" if json_response else "Accept: text/plain",
                   "--write-out", "\n%{http_code}", "--config", "-", url]
        # The token is sent through stdin, never command-line arguments, files,
        # logs, review records, or the environment inherited by curl.
        config = f'header = "Authorization: Bearer {self.token}"\n' if self.token else ""
        env = {k: v for k, v in os.environ.items() if k not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}}
        try:
            result = subprocess.run(command, input=config.encode(), capture_output=True, timeout=request_seconds+5, env=env)
        except subprocess.TimeoutExpired as error:
            raise LabError("The small dataset request timed out; no corpus was installed.") from error
        self.requests += 1
        body, _, status_bytes = result.stdout.rpartition(b"\n")
        self.remaining -= len(body); self.received += len(body)
        if result.returncode == 28:
            if _retry and self.remaining >= 1024:
                return self.get(url, json_response=json_response, _retry=False)
            raise LabError(f"The bounded dataset request to {parsed.path} timed out; no full-dataset fallback is attempted.")
        if result.returncode or len(body) > limit or self.remaining < 0:
            raise LabError("Dataset request failed or exceeded its strict response limit; no full-dataset fallback is attempted.")
        if status_bytes in {b"401", b"403"}:
            raise HubAccessRequired("Hugging Face requires approved dataset access and a read-only token. Run import-text with --token-prompt in your own terminal; do not paste a token into chat.")
        if status_bytes != b"200":
            if status_bytes in {b"500", b"502", b"503", b"504"} and _retry and self.remaining >= 1024:
                return self.get(url, json_response=json_response, _retry=False)
            status = int(status_bytes) if status_bytes.isdigit() else 0
            raise HubHTTPError(parsed.path, status)
        try:
            text = body.decode("utf-8")
            return json.loads(text) if json_response else text
        except (UnicodeError, ValueError) as error:
            raise LabError("The API returned invalid UTF-8/JSON.") from error


def private_token(prompt=False):
    if prompt:
        if not sys.stdin.isatty():
            raise LabError("Run --token-prompt in your own interactive terminal. Token entry is disabled for pipes or chat tools.")
        return getpass.getpass("Hugging Face read-only token (not saved): ").strip()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def transcript_issue(row):
    """Describe a rejected shape without logging source values or identifiers."""
    if not isinstance(row, dict):
        return "invalid_row"
    for field in ("transcript", "speaker_id"):
        if field not in row:
            return f"missing_{field}"
        if not isinstance(row[field], str):
            return f"non_text_{field}"
        if not row[field].strip():
            return f"empty_{field}"
    return None


def increment(counts, key):
    counts[key] = counts.get(key, 0) + 1


def save_preview_diagnostic(source, spec, revision, card, rows, diagnostics, records, client):
    """Retain only eligible text and aggregate rejection details, never raw JSON.

    A rejected import is not a training corpus. This receipt lets us inspect
    an authenticated preview later without retaining credentials/media links.
    Content-addressed paths preserve earlier observations and deduplicate retries.
    """
    snapshot = {"format": 1, "source": spec["dataset"], "revision": revision,
                "license_declared": spec["license"], "restrictions": spec["restrictions"],
                "requested_records": rows, "splits": diagnostics, "eligible_preview_records": records,
                "source_card": card, "audio_downloaded": False, "training_sample_installed": False,
                "review_status": "pending", "http_body_bytes": client.received,
                "http_budget_bytes": MAX_TRANSFER,
                "snapshot_note": "Hub head matched before/after; viewer row revision is not independently guaranteed."}
    # The receipt has its own output cap; JSON pretty-printing can add overhead.
    if len(json.dumps(snapshot, ensure_ascii=False, indent=2).encode()) > MAX_TRANSFER:
        snapshot["eligible_preview_records"] = []
        snapshot["text_records_omitted_for_size"] = True
    digest = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    path = owned(LAB / "imports" / f"{source}-preview-{digest}.json")
    if not path.exists():
        write_json(path, snapshot)
    return path


def text_record(spec, revision, split, row_index, row):
    if not isinstance(row, dict):
        return None
    if spec["kind"] in {"pair", "instruction"}:
        if spec["kind"] == "instruction":
            if row.get("language_code") != "ibo" or row.get("annotation_type") != "original-annotations":
                return None
            user, answer = row.get("inputs"), row.get("targets")
            content_type = "instruction"
        else:
            if row.get("type") not in {"conversation", "translation", "qa", "grammar", "culture", "story", "proverb"}:
                return None
            user, answer = row.get("input"), row.get("output")
            content_type = row.get("type")
        if not isinstance(user, str) or not isinstance(answer, str):
            return None
        content = {"input": user.strip(), "output": answer.strip(), "type": content_type}
        if not content["input"] or not content["output"]:
            return None
        group_value = re.sub(r"\s+", " ", user.casefold()).strip(" .?!")
    elif spec["kind"] == "mixed-text":
        text=row.get("text")
        if (not isinstance(text,str) or not 5000<=row_index<5032
                or not text.startswith(str(row_index-4999)+". ")):return None
        content={"text":text,"type":"published-scripted-mixed-request"}
        group_value=spec["source_file"]
    elif spec["kind"] == "intent":
        text, intent, example = row.get("text"), row.get("intent"), row.get("example_id")
        example_prefix = r"(?:dev|validation)" if split == "validation" else re.escape(split)
        if (not isinstance(text,str) or not text.strip() or intent not in spec["intents"]
                or not isinstance(example,str) or not re.fullmatch(example_prefix + r"-[0-9]{8}",example)):
            return None
        content = {"text":text, "intent":intent, "type":"elicited-user-request"}
        group_value = re.sub(r"[^\w]+", " ", unicodedata.normalize("NFC",text).casefold()).strip()
        if not group_value:
            return None
    else:
        transcript = row.get("transcript")
        speaker = row.get("speaker_id")
        if transcript_issue(row):
            return None
        content = {"text": transcript.strip(), "type": "speech-transcript"}
        if not content["text"]:
            return None
        group_value = speaker
    serialized = json.dumps(content, ensure_ascii=False, sort_keys=True)
    if len(serialized.encode()) > 16_384 or any("\x00" in value for value in content.values() if isinstance(value, str)):
        return None
    # Keep official holdout rows even if their text repeats a training row.
    # Group by normalized text, then explicitly exclude conflicts during review.
    identity = serialized + "/" + split + "/" + example if spec["kind"] == "intent" else serialized
    key = hashlib.sha256((spec["dataset"] + "/" + identity).encode()).hexdigest()
    record = {
        "id": key[:24], "group": hashlib.sha256((spec["dataset"] + "/" + group_value).encode()).hexdigest()[:24],
        "source": spec["dataset"], "source_url": f"https://huggingface.co/datasets/{spec['dataset']}",
        "source_revision": revision, "source_split": split, "source_row_index": row_index,
        "source_revision_note": "Hub head seen before/after import; indexed row revision not independently guaranteed.",
        "license": spec["license"], "origin": spec["origin"], "demo_only": False,
        "content_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "content": content, "review_status": "pending", "audio_downloaded": False,
    }
    if spec["kind"] == "transcript":
        record["split"] = {"train": "train", "dev": "validation", "dev_test": "test"}[split]
    if spec["kind"] == "instruction":
        record["source_language_code"] = "ibo"
        record["source_annotation_type"] = "original-annotations"
    if spec["kind"] == "intent":
        record.update(source_intent=intent,source_example_id=example,source_config=spec["config"],source_language_code=spec.get("language_code","ibo"),split=split)
        if spec.get("retain_slots"):record.update(slot_metadata(row))
    if spec["kind"] == "mixed-text":
        record.update(source_file=spec["source_file"],source_config=spec["config"],source_language_code="ig-en",
                      source_language_note="Publisher Igbo-English file section; per-record language review required.",split="train")
    return record


def sample_directory(source, offset=0):
    if source not in SOURCES or type(offset) is not int or not 0 <= offset <= 1_000_000:
        raise LabError("Choose a registered source and a row offset from 0 to 1,000,000.")
    if offset and SOURCES[source]["kind"] in {"transcript","intent"}:
        raise LabError("Transcript/intent previews do not support later row offsets; the existing sample is preserved.")
    base = owned(LAB / "imports" / source)
    return owned(base / "batches" / f"offset-{offset:07d}") if offset else base


def import_text(source="public-igbo", rows=100, token_prompt=False, client=None, offset=0):
    if source not in SOURCES or not 1 <= rows <= 500:
        raise LabError("Choose a registered text source and 1–500 records.")
    if SOURCES[source]["kind"]=="mixed-text" and (rows!=32 or offset!=5000):
        raise LabError("The mixed-text pilot is pinned to --rows 32 --offset 5000; do not guess another language's row range.")
    destination = sample_directory(source, offset)
    if destination.exists():
        raise LabError("This sample already exists. Its raw records and review provenance are preserved.")
    if SOURCES[source]["kind"] == "intent":
        from .request_data import import_intent_sample
        return import_intent_sample(source, rows, token_prompt=token_prompt, client=client)
    ensure_capacity(MAX_TRANSFER * 3)
    spec = SOURCES[source]
    if spec["kind"] == "transcript" and rows < 3:
        raise LabError("Use at least three transcript rows so the official train/dev/dev_test splits are represented.")
    client = client or BoundedHubClient(private_token(token_prompt))
    metadata_url = f"https://huggingface.co/api/datasets/{spec['dataset']}"
    metadata = client.get(metadata_url)
    revision = metadata.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise LabError("Dataset revision could not be recorded.")
    if metadata.get("cardData", {}).get("license") not in (spec["license"],[spec["license"]]):
        raise LabError("The publisher's license label changed. Review the selected release before importing.")
    if spec.get("required_revision",revision)!=revision:
        raise LabError("The pinned text source revision changed; inspect the new release before importing.")
    card = client.get(f"https://huggingface.co/datasets/{spec['dataset']}/raw/{revision}/README.md", json_response=False)
    split_info = client.get("https://datasets-server.huggingface.co/splits?" + urlencode({"dataset": spec["dataset"]}))
    available = {(item["config"], item["split"]) for item in split_info.get("splits", [])}
    records, seen, scanned = [], set(), 0
    preview_only, methods, matching_rows = False, {}, {}
    diagnostics, preview_records, shortfalls = {}, [], []
    if spec["kind"] == "transcript":
        holdout = max(1, rows // 10)
        quotas = [rows - 2 * holdout, holdout, holdout]
    else:
        quotas = [rows]
    for split, quota in zip(spec["splits"], quotas):
        if (spec["config"], split) not in available:
            raise LabError("Expected dataset subset/split is unavailable; no full-dataset fallback is attempted.")
        page_offset = offset
        split_start = len(records)
        stats = {"requested": quota, "rows_returned": 0, "rows_processed": 0, "eligible_unique_rows": 0,
                 "selected": 0, "excluded": {}, "field_types": {"transcript": {}, "speaker_id": {}},
                 "columns_seen": [], "preview_truncated": False}
        diagnostics[split] = stats
        while len(records) - split_start < quota and scanned < 2000:
            length = min(100, quota - (len(records) - split_start)) if spec["kind"] == "transcript" else spec.get("page_rows", 100)
            query = {"dataset": spec["dataset"], "config": spec["config"], "split": split, "offset": page_offset, "length": length}
            endpoint = spec.get("viewer_endpoint", "rows")
            if endpoint == "filter":
                query["where"] = spec["viewer_where"]
            if client.remaining < MAX_PAGE:
                break
            if not preview_only:
                try:
                    page = client.get(f"https://datasets-server.huggingface.co/{endpoint}?" + urlencode(query))
                except HubHTTPError as error:
                    if error.status != 501 or spec["kind"] != "transcript" or page_offset:
                        raise
                    preview_only = True
                    print("The row-slice API is unavailable (HTTP 501); trying the bounded JSON preview. Audio stays remote.")
            if preview_only:
                preview_query = {key: query[key] for key in ("dataset", "config", "split")}
                try:
                    page = client.get("https://datasets-server.huggingface.co/first-rows?" + urlencode(preview_query))
                except HubHTTPError as error:
                    raise LabError(f"The row-slice API is unavailable and the preview returned HTTP {error.status}. No sample was installed; audio and full dataset files were not requested.") from error
            methods[split] = "first-rows" if preview_only else endpoint
            if not isinstance(page, dict) or any(page.get(key, value) != value for key, value in query.items() if key in {"dataset", "config", "split"}):
                raise LabError("The viewer returned an unexpected dataset/subset/split; no sample was installed.")
            if page.get("dataset_git_revision") not in {None, revision}:
                raise LabError("The indexed rows identify a different dataset revision; no sample was installed.")
            if type(page.get("num_rows_total")) is int:
                matching_rows[split] = page["num_rows_total"]
            page_rows = page.get("rows", [])
            if not isinstance(page_rows, list) or len(page_rows) > 100:
                raise LabError("The viewer returned an invalid row page; no sample was installed.")
            if spec["kind"]=="mixed-text" and (page.get("partial") is True or page.get("num_rows_total")!=15000
                    or len(page_rows)!=32 or any(not isinstance(e,dict) or type(e.get("row_idx")) is not int
                    or e["row_idx"]!=5000+i for i,e in enumerate(page_rows))):
                raise LabError("The pinned mixed-text file layout is incomplete or changed; no sample installed.")
            stats["rows_returned"] += len(page_rows)
            stats["preview_truncated"] = stats["preview_truncated"] or page.get("truncated") is True
            if not page_rows:
                break
            required = ({"transcript", "speaker_id"} if spec["kind"] == "transcript" else
                        {"inputs", "targets", "language_code", "annotation_type"} if spec["kind"] == "instruction" else
                        {"text"} if spec["kind"]=="mixed-text" else {"input", "output", "type"})
            for entry in page_rows:
                if scanned >= 2000:
                    break
                scanned += 1
                stats["rows_processed"] += 1
                if not isinstance(entry, dict):
                    increment(stats["excluded"], "invalid_row_envelope")
                    continue
                row = entry.get("row")
                if isinstance(row, dict):
                    stats["columns_seen"] = sorted(set(stats["columns_seen"]) | {key for key in row if isinstance(key, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", key)})[:64]
                    for field in stats["field_types"]:
                        kind = type(row[field]).__name__ if field in row else "missing"
                        increment(stats["field_types"][field], kind)
                truncated = entry.get("truncated_cells", [])
                if not isinstance(truncated, list) or not all(isinstance(column, str) for column in truncated):
                    increment(stats["excluded"], "invalid_truncation_metadata")
                    continue
                if required.intersection(truncated):
                    for field in sorted(required.intersection(truncated)):
                        increment(stats["excluded"], f"truncated_{field}")
                    continue
                if spec["kind"] == "transcript" and (issue := transcript_issue(row)):
                    increment(stats["excluded"], issue)
                    continue
                row_index = entry.get("row_idx")
                if type(row_index) is not int or row_index < 0:
                    increment(stats["excluded"], "invalid_source_row_index")
                    continue
                record = text_record(spec, revision, split, row_index, row)
                if not record:
                    increment(stats["excluded"], "ineligible_or_invalid_text")
                    continue
                if record["id"] in seen:
                    increment(stats["excluded"], "duplicate_text")
                    continue
                seen.add(record["id"])
                stats["eligible_unique_rows"] += 1
                if preview_only:
                    preview_records.append(record)
                if len(records) - split_start < quota:
                    records.append(record)
                    stats["selected"] += 1
                if (not preview_only and len(records) - split_start >= quota) or scanned >= 2000:
                    break
            page_offset += len(page_rows)
            if preview_only or len(page_rows) < length or spec["kind"]=="mixed-text":
                break
        if spec["kind"] == "transcript" and len(records) - split_start < quota:
            shortfalls.append(split)
        if len(records) >= rows or scanned >= 2000:
            break
    # Viewer APIs serve their current indexed snapshot. Check the Hub head again
    # and state the limitation instead of pretending this proves immutable rows.
    if client.get(metadata_url).get("sha") != revision:
        raise LabError("The dataset changed during import. Retry a consistent small sample.")
    if shortfalls:
        path = save_preview_diagnostic(source, spec, revision, card, rows, diagnostics, preview_records, client)
        for split, stats in diagnostics.items():
            counts = ", ".join(f"{key}={value}" for key, value in sorted(stats["excluded"].items())) or "none"
            print(f"{split}: selected {stats['selected']}/{stats['requested']}; returned {stats['rows_returned']} rows; excluded: {counts}.")
        print(f"Text-only diagnostic saved: {path.relative_to(LAB.parent.parent)}")
        raise LabError("The preview could not fill the official split quotas; no training sample was installed. The diagnostic records missing/invalid fields, duplicates and truncated text. Inspect it before retrying; a smaller --rows value may not help.")
    if not records:
        raise LabError("No eligible text records were found within the bounded sample.")
    if spec["kind"]=="mixed-text" and len(records)!=rows:
        raise LabError("The pinned mixed-text row quota was not filled; no partial corpus installed.")
    if destination.exists():
        raise LabError("This sample already exists. Its raw records and review provenance are preserved.")
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode()
    if len(payload) > MAX_TRANSFER:
        raise LabError("The text-only output exceeds the 5 MiB sample cap.")
    ensure_capacity(len(payload) + len(card.encode()) + MIB)
    report = {"source": spec["dataset"], "revision": revision, "license_declared": spec["license"],
              "origin": spec["origin"], "restrictions": spec["restrictions"], "records": len(records), "rows_scanned": scanned,
              "http_body_bytes": client.received, "http_budget_bytes": MAX_TRANSFER, "requests": client.requests,
              "saved_text_bytes": len(payload), "audio_downloaded": False, "review_status": "pending",
              "sample_counts_by_source_split": {split:sum(record["source_split"] == split for record in records) for split in spec["splits"]},
              "viewer_endpoints_by_source_split": methods,
              "snapshot_note": "Hub head matched before/after; viewer row revision is not independently guaranteed.",
              "candidates_sha256": hashlib.sha256(payload).hexdigest()}
    report["requested_records"] = rows
    report["batch_start_offset"] = offset
    report["scan_limit_reached"] = scanned >= 2000
    if matching_rows:
        report["indexed_matching_rows_by_split"] = matching_rows
    if spec["kind"] == "instruction":
        report["selection_note"] = "Bounded mixed-language row scan; only exact language_code=ibo and annotation_type=original-annotations retained. No demographics or annotator identifiers saved."
    if spec.get("viewer_endpoint") == "filter":
        report["selection_note"] = "Server-side Igbo original-annotation filter, rechecked on every row locally. No unfiltered fallback; demographics/annotator identifiers discarded."
        report["batch_offset_kind"] = "Offset in filtered matches; source_row_index retains the original dataset row index."
        report["viewer_filter"] = spec["viewer_where"]
    from .resources import remove_generated
    stage = owned(LAB / "imports/import-staging")
    stage.parent.mkdir(exist_ok=True)
    remove_generated(stage); stage.mkdir()
    try:
        (stage / "candidates.jsonl").write_bytes(payload)
        (stage / "source-card.md").write_text(card, encoding="utf-8")
        write_json(stage / "manifest.json", report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
    finally:
        remove_generated(stage)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
