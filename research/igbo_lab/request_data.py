"""Balanced, text-only native-speaker request samples, without fabricated replies."""
import hashlib
import json
import os
import re
from urllib.parse import urlencode

from . import importer
from .resources import LAB, MIB, LabError, ensure_capacity, owned, remove_generated, write_json

PAPER = "https://aclanthology.org/2025.acl-long.464/"
# The observed release stores equal-sized, alphabetical intent blocks. Check
# the returned split size, original indices and every label instead of trusting
# that layout after a publisher change. This is a declared row-slice strategy,
# not an automatic fallback from a failed filtered/full-dataset download.
INTENT_ORDER = tuple(sorted("alarm balance bill_balance book_flight book_hotel calendar_update cancel_reservation car_rental confirm_reservation cook_time exchange_rate food_last freeze_account ingredients_list interest_rate international_visa make_call meal_suggestion min_payment pay_bill pin_change play_music plug_type recipe restaurant_reservation restaurant_reviews restaurant_suggestion share_location shopping_list_update spending_history text time timezone transactions transfer translate travel_notification travel_suggestion update_playlist weather".split()))
BLOCK_ROWS = {"train":56,"validation":8,"test":16}


def import_intent_sample(source, rows, token_prompt=False, client=None):
    spec = importer.SOURCES[source]
    if spec.get("adapter")=="clinc":
        from .english_data import import_english_requests
        return import_english_requests(source,rows,token_prompt,client)
    destination = importer.sample_directory(source)
    if type(rows) is not int or not 80 <= rows <= 480 or rows % 80:
        raise LabError("Use 80, 160, 240, 320, 400 or 480 requests: balanced across eight intents and official 70/10/20 splits.")
    if destination.exists():
        raise LabError("This sample already exists; original requests are preserved.")
    ensure_capacity(15*MIB)
    client = client or importer.BoundedHubClient(importer.private_token(token_prompt))
    metadata_url = f"https://huggingface.co/api/datasets/{spec['dataset']}"
    metadata = client.get(metadata_url)
    revision = metadata.get("sha")
    if not isinstance(revision,str) or not re.fullmatch(r"[a-f0-9]{40}",revision):
        raise LabError("The request dataset revision could not be recorded.")
    if spec.get("required_revision",revision)!=revision:
        raise LabError("The publisher revision changed before the planned request extension; inspect the new release first.")
    if metadata.get("cardData",{}).get("license") != spec["license"]:
        raise LabError("The request dataset license label changed; review the release first.")
    card = client.get(f"https://huggingface.co/datasets/{spec['dataset']}/raw/{revision}/README.md",json_response=False)
    splits = client.get("https://datasets-server.huggingface.co/splits?" + urlencode({"dataset":spec["dataset"]}))
    available = {(r.get("config"),r.get("split")) for r in splits.get("splits",[])}
    if any((spec["config"],split) not in available for split in spec["splits"]):
        raise LabError("All official Igbo request splits must be available; no substitute subset is used.")
    records, seen, counts, indexed = [], set(), {}, {}
    scale = rows // 80
    for split, per_intent in (("train",7*scale),("validation",scale),("test",2*scale)):
        counts[split], indexed[split] = {}, {}
        for intent in spec["intents"]:
            within=spec.get("within_class_start",{}).get(split,0)
            if within+per_intent>BLOCK_ROWS[split]:
                raise LabError("The planned request slice exceeds its source class block.")
            start = INTENT_ORDER.index(intent)*BLOCK_ROWS[split]+within
            query = {"dataset":spec["dataset"],"config":spec["config"],"split":split,
                     "offset":start,"length":per_intent}
            page = client.get("https://datasets-server.huggingface.co/rows?" + urlencode(query))
            if (not isinstance(page,dict) or any(page.get(k,v) != v for k,v in query.items() if k in {"dataset","config","split"})
                    or page.get("dataset_git_revision") not in {None,revision}):
                raise LabError("The request viewer returned a different source/subset/split/revision.")
            entries = page.get("rows")
            if (not isinstance(entries,list) or len(entries) != per_intent or page.get("partial") is True
                    or type(page.get("num_rows_total")) is not int or page["num_rows_total"] != len(INTENT_ORDER)*BLOCK_ROWS[split]):
                raise LabError("The bounded request page could not fill its balanced quota; no partial sample installed.")
            indexed[split][intent] = BLOCK_ROWS[split]
            for local_index,entry in enumerate(entries):
                if not isinstance(entry,dict):
                    raise LabError("Invalid request row envelope; no sample installed.")
                truncated = entry.get("truncated_cells",[])
                if (not isinstance(truncated,list) or not all(isinstance(c,str) for c in truncated)
                        or ({"text","intent","example_id"}|({"spans"} if spec.get("retain_slots") else set())).intersection(truncated)):
                    raise LabError("Truncated request text/label/identity refused; no sample installed.")
                row, index = entry.get("row"),entry.get("row_idx")
                if not isinstance(row,dict) or row.get("intent") != intent or type(index) is not int or index != start+local_index:
                    raise LabError("A request slice has an invalid label or original row index; the source layout may have changed.")
                record = importer.text_record(spec,revision,split,index,row)
                if record is None or record["id"] in seen:
                    raise LabError("A request is invalid or repeats its source identity; no sample installed.")
                seen.add(record["id"]); records.append(record)
            counts[split][intent] = per_intent
        print(f"{split}: {sum(counts[split].values())} requests checked across eight intent labels.",flush=True)
    if client.get(metadata_url).get("sha") != revision:
        raise LabError("The request dataset changed during import; no sample installed.")
    payload = "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records).encode()
    if len(payload) > importer.MAX_TRANSFER:
        raise LabError("Request sample exceeds its 5 MiB output cap.")
    report = {"source":spec["dataset"],"revision":revision,"license_declared":spec["license"],
              "license_note":"Hub metadata declares Apache-2.0; source paper states dataset CC BY 4.0. Retain both and author/CLINC attribution.",
              "paper":PAPER,"origin":spec["origin"],"restrictions":spec["restrictions"],"source_config":spec["config"],
              "records":len(records),"rows_scanned":len(records),"requested_records":rows,"batch_start_offset":0,
              "sample_counts_by_source_split":{split:sum(counts[split].values()) for split in counts},
              "sample_counts_by_split_intent":counts,"indexed_counts_by_split_intent":indexed,
              "selection_note":"Fixed class-block row slices in official ibo splits; validate split totals, row indices and every label. A small stratified convenience sample, not random or all-domain coverage. No English translations/replies generated.",
              "viewer_endpoints_by_source_split":{split:"rows" for split in spec["splits"]},
              "snapshot_note":"Hub head matched before/after; indexed row revision not independently guaranteed.",
              "http_body_bytes":client.received,"http_budget_bytes":importer.MAX_TRANSFER,"requests":client.requests,
              "saved_text_bytes":len(payload),"candidates_sha256":hashlib.sha256(payload).hexdigest(),
              "review_status":"pending","purpose":"user-request understanding; no assistant reply targets",
              "slot_annotations_retained":False,"audio_downloaded":False}
    if spec.get("retain_slots"):
        report.update(slot_annotations_retained=True,slot_semantics_reviewed=False,
                      within_class_start=spec.get("within_class_start",{}),
                      slot_status_counts={status:sum(r["source_slot_status"]==status for r in records) for status in sorted({r["source_slot_status"] for r in records})})
    ensure_capacity(len(payload)+len(card.encode())+MIB)
    stage = owned(LAB/"imports/import-staging")
    stage.parent.mkdir(parents=True,exist_ok=True)
    remove_generated(stage);stage.mkdir()
    try:
        (stage/"candidates.jsonl").write_bytes(payload)
        (stage/"source-card.md").write_text(card,encoding="utf-8")
        write_json(stage/"manifest.json",report)
        if destination.exists():
            raise LabError("An existing request sample appeared; refusing to replace it.")
        os.replace(stage,destination)
    finally:
        remove_generated(stage)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report
