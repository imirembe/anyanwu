"""Bounded official CLINC original-OOS rows, with no filter/full-file fallback."""
import hashlib
import json
import os
import re
from urllib.parse import urlencode

from . import importer
from .resources import LAB,MIB,LabError,ensure_capacity,owned,remove_generated,write_json

LAYOUT={"train":(15000,64,15250),"validation":(3000,16,3100),"test":(4500,16,5500)}


def import_original_oos(source,rows,token_prompt=False,client=None):
    spec=importer.SOURCES[source]
    if type(rows) is not int or rows!=96:raise LabError("The declared original-OOS sample is exactly 96 records: 64/16/16.")
    destination=importer.sample_directory(source)
    if destination.exists():raise LabError("The original-OOS sample already exists and is preserved.")
    ensure_capacity(15*MIB)
    client=client or importer.BoundedHubClient(importer.private_token(token_prompt))
    url=f"https://huggingface.co/api/datasets/{spec['dataset']}"
    metadata=client.get(url);revision=metadata.get("sha")
    if revision!=spec["required_revision"]:raise LabError("The pinned original-OOS source revision changed.")
    license_label=metadata.get("cardData",{}).get("license")
    if license_label not in (spec["license"],[spec["license"]]):raise LabError("Original-OOS source license label changed.")
    card=client.get(f"https://huggingface.co/datasets/{spec['dataset']}/raw/{revision}/README.md",json_response=False)
    splits=client.get("https://datasets-server.huggingface.co/splits?"+urlencode({"dataset":spec["dataset"]}))
    available={(s.get("config"),s.get("split")) for s in splits.get("splits",[])}
    if any((spec["config"],s) not in available for s in LAYOUT):raise LabError("All official OOS folds are required.")
    info=client.get("https://datasets-server.huggingface.co/info?"+urlencode({"dataset":spec["dataset"],"config":spec["config"]}))
    feature=info.get("dataset_info",{}).get("features",{}).get("intent",{});labels=feature.get("names")
    if (info.get("partial") is True or feature.get("_type")!="ClassLabel" or not isinstance(labels,list)
            or not 8<=len(labels)<=256
            or not all(isinstance(x,str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}",x) for x in labels)
            or len(set(labels))!=len(labels) or "oos" not in labels):
        raise LabError("Official OOS ClassLabel map is missing or invalid.")
    label_id=labels.index("oos");records=[]
    for split,(start,count,total) in LAYOUT.items():
        query={"dataset":spec["dataset"],"config":spec["config"],"split":split,"offset":start,"length":count}
        page=client.get("https://datasets-server.huggingface.co/rows?"+urlencode(query))
        if (not isinstance(page,dict) or page.get("partial") is True or page.get("num_rows_total")!=total
                or any(page.get(k,v)!=v for k,v in query.items() if k in {"dataset","config","split"})
                or page.get("dataset_git_revision") not in {None,revision}):raise LabError("The declared OOS row layout changed.")
        names=next((f.get("type",{}).get("names") for f in page.get("features",[]) if isinstance(f,dict) and f.get("name")=="intent"),None)
        entries=page.get("rows")
        if names!=labels or not isinstance(entries,list) or len(entries)!=count:raise LabError("OOS quota/map incomplete; no partial corpus installed.")
        for i,e in enumerate(entries):
            if (not isinstance(e,dict) or type(e.get("row_idx")) is not int or e["row_idx"]!=start+i
                    or not isinstance(e.get("row"),dict) or type(e["row"].get("intent")) is not int
                    or e["row"]["intent"]!=label_id or not isinstance(e.get("truncated_cells",[]),list)
                    or bool(e.get("truncated_cells",[]))):raise LabError("Wrong/truncated OOS class or original row index.")
            index=e["row_idx"]
            record=importer.text_record(spec,revision,split,index,{"text":e["row"].get("text"),"intent":"oos","example_id":f"{split}-{index:08d}"})
            if record is None:raise LabError("Invalid OOS text; no partial sample installed.")
            record.update(source_intent_id=label_id,source_example_id_kind="derived-from-official-split-and-row-index; not a publisher example_id")
            records.append(record)
        print(f"Original OOS {split}: {count} source rows checked.",flush=True)
    if client.get(url).get("sha")!=revision:raise LabError("Original-OOS source changed during import.")
    blob="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records).encode()
    if len(blob)>importer.MAX_TRANSFER:raise LabError("The OOS output exceeds the 5 MiB cap.")
    report={"source":spec["dataset"],"revision":revision,"license_declared":spec["license"],"origin":spec["origin"],"restrictions":spec["restrictions"],
            "records":len(records),"requested_records":rows,"rows_scanned":len(records),"batch_start_offset":0,"source_config":spec["config"],
            "sample_counts_by_source_split":{s:n for s,(o,n,t) in LAYOUT.items()},"source_row_offsets":{s:o for s,(o,n,t) in LAYOUT.items()},
            "source_class_names":labels,"source_oos_class_id":label_id,"selection_note":"Explicit pinned original-OOS tail slices; check every label/map/index and full split size. No filter query or full-file fallback.",
            "http_body_bytes":client.received,"http_budget_bytes":importer.MAX_TRANSFER,"requests":client.requests,"saved_text_bytes":len(blob),
            "candidates_sha256":hashlib.sha256(blob).hexdigest(),"review_status":"pending","audio_downloaded":False,"slot_annotations_retained":False,
            "snapshot_note":"Hub head checked before/after; viewer row revision not independently guaranteed."}
    ensure_capacity(len(blob)+len(card.encode())+MIB)
    stage=owned(LAB/"imports/import-staging");remove_generated(stage);stage.mkdir(parents=True)
    try:
        (stage/"candidates.jsonl").write_bytes(blob);(stage/"source-card.md").write_text(card,encoding="utf-8");write_json(stage/"manifest.json",report)
        if destination.exists():raise LabError("An OOS sample appeared; refusing to replace it.")
        os.replace(stage,destination)
    finally:remove_generated(stage)
    print(json.dumps({k:report[k] for k in ("source","records","sample_counts_by_source_split","http_body_bytes","requests","audio_downloaded")},indent=2))
    return report
