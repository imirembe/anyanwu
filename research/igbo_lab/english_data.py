"""Small published CLINC English request samples with official ClassLabel mapping."""
import hashlib
import json
import os
import re
from urllib.parse import urlencode

from . import importer
from .resources import LAB, MIB, LabError, ensure_capacity, owned, remove_generated, write_json


def import_english_requests(source,rows,token_prompt=False,client=None):
    spec=importer.SOURCES[source]
    if spec.get("original_oos"):
        from .english_oos import import_original_oos
        return import_original_oos(source,rows,token_prompt,client)
    if type(rows) is not int or not 96<=rows<=480 or rows%96:
        raise LabError("Use 96, 192, 288, 384 or 480 English requests: eight intents with official train/validation/test quotas.")
    scale=rows//96
    lookup_rows=spec.get("class_lookup_rows",2)
    if type(lookup_rows) is not int or lookup_rows not in {1,2}:raise LabError("Use a one- or two-row class-position lookup.")
    for split,quota in (("train",8*scale),("validation",2*scale),("test",2*scale)):
        within=spec.get("within_class_start",{}).get(split,0)
        if type(within) is not int or within<0 or within+quota>{"train":100,"validation":20,"test":30}[split]:
            raise LabError("The planned English slice exceeds its source class block.")
    declared_blocks=spec.get("class_block_positions")
    if declared_blocks is not None and (set(declared_blocks)!=set(spec["intents"])
            or any(type(v) is not int or not 0<=v<150 for v in declared_blocks.values()) or not spec.get("required_revision")):
        raise LabError("Declared English class positions require every label and a pinned revision.")
    destination=importer.sample_directory(source)
    if destination.exists():raise LabError("This English sample already exists and is preserved.")
    ensure_capacity(15*MIB)
    client=client or importer.BoundedHubClient(importer.private_token(token_prompt))
    url=f"https://huggingface.co/api/datasets/{spec['dataset']}"
    meta=client.get(url);revision=meta.get("sha");license_label=meta.get("cardData",{}).get("license")
    if not isinstance(revision,str) or not re.fullmatch(r"[a-f0-9]{40}",revision):raise LabError("English source revision unavailable.")
    if spec.get("required_revision",revision)!=revision:raise LabError("The pinned English source revision changed; inspect the release first.")
    if license_label not in (spec["license"],[spec["license"]]):raise LabError("English source license label changed; inspect the release first.")
    card=client.get(f"https://huggingface.co/datasets/{spec['dataset']}/raw/{revision}/README.md",json_response=False)
    splits=client.get("https://datasets-server.huggingface.co/splits?"+urlencode({"dataset":spec["dataset"]}))
    available={(s.get("config"),s.get("split")) for s in splits.get("splits",[])}
    if any((spec["config"],s) not in available for s in spec["splits"]):raise LabError("All official English source splits are required.")
    info=client.get("https://datasets-server.huggingface.co/info?"+urlencode({"dataset":spec["dataset"],"config":spec["config"]}))
    feature=info.get("dataset_info",{}).get("features",{}).get("intent",{})
    labels=feature.get("names")
    if (info.get("partial") is True or feature.get("_type")!="ClassLabel" or not isinstance(labels,list)
            or not 8<=len(labels)<=256 or not all(isinstance(x,str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}",x) for x in labels)
            or len(set(labels))!=len(labels) or not set(spec["intents"]).issubset(labels)):
        raise LabError("The official English ClassLabel map is missing/invalid; numeric labels are never guessed.")
    records=[];seen=set();positions=set();counts={};totals={};blocks=dict(declared_blocks or {})
    for split,quota in (("train",8*scale),("validation",2*scale),("test",2*scale)):
        counts[split]={};totals[split]={}
        for label in spec["intents"]:
            label_id=labels.index(label)
            block_rows={"train":100,"validation":20,"test":30}[split]
            print(f"English {split}/{label}: checking bounded source rows.",flush=True)
            if split=="train" and declared_blocks is None:
                # A bounded lookup establishes the source class position. Fetch
                # the actual quota through row slices, also used for holdouts.
                lookup={"dataset":spec["dataset"],"config":spec["config"],"split":split,"where":f'"intent"={label_id}',"offset":0,"length":lookup_rows}
                found=client.get("https://datasets-server.huggingface.co/filter?"+urlencode(lookup))
                entries=found.get("rows",[]) if isinstance(found,dict) else []
                names=next((f.get("type",{}).get("names") for f in found.get("features",[]) if isinstance(f,dict) and f.get("name")=="intent"),None) if isinstance(found,dict) else None
                if (not isinstance(found,dict) or found.get("partial") is True or found.get("num_rows_total")!=100 or names!=labels or len(entries)!=lookup_rows
                        or any(found.get(k,v)!=v for k,v in lookup.items() if k in {"dataset","config","split"})
                        or found.get("dataset_git_revision") not in {None,revision}
                        or any(not isinstance(e,dict) or type(e.get("row_idx")) is not int
                        or not isinstance(e.get("truncated_cells",[]),list) or bool(e.get("truncated_cells",[]))
                        or not isinstance(e.get("row"),dict) or type(e["row"].get("intent")) is not int or e["row"]["intent"]!=label_id for e in entries)
                        or entries[0]["row_idx"]%100 or any(e["row_idx"]!=entries[0]["row_idx"]+i for i,e in enumerate(entries))):
                    raise LabError("The English class-position lookup was incomplete or changed.")
                blocks[label]=entries[0]["row_idx"]//100
            within=spec.get("within_class_start",{}).get(split,0)
            query={"dataset":spec["dataset"],"config":spec["config"],"split":split,"offset":blocks[label]*block_rows+within,"length":quota}
            endpoint="rows"
            page=client.get(f"https://datasets-server.huggingface.co/{endpoint}?"+urlencode(query))
            if (not isinstance(page,dict) or any(page.get(k,v)!=v for k,v in query.items() if k in {"dataset","config","split"})
                    or page.get("dataset_git_revision") not in {None,revision} or page.get("partial") is True):
                raise LabError("The English viewer returned an inconsistent source/subset/split/revision.")
            features=page.get("features",[])
            returned=next((f.get("type",{}).get("names") for f in features if isinstance(f,dict) and f.get("name")=="intent"),None)
            if returned!=labels:raise LabError("The row-page label map differs from the official English map.")
            entries=page.get("rows")
            if not isinstance(entries,list) or len(entries)!=quota:raise LabError("The bounded English quota was not filled; no partial sample installed.")
            if split=="train":
                first=entries[0].get("row_idx") if isinstance(entries[0],dict) else None
                if type(first) is not int or first!=query["offset"] or page.get("num_rows_total")!=15250:
                    raise LabError("The observed English training class-block layout changed.")
                expected_start=first
            else:
                expected_start=query["offset"]
                if page.get("num_rows_total")!={"validation":3100,"test":5500}[split]:
                    raise LabError("The observed English holdout size changed; inspect its layout first.")
            totals[split][label]=block_rows
            for local_index,entry in enumerate(entries):
                if not isinstance(entry,dict):raise LabError("Invalid English row envelope.")
                truncated=entry.get("truncated_cells",[])
                if not isinstance(truncated,list) or not all(isinstance(c,str) for c in truncated) or {"text","intent"}.intersection(truncated):
                    raise LabError("Truncated English text or label refused.")
                row,index=entry.get("row"),entry.get("row_idx")
                if (not isinstance(row,dict) or type(row.get("intent")) is not int or row["intent"]!=label_id
                        or type(index) is not int or index!=expected_start+local_index or (split,index) in positions):
                    raise LabError("English row has a wrong class ID, repeated identity or invalid original index.")
                positions.add((split,index))
                derived={"text":row.get("text"),"intent":label,"example_id":f"{split}-{index:08d}"}
                record=importer.text_record(spec,revision,split,index,derived)
                if record is None or record["id"] in seen:raise LabError("Invalid/repeated English request; no sample installed.")
                record.update(source_intent_id=label_id,source_example_id_kind="derived-from-official-split-and-row-index; not a publisher example_id")
                seen.add(record["id"]);records.append(record)
            counts[split][label]=quota
        print(f"English {split}: {sum(counts[split].values())} published requests checked.",flush=True)
    if client.get(url).get("sha")!=revision:raise LabError("English dataset changed during import; no sample installed.")
    payload="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records).encode()
    if len(payload)>importer.MAX_TRANSFER:raise LabError("English output exceeds the 5 MiB sample cap.")
    report={"source":spec["dataset"],"revision":revision,"license_declared":spec["license"],"license_label_observed":license_label,
            "paper":"https://aclanthology.org/D19-1131/","origin":spec["origin"],"restrictions":spec["restrictions"],"source_config":spec["config"],
            "records":len(records),"rows_scanned":len(records),"requested_records":rows,"batch_start_offset":0,
            "sample_counts_by_source_split":{s:sum(c.values()) for s,c in counts.items()},"sample_counts_by_split_intent":counts,
            "indexed_counts_by_split_intent":totals,"source_class_names":labels,
            "selection_note":"Class positions are declared from a pinned prior verified sample or established by two-row training lookups. Every quota uses small row slices; verify page class maps, numeric labels, indices and fixed split sizes. No full-file fallback; convenience sample, not random.",
            "within_class_start":{s:spec.get("within_class_start",{}).get(s,0) for s in spec["splits"]},
            "class_position_origin":"pinned-prior-verified-class-blocks" if declared_blocks is not None else f"{lookup_rows}-row-class-filter-lookups",
            "class_lookup_rows":lookup_rows if declared_blocks is None else 0,
            "source_class_block_positions":blocks,"viewer_endpoints_by_source_split":{"train":"rows" if declared_blocks is not None else "filter-position-lookup+rows","validation":"rows","test":"rows"},
            "snapshot_note":"Hub head matched before/after; indexed row revision not independently guaranteed.",
            "http_body_bytes":client.received,"http_budget_bytes":importer.MAX_TRANSFER,"requests":client.requests,"saved_text_bytes":len(payload),
            "candidates_sha256":hashlib.sha256(payload).hexdigest(),"review_status":"pending","slot_annotations_retained":False,"audio_downloaded":False,
            "purpose":"Published English request understanding; no assistant reply targets"}
    ensure_capacity(len(payload)+len(card.encode())+MIB)
    stage=owned(LAB/"imports/import-staging");stage.parent.mkdir(parents=True,exist_ok=True);remove_generated(stage);stage.mkdir()
    try:
        (stage/"candidates.jsonl").write_bytes(payload);(stage/"source-card.md").write_text(card,encoding="utf-8")
        write_json(stage/"manifest.json",report)
        if destination.exists():raise LabError("An existing English sample appeared; refusing to replace it.")
        os.replace(stage,destination)
    finally:remove_generated(stage)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report
