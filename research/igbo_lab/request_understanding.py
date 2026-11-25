"""Versioned intent-and-raw-details experiment, separate from frozen v1.

No pretrained weights, generated replies, recordings or phone actions. Prepare
fits training text and calibrates on validation; evaluation is a one-shot step.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import time

from . import details, intent
from .experiments import assembly_identity
from .resources import LAB, MIB, LabError, ensure_capacity, owned, project_input, write_json
from .review import load_jsonl, prompt_key

NAME="request-understanding-v2"
SNAPSHOT=LAB/"request_data_snapshots/88f830477b4f9807"
REVIEW=Path(__file__).parent/"data/request_details_v2_review.jsonl"
MAX_SECONDS=60
BASE_ARTIFACTS={"plan.json","dataset.jsonl","quarantine.jsonl","detail_review.jsonl","calibration.json"}
COMPLETE_ARTIFACTS=BASE_ARTIFACTS|{"result.json"}


def sha(path):
    path=project_input(path)
    if path.stat().st_size>5*MIB: raise LabError("Request artifacts must stay below 5 MiB each.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementations():
    return {name:sha(Path(__file__).parent/name) for name in ("request_understanding.py","details.py","intent.py")}


def folder_for(name):
    if not isinstance(name,str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}",name):
        raise LabError("Use a short request experiment name without paths.")
    return owned(LAB/"request_experiments"/name)


def read_json(path):
    path=owned(path)
    if not path.is_file() or path.stat().st_size>2*MIB: raise LabError("A request experiment artifact is missing or oversized.")
    return json.loads(path.read_text())


def validate_rows(rows):
    if not 27<=len(rows)<=1000: raise LabError("Use a bounded request snapshot with every label in each fold.")
    ids,texts,groups=set(),set(),{}
    for r in rows:
        intent.canonical(r.get("text"))
        label,split=r.get("label"),r.get("split")
        if label not in (*intent.LABELS,"unknown") or split not in {"train","validation","test"}:
            raise LabError("Invalid request label or official split.")
        if r.get("source_language_code") not in {"ibo","eng","ig-en"}:
            raise LabError("Request language provenance is missing.")
        key=prompt_key(r["text"]);group=(r.get("source"),r.get("group"))
        if not r.get("id") or r["id"] in ids or key in texts or not all(group):
            raise LabError("Request IDs/text must be unique and source groups explicit.")
        if group in groups and groups[group]!=split: raise LabError("A request source group crosses splits.")
        ids.add(r["id"]);texts.add(key);groups[group]=split
    kept=[r for r in rows if r["source_language_code"]!="ig-en"]
    quarantine=[r for r in rows if r["source_language_code"]=="ig-en"]
    if any(r["label"]!="unknown" or r["split"]!="train" for r in quarantine):
        raise LabError("The declared mixed-source quarantine layout changed.")
    for split in ("train","validation","test"):
        if {r["label"] for r in kept if r["split"]==split}!=set(intent.LABELS)|{"unknown"}:
            raise LabError("Every experimental fold needs eight supported classes and unknown requests.")
    return kept,quarantine


def load_snapshot(path):
    path=owned(Path(path).resolve())
    manifest=read_json(path/"manifest.json")
    if sha(path/"dataset.jsonl")!=manifest.get("dataset_sha256") or sha(path/"test.jsonl")!=manifest.get("test_sha256"):
        raise LabError("The frozen request snapshot changed.")
    rows=load_jsonl(path/"dataset.jsonl")
    if [r for r in rows if r["split"]=="test"]!=load_jsonl(path/"test.jsonl"):
        raise LabError("The frozen test file differs from the snapshot test fold.")
    _,assembly=assembly_identity(owned(LAB/manifest["assembly"]))
    if (assembly["assembly_sha256"]!=manifest.get("assembly_sha256")
            or assembly["records_sha256"]!=manifest.get("assembly_records_sha256")):
        raise LabError("The request snapshot no longer matches its source assembly.")
    original={r["id"]:r for r in load_jsonl(LAB/manifest["assembly"]/"records.jsonl")}
    if set(original)!={r.get("id") for r in rows}: raise LabError("Snapshot and assembly identities differ.")
    for r in rows:
        parent=original[r["id"]]
        if (any(r.get(k)!=parent.get(k) for k in ("text","source","group","split","source_content_sha256","review","source_language_code"))
                or r.get("label")!=parent["review"].get("request_scope")):
            raise LabError("Snapshot text/labels/reviews do not match their curated source.")
    # Old test/validation inputs never become candidate training data.
    protected={prompt_key(r["text"]) for r in load_jsonl(LAB/"intent_experiments/intent-prototype-v1/dataset.jsonl")}
    protected.update(prompt_key(r["prompt"]) for r in load_jsonl(intent.FROZEN_CASES))
    if any(prompt_key(r["text"]) in protected for r in rows): raise LabError("The fresh request data overlaps an earlier protected query.")
    kept,quarantine=validate_rows(rows)
    return kept,quarantine,manifest


def load_detail_review(path,rows):
    selected={r["id"]:r for r in rows if r["split"]!="train"}
    output={}
    for record in load_jsonl(path):
        row=selected.get(record.get("id"))
        if (not row or row["id"] in output or record.get("review_type")!="AI" or not record.get("reviewer")
                or record.get("confidence") not in {"medium","high"} or not record.get("reason")
                or record.get("text_sha256")!=details.offset_reference(row["text"])["transcript_sha256"]
                or record.get("source_content_sha256")!=row.get("source_content_sha256")
                or record.get("split")!=row["split"] or record.get("label")!=row["label"]):
            raise LabError("Every held-out detail reference needs a matching explicit AI span review.")
        raw=details.transcript(row["text"]).encode();spans=record.get("spans")
        if not isinstance(spans,list) or len(spans)>16: raise LabError("Invalid detail reference list.")
        seen=set()
        for s in spans:
            if not isinstance(s,dict): raise LabError("Invalid detail reference span.")
            a,b=s.get("start_byte"),s.get("limit_byte")
            if (s.get("field") not in details.SCOPE.get(row["label"],()) or type(a) is not int or type(b) is not int
                    or not 0<=a<b<=len(raw) or not isinstance(s.get("value"),str)):
                raise LabError("Detail references require allowed fields and UTF-8 boundaries.")
            try: value=raw[a:b].decode()
            except UnicodeDecodeError as error: raise LabError("Detail reference splits a UTF-8 character.") from error
            key=(s["field"],a,b)
            if value!=s["value"] or key in seen: raise LabError("Detail reference text/identity mismatch.")
            seen.add(key)
        output[row["id"]]=record
    if set(output)!=set(selected): raise LabError("Review every validation/test record, including empty spans in the declared scope.")
    return output


def load_run(name,check_implementation=True):
    folder=folder_for(name);state=read_json(folder/"run.json")
    if state.get("status") not in {"prepared","complete"}:
        raise LabError("This request experiment is incomplete/failed; it is not retried automatically.")
    expected=COMPLETE_ARTIFACTS if state["status"]=="complete" else BASE_ARTIFACTS
    if set(state.get("artifact_sha256",{}))!=expected: raise LabError("Request artifact inventory changed.")
    for name,digest in state["artifact_sha256"].items():
        if sha(owned(folder/name))!=digest: raise LabError("A request experiment artifact changed after preparation.")
    plan=read_json(folder/"plan.json")
    if check_implementation and plan["implementation_sha256"]!=implementations():
        raise LabError("Request implementation changed after registration; preserve the run and declare a different experiment.")
    return folder,state,plan


def runtime_guard(start):
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024)
    ensure_capacity()
    if time.perf_counter()-start>MAX_SECONDS or peak>512*MIB: raise LabError("Request experiment exceeded its time/memory limit.")
    return peak


def make_plan(name,rows,quarantine,manifest,review_path,baseline):
    return {"format":1,"name":name,"registered_at":datetime.now(timezone.utc).isoformat(),
            "source_dataset_sha256":manifest["dataset_sha256"],"source_test_sha256":manifest["test_sha256"],
            "source_assembly_sha256":manifest["assembly_sha256"],"detail_review_sha256":sha(review_path),
            "implementation_sha256":implementations(),
            "algorithm":"Frozen v1 TF-IDF feature/index algorithm; new training data and validation-only calibration. Independent deterministic raw-reference parser.",
            "features":intent.FEATURES,"threshold_grid":intent.GRID,
            "threshold_selection":"Unchanged v1 objective: validation accepted errors, correct known acceptance, then lower gate thresholds.",
            "splits":dict(Counter(r["split"] for r in rows)),"excluded_mixed_training_records":len(quarantine),
            "quarantine_reason":"Mixed source has only unknown examples from one scripted file. Exclude it entirely from fitting/calibration/testing to avoid coupling mixed language with rejection.",
            "detail_fields":list(details.FIELDS),"detail_scope":details.SCOPE,
            "detail_reference_conventions":"AI references include explicit raw clock/daypart/date phrases and geographic references, including ones the parser cannot recognise. Strip surrounding prepositions where separable. Preserve relations as contact references; do not resolve names, dates or spelling. Empty lists mean no target in the declared scope, not no other entities.",
            "details_policy":"Rule-based tentative raw references, NFC/UTF-8 spans. No publisher slot labels used to fit or score: byte-valid offsets can still be semantically wrong. Fresh explicit AI span references only; no new language authored.",
            "evaluation_policy":"Fit IDF/neighbours on train only, calibrate on validation only. Freeze preparation artifacts before one shared fresh 59-case baseline/candidate evaluation. Report intent by language/class; detail exact spans both with true intent and end to end. Do not tune or retry after test results.",
            "baseline_artifact_sha256":{p:sha(baseline/p) for p in ("plan.json","dataset.jsonl","calibration.json","result.json")},
            "baseline_name":intent.NAME,"runtime":{"python":platform.python_version()},
            "limits":{"seconds_per_phase":MAX_SECONDS,"peak_process_bytes":512*MIB,"persisted_weights":False,"additional_checkpoint_slots":0},
            "coverage_limits":["Igbo and English task text only; no supported mixed tasks or independent mixed test.","English unknown rejection is untested: unknown validation/test records are Igbo only.","Four raw detail-reference types only; message body, event, music and shopping-item parsing are not implemented.","AI span review is not native-speaker validation; no dates, contacts or locations are resolved.","No speech recognition, Igbo reply generation, TTS or phone actions are evaluated."],
            "phone_actions_enabled":False,"production_ready":False}


def prepare_experiment(snapshot=SNAPSHOT,review_path=REVIEW,name=NAME):
    folder=folder_for(name)
    if folder.exists():
        folder,state,_=load_run(name)
        report={"name":name,"status":state["status"],"reused":True,"test_scored":state["status"]=="complete"}
        print(json.dumps(report,indent=2));return report
    rows,quarantine,manifest=load_snapshot(snapshot)
    review_path=project_input(review_path);gold=load_detail_review(review_path,rows)
    baseline,_,_,_=intent.load_completed(intent.NAME)
    plan=make_plan(name,rows,quarantine,manifest,review_path,baseline)
    ensure_capacity(8*MIB);folder.mkdir(parents=True)
    write_json(folder/"plan.json",plan)
    for filename,data in (("dataset.jsonl",rows),("quarantine.jsonl",quarantine),("detail_review.jsonl",list(gold.values()))):
        blob="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in data).encode()
        if len(blob)>2*MIB: raise LabError("Request preparation artifact exceeds its 2 MiB cap.")
        (folder/filename).write_bytes(blob)
    state={"name":name,"status":"preparing"};write_json(folder/"run.json",state);start=time.perf_counter()
    try:
        train=[r for r in rows if r["split"]=="train"];validation=[r for r in rows if r["split"]=="validation"]
        model=intent.IntentIndex(train);fit_seconds=time.perf_counter()-start;runtime_guard(start)
        scores=[]
        for row in validation: scores.append(model.score(row["text"]));runtime_guard(start)
        calibration=intent.calibrate(validation,scores)
        thresholds=calibration["selected"]["thresholds"]
        calibration.update(validation_details_oracle=details.detail_metrics(validation,[details.extract_details(r["text"],r["label"]) for r in validation],gold),
                           validation_details_end_to_end=details.detail_metrics(validation,[details.extract_details(r["text"],intent.decide(s,thresholds)) for r,s in zip(validation,scores)],gold),
                           training_examples=len(train),features=len(model.idf),derived_coefficients=model.coefficients,
                           fit_seconds=round(fit_seconds,4),process_peak_bytes=runtime_guard(start),test_scored=False)
        write_json(folder/"calibration.json",calibration)
        state.update(status="prepared",artifact_sha256={f:sha(folder/f) for f in BASE_ARTIFACTS});write_json(folder/"run.json",state)
        report={"name":name,"status":"prepared","splits":plan["splits"],"quarantined_mixed":len(quarantine),
                "validation":calibration["selected"]["metrics"],"test_scored":False,"directory":str(folder)}
        print(json.dumps(report,ensure_ascii=False,indent=2));return report
    except BaseException as error:
        write_json(folder/"run.json",{**state,"status":"failed","error_type":type(error).__name__});raise


def intent_report(rows,scores,thresholds):
    overall=intent.metrics(rows,scores,thresholds)
    by_language={}
    for language in sorted({r["source_language_code"] for r in rows}):
        pairs=[(r,s) for r,s in zip(rows,scores) if r["source_language_code"]==language]
        by_language[language]=intent.metrics([r for r,s in pairs],[s for r,s in pairs],thresholds)
    return {"overall":overall,"by_language":by_language}


def evaluate_experiment(name=NAME):
    folder,state,plan=load_run(name)
    if state["status"]=="complete":
        result=read_json(folder/"result.json");print(json.dumps({"reused_completed_experiment":True,"result":result},ensure_ascii=False,indent=2));return result
    baseline,_,base_calibration,_=intent.load_completed(plan["baseline_name"])
    if any(sha(baseline/f)!=h for f,h in plan["baseline_artifact_sha256"].items()): raise LabError("The frozen baseline changed.")
    rows=load_jsonl(folder/"dataset.jsonl");validate_rows(rows)
    gold=load_detail_review(folder/"detail_review.jsonl",rows)
    calibration=read_json(folder/"calibration.json");thresholds=calibration["selected"]["thresholds"]
    state.update(status="evaluating");write_json(folder/"run.json",state);start=time.perf_counter()
    try:
        model=intent.IntentIndex([r for r in rows if r["split"]=="train"])
        base=intent.IntentIndex([r for r in load_jsonl(baseline/"dataset.jsonl") if r["split"]=="train"])
        test=[r for r in rows if r["split"]=="test"];scores=[];base_scores=[];durations=[]
        for r in test:
            began=time.perf_counter();scores.append(model.score(r["text"]));durations.append(time.perf_counter()-began)
            base_scores.append(base.score(r["text"]));runtime_guard(start)
        predictions=[intent.decide(s,thresholds) for s in scores]
        oracle=[details.extract_details(r["text"],r["label"]) for r in test]
        end_to_end=[details.extract_details(r["text"],p) for r,p in zip(test,predictions)]
        result={"name":name,"plan_sha256":sha(folder/"plan.json"),"thresholds":thresholds,
                "candidate":intent_report(test,scores,thresholds),
                "baseline_on_same_fresh_test":intent_report(test,base_scores,base_calibration["selected"]["thresholds"]),
                "details_oracle_intent":details.detail_metrics(test,oracle,gold),
                "details_end_to_end":details.detail_metrics(test,end_to_end,gold),
                "test_predictions":[{"id":r["id"],"expected":r["label"],"prediction":p,"similarity":s["similarity"],"margin":s["margin"],"details":d} for r,p,s,d in zip(test,predictions,scores,end_to_end)],
                "mean_candidate_core_inference_ms":round(sum(durations)/len(durations)*1000,3),
                "elapsed_seconds":round(time.perf_counter()-start,4),"process_peak_bytes":runtime_guard(start),
                "persisted_weights":False,"test_scored_once":True,"original_v1_test_repeated":False,"frozen_reply_benchmark_repeated":False,
                "native_speaker_validated":False,"audio_evaluated":False,"phone_actions_enabled":False,"production_ready":False,
                "limitations":plan["coverage_limits"]}
        write_json(folder/"result.json",result)
        state.update(status="complete",artifact_sha256={f:sha(folder/f) for f in COMPLETE_ARTIFACTS});write_json(folder/"run.json",state)
        write_json(LAB/"latest_request_experiment.json",{"name":name,"directory":str(folder),"result_sha256":sha(folder/"result.json"),"production_ready":False})
        print(json.dumps({k:result[k] for k in ("name","candidate","baseline_on_same_fresh_test","details_oracle_intent","details_end_to_end","production_ready")},ensure_ascii=False,indent=2));return result
    except BaseException as error:
        write_json(folder/"run.json",{**state,"status":"failed","error_type":type(error).__name__});raise


def understand_request(text,name=NAME):
    folder,_,plan=load_run(name)
    rows=load_jsonl(folder/"dataset.jsonl")
    model=intent.IntentIndex([r for r in rows if r["split"]=="train"])
    score=model.score(text);calibration=read_json(folder/"calibration.json")
    label=intent.decide(score,calibration["selected"]["thresholds"])
    result={"experiment":name,"intent":None if label=="unknown" else label,
            "decision":"unknown" if label=="unknown" else "candidate",
            "tentative_details":details.extract_details(text,label),"offset_reference":details.offset_reference(text),
            "similarity":round(score["similarity"],4),"margin":round(score["margin"],4),
            "score_kind":"Cosine similarity, not a calibrated probability.",
            "entities_resolved":False,"operation_resolved":False,"requires_review":True,
            "limitations":plan["coverage_limits"],"action_executed":False,"transcript_saved":False,"production_ready":False}
    print(json.dumps(result,ensure_ascii=False,indent=2));return result
