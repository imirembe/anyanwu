"""Small TF-IDF request classifier, fitted in RAM from published training text.

No neural checkpoints, pretrained weights, recordings, replies or phone actions.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import resource
import time
import unicodedata

from .assembly import assemble_reviewed
from .experiments import assembly_identity
from .importer import SOURCES
from .resources import LAB, MIB, MAX_PARAMETERS, LabError, ensure_capacity, owned, project_input, write_json
from .review import load_jsonl, prompt_key

LABELS = tuple(SOURCES["injongo-igbo"]["intents"])
FEATURES = {"character_ngram_lengths":[2,3,4,5], "word_weight":1.0, "character_weight":0.5,
            "idf":"log((1 + train documents)/(1 + train document frequency)) + 1",
            "maximum_features":100_000, "maximum_coefficients":500_000,
            "maximum_query_characters":2048, "maximum_training_documents":512}
GRID = {"minimum_similarity":[0.15,0.25,0.35,0.45,0.55,0.65], "minimum_margin":[0.0,0.03,0.06,0.10,0.15],
        "minimum_feature_coverage":0.35}
NAME = "intent-prototype-v1"
MAX_SECONDS = 60
FROZEN_CASES = Path(__file__).parent/"data/voice_eval_v1.jsonl"


def sha(path):
    if path.stat().st_size>5*MIB:
        raise LabError("An intent artifact exceeds the 5 MiB hash/read budget.")
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        while chunk:=handle.read(64*1024):digest.update(chunk)
    return digest.hexdigest()


def canonical(text):
    if not isinstance(text,str) or len(text)>FEATURES["maximum_query_characters"] or "\x00" in text:
        raise LabError("Use a speech transcript of at most 2,048 characters without NULs.")
    return re.sub(r"\s+"," ",unicodedata.normalize("NFC",text).casefold().replace("’","'")).strip()


def features(text):
    result=Counter()
    for word in canonical(text).split():
        word=word.strip('.,!?;:"“”()[]{}')
        if not word:continue
        result["w:"+word]+=1
        padded="^"+word+"$"
        for size in FEATURES["character_ngram_lengths"]:
            for start in range(len(padded)-size+1):result["c:"+padded[start:start+size]]+=1
    return result


class IntentIndex:
    """Nearest training example per label, with a published-question background."""
    def __init__(self, training):
        if not training or len(training)>FEATURES["maximum_training_documents"] or any(r["split"]!="train" for r in training):
            raise LabError("Fit the intent index on bounded training rows only.")
        if {r["label"] for r in training} != set(LABELS)|{"unknown"}:
            raise LabError("Training needs all eight request labels and background examples.")
        raw=[];df=Counter();nonzero=0
        for row in training:
            vector=features(row["text"]);nonzero+=len(vector);df.update(vector.keys())
            if len(df)>FEATURES["maximum_features"] or nonzero>FEATURES["maximum_coefficients"]:
                raise LabError("The intent feature budget was exceeded.")
            raw.append(vector)
        self.idf={key:math.log((1+len(training))/(1+count))+1 for key,count in df.items()}
        self.documents=[(r["label"],r["id"],self.vector(vector)[0]) for r,vector in zip(training,raw)]
        self.coefficients=len(self.idf)+sum(len(v) for _,_,v in self.documents)
        if self.coefficients>min(FEATURES["maximum_coefficients"],MAX_PARAMETERS):
            raise LabError("The derived intent coefficients exceed the pilot budget.")

    def vector(self,counts):
        total=sum(counts.values())
        coverage=sum(n for key,n in counts.items() if key in self.idf)/total if total else 0.0
        vector={key:(1+math.log(n))*self.idf[key]*(FEATURES["word_weight"] if key.startswith("w:") else FEATURES["character_weight"])
                for key,n in counts.items() if key in self.idf}
        norm=math.sqrt(sum(v*v for v in vector.values()))
        return ({key:v/norm for key,v in vector.items()} if norm else {}),coverage

    def score(self,text):
        query,coverage=self.vector(features(text))
        scores={label:0.0 for label in (*LABELS,"unknown")}
        nearest={label:None for label in scores}
        for label,identifier,document in self.documents:
            small,large=(query,document) if len(query)<len(document) else (document,query)
            similarity=max(0.0,min(1.0,sum(value*large.get(key,0.0) for key,value in small.items())))
            if similarity>scores[label]:scores[label],nearest[label]=similarity,identifier
        ordered=sorted(LABELS,key=lambda label:(-scores[label],label))
        first,second=ordered[:2]
        return {"nearest_known_label":first,"similarity":scores[first],"margin":scores[first]-max(scores[second],scores["unknown"]),
                "background_similarity":scores["unknown"],"feature_coverage":coverage,"nearest_training_id":nearest[first],
                "scores":scores}


def decide(scored,thresholds):
    accepted=(scored["similarity"]>0 and scored["similarity"]>=thresholds["minimum_similarity"]
              and scored["margin"]>0 and scored["margin"]>=thresholds["minimum_margin"]
              and scored["feature_coverage"]>=thresholds["minimum_feature_coverage"])
    return scored["nearest_known_label"] if accepted else "unknown"


def metrics(rows,scores,thresholds):
    predictions=[decide(s,thresholds) for s in scores]
    known=[i for i,r in enumerate(rows) if r["label"]!="unknown"]
    unknown=[i for i,r in enumerate(rows) if r["label"]=="unknown"]
    accepted=[i for i,p in enumerate(predictions) if p!="unknown"]
    correct=sum(predictions[i]==rows[i]["label"] for i in accepted)
    errors=len(accepted)-correct
    per_label={}
    for label in (*LABELS,"unknown"):
        support=sum(r["label"]==label for r in rows)
        tp=sum(r["label"]==label and p==label for r,p in zip(rows,predictions))
        proposed=sum(p==label for p in predictions)
        per_label[label]={"support":support,"correct":tp,"predicted":proposed,
                          "recall":tp/support if support else None,"precision":tp/proposed if proposed else None}
    return {"examples":len(rows),"known_examples":len(known),"unknown_examples":len(unknown),
            "correct":sum(r["label"]==p for r,p in zip(rows,predictions)),
            "known_correct":sum(predictions[i]==rows[i]["label"] for i in known),
            "known_abstained":sum(predictions[i]=="unknown" for i in known),
            "known_wrong_label":sum(predictions[i] not in {rows[i]["label"],"unknown"} for i in known),
            "closed_set_known_correct":sum(scores[i]["nearest_known_label"]==rows[i]["label"] for i in known),
            "unknown_rejected":sum(predictions[i]=="unknown" for i in unknown),
            "unknown_false_accepted":sum(predictions[i]!="unknown" for i in unknown),
            "accepted":len(accepted),"accepted_errors":errors,"accepted_precision":correct/len(accepted) if accepted else None,
            "accuracy":sum(r["label"]==p for r,p in zip(rows,predictions))/len(rows) if rows else None,
            "per_label":per_label}


def calibrate(rows,scores):
    if not rows or any(r["split"]!="validation" for r in rows) or not any(r["label"]=="unknown" for r in rows):
        raise LabError("Calibrate with validation-only requests and background questions.")
    grid=[]
    for similarity in GRID["minimum_similarity"]:
        for margin in GRID["minimum_margin"]:
            thresholds={"minimum_similarity":similarity,"minimum_margin":margin,"minimum_feature_coverage":GRID["minimum_feature_coverage"]}
            report=metrics(rows,scores,thresholds)
            grid.append({"thresholds":thresholds,"metrics":report})
    # Fix the objective before looking at test results: minimise accepted errors,
    # then maximise correct accepted requests, then prefer the less strict gate.
    best=min(grid,key=lambda row:(row["metrics"]["accepted_errors"],-row["metrics"]["known_correct"],
                                  row["thresholds"]["minimum_similarity"],row["thresholds"]["minimum_margin"]))
    return {"selection":"Validation only: fewest accepted errors, most correct accepted requests, then lower similarity/margin thresholds.",
            "selected":best,"grid":grid}


def build_dataset(request_folder,background_folder,review_path):
    _,requests=assembly_identity(request_folder)
    _,background=assembly_identity(background_folder)
    request_rows=load_jsonl(request_folder/"records.jsonl")
    background_rows={r["id"]:r for r in load_jsonl(background_folder/"records.jsonl")}
    decisions=load_jsonl(review_path)
    rows=[];seen=set();groups={};texts={}
    for r in request_rows:
        if r.get("source_intent") not in LABELS or r.get("source_config")!="ibo" or r["source"]!="masakhane/InjongoIntent":
            raise LabError("This prototype accepts the reviewed eight-intent Igbo request corpus only.")
        rows.append({"id":r["id"],"label":r["source_intent"],"text":r["text"],"split":r["split"],"group":r["group"],
                     "source":r["source"],"source_content_sha256":r["source_content_sha256"],"review_type":r["review"]["type"]})
    for d in decisions:
        r=background_rows.get(d.get("id"))
        if (not r or d["id"] in seen or d.get("expected_intent")!="unknown" or d.get("review_type")!="AI"
                or d.get("confidence") not in {"high","medium"} or not d.get("reason")
                or any(d.get(k)!=r.get(k) for k in ("source","group","split","source_content_sha256"))
                or not r["text"].startswith("<|user|>") or "<|assistant|>" not in r["text"]):
            raise LabError("Every background question needs explicit matching published-source/unknown review.")
        seen.add(d["id"])
        query=r["text"][len("<|user|>"):].split("<|assistant|>",1)[0]
        rows.append({"id":r["id"],"label":"unknown","text":query,"split":r["split"],"group":r["group"],
                     "source":r["source"],"source_content_sha256":r["source_content_sha256"],"review_type":"AI"})
    for r in rows:
        canonical(r["text"])
        group=(r["source"],r["group"]);text=prompt_key(r["text"])
        if r["split"] not in {"train","validation","test"} or (group in groups and groups[group]!=r["split"]):
            raise LabError("An intent source group crosses official splits.")
        if text in texts:
            raise LabError("Duplicate intent/background queries must be reviewed before this experiment.")
        groups[group]=r["split"];texts[text]=r["label"]
    if len(rows)>1000 or any({r["label"] for r in rows if r["split"]==split}!=set(LABELS)|{"unknown"} for split in ("train","validation","test")):
        raise LabError("Supply all request classes plus background in each bounded split.")
    return rows,{"requests":requests["assembly_sha256"],"background":background["assembly_sha256"],"background_review_sha256":sha(review_path)}


def folder_for(name):
    if not isinstance(name,str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}",name):
        raise LabError("Use a short intent experiment name with letters, digits and hyphens.")
    return owned(LAB/"intent_experiments"/name)


def load_completed(name,check_implementation=True):
    folder=folder_for(name)
    def read(filename):
        path=owned(folder/filename)
        if not path.is_file() or path.stat().st_size>2*MIB:raise LabError("An intent experiment artifact is missing/oversized.")
        return json.loads(path.read_text())
    state=read("run.json")
    if state.get("status")!="complete":raise LabError("This intent experiment is incomplete; it is not retried automatically.")
    for filename,digest in state["artifact_sha256"].items():
        if filename not in {"plan.json","dataset.jsonl","calibration.json","result.json"} or sha(owned(folder/filename))!=digest:
            raise LabError("An intent experiment artifact changed after completion.")
    plan=read("plan.json")
    if check_implementation and plan["implementation_sha256"]!=sha(Path(__file__)):
        raise LabError("The intent implementation changed since this run. Inspect its recorded results before declaring another experiment.")
    return folder,plan,read("calibration.json"),read("result.json")


def run_intent_experiment(request_folder,background_folder,review_path,name=NAME):
    folder=folder_for(name)
    if folder.exists():
        _,_,_,result=load_completed(name,check_implementation=False)
        print(json.dumps({"reused_completed_experiment":True,"result":result},ensure_ascii=False,indent=2))
        return result
    request_folder=owned(request_folder);background_folder=owned(background_folder);review_path=project_input(review_path)
    # Recheck raw/review provenance before freezing the experimental snapshot.
    for source in (request_folder,background_folder):
        _,manifest=assembly_identity(source)
        cases=project_input(FROZEN_CASES)
        rebuilt=assemble_reviewed([p["sample"] for p in manifest["parents"]],cases)
        if rebuilt!=manifest:raise LabError("The intent source assembly no longer matches its reviews.")
    rows,source_hashes=build_dataset(request_folder,background_folder,review_path)
    payload="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode()
    if len(payload)>2*MIB:raise LabError("Keep this intent snapshot under 2 MiB.")
    ensure_capacity(len(payload)+4*MIB)
    plan={"format":1,"name":name,"registered_at":datetime.now(timezone.utc).isoformat(),"algorithm":"Train-only TF-IDF nearest-neighbour per label, including published-question background; no pretrained model.",
          "features":FEATURES,"threshold_grid":GRID,"threshold_selection":"Validation-only minimum accepted errors, maximum correct accepted requests, lower similarity/margin tie-break.",
          "labels":list(LABELS),"source_assemblies":source_hashes,"dataset_sha256":hashlib.sha256(payload).hexdigest(),"implementation_sha256":sha(Path(__file__)),
          "runtime":{"python":platform.python_version(),"unicode":unicodedata.unidata_version},
          "splits":dict(Counter(r["split"] for r in rows)),"split_policy":"Preserve source train/validation/test groups. Fit IDF/index on train only; calibrate on validation only; score test once.",
          "maximum_seconds":MAX_SECONDS,"maximum_process_bytes":512*MIB,"maximum_derived_coefficients":FEATURES["maximum_coefficients"],
          "weight_retention":"No persisted intent weights. Rebuild this small index from frozen training rows for each CLI session; preserve the two existing neural checkpoints.",
          "unknown_limit":"Background consists of cultural/factual questions, not unseen action categories. English/mixed speech, arbitrary unknown intents and real audio remain unvalidated.",
          "phone_actions_enabled":False,"production_ready":False}
    folder.mkdir(parents=True)
    write_json(folder/"plan.json",plan)
    (folder/"dataset.jsonl").write_bytes(payload)
    state={"status":"fitting","name":name};write_json(folder/"run.json",state)
    start=time.perf_counter()
    def guard():
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024)
        ensure_capacity()
        if time.perf_counter()-start>MAX_SECONDS or peak>512*MIB:raise LabError("The intent experiment reached its time/memory budget.")
        return peak
    try:
        train=[r for r in rows if r["split"]=="train"]
        validation=[r for r in rows if r["split"]=="validation"]
        test=[r for r in rows if r["split"]=="test"]
        model=IntentIndex(train);fit_seconds=time.perf_counter()-start;guard()
        validation_scores=[]
        for r in validation:validation_scores.append(model.score(r["text"]));guard()
        calibration=calibrate(validation,validation_scores);write_json(folder/"calibration.json",calibration)
        state["status"]="evaluating";write_json(folder/"run.json",state)
        test_scores=[];durations=[]
        for r in test:
            began=time.perf_counter();test_scores.append(model.score(r["text"]));durations.append(time.perf_counter()-began);guard()
        thresholds=calibration["selected"]["thresholds"]
        result={"name":name,"plan_sha256":sha(folder/"plan.json"),"dataset_sha256":plan["dataset_sha256"],"thresholds":thresholds,
                "training_examples":len(train),"features":len(model.idf),"derived_coefficients":model.coefficients,
                "validation":calibration["selected"]["metrics"],"test":metrics(test,test_scores,thresholds),
                "test_predictions":[{"id":r["id"],"expected":r["label"],"prediction":decide(s,thresholds),**s} for r,s in zip(test,test_scores)],
                "fit_seconds":round(fit_seconds,4),"elapsed_seconds":round(time.perf_counter()-start,4),
                "mean_test_inference_ms":round(sum(durations)/len(durations)*1000,3),"process_peak_bytes":guard(),
                "persisted_intent_weights":False,"neural_checkpoints_changed":False,"frozen_reply_benchmark_repeated":False,
                "native_speaker_validated":False,"audio_evaluated":False,"phone_actions_enabled":False,"production_ready":False,
                "limitations":[plan["unknown_limit"],"Tiny convenience sample; similarity is not a calibrated probability.","Request recognition does not generate Igbo replies or authorize actions."]}
        write_json(folder/"result.json",result)
        state.update(status="complete",artifact_sha256={name:sha(folder/name) for name in ("plan.json","dataset.jsonl","calibration.json","result.json")})
        write_json(folder/"run.json",state)
        write_json(LAB/"latest_intent_experiment.json",{"name":name,"directory":str(folder),"result_sha256":sha(folder/"result.json"),"production_ready":False})
        print(json.dumps({"name":name,"training_examples":len(train),"test":result["test"],"result":str(folder/"result.json"),"production_ready":False},ensure_ascii=False,indent=2))
        return result
    except BaseException as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error)[:500]);write_json(folder/"run.json",state)
        raise


def classify_intent(text,name=NAME):
    folder,plan,calibration,_=load_completed(name)
    rows=load_jsonl(folder/"dataset.jsonl")
    model=IntentIndex([r for r in rows if r["split"]=="train"])
    score=model.score(text)
    label=decide(score,calibration["selected"]["thresholds"])
    result={"intent":None if label=="unknown" else label,"decision":"unknown" if label=="unknown" else "candidate",
            "similarity":round(score["similarity"],4),"margin":round(score["margin"],4),"feature_coverage":round(score["feature_coverage"],4),
            "score_kind":"Cosine similarity, not a calibrated probability.","experiment":plan["name"],
            "supported_scope":"Experimental Igbo requests; English/mixed speech and general unknown-intent rejection unvalidated.",
            "action_executed":False,"transcript_saved":False,"production_ready":False}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
