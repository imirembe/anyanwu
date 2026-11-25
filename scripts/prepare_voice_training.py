"""Prepare a cloud readiness audit for the requested training; no local media."""
from pathlib import Path
import base64
import json

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/".build/VoiceResearch/training/anyanwu-female-v1"
folder.mkdir(parents=True,exist_ok=True)
review=json.loads((ROOT/".build/VoiceResearch/bbc-igbo-female-sources-v1/cloud-review-result.json").read_text())
sources=json.loads((ROOT/".build/VoiceResearch/bbc-igbo-female-sources-v1/sources.json").read_text())
plan={"id":"anyanwu-female-training-v1","request":"User requested start training", "status":"readiness_audit",
      "purpose":"Research a natural female Igbo TTS voice; no named-presenter identity target", "permission":review["permission"],
      "data_plan":{"initial_target_utterances":100,"proposed_split_counts":{"train":80,"validation":10,"test":10},
                   "target_note":"Initial data-planning target only, not a universal minimum or guarantee of quality",
                   "split_policy":"Keep source passages and near-duplicate sentences in one fold; retain speaker/source IDs",
                   "current_pilot_role":"59-second convenience audit, not labelled training data"},
      "execution_policy":{"cloud_only":True,"no_paid_resources_without_budget":True,"no_training_on_unverified_pairs":True,
                          "preflight_network_requests":0,"preflight_model_downloads":0,"preflight_minutes":2},
      "base_model":{"selected":None,"reason":"Igbo phoneme/token support, female adaptation recipe and training-checkpoint compatibility must be verified before weights are fetched"},
      "clips":[]}
for row in review["sources"]:
    asset=next(a for a in row["assets"] if a["filename"]=="speech-12s.wav")
    source=next(s for s in sources["sources"] if s["id"]==row["id"])
    plan["clips"].append({"id":row["id"],"source_url":source["url"],"sha256":asset["sha256"]})
(folder/"plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
module=(ROOT/"backend/voice_research/training_preflight.py").read_text()
tail='''
from IPython.display import Javascript, display, Markdown
try:
    report=preflight(PLAN)
except Exception as error:
    report={"phase":"training_preflight_failed","trained":False,"optimizer_updates":0,"error_type":type(error).__name__,"error":str(error)[:180] if isinstance(error,RuntimeError) else "Inspect cloud error"}
display(Markdown("# Anyanwụ voice training readiness\\n**Training has not started.** This cell verifies the actual data and runtime before any model install or optimizer run."))
print(json.dumps(report,ensure_ascii=False,indent=2))
display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':report})+', "https://colab.research.google.com");'))
'''
cell="# Anyanwu bounded cloud diagnostics: requested training preflight\n"+module+"\nPLAN="+repr(plan)+"\n"+tail
compile(cell,"voice_training_preflight_cell.py","exec")
(folder/"training-preflight-cell.py").write_text(cell,encoding="utf-8")
(ROOT/".build/VoiceResearch/colab-diagnostic.b64").write_text(base64.b64encode(cell.encode()).decode())
print(json.dumps({"plan":str(folder.relative_to(ROOT)/"plan.json"),"clips":len(plan["clips"]),"models_downloaded":False,"trained":False}))
