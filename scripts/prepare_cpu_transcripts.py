"""Create the bounded cloud transcription cell; never installs/downloads locally."""
from pathlib import Path
import base64
import json

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/".build/VoiceResearch/training/anyanwu-female-v1"
plan=json.loads((folder/"plan.json").read_text())
plan["transcript_draft_plan"]={"model":"omniASR_CTC_300M_v2","library":"omnilingual-asr==0.2.0","purpose":"Unverified source-derived transcript drafts, not TTS training",
                               "runtime":"Existing Colab CPU only","clips":5,"input_audio_seconds_cap":60,"minutes_cap":20,"cloud_disk_increase_gib_cap":8,"model_cache_gib_cap":2,"process_rss_gib_cap":6,"log_kib_cap":256,
                               "private_credentials":False,"new_paid_resource":False,"training_optimizer_updates":0,
                               "sources":["https://github.com/facebookresearch/omnilingual-asr","https://github.com/facebookresearch/fairseq2#variants"]}
(folder/"plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n")
setup=(ROOT/"backend/voice_research/cpu_transcript_setup.py").read_text()
worker=(ROOT/"backend/voice_research/cpu_transcript_worker.py").read_text()
tail='''
from IPython.display import Javascript, display, Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown("# Preparing source transcript drafts on CPU\\nThis is **data preparation, not voice training**. All model/dependency files stay in this temporary cloud VM. Drafts need review before becoming audio/text targets."))
try:
    report=run(PLAN,WORKER_SOURCE,publish)
except Exception as error:
    report={"phase":"cpu_transcript_drafts_stopped","trained":False,"error_type":type(error).__name__,"error":str(error)[:180] if isinstance(error,RuntimeError) else "Cloud setup error"}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell="# Anyanwu bounded cloud diagnostics: CPU source transcript drafts\n"+setup+"\nPLAN="+repr(plan)+"\nWORKER_SOURCE="+repr(worker)+"\n"+tail
compile(cell,"cpu-transcript-cell.py","exec")
(folder/"cpu-transcript-cell.py").write_text(cell,encoding="utf-8")
(ROOT/".build/VoiceResearch/colab-diagnostic.b64").write_text(base64.b64encode(cell.encode()).decode())
print(json.dumps({"phase":"cpu_transcript_cell_prepared","trained":False,"local_weights":False}))
