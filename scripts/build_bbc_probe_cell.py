"""Build a small cloud-only diagnostic cell; performs no network calls."""
from pathlib import Path
import base64
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/"backend/voice_research/bbc_probe.py").read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0]
compile(source, "bbc_probe.py", "exec")
bridge = '''
from IPython.display import Javascript, display
display(Javascript("window.top.postMessage(" + json.dumps({"type":"anyanwu-cloud-diagnostic","nonce":"anyanwu-20261122-setup-r2","payload":report}) + ", \\\"https://colab.research.google.com\\\");"))
'''
if sys.argv[1:] == ["samples"]:
    samples = (ROOT/"backend/voice_research/bbc_samples.py").read_text(encoding="utf-8")
    samples = samples.replace("from backend.voice_research.bbc_probe import SOURCES, Transport, attributes, checked_url, cloud_guard", "# BBC transport helpers embedded above; no project installation needed.")
    source += "\n"+samples
    run = '''
try:
    report = prepare_samples()
except Exception as error:
    report = {"stage":"bbc_cloud_sample_stopped", "error_type":type(error).__name__, "error":str(error)[:180] if isinstance(error,RuntimeError) else "Stopped; inspect the cloud output", "trained":False}
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
    listen = '''
from IPython.display import Audio, Image, Markdown
for item in report.get("sources",[]):
    display(Markdown("### "+item["id"]+" · source 00:12–00:24"))
    if item["status"] != "sample_ready_for_listening_and_caption_review":
        print(item.get("error","Sample unavailable"))
        continue
    display(Audio(filename=str(ROOT/item["id"]/"speech-12s.wav")))
    for asset in item["assets"]:
        if asset["filename"].endswith(".jpg"):
            display(Image(filename=str(ROOT/item["id"]/asset["filename"]),width=280))
print("Review the spoken language, word-for-word subtitle match, background music and speaker changes. No training has started. Samples live only in this temporary Colab runtime.")
'''
    cell = "# Anyanwu bounded cloud diagnostics: BBC cloud-only 12-second samples\n"+source+run+bridge+listen
    folder = ROOT/".build/VoiceResearch/bbc-igbo-female-sources-v1"
    instructions = """# Anyanwụ: BBC female Igbo speech audit

Use a **hosted CPU** runtime: Runtime → Change runtime type → CPU. If Colab shows a GPU quota warning, choose **Connect without GPU**. Do not run the older Naija TTS notebook's model cells for this audit.

Uses the five BBC links you supplied and your confirmation of training permission. Retrieves only public, accessible streams with no login, geo override or DRM handling. Stops on unsupported streams; there is no full-video fallback.

Takes **12 seconds per source (00:12–00:24)**, at most 60 seconds combined, plus four caption frames each. Limits: 32 MiB total response bodies (6 MiB/source), 64 MiB cloud staging, 5 GiB cloud free-space floor and eight minutes of network work. No model weights, API keys, private conversations or package installs. Media stays in the temporary Colab VM; no download to your laptop is triggered.

This is a data audit, **not training or a new Anyanwụ voice**. Check whether speech is actually Igbo, whether captions are verbatim or translated, and whether each passage has one clean speaker without music/overlap. AI/OCR labels remain provisional. Publisher/speaker identities are not the final brand voice.
"""
    notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{"colab":{"name":"Anyanwu-BBC-Igbo-audit.ipynb"},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"markdown","metadata":{},"source":instructions.splitlines(keepends=True)},{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":cell.splitlines(keepends=True)},{"cell_type":"markdown","metadata":{},"source":["When finished, save your text observations, then use **Runtime → Disconnect and delete runtime**. Audio and frame files are temporary and disappear when the VM is deleted. No model training is run by this notebook."]}]}
    for index, notebook_cell in enumerate(notebook["cells"]):
        notebook_cell["id"] = "bbc-audit-"+str(index)
    encoded = json.dumps(notebook,ensure_ascii=False,indent=2)+"\n"
    (folder/"Anyanwu-BBC-Igbo-audit.ipynb").write_text(encoded,encoding="utf-8")
    (ROOT/"Voice Research/Anyanwu-BBC-Igbo-audit.ipynb").write_text(encoded,encoding="utf-8")
else:
    run = '''
report, BBC_MANIFESTS = probe()
for row in report["sources"]:
    row["variant_count"] = len(row.get("variants",[]))
    row["variants"] = row.get("variants",[])[:2]
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
    cell = "# Anyanwu bounded cloud diagnostics: BBC manifests, no media segments\n"+source+run+bridge
compile(cell, "bbc_probe_cell.py", "exec")
(ROOT/".build/VoiceResearch/colab-diagnostic.b64").write_text(base64.b64encode(cell.encode()).decode())
