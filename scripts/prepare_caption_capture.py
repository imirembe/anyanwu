"""Build a cloud-only capture cell using the existing bounded BBC transport."""
from pathlib import Path
import base64

ROOT=Path(__file__).resolve().parents[1]
transport=(ROOT/"backend/voice_research/bbc_probe.py").read_text().split('\nif __name__ == "__main__":')[0]
samples=(ROOT/"backend/voice_research/bbc_samples.py").read_text().replace("from backend.voice_research.bbc_probe import SOURCES, Transport, attributes, checked_url, cloud_guard", "# Bounded BBC helpers are embedded above.")
capture=(ROOT/"backend/voice_research/caption_capture.py").read_text().replace("from backend.voice_research.bbc_probe import SOURCES, Transport, cloud_guard", "# Bounded BBC transport above.").replace("from backend.voice_research.bbc_samples import master_selection, acquire_track, ffmpeg", "# Existing local-only decoder above.")
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown("# Capturing the user-reviewed BBC subtitles\\nOnly the existing 12-second windows are fetched, up to 32 MiB. Frames stay in Colab. ASR drafts are not substituted for caption text; TTS training has not started."))
try:
    report=capture(publish)
except Exception as error:
    report={'phase':'caption_capture_stopped','trained':False,'error_type':type(error).__name__,'error':str(error)[:180] if isinstance(error,RuntimeError) else 'Inspect the cloud capture'}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell="# Anyanwu bounded cloud diagnostics: user-reviewed source caption pictures\n"+transport+"\n"+samples+"\n"+capture+tail
compile(cell,"caption-capture-cell.py","exec")
folder=ROOT/".build/VoiceResearch/training/anyanwu-female-v1"
(folder/"caption-capture-cell.py").write_text(cell)
(ROOT/".build/VoiceResearch/colab-diagnostic.b64").write_text(base64.b64encode(cell.encode()).decode())
