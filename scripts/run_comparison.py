"""Stage the inference-only comparison in the existing fourth Colab cell."""
from pathlib import Path
import base64
import json

root=Path(__file__).resolve().parents[1]
folder=root/'.build/VoiceResearch/auditions/waxal-female5-adapter-comparison-v1'
runner=(root/'backend/voice_research/comparison_run.py').read_text()
worker=(root/'backend/voice_research/comparison_worker.py').read_text()
generator=(root/'backend/voice_research/flash_generation.py').read_text()
tail='''
from IPython.display import Javascript,display,Markdown
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
display(Markdown('# Generate the listening comparison\\nThree published texts, one original-base and one trained-adapter take each. Same female source reference, seed and frozen decoder. No optimizer updates or automatic retakes.'))
try: report=run_comparison(WORKER,GENERATOR,publish)
except Exception as error: report={'phase':'comparison_guard_stopped','error':str(error)[:400]}
publish(report)
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
cell='# Anyanwu bounded cloud diagnostics: generate paired voice comparison\n'+runner+'\nWORKER='+repr(worker)+'\nGENERATOR='+repr(generator)+'\n'+tail
for name,source in [('worker',worker),('generator',generator),('cell',cell)]:compile(source,name,'exec')
(folder/'generation-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Staged six one-take generated WAVs in Colab; no local model/audio download.')
