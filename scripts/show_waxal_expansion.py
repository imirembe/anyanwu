"""Back up new source audio and show its review in the existing Colab cell."""
from pathlib import Path
import base64
import json

root=Path(__file__).resolve().parents[1];folder=root/'.build/VoiceResearch/datasets/waxal-female5-expansion-v1'
archive=(root/'backend/voice_research/source_expansion_archive.py').read_text()
review=(root/'backend/voice_research/source_expansion_review.py').read_text()
extras={'SOURCE_CARD.txt':(root/'.build/VoiceResearch/training/anyanwu-female-v1/waxal-card.txt').read_text(),
  'ATTRIBUTION.txt':'Google WAXAL / Media Trust Igbo TTS speaker5, publisher gender Female.\nCC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/\nhttps://huggingface.co/datasets/google/WaxalNLP/blob/5f4d8ca24f2b9d168b2ee545f1febaaff4b40580/README.md\nChanges: selected whole lossy MP3 previews and decoded to16k mono PCM. No clipping, text correction, voice cloning or model training in this stage.\nPublished speech labels are preserved as pronunciation targets, not verified current factual assistant knowledge. Human review pending.\nNo speaker consent audit, final unique voice identity, base weights, private conversations or credentials included.\n',
  'source_expansion_review.py':review,'waxal_expansion_data.py':(root/'backend/voice_research/waxal_expansion_data.py').read_text()}
tail='''
from IPython.display import Javascript,display
def publish(value):
    display(Javascript('window.top.postMessage('+json.dumps({'type':'anyanwu-cloud-diagnostic','nonce':'anyanwu-20261122-setup-r2','payload':value})+', "https://colab.research.google.com");'))
try:
    backup=backup_source_expansion(EXTRAS)
    publish(backup)
    print('Whole source recordings and provenance are verified in Google Drive. No new training has started.')
    show_source_expansion(backup)
except Exception as error:
    publish({'phase':'expanded_source_review_stopped','error':str(error)[:500]})
    raise
'''
cell='# Anyanwu bounded cloud diagnostics: review the expanded female source sample\n'+archive+'\n'+review+'\nEXTRAS='+repr(extras)+'\n'+tail
compile(cell,'expanded-source-review.py','exec');(folder/'review-cell.py').write_text(cell)
(root/'.build/VoiceResearch/colab-diagnostic.b64').write_text(base64.b64encode(cell.encode()).decode())
print('Prepared verified Drive archive and source review with explicit checks and protected drafts.')
