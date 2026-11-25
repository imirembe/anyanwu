"""Make a small audio-only CPU review notebook from the verified backup receipt."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    folder = ROOT / '.build/VoiceResearch/auditions/waxal-female5-expanded-comparison-v2'
    receipt = json.loads((folder / 'backup-result.json').read_text())
    expected = '/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research/auditions/waxal-female5-expanded-comparison-v2/anyanwu-female-expanded-comparison-v2.zip'
    if receipt['drive_archive'] != expected or receipt['audio_clips'] != 8 or not receipt['durable_backup_verified'] or not 0 < receipt['archive_bytes'] <= 16 * 1024**2 or not re.fullmatch('[a-f0-9]{64}', receipt['archive_sha256']):
        raise RuntimeError('Verified eight-clip Drive backup receipt required')
    review = (ROOT / 'backend/voice_research/expanded_comparison_review.py').read_text()
    tail = '''
import importlib.util,platform
if platform.system()!='Linux' or not Path('/content').is_dir() or importlib.util.find_spec('google.colab') is None:
 raise RuntimeError('Use a hosted Colab CPU runtime; no laptop audio download or local execution')
from google.colab import drive
from IPython.display import Markdown,display
if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive',timeout_ms=300000)
restore_comparison(RECEIPT)
display(Markdown('# Previous and expanded female voices\\nAudio-only recovery. No model loading, generation or training. Your browser drafts remain available until matching Drive saves are confirmed.'))
show_comparison(RECEIPT)
'''
    cell = '# Anyanwu audio-only recovery: previous and expanded female comparison\n' + review + '\nRECEIPT=' + repr(receipt) + '\n' + tail
    compile(cell, 'expanded-audio-only-recovery', 'exec')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
                'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# Anyanwụ: recover the eight-clip voice review\nA hosted CPU runtime is sufficient. Connect Google Drive and run the cell. It restores only the checksum-verified small listening archive, never models or training. Do not replace an active review merely to test recovery.\n']},
                          {'cell_type': 'code', 'metadata': {}, 'source': cell.splitlines(True), 'execution_count': None, 'outputs': []}]}
    (folder / 'audio-recovery-cell.py').write_text(cell)
    path = ROOT / 'Voice Research/Anyanwu-Expanded-Voice-Review.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    result = {'notebook': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'cell_sha256': hashlib.sha256(cell.encode()).hexdigest(),
              'archive_sha256': receipt['archive_sha256'], 'gpu_required': False, 'models_loaded': False, 'generation': False, 'training': False}
    (folder / 'audio-recovery-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
