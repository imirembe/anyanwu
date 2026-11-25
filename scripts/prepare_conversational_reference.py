"""Prepare a pinned, six-recording cloud-only listening reference, without fetching media."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / '.build/VoiceResearch/conversational-igbo-v1'
head = json.loads((folder / 'source-head.json').read_text())
metadata = (folder / 'metadata.csv').read_bytes()
card = (folder / 'source-card.md').read_bytes()
rows = list(csv.DictReader(metadata.decode('utf-8-sig').splitlines()))
chosen = [
    ('IGB_004', '2abd4ede-5762-485d-b38c-a800b744f8e6', 'Shopping at the market'),
    ('IGB_004', '11bac2b4-8136-4c60-924b-72d07be7e0d7', 'A rainy day'),
    ('IGB_005', '7c948456-ad30-4f0a-b42f-327b306f633b', 'Discussing prices'),
    ('IGB_005', '20de3b2e-6c11-40f9-915b-97915b607d97', 'Difficulties at work'),
    ('IGB_001', '86d55ded-7c7b-414e-b6cb-4481643ecbeb', 'Places to eat'),
    ('IGB_001', '8c7969ef-b1d8-44ff-9281-33f5374e0f65', 'Technology and learning'),
]
clips = []
for index, (speaker, script, topic) in enumerate(chosen, 1):
    matches = [r for r in rows if r['speaker_id'] == speaker and r['script_id'] == script]
    if len(matches) != 1:
        raise RuntimeError('Source selection must be unique')
    row = matches[0]
    if row['gender'] != 'female' or row['dialect'] != 'Nigeria - Owerri' or row['type_of_script'] != 'free_speech':
        raise RuntimeError('Unexpected source labels')
    clips.append({'id': f'conversation-ref-{index:02}', 'speaker_id': speaker,
                  'script_id': script, 'file_name': row['file_name'],
                  'publisher_seconds': float(row['duration']), 'topic': topic})
plan = {
    'id': 'conversational-igbo-v1', 'repository': head['repository'], 'revision': head['revision'],
    'metadata_path': 'igbo_nigeria/free_speech/metadata.csv',
    'metadata_sha256': hashlib.sha256(metadata).hexdigest(),
    'card_sha256': hashlib.sha256(card).hexdigest(),
    'attribution': 'Silencio Network, Igbo Speech Dataset, 2026. CC BY-NC 4.0.',
    'purpose': 'Noncommercial qualitative listening to human spontaneous delivery; not a commercial-product benchmark',
    'limitations': ['Six source recordings are not a representative accent sample or a training corpus.',
                    'Publisher labels are not independent listening verification.',
                    'No transcripts: elicitation questions must never be used as spoken labels.',
                    'No voice-identity imitation, model training, synthesis or app integration in this phase.'],
    'caps': {'clips': 6, 'source_bytes_per_file': 8*1024**2, 'source_bytes_total': 48*1024**2,
             'http_body_bytes': 50*1024**2, 'http_requests_including_redirects': 40,
             'network_seconds': 480, 'source_seconds_per_file': 45,
             'listening_bytes_total': 12*1024**2, 'cloud_folder_bytes': 128*1024**2,
             'cloud_free_reserve_bytes': 5*1024**3, 'laptop_audio_bytes': 0},
    'clips': clips, 'training': False, 'review_status': 'pending_user_listening',
}
(folder / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
module = (ROOT / 'backend/voice_research/conversational_reference.py').read_text()
cell = '# Anyanwu spontaneous Igbo reference listening — CPU, no model/training\n' + module + '\nPLAN = ' + repr(plan) + '\nrun_reference(PLAN)\n'
compile(cell, 'conversational-reference-cell', 'exec')
(folder / 'reference-cell.py').write_text(cell)
notebook = {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'colab': {'name': 'Anyanwu-Conversational-Igbo-Reference.ipynb'},
                         'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
            'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [
                '# Anyanwụ · natural conversational delivery\n',
                'Use a hosted CPU runtime and connect the existing research Google Drive. '
                'Six whole human spontaneous responses, no model downloads or training. '
                'Audio remains in Colab/Drive. Source prompts are not transcripts.\n']},
                {'cell_type': 'code', 'metadata': {}, 'source': cell.splitlines(keepends=True),
                 'execution_count': None, 'outputs': []}]}
raw = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
(folder / 'Anyanwu-Conversational-Igbo-Reference.ipynb').write_text(raw)
(ROOT / 'Voice Research/Anyanwu-Conversational-Igbo-Reference.ipynb').write_text(raw)
receipt = {'phase': 'spontaneous_reference_prepared', 'source_revision': plan['revision'],
           'cell_sha256': hashlib.sha256(cell.encode()).hexdigest(), 'clips': 6,
           'estimated_source_seconds': sum(c['publisher_seconds'] for c in clips),
           'media_downloaded_to_laptop': False, 'training_started': False}
(folder / 'source-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
