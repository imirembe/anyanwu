"""Offline archive, data separation, platform and review regression checks."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.voice_research.expanded_comparison_cloud import read_comparison_archive, restore_expanded_comparison, run_expanded_comparison
from backend.voice_research.expanded_comparison_review import validate_comparison_feedback


class ExpandedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / '.build/VoiceResearch/auditions/waxal-female5-expanded-comparison-v2'
        self.plan = json.loads((self.folder / 'plan.json').read_text())
        self.fixtures = ROOT / '.build/VoiceResearch/test-fixtures'
        self.fixtures.mkdir(exist_ok=True)

    def test_four_reserved_texts_and_two_exact_adapters(self):
        training = json.loads((ROOT / '.build/VoiceResearch/training/waxal-female5-expanded-lora-v2/plan.json').read_text())
        self.assertEqual(self.plan['texts'], training['fresh_synthesis_comparison']['texts'])
        self.assertFalse({c['id'] for c in self.plan['texts']} & {c['id'] for c in training['cases']})
        self.assertEqual(self.plan['outputs'], 8)
        self.assertNotEqual(self.plan['adapter_hashes']['base'], self.plan['adapter_hashes']['adapter'])
        self.assertEqual(self.plan['optimizer_updates_this_stage'], 0)
        self.assertLess(sum(f['size'] for f in self.plan['model_files']), 4 * 1024**3)

    def test_mac_guard_precedes_gpu_process_or_install(self):
        with patch('backend.voice_research.expanded_comparison_cloud.platform.system', return_value='Darwin'), patch('backend.voice_research.expanded_comparison_cloud.subprocess.run', side_effect=AssertionError('No local subprocess')):
            with self.assertRaisesRegex(RuntimeError, 'Hosted Colab only'):
                run_expanded_comparison({}, '', '', '', '', lambda _: None)

    def test_corrupt_adapter_archive_rejected_before_read(self):
        with tempfile.TemporaryDirectory(dir=self.fixtures) as directory:
            path = Path(directory) / 'bad.zip'
            path.write_bytes(b'fixture only')
            with self.assertRaisesRegex(RuntimeError, 'integrity'):
                read_comparison_archive({'path': str(path), 'bytes': path.stat().st_size, 'sha256': '0' * 64}, ['adapter.safetensors'])

    def test_verified_restore_preserves_both_variants_and_only_common_prompt(self):
        with tempfile.TemporaryDirectory(dir=self.fixtures) as directory:
            folder = Path(directory)
            sha = lambda body: hashlib.sha256(body).hexdigest()
            old, new, wav, mp3 = b'previous fixture', b'expanded fixture', b'wav fixture', b'mp3 fixture'
            source = {'id': 'prompt', 'speaker_id': '5', 'gender': 'Female', 'adapter_split': 'train', 'audio_sha256': sha(wav), 'mp3_sha256': sha(mp3)}
            common = {'plan.json': json.dumps({'clips': [{'id': 'prompt'}]}).encode(), 'training-result.json': json.dumps({'optimizer_updates': 60}).encode(), 'adapter.safetensors': old,
                      'data-result.json': json.dumps({'clips': [source]}).encode(), 'speech-targets.json': json.dumps([{'id': 'prompt', 'speech_ids': [1, 2], 'text_ids': [3]}]).encode(), 'data/prompt.wav': wav, 'data/prompt.mp3': mp3}
            expanded = {'plan.json': json.dumps({'cases': [{'id': 'other'}]}).encode(), 'training-result.json': json.dumps({'optimizer_updates': 60, 'cumulative_adapter_updates': 120, 'starting_adapter_sha256': sha(old)}).encode(), 'adapter.safetensors': new}
            def package(name, entries):
                path = folder / name
                with zipfile.ZipFile(path, 'w') as out:
                    for member, body in entries.items():
                        out.writestr(member, body)
                    out.writestr('manifest.json', json.dumps({'files': {k: {'bytes': len(v), 'sha256': sha(v)} for k, v in entries.items()}}))
                return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())}
            plan = {'prompt_id': 'prompt', 'texts': [{'id': str(i)} for i in range(4)], 'adapter_hashes': {'base': sha(old), 'adapter': sha(new)}, 'archives': {'previous': package('old.zip', common), 'expanded': package('new.zip', expanded)}}
            result = restore_expanded_comparison(plan, folder / 'restored')
            self.assertEqual(result['optimizer_updates_this_stage'], 0)
            self.assertEqual((folder / 'restored/assets/previous-adapter.safetensors').read_bytes(), old)
            self.assertEqual((folder / 'restored/assets/adapter.safetensors').read_bytes(), new)
            self.assertEqual(len(list((folder / 'restored/assets/data').iterdir())), 2)
            plan['texts'][0]['id'] = 'other'
            with self.assertRaisesRegex(RuntimeError, 'overlap'):
                restore_expanded_comparison(plan, folder / 'rejected')

    def test_review_rejects_stale_audio_and_keeps_unicode_unreviewed(self):
        allowed = {'case': {'base': 'a' * 64, 'adapter': 'b' * 64}}
        payload = {'id': 'case', 'nonce': 'session', 'audio_hashes': allowed['case'], 'preference': 'unreviewed', 'words_base': 'unreviewed', 'words_adapter': 'unreviewed', 'cadence_base': 'unreviewed', 'cadence_adapter': 'unreviewed', 'notes': 'ụ́ Ọ bụ m.'}
        self.assertEqual(validate_comparison_feedback(payload, allowed, 'session')['notes'], payload['notes'])
        with self.assertRaisesRegex(ValueError, 'Unknown comparison audio'):
            validate_comparison_feedback({**payload, 'audio_hashes': {'base': 'old', 'adapter': 'b' * 64}}, allowed, 'session')

    def test_embedded_download_script_and_real_review_javascript(self):
        source = (ROOT / 'backend/voice_research/expanded_comparison_cloud.py').read_text()
        tree = ast.parse(source)
        download = next(n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'download_source' for t in n.targets))
        compile(download, 'cloud-download-child', 'exec')
        review = ast.parse((ROOT / 'backend/voice_research/expanded_comparison_review.py').read_text())
        template = next(n.value.value for n in ast.walk(review) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'template' for t in n.targets))
        pairs = [{'id': 'case' + str(i), 'text': 'ụ́ fixture', 'review': {}, 'clips': {v: {'sha256': v + str(i), 'seconds': 1.0, 'warnings': [], 'audio': ''} for v in ('base', 'adapter')}} for i in range(4)]
        script = template.split('<script>', 1)[1].split('</script>', 1)[0].replace('__DATA__', json.dumps(pairs)).replace('__NONCE__', '"fixture"').replace('__BACKUP__', '{"durable_backup_verified":true}')
        result = subprocess.run(['node', str(ROOT / 'scripts/test_expanded_comparison_drafts.mjs')], input=json.dumps({'script': script}), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
