"""Offline invariants for comparison isolation, feedback and safe recovery."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.voice_research.comparison_review import validate_comparison_feedback,restore_comparison
from backend.voice_research.comparison_run import run_comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.allowed={'case':{'base':'a'*64,'adapter':'b'*64}};self.nonce='c'*24
        self.payload={'id':'case','nonce':self.nonce,'audio_hashes':self.allowed['case'],'preference':'unreviewed',
          'words_base':'unreviewed','words_adapter':'unreviewed','cadence_base':'unreviewed','cadence_adapter':'unreviewed','notes':'ụ́'}

    def test_unreviewed_is_retained_without_approval(self):
        result=validate_comparison_feedback(self.payload,self.allowed,self.nonce)
        self.assertEqual(result['preference'],'unreviewed');self.assertEqual(result['notes'],'ụ́');self.assertNotIn('nonce',result)

    def test_stale_audio_session_and_invalid_fields_rejected(self):
        for replacement in ({'nonce':'old'},{'audio_hashes':{'base':'new','adapter':'b'*64}},
                            {'preference':'accepted'},{'cadence_base':5},{'words_adapter':True},{'notes':'x'*501}):
            with self.subTest(replacement=replacement),self.assertRaises(ValueError):validate_comparison_feedback({**self.payload,**replacement},self.allowed,self.nonce)

    def test_published_comparison_excluded_from_adapter_sample(self):
        folder=ROOT/'.build/VoiceResearch'
        plan=json.loads((folder/'auditions/waxal-female5-adapter-comparison-v1/plan.json').read_text())
        train=json.loads((folder/'training/waxal-female5-lora-v1/plan.json').read_text())
        self.assertEqual(plan['outputs'],6);self.assertFalse({x['id'] for x in plan['texts']}&{x['id'] for x in train['clips']})
        self.assertEqual(plan['limits']['optimizer_updates'],0);self.assertFalse(plan['limits']['laptop_audio_or_models'])

    def test_laptop_generation_guard_precedes_any_worker(self):
        if sys.platform=='darwin':
            with self.assertRaisesRegex(RuntimeError,'Hosted Colab only'):run_comparison('raise AssertionError()','',lambda value:None)

    def test_corrupt_archive_rejected_before_extraction(self):
        tests=ROOT/'.build/VoiceResearch/test-fixtures';tests.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tests) as directory:
            path=Path(directory)/'fixture.zip';path.write_bytes(b'not-an-archive')
            with self.assertRaisesRegex(RuntimeError,'integrity'):
                restore_comparison({'drive_archive':str(path),'archive_bytes':path.stat().st_size,'archive_sha256':'0'*64})


if __name__=='__main__':unittest.main()
