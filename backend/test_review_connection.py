"""Isolated fixture feedback only; never uses the real review paths or Google."""
from pathlib import Path
import json
import tempfile
import types
import unittest
from unittest.mock import patch
from backend.voice_research.review_connection import register_review_connection


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        scratch=Path('.build/VoiceResearch/review-connection-tests')
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.callbacks={}
        fake=types.ModuleType('google.colab')
        fake.output=types.SimpleNamespace(register_callback=lambda name,fn:self.callbacks.__setitem__(name,fn))
        display=types.ModuleType('IPython.display')
        display.JSON=lambda data:data
        self.modules=patch.dict('sys.modules',{'google.colab':fake,'IPython.display':display})
        self.modules.start();self.addCleanup(self.modules.stop)
        def mapped(value):
            if value=='/content/anyanwu-bbc-expansion-v1':return self.root/'runtime'
            if value=='/content/drive/MyDrive/Anyanwu Voice Research/datasets/bbc-expansion-v1':return self.root/'drive'
            raise AssertionError('Fixture attempted unexpected path: '+str(value))
        self.paths=patch('backend.voice_research.review_connection.Path',side_effect=mapped)
        self.paths.start();self.addCleanup(self.paths.stop)
        self.alignment={'experiment_id':'fixture-only','cases':[{'id':f'case{i}','cloud_audio_path':'fixture.wav',
                          'candidate_audio_sha256':f'hash{i}','timing_warnings':[]} for i in range(7)]}

    def payload(self,nonce,case='case0'):
        return {'id':case,'nonce':nonce,'words_match':'unreviewed','complete_edges':'unreviewed',
                'clean_female_voice':'unreviewed','notes':'Fixture only: ọ́ ụ̄'}

    def test_readonly_recovery_probe_does_not_bind_or_create_feedback(self):
        register_review_connection(self.alignment,recover_existing_form=True)
        probe=self.callbacks['anyanwu.expanded_cuts.check']('B'*24)
        self.assertTrue(probe['ready']);self.assertFalse(probe['feedback_records_changed'])
        self.assertFalse((self.root/'runtime/human-reviews.json').exists())
        result=self.callbacks['anyanwu.expanded_cuts.save'](self.payload('A'*24))
        self.assertTrue(result['saved']);self.assertFalse(result['durable'])
        self.assertEqual(result['status'],'pending')
        self.assertFalse(self.callbacks['anyanwu.expanded_cuts.save'](self.payload('B'*24,'case1'))['saved'])

    def test_invalid_first_request_cannot_bind_recovered_form(self):
        register_review_connection(self.alignment,recover_existing_form=True)
        save=self.callbacks['anyanwu.expanded_cuts.save']
        self.assertFalse(save(self.payload('B'*24,'unknown'))['saved'])
        self.assertTrue(save(self.payload('A'*24))['saved'])

    def test_drive_save_roundtrip_preserves_unicode_and_reload(self):
        (self.root/'drive').mkdir()
        nonce,_=register_review_connection(self.alignment)
        save=self.callbacks['anyanwu.expanded_cuts.save']
        self.assertFalse(save(self.payload('wrong'))['saved'])
        result=save(self.payload(nonce))
        self.assertTrue(result['saved'] and result['durable'])
        saved=json.loads((self.root/'drive/human-reviews.json').read_text())
        self.assertEqual(saved['records']['case0']['notes'],'Fixture only: ọ́ ụ̄')
        _,restored=register_review_connection(self.alignment,recover_existing_form=True)
        self.assertEqual(restored,saved)
        self.assertFalse(result['training_started'])


if __name__=='__main__':unittest.main()
