"""Offline source separation, cloud guards and user-decision validation."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.voice_research.waxal_expansion_data import validate_expansion_plan,acquire_expansion
from backend.voice_research.source_expansion_review import validate_source_feedback
from backend.voice_research import source_expansion_review


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((ROOT/'.build/VoiceResearch/datasets/waxal-female5-expansion-v1/plan.json').read_text())
        self.allowed={'fixture':'a'*64};self.nonce='b'*24
        self.payload={'nonce':self.nonce,'id':'fixture','audio_sha256':'a'*64,'words_match':'unreviewed','complete_utterance':'unreviewed','clean_female_voice':'unreviewed','notes':'ụ́'}

    def test_source_excludes_old_and_reserved_text(self):
        validate_expansion_plan(self.plan)
        self.assertEqual(len(self.plan['clips']),17);self.assertEqual(len(self.plan['reserved_evaluation_texts']),4)
        changed=copy.deepcopy(self.plan);changed['clips'][0]['text']=changed['reserved_evaluation_texts'][0]['text']
        with self.assertRaisesRegex(ValueError,'overlap'):validate_expansion_plan(changed)

    def test_publisher_gender_or_truncation_change_rejected(self):
        for values in ({'gender':'Male'},{'speaker_id':'7'},{'truncated_cells':['text']}):
            changed=copy.deepcopy(self.plan);changed['clips'][0].update(values)
            with self.subTest(values=values),self.assertRaises(ValueError):validate_expansion_plan(changed)

    def test_unchecked_does_not_become_accepted(self):
        self.assertEqual(validate_source_feedback(self.payload,self.allowed,self.nonce)['status'],'pending')
        partial={**self.payload,'words_match':'yes','complete_utterance':'yes'}
        self.assertEqual(validate_source_feedback(partial,self.allowed,self.nonce)['status'],'pending')
        self.assertEqual(validate_source_feedback({**partial,'clean_female_voice':'yes'},self.allowed,self.nonce)['status'],'accepted_for_research')
        self.assertEqual(validate_source_feedback({**partial,'clean_female_voice':'no'},self.allowed,self.nonce)['status'],'held_by_user')

    def test_stale_hash_session_and_oversized_note_rejected(self):
        for change in ({'nonce':'old'},{'audio_sha256':'c'*64},{'id':'unknown'},{'words_match':True},{'notes':'x'*501}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_source_feedback({**self.payload,**change},self.allowed,self.nonce)

    def test_mac_acquisition_stops_before_network(self):
        if sys.platform=='darwin':
            with self.assertRaisesRegex(RuntimeError,'Hosted Colab only'):acquire_expansion(self.plan,{},lambda value:None)

    def test_recovery_probe_does_not_bind_and_only_valid_save_binds(self):
        fixture_root=ROOT/'.build/VoiceResearch/test-fixtures';fixture_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixture_root) as directory:
            base=Path(directory);cloud=base/'source';drive=base/'drive';cloud.mkdir();drive.mkdir()
            (cloud/'plan.json').write_text(json.dumps({'id':'fixture-experiment'}))
            (cloud/'data-result.json').write_text(json.dumps({'clips':[{'id':'fixture','audio_sha256':'a'*64}]}))
            callbacks={};google=types.ModuleType('google');colab=types.ModuleType('google.colab');output=types.ModuleType('google.colab.output')
            output.register_callback=lambda name,callback:callbacks.__setitem__(name,callback)
            colab.output=output;google.colab=colab
            ipython=types.ModuleType('IPython');display=types.ModuleType('IPython.display');display.JSON=lambda value:value;ipython.display=display
            def mapped_path(value):
                raw=str(value)
                if raw=='/content/anyanwu-waxal-female5-expansion-v1':return cloud
                if raw=='/content/drive/MyDrive':return drive
                if raw=='/content/drive/MyDrive/Anyanwu Voice Research/datasets/waxal-female5-expansion-v1':return drive/'reviews'
                raise AssertionError('Unexpected fixture path '+raw)
            with patch.object(source_expansion_review,'Path',mapped_path),patch.dict(sys.modules,{'google':google,'google.colab':colab,'google.colab.output':output,'IPython':ipython,'IPython.display':display}):
                nonce,state,counts=source_expansion_review.register_source_review_connection(recover_existing_form=True)
                check=callbacks['anyanwu.waxal_expansion.check'];save=callbacks['anyanwu.waxal_expansion.save']
                self.assertIsNone(nonce);self.assertTrue(check('0'*24)['ready']);self.assertFalse(check('0'*24)['session_bound'])
                invalid=save({**self.payload,'nonce':'c'*24,'audio_sha256':'invalid'})
                self.assertFalse(invalid['saved']);self.assertFalse(check('d'*24)['session_bound']);self.assertEqual(state,{})
                user={**self.payload,'nonce':'d'*24,'words_match':'yes','complete_utterance':'yes','clean_female_voice':'yes'}
                confirmed=save(user);self.assertTrue(confirmed['saved']);self.assertTrue(confirmed['durable'])
                self.assertEqual(json.loads((drive/'reviews/source-reviews.json').read_text())['records']['fixture']['status'],'accepted_for_research')
                self.assertTrue(check('d'*24)['ready']);self.assertFalse(check('e'*24)['ready']);self.assertFalse(save({**user,'nonce':'e'*24})['saved'])
                self.assertTrue(check({'read_only_health':True})['ready']);self.assertTrue(check({'read_only_health':True})['session_bound'])
                self.assertEqual(len(state),1)


if __name__=='__main__':unittest.main()
