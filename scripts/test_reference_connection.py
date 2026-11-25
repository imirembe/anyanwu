"""Fixture checks for reconnecting existing review cards without resubmitting them."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import ast,importlib.util
spec=importlib.util.spec_from_file_location('reference_review_fixture',ROOT/'.build/VoiceResearch/auditions/conversational-reference-conditioning-v1/reference_review.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
register_reference_conditioning_connection=module.register_reference_conditioning_connection


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        fixtures=ROOT/'.build/VoiceResearch/test-fixtures';fixtures.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=fixtures);self.addCleanup(self.temp.cleanup)
        folder=Path(self.temp.name);self.root=folder/'audio';self.root.mkdir();self.mount=folder/'mount';self.mount.mkdir();self.drive=self.mount/'reviews'
        clips=[{'text_id':'case'+str(i),'variant':v,'audio_sha256':hashlib.sha256((v+str(i)).encode()).hexdigest()} for i in range(2) for v in ('base','adapter')]
        (self.root/'plan.json').write_text(json.dumps({'id':'fixture','texts':[{'id':'case'+str(i)} for i in range(2)]}))
        (self.root/'generation-result.json').write_text(json.dumps({'completed_wavs':4,'clips':clips}))
        self.callbacks={}
        output=types.SimpleNamespace(register_callback=lambda name,fn:self.callbacks.__setitem__(name,fn))
        colab=types.ModuleType('google.colab');colab.output=output
        display=types.ModuleType('IPython.display');display.JSON=lambda data:types.SimpleNamespace(data=data)
        self.modules=patch.dict(sys.modules,{'google':types.ModuleType('google'),'google.colab':colab,'IPython':types.ModuleType('IPython'),'IPython.display':display})
        self.modules.start();self.addCleanup(self.modules.stop)
        register_reference_conditioning_connection(True,root=self.root,drive=self.drive,mount_root=self.mount)
        self.save=self.callbacks['anyanwu.reference_conditioning.save'];self.check=self.callbacks['anyanwu.reference_conditioning.check']
        self.payload={'id':'case0','nonce':'a'*24,'audio_hashes':{c['variant']:c['audio_sha256'] for c in clips if c['text_id']=='case0'},'preference':'tie','words_base':'yes','words_adapter':'yes','cadence_base':'3','cadence_adapter':'4','notes':'Fixture: ụ́ Ọ bụ m.'}

    def test_recovery_cell_calls_the_exported_connection_function(self):
        source=(ROOT/'.build/VoiceResearch/auditions/conversational-reference-conditioning-v1/audio-connection-cell.py').read_text()
        calls=[node.value for node in ast.parse(source).body if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id.startswith('register_')]
        self.assertEqual(len(calls),1)
        function=getattr(module,calls[0].func.id)
        function(True,root=self.root,drive=self.drive,mount_root=self.mount)
        self.assertTrue(self.callbacks['anyanwu.reference_conditioning.check']({'read_only_health':True}).data['ready'])

    def test_read_only_health_does_not_bind_or_create_feedback(self):
        r=self.check({'read_only_health':True}).data
        self.assertTrue(r['ready']);self.assertFalse(r['session_bound']);self.assertFalse(r['feedback_changed']);self.assertEqual(r['saved_records'],0)
        self.assertFalse((self.drive/'human-comparison.json').exists())

    def test_old_valid_nonce_is_adopted_and_feedback_is_durable(self):
        r=self.save(self.payload).data
        self.assertTrue(r['saved']);self.assertTrue(r['durable']);self.assertEqual(r['complete_reviews'],1)
        saved=json.loads((self.drive/'human-comparison.json').read_text())
        self.assertEqual(saved['records']['case0']['notes'],self.payload['notes']);self.assertFalse(saved['automatic_training'])
        self.assertTrue(self.check({'read_only_health':True}).data['session_bound'])

    def test_invalid_hash_request_does_not_capture_session(self):
        r=self.save({**self.payload,'audio_hashes':{'base':'bad','adapter':'bad'}}).data
        self.assertFalse(r['saved']);self.assertFalse(self.check({'read_only_health':True}).data['session_bound'])
        self.assertTrue(self.save({**self.payload,'nonce':'b'*24}).data['saved'])

    def test_invalid_choice_does_not_capture_session(self):
        self.assertFalse(self.save({**self.payload,'cadence_base':5}).data['saved'])
        self.assertFalse(self.check({'read_only_health':True}).data['session_bound'])
        self.assertTrue(self.save({**self.payload,'nonce':'b'*24}).data['saved'])

    def test_second_form_nonce_is_rejected_after_binding(self):
        self.assertTrue(self.save(self.payload).data['saved'])
        self.assertFalse(self.save({**self.payload,'nonce':'b'*24}).data['saved'])
        self.assertEqual(self.check({'read_only_health':True}).data['saved_records'],1)

    def test_external_feedback_change_is_preserved(self):
        self.assertTrue(self.save(self.payload).data['saved'])
        path=self.drive/'human-comparison.json';value=json.loads(path.read_text());value['records']['case0']['notes']='External fixture edit';path.write_text(json.dumps(value))
        r=self.save(self.payload).data
        self.assertFalse(r['saved']);self.assertIn('changed',r['error'])
        self.assertEqual(json.loads(path.read_text())['records']['case0']['notes'],'External fixture edit')


if __name__=='__main__':unittest.main()
