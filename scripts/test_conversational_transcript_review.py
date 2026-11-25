"""Offline tests of actual review script and durable callback behaviour."""
from pathlib import Path
import ast
import json
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.voice_research import conversational_transcript_review as review


class ReviewTests(unittest.TestCase):
    def test_actual_editor_drafts(self):
        tree=ast.parse((ROOT/'backend/voice_research/conversational_transcript_review.py').read_text())
        template=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
        items=[{'id':f'conversation-ref-0{i}','audio_sha256':f'a{i}','draft_sha256':f'd{i}','draft':'Fixture Igbo ụ́','seconds':1,'review':{},'pauses':[],'audio':''} for i in (3,4)]
        script=template.split('<script>',1)[1].split('</script>',1)[0].replace('__DATA__',json.dumps(items)).replace('__NONCE__','"fixture"')
        result=subprocess.run(['node',str(ROOT/'scripts/test_conversational_transcript_drafts.mjs')],input=json.dumps({'script':script}),text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_callback_pins_valid_old_session_saves_and_preserves_external_changes(self):
        callbacks={};google=types.ModuleType('google');colab=types.ModuleType('google.colab');display=types.ModuleType('IPython.display')
        colab.output=types.SimpleNamespace(register_callback=lambda name,fn:callbacks.update({name:fn}));display.JSON=lambda value:value
        fixtures=ROOT/'.build/VoiceResearch/test-fixtures';fixtures.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixtures) as temp:
            folder=Path(temp);(folder/'transcription-v1').mkdir()
            report={'plan_id':'fixture','records':[{'id':f'conversation-ref-0{i}','listening_sha256':f'a{i}','draft_sha256':f'd{i}'} for i in (3,4)]}
            with patch.dict(sys.modules,{'google':google,'google.colab':colab,'IPython.display':display}),patch.object(review,'load_transcript_review',return_value=(folder,report)):
                review.register_transcript_review(True,mount_root=folder)
                rows=[{'id':r['id'],'audio_sha256':r['listening_sha256'],'draft_sha256':r['draft_sha256'],'transcript':'Ụ́tụtụ ọma.','words_match':'yes','clean_complete_speech':'yes','notes':''} for r in report['records']]
                save=callbacks['anyanwu.conversational_transcripts.save'];health=callbacks['anyanwu.conversational_transcripts.check']
                bad={'nonce':'0'*24,'records':[]};self.assertFalse(save(bad)['saved'])
                self.assertTrue(health({'read_only_health':True})['ready'])
                payload={'nonce':'1'*24,'records':rows};self.assertTrue(save(payload)['durable'])
                saved=folder/'transcription-v1/human-transcripts.json';self.assertIn('Ụ́tụtụ',saved.read_text())
                self.assertFalse(save({**payload,'nonce':'2'*24})['saved'])
                original=saved.read_bytes();saved.write_bytes(original+b' ')
                self.assertFalse(save(payload)['saved']);self.assertEqual(saved.read_bytes(),original+b' ')


if __name__=='__main__':unittest.main()
