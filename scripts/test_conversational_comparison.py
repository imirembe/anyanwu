"""Offline integrity, evaluation separation and actual review-JS checks."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.voice_research.conversational_comparison_cloud import restore_conversational_comparison,run_conversational_comparison


class ComparisonTests(unittest.TestCase):
    def test_verified_restoration_and_new_training_counts(self):
        fixtures=ROOT/'.build/VoiceResearch/test-fixtures';fixtures.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixtures) as directory:
            folder=Path(directory);sha=lambda b:hashlib.sha256(b).hexdigest();old,new,wav,mp3=b'v1 fixture',b'candidate fixture',b'wave fixture',b'mp3 fixture'
            source={'id':'prompt','speaker_id':'5','gender':'Female','adapter_split':'train','audio_sha256':sha(wav),'mp3_sha256':sha(mp3)}
            previous={'plan.json':json.dumps({'clips':[{'id':'prompt'}]}).encode(),'training-result.json':b'{"optimizer_updates":60}',
                      'adapter.safetensors':old,'data-result.json':json.dumps({'clips':[source]}).encode(),
                      'speech-targets.json':json.dumps([{'id':'prompt','speech_ids':[1],'text_ids':[2]}]).encode(),
                      'data/prompt.wav':wav,'data/prompt.mp3':mp3}
            candidate={'plan.json':json.dumps({'schedule':['conversation-ref-03'],'starting_adapter_sha256':sha(old)}).encode(),
                       'training-result.json':b'{"optimizer_updates":30,"cumulative_lora_updates":90,"baseline_unchanged":true,"adapter_reloaded":true}',
                       'adapter.safetensors':new}
            def package(name,files):
                path=folder/name
                with zipfile.ZipFile(path,'w') as out:
                    for key,body in files.items():out.writestr(key,body)
                    out.writestr('manifest.json',json.dumps({'files':{key:{'bytes':len(body),'sha256':sha(body)} for key,body in files.items()}}))
                return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())}
            plan={'prompt_id':'prompt','texts':[{'id':f'fresh{i}'} for i in range(4)],'adapter_hashes':{'base':sha(old),'adapter':sha(new)},
                  'archives':{'previous':package('old.zip',previous),'candidate':package('new.zip',candidate)}}
            result=restore_conversational_comparison(plan,root=folder/'comparison',warm=folder/'warm')
            self.assertEqual(result['optimizer_updates_this_stage'],0)
            self.assertEqual((folder/'comparison/assets/previous-adapter.safetensors').read_bytes(),old)
            self.assertEqual((folder/'comparison/assets/adapter.safetensors').read_bytes(),new)
            self.assertEqual(len(json.loads((folder/'comparison/assets/speech-targets.json').read_text())),1)
            plan['texts'][0]['id']='conversation-ref-03'
            with self.assertRaisesRegex(RuntimeError,'overlap'):restore_conversational_comparison(plan,root=folder/'bad',warm=folder/'warm')

    def test_mac_guard_precedes_models_or_installs(self):
        with patch('backend.voice_research.conversational_comparison_cloud.platform.system',return_value='Darwin'),patch('backend.voice_research.conversational_comparison_cloud.subprocess.run',side_effect=AssertionError('No local install')):
            with self.assertRaisesRegex(RuntimeError,'Hosted Colab only'):run_conversational_comparison({},'','','','',lambda _:None)

    def test_actual_review_script_and_download_child(self):
        module=ast.parse((ROOT/'backend/voice_research/conversational_comparison_cloud.py').read_text())
        download=next(n.value.value for n in ast.walk(module) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='download' for t in n.targets))
        compile(download,'download-child','exec')
        review=ast.parse((ROOT/'.build/VoiceResearch/auditions/conversational-voice-comparison-v1/comparison_review.py').read_text())
        template=next(n.value.value for n in ast.walk(review) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
        pairs=[{'id':'case'+str(i),'text':'ụ́ fixture','review':{},'clips':{v:{'sha256':v+str(i),'seconds':1,'warnings':[],'audio':''} for v in ('base','adapter')}} for i in range(4)]
        script=template.split('<script>',1)[1].split('</script>',1)[0].replace('__DATA__',json.dumps(pairs)).replace('__NONCE__','"fixture"').replace('__BACKUP__','{}')
        runner=(ROOT/'scripts/test_expanded_comparison_drafts.mjs').read_text().replace('anyanwu:expanded-comparison-drafts:v2','anyanwu:conversational-comparison-drafts:v1')
        fixtures=ROOT/'.build/VoiceResearch/test-fixtures';fixtures.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixtures) as directory:
            path=Path(directory)/'review-fixture.mjs';path.write_text(runner)
            result=subprocess.run(['node',str(path)],input=json.dumps({'script':script}),text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_fixed_one_take_plan_and_no_training_code(self):
        folder=ROOT/'.build/VoiceResearch/auditions/conversational-voice-comparison-v1';plan=json.loads((folder/'plan.json').read_text())
        self.assertEqual(plan['outputs'],8);self.assertEqual(plan['seed'],17);self.assertEqual(plan['optimizer_updates_this_stage'],0)
        self.assertEqual([t['id'] for t in plan['texts']],['ibo_1703','ibo_1715','ibo_1755','ibo_1790'])
        self.assertNotIn('optimizer.step',(folder/'comparison_worker.py').read_text())
        self.assertIn("'adapter':90",(folder/'comparison_worker.py').read_text())
        self.assertNotIn('anyanwu.expanded_comparison',(folder/'comparison_review.py').read_text())


if __name__=='__main__':unittest.main()
