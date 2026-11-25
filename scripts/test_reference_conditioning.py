"""Offline checks for fixed-v1 inputs, whole-reference limits and review recovery."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.voice_research.reference_conditioning import restore_reference_inputs,context_positions,match_reference_boundary
FOLDER=ROOT/'.build/VoiceResearch/auditions/conversational-reference-conditioning-v1'


class ReferenceTests(unittest.TestCase):
    def test_whole_reference_capacity_and_edge_rounding(self):
        self.assertEqual(context_positions(7,123,611,70,400,1536),1214)
        with self.assertRaisesRegex(RuntimeError,'no truncation'):context_positions(7,123,611,500,400,1536)
        self.assertEqual(match_reference_boundary(611,1221),610)
        with self.assertRaisesRegex(RuntimeError,'rounding'):match_reference_boundary(611,1000)

    def test_fixed_baseline_restore_and_reject_changed_transcript(self):
        fixtures=ROOT/'.build/VoiceResearch/test-fixtures';fixtures.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixtures) as directory:
            folder=Path(directory);sha=lambda b:hashlib.sha256(b).hexdigest()
            v1,wav,mp3,conversation=b'v1 fixture',b'wav fixture',b'mp3 fixture',b'full conversation fixture'
            text='ụ́ Original like English';target={'id':'conversation-ref-03','text':text,'speech_ids':[1]*611,'text_ids':[2]*123}
            review={'records':{'conversation-ref-03':{'transcript':text,'words_match':'yes','clean_complete_speech':'yes'}}}
            snapshot=folder/'review.json';snapshot.write_text(json.dumps(review))
            targets=folder/'targets.json';targets.write_text(json.dumps([target]))
            audio=folder/'reference.wav';audio.write_bytes(conversation)
            source={'id':'ibo_616','speaker_id':'5','gender':'Female','adapter_split':'train','audio_sha256':sha(wav),'mp3_sha256':sha(mp3)}
            old={'plan.json':b'{"clips":[{"id":"ibo_616"}]}','training-result.json':b'{"optimizer_updates":60}',
                 'adapter.safetensors':v1,'data-result.json':json.dumps({'clips':[source]}).encode(),
                 'speech-targets.json':b'[{"id":"ibo_616","speech_ids":[1],"text_ids":[2]}]',
                 'data/ibo_616.wav':wav,'data/ibo_616.mp3':mp3}
            condition={'reference_id':'conversation-ref-03','speaker_id':'IGB_005','submitted_transcript_snapshot_sha256':sha(snapshot.read_bytes()),
                'encoded_targets_sha256':sha(targets.read_bytes()),'transcript_sha256':sha(text.encode()),'listening_audio_sha256':sha(conversation),'analysis_sha256':'a'*64}
            plan={'checkpoint_sha256':sha(v1),'archives':{'previous':{}},'texts':[{'id':'fresh0'},{'id':'fresh1'}],
                'conditions':[{},condition],'previously_reviewed_or_trained_ids':['ibo_616'],
                'submitted_snapshot':str(snapshot),'encoded_targets_path':str(targets),'reference_audio_path':str(audio)}
            with patch('backend.voice_research.expanded_comparison_cloud.read_comparison_archive',return_value=old):
                restore_reference_inputs(plan,root=folder/'run',warm=folder/'warm')
                self.assertEqual((folder/'run/assets/adapter.safetensors').read_bytes(),v1)
                self.assertEqual((folder/'run/assets/previous-adapter.safetensors').read_bytes(),v1)
                cases=json.loads((folder/'run/assets/speech-targets.json').read_text())
                self.assertEqual(cases[1]['text'],text);self.assertEqual(len(cases[1]['speech_ids']),611)
                plan['texts'][0]['id']='ibo_616'
                with self.assertRaisesRegex(RuntimeError,'overlaps'):restore_reference_inputs(plan,root=folder/'bad',warm=folder/'warm')
                plan['texts'][0]['id']='fresh0'
                review['records']['conversation-ref-03']['transcript']='translated fixture';snapshot.write_text(json.dumps(review))
                condition['submitted_transcript_snapshot_sha256']=sha(snapshot.read_bytes())
                with self.assertRaisesRegex(RuntimeError,'user review'):restore_reference_inputs(plan,root=folder/'bad2',warm=folder/'warm')

    def test_no_local_model_work_and_embedded_sources_compile(self):
        cell=ast.parse((FOLDER/'reference-cell.py').read_text())
        namespace={'__name__':'offline_fixture'}
        cloud=(FOLDER/'reference_cloud.py').read_text();exec(compile(cloud,'cloud','exec'),namespace)
        with patch.object(namespace['platform'],'system',return_value='Darwin'),patch.object(namespace['subprocess'],'run',side_effect=AssertionError('Local model work')):
            with self.assertRaisesRegex(RuntimeError,'Hosted Colab only'):namespace['run_reference_comparison']({},'','','','',lambda _:None)
        download=next(n.value.value for n in ast.walk(ast.parse(cloud)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='download' for t in n.targets))
        compile(download,'download','exec')
        plan=json.loads((FOLDER/'plan.json').read_text())
        self.assertEqual(len(set(plan['adapter_hashes'].values())),1)
        self.assertEqual(plan['outputs'],4);self.assertEqual(plan['optimizer_updates_this_stage'],0)
        worker=(FOLDER/'reference_worker.py').read_text();self.assertNotIn('optimizer.step',worker)
        self.assertIn('context_positions(',worker);self.assertIn("references[item['variant']]",worker)

    def test_actual_two_pair_draft_recovery_script(self):
        module=ast.parse((FOLDER/'reference_review.py').read_text())
        template=next(n.value.value for n in ast.walk(module) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
        pairs=[{'id':'case'+str(i),'text':'ụ́ fixture','review':{},'clips':{v:{'sha256':v+str(i),'seconds':1,'warnings':[],'audio':''} for v in ('base','adapter')}} for i in range(2)]
        script=template.split('<script>',1)[1].split('</script>',1)[0].replace('__DATA__',json.dumps(pairs)).replace('__NONCE__','"fixture"').replace('__BACKUP__','{}')
        runner=(ROOT/'scripts/test_expanded_comparison_drafts.mjs').read_text().replace('anyanwu:expanded-comparison-drafts:v2','anyanwu:reference-conditioning-drafts:v1').replace('pairs:4','pairs:2').replace('audio_players,8','audio_players,4')
        fixtures=ROOT/'.build/VoiceResearch/test-fixtures';fixtures.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixtures) as directory:
            path=Path(directory)/'review.mjs';path.write_text(runner)
            result=subprocess.run(['node',str(path)],input=json.dumps({'script':script}),text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
