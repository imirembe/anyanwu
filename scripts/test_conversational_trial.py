"""Offline source-integrity, context, separation and GPU gate tests."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.voice_research.conversational_trial_restore import restore_conversational_trial
from backend.voice_research.conversational_trial_setup import run_conversational_trial

sha=lambda raw:hashlib.sha256(raw).hexdigest()


class TrialTests(unittest.TestCase):
    def fixture(self,folder):
        ids=['ibo_616']+[f'ibo_{i:03}' for i in range(1,24)]
        old=[{'id':key,'split':'train' if i<20 else 'validation','text_ids':[1],'speech_ids':[2,3],'seconds':1} for i,key in enumerate(ids)]
        text='Ndewo ụ́ so.'
        entries={'plan.json':b'{}','training-result.json':b'{"optimizer_updates":60}',
                 'adapter.safetensors':b'fixture baseline only','speech-targets.json':json.dumps(old).encode(),
                 'data-result.json':json.dumps({'clips':[{'id':key,'text':text} for key in ids]}).encode(),
                 'data/ibo_616.wav':b'fixture waveform','data/ibo_616.mp3':b'fixture source'}
        archive=folder/'baseline.zip'
        with zipfile.ZipFile(archive,'w') as package:
            for name,raw in entries.items():package.writestr(name,raw)
            package.writestr('manifest.json',json.dumps({'files':{name:{'bytes':len(raw),'sha256':sha(raw)} for name,raw in entries.items()}}))
        new=[{'id':f'conversation-ref-0{i}','text':text,'text_ids':[1],'speech_ids':[2]*count,
              'transcript_sha256':sha(text.encode())} for i,count in ((3,611),(4,654))]
        targets=folder/'targets.json';targets.write_text(json.dumps(new))
        snapshot=folder/'review.json';snapshot.write_text(json.dumps({'records':{r['id']:{'transcript':text,'words_match':'yes','clean_complete_speech':'yes'} for r in new}}))
        plan={'baseline_archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive.read_bytes())},
              'starting_adapter_sha256':sha(entries['adapter.safetensors']),'submitted_snapshot':str(snapshot),
              'submitted_review_sha256':sha(snapshot.read_bytes()),'encoded_targets':{'path':str(targets),'sha256':sha(targets.read_bytes())},
              'comparison':{'texts':[{'id':'fresh'}]},'schedule':[key for anchor in ids[:10] for key in ('conversation-ref-03','conversation-ref-04',anchor)]}
        return plan,targets

    def test_whole_targets_two_source_groups_and_no_baseline_overwrite(self):
        fixture_root=ROOT/'.build/VoiceResearch/test-fixtures';fixture_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixture_root) as directory:
            folder=Path(directory);plan,_=self.fixture(folder)
            result=restore_conversational_trial(plan,root=folder/'restored')
            cases=json.loads((folder/'restored/training-cases.json').read_text())
            self.assertEqual(result['source_groups'],2)
            self.assertEqual([len(c['speech_ids']) for c in cases if c['source_index']==1],[611,654])
            self.assertEqual(sum(c['split']=='validation' for c in cases),4)
            self.assertEqual((folder/'restored/baseline/adapter.safetensors').read_bytes(),b'fixture baseline only')
            self.assertTrue(all(c['text']=='Ndewo ụ́ so.' for c in cases))

    def test_modified_review_or_oversized_target_is_rejected(self):
        fixture_root=ROOT/'.build/VoiceResearch/test-fixtures';fixture_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixture_root) as directory:
            folder=Path(directory);plan,targets=self.fixture(folder)
            original=Path(plan['submitted_snapshot']).read_bytes();Path(plan['submitted_snapshot']).write_bytes(original+b' ')
            with self.assertRaisesRegex(RuntimeError,'Reviewed text'):restore_conversational_trial(plan,root=folder/'bad-review')
            Path(plan['submitted_snapshot']).write_bytes(original)
            rows=json.loads(targets.read_text());rows[0]['speech_ids']=[2]*801;targets.write_text(json.dumps(rows));plan['encoded_targets']['sha256']=sha(targets.read_bytes())
            with self.assertRaisesRegex(RuntimeError,'Whole target exceeds'):restore_conversational_trial(plan,root=folder/'bad-context')

    def test_cpu_stops_before_model_restore_or_installs(self):
        with patch('backend.voice_research.conversational_trial_setup.platform.system',return_value='Linux'),patch('backend.voice_research.conversational_trial_setup.Path.is_dir',return_value=True),patch('backend.voice_research.conversational_trial_setup.importlib.util.find_spec',return_value=object()),patch('backend.voice_research.conversational_trial_setup.shutil.which',return_value=None),patch('backend.voice_research.conversational_trial_setup.subprocess.run',side_effect=AssertionError('No CPU install or GPU allocation')):
            with self.assertRaisesRegex(RuntimeError,'current CPU is preserved'):run_conversational_trial({},'','',lambda _:None)

    def test_registered_schedule_and_fresh_comparison_separation(self):
        plan=json.loads((ROOT/'.build/VoiceResearch/training/conversational-style-lora-v1/plan.json').read_text())
        self.assertEqual(plan['schedule'].count('conversation-ref-03'),10);self.assertEqual(plan['schedule'].count('conversation-ref-04'),10)
        self.assertEqual(len(plan['schedule']),30);self.assertEqual(plan['position_cap'],1024)
        self.assertLess(max(c['packed_positions'] for c in plan['capacity_cases']),1024)
        self.assertFalse(set(plan['schedule'])&{c['id'] for c in plan['comparison']['texts']})
        self.assertFalse({t['id'] for t in plan['comparison']['texts']}&set(plan['comparison']['prior_comparison_texts_excluded']))
        self.assertEqual(plan['starting_adapter_sha256'],'d7d6a2d60501c100ee17e08f1ae008a1febe41e84882904626c6256a2c7435fd')


if __name__=='__main__':unittest.main()
