"""Offline restore integrity, source-review gating and laptop safety fixtures."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.voice_research import expanded_train_restore as restore_module
from backend.voice_research.expanded_train_setup import run_expanded_training


class ExpandedTrainingTests(unittest.TestCase):
    def fixture(self,base):
        plan=copy.deepcopy(json.loads((ROOT/'.build/VoiceResearch/training/waxal-female5-expanded-lora-v2/plan.json').read_text()))
        filesets={};datasets={}
        for which,selected in [('baseline',plan['cases'][:24]),('new_source',plan['cases'][24:])]:
            files={};clips=[]
            for c in selected:
                item={**c,'seconds':1.0,'audio_bytes':12,'mp3_bytes':12}
                for suffix,key in (('.wav','audio_sha256'),('.mp3','mp3_sha256')):
                    body=('fixture-'+c['id']+suffix).encode();files['data/'+c['id']+suffix]=body;item[key]=hashlib.sha256(body).hexdigest()
                c['audio_sha256']=item['audio_sha256'];clips.append(item)
            files['plan.json']=json.dumps({'fixture':True}).encode();files['data-result.json']=json.dumps({'clips':clips}).encode()
            if which=='baseline':
                files['adapter.safetensors']=b'explicit-nonmodel-fixture';files['speech-targets.json']=b'[]';files['training-result.json']=b'{}'
                plan['starting_adapter_sha256']=hashlib.sha256(files['adapter.safetensors']).hexdigest()
            manifest={'files':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in files.items()}}
            archive=base/(which+'.zip')
            with zipfile.ZipFile(archive,'w') as out:
                for name,body in files.items():out.writestr(name,body)
                out.writestr('manifest.json',json.dumps(manifest))
            plan['archives'][which]={'drive_archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
            filesets[which]=files;datasets[which]=clips
        records={c['id']:{'id':c['id'],'audio_sha256':c['audio_sha256'],'words_match':'yes','complete_utterance':'yes','clean_female_voice':'yes'} for c in datasets['new_source']}
        feedback={'experiment':'anyanwu-waxal-female5-expansion-v1','records':records}
        path=base/'feedback.json';path.write_text(json.dumps(feedback));plan['source_feedback_snapshot']=str(path);plan['feedback_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        plan['combined_train_seconds']=34.0
        def mapped_path(value):
            if str(value)=='/content/anyanwu-waxal-female5-expanded-lora-v2':return base/'run'
            path=Path(value)
            if not path.is_relative_to(base):raise AssertionError('Unexpected fixture path '+str(value))
            return path
        return plan,mapped_path,feedback

    def test_hash_checked_restore_preserves_split_and_adapter(self):
        parent=ROOT/'.build/VoiceResearch/test-fixtures';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            base=Path(directory);plan,mapped,_=self.fixture(base)
            with patch.object(restore_module,'Path',mapped):result=restore_module.restore_expanded_training(plan,lambda value:None)
            self.assertEqual((result['train_clips'],result['validation_clips']),(34,4));self.assertEqual(len(result['clips']),38)
            self.assertEqual(hashlib.sha256((base/'run/baseline/adapter.safetensors').read_bytes()).hexdigest(),plan['starting_adapter_sha256'])
            self.assertFalse(set(result['reserved_evaluation_ids'])&{c['id'] for c in result['clips']})

    def test_rejected_feedback_never_installs_training_receipt(self):
        parent=ROOT/'.build/VoiceResearch/test-fixtures';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            base=Path(directory);plan,mapped,feedback=self.fixture(base)
            next(iter(feedback['records'].values()))['words_match']='no'
            path=Path(plan['source_feedback_snapshot']);path.write_text(json.dumps(feedback));plan['feedback_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(restore_module,'Path',mapped),self.assertRaisesRegex(RuntimeError,'not explicitly accepted'):
                restore_module.restore_expanded_training(plan,lambda value:None)
            self.assertFalse((base/'run/data-result.json').exists())

    def test_corrupt_archive_rejected_before_restore(self):
        parent=ROOT/'.build/VoiceResearch/test-fixtures';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            base=Path(directory);plan,mapped,_=self.fixture(base)
            Path(plan['archives']['baseline']['drive_archive']).write_bytes(b'corrupt')
            with patch.object(restore_module,'Path',mapped),self.assertRaisesRegex(RuntimeError,'checksum/size'):
                restore_module.restore_expanded_training(plan,lambda value:None)

    def test_mac_guard_precedes_restore_install_and_workers(self):
        if sys.platform=='darwin':
            with self.assertRaisesRegex(RuntimeError,'hosted Colab'):
                run_expanded_training({},'raise AssertionError()','',lambda value:None)


if __name__=='__main__':unittest.main()
