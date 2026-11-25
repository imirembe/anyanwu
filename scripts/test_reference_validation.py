"""Check constant-gain copies and the actual six-pair review script."""
from pathlib import Path
import ast
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.voice_research.reference_loudness import choose_loudness_target,normalize_listening_copies
FOLDER=ROOT/'.build/VoiceResearch/auditions/conversational-reference-validation-v2'


class ValidationTests(unittest.TestCase):
    def test_common_target_preserves_peak_headroom_and_rejects_silence(self):
        rows=[{'integrated_lufs':-18.,'true_peak_db':-1.},{'integrated_lufs':-39.,'true_peak_db':-21.}]
        target,gains=choose_loudness_target(rows)
        self.assertEqual(target,-20.)
        for row,gain in zip(rows,gains):self.assertLessEqual(row['true_peak_db']+gain,-1.5);self.assertLessEqual(gain,24)
        with self.assertRaisesRegex(RuntimeError,'finite'):choose_loudness_target([{'integrated_lufs':-math.inf,'true_peak_db':-math.inf}])
        with self.assertRaisesRegex(RuntimeError,'too quiet'):choose_loudness_target([{'integrated_lufs':-50.,'true_peak_db':-4.}])

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg unavailable for tiny non-speech fixture')
    def test_real_gain_matching_preserves_raw_bytes_and_frame_counts(self):
        fixtures=ROOT/'.build/VoiceResearch/test-fixtures';fixtures.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixtures) as directory:
            root=Path(directory);clips=[]
            for i in range(6):
                for variant,amplitude in [('base',.2),('adapter',.02)]:
                    path=root/(variant+'-case'+str(i)+'.wav')
                    samples=b''.join(struct.pack('<h',round(amplitude*32767*math.sin(2*math.pi*(300+50*i)*n/24000))) for n in range(24000))
                    with wave.open(str(path),'wb') as writer:
                        writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(24000);writer.writeframes(samples)
                    clips.append({'variant':variant,'text_id':'case'+str(i),'audio_filename':path.name,'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            original={'completed_wavs':12,'clips':clips,'optimizer_updates_this_stage':0,'source_checkpoints_unchanged':True}
            (root/'generation-result.json').write_text(json.dumps(original))
            plan={'outputs':12,'listening_loudness':{'requested_integrated_lufs':-20.,'true_peak_headroom_db':-1.5,'maximum_gain_db':24.}}
            receipt=normalize_listening_copies(root,plan,lambda _:None)
            self.assertLessEqual(receipt['maximum_spread_lu'],1.)
            result=json.loads((root/'generation-result.json').read_text())
            for old,new in zip(clips,result['clips']):
                self.assertEqual(hashlib.sha256((root/old['audio_filename']).read_bytes()).hexdigest(),old['audio_sha256'])
                self.assertEqual(new['listening_frames'],24000)
                self.assertLessEqual(new['listening_loudness']['true_peak_db'],-1.)
            self.assertEqual(normalize_listening_copies(root,plan,lambda _:None),receipt)
            (root/result['clips'][0]['listening_audio_filename']).write_bytes(b'corrupt fixture')
            with self.assertRaisesRegex(RuntimeError,'changed'):normalize_listening_copies(root,plan,lambda _:None)

    def test_actual_six_page_draft_save_script(self):
        module=ast.parse((FOLDER/'validation_review.py').read_text())
        template=next(n.value.value for n in ast.walk(module) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
        pairs=[{'id':'case'+str(i),'text':'ụ́ fixture','review':{},'clips':{v:{'sha256':v+str(i),'seconds':1,'warnings':[],'audio':''} for v in ('base','adapter')}} for i in range(6)]
        script=template.split('<script>',1)[1].split('</script>',1)[0].replace('__DATA__',json.dumps(pairs)).replace('__NONCE__','"fixture"').replace('__BACKUP__','{}')
        runner=(ROOT/'scripts/test_expanded_comparison_drafts.mjs').read_text().replace('anyanwu:expanded-comparison-drafts:v2','anyanwu:reference-validation-drafts:v2').replace('pairs:4','pairs:6').replace('audio_players,8','audio_players,12').replace('card:pairs.children[0]','card:pairs.children[1]')
        runner=runner.replace('this.children.push(...children);','this.children.push(...children);for(const c of children)if(c instanceof Element)c.parent=this;')
        old=" querySelectorAll(tag){return this.children.flatMap(c=>c instanceof Element?[...(c.tag===tag?[c]:[]),...c.querySelectorAll(tag)]:[]);}"
        new=""" getBoundingClientRect(){let p=this;while(p){if(p.style.display==='none')return {height:0};p=p.parent;}return {height:20};}
 querySelectorAll(tag){const matches=c=>tag.split(',').some(t=>t.startsWith('[')?c[t.slice(1,-1)]!==undefined:c.tag===t);return this.children.flatMap(c=>c instanceof Element?[...(matches(c)?[c]:[]),...c.querySelectorAll(tag)]:[]);}"""
        self.assertIn(old,runner);runner=runner.replace(old,new).replace('const context={document:',"const context={getComputedStyle:()=>({color:'rgb(23, 63, 52)'}),document:")
        marker="const key='anyanwu:reference-validation-drafts:v2';"
        navigation="""assert.equal(view.notices.at(-1).payload.feedback_fields,30);assert.equal(view.notices.at(-1).payload.visible_feedback_fields,5);assert.equal(view.notices.at(-1).payload.visible_save_buttons,1);const pager=view.pairs.children[0];for(let i=0;i<5;i++)pager.children[2].onclick();assert.equal(pager.children[2].disabled,true);assert.equal(view.pairs.children[6].style.display,'block');for(let i=0;i<5;i++)pager.children[0].onclick();assert.equal(view.pairs.children[1].style.display,'block');
"""
        runner=runner.replace(marker,navigation+marker)
        fixtures=ROOT/'.build/VoiceResearch/test-fixtures';fixtures.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixtures) as directory:
            path=Path(directory)/'review.mjs';path.write_text(runner)
            result=subprocess.run(['node',str(path)],input=json.dumps({'script':script}),text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_new_texts_fixed_model_and_mac_guard(self):
        plan=json.loads((FOLDER/'plan.json').read_text())
        self.assertEqual(plan['outputs'],12);self.assertEqual(len(plan['texts']),6);self.assertEqual(len(set(plan['adapter_hashes'].values())),1)
        self.assertFalse({t['id'] for t in plan['texts']}&set(plan['previously_reviewed_or_trained_ids']))
        self.assertNotIn('optimizer.step',(FOLDER/'validation_worker.py').read_text())
        namespace={};exec(compile((FOLDER/'validation_cloud.py').read_text(),'cloud','exec'),namespace)
        with patch.object(namespace['platform'],'system',return_value='Darwin'),patch.object(namespace['subprocess'],'run',side_effect=AssertionError('No local models')):
            with self.assertRaisesRegex(RuntimeError,'Hosted Colab only'):namespace['run_reference_validation']({},'','','','',lambda _:None)
        node=ast.parse((FOLDER/'validation_cloud.py').read_text());download=next(n.value.value for n in ast.walk(node) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='download' for t in n.targets));compile(download,'download','exec')


if __name__=='__main__':unittest.main()
