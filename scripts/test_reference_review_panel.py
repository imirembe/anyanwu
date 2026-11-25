"""Exercise the real paged review JS without submitting user feedback."""
from pathlib import Path
import ast
import json
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ReviewPanelTests(unittest.TestCase):
    def test_navigation_and_durable_drafts(self):
        path=ROOT/'.build/VoiceResearch/auditions/conversational-reference-conditioning-v1/combined_review_panel.py'
        module=ast.parse(path.read_text())
        template=next(n.value.value for n in ast.walk(module) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
        pairs=[{'id':'case'+str(i),'text':'ụ́ fixture','review':{},'clips':{v:{'sha256':v+str(i),'seconds':1,'warnings':[],'audio':''} for v in ('base','adapter')}} for i in range(2)]
        script=template.split('<script>',1)[1].split('</script>',1)[0].replace('__DATA__',json.dumps(pairs)).replace('__NONCE__','"fixture"').replace('__BACKUP__','{}')
        runner=(ROOT/'scripts/test_expanded_comparison_drafts.mjs').read_text()
        runner=runner.replace('anyanwu:expanded-comparison-drafts:v2','anyanwu:reference-conditioning-drafts:v1').replace('pairs:4','pairs:2').replace('audio_players,8','audio_players,4').replace('card:pairs.children[0]','card:pairs.children[1]')
        runner=runner.replace('this.children.push(...children);','this.children.push(...children);for(const c of children)if(c instanceof Element)c.parent=this;')
        old=" querySelectorAll(tag){return this.children.flatMap(c=>c instanceof Element?[...(c.tag===tag?[c]:[]),...c.querySelectorAll(tag)]:[]);}";
        new=""" getBoundingClientRect(){let p=this;while(p){if(p.style.display==='none')return {height:0};p=p.parent;}return {height:20};}
 querySelectorAll(tag){const matches=c=>tag.split(',').some(t=>t.startsWith('[')?c[t.slice(1,-1)]!==undefined:c.tag===t);return this.children.flatMap(c=>c instanceof Element?[...(matches(c)?[c]:[]),...c.querySelectorAll(tag)]:[]);}"""
        self.assertIn(old,runner);runner=runner.replace(old,new)
        runner=runner.replace('const context={document:',"const context={getComputedStyle:()=>({color:'rgb(23, 63, 52)'}),document:")
        marker="const key='anyanwu:reference-conditioning-drafts:v1';"
        navigation="""assert.equal(view.notices.at(-1).payload.feedback_fields,10);assert.equal(view.notices.at(-1).payload.visible_feedback_fields,5);assert.equal(view.notices.at(-1).payload.visible_save_buttons,1);
const pager=view.pairs.children[0];assert.equal(pager.children[0].disabled,true);pager.children[2].onclick();assert.equal(view.pairs.children[1].style.display,'none');assert.equal(view.pairs.children[2].style.display,'block');assert.equal(pager.children[2].disabled,true);pager.children[0].onclick();assert.equal(view.pairs.children[1].style.display,'block');
"""
        runner=runner.replace(marker,navigation+marker)
        fixtures=ROOT/'.build/VoiceResearch/test-fixtures';fixtures.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixtures) as directory:
            file=Path(directory)/'paged-review.mjs';file.write_text(runner)
            result=subprocess.run(['node',str(file)],input=json.dumps({'script':script}),text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
