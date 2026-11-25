"""Extract and execute the real embedded UI against an isolated JS fixture."""
from pathlib import Path
import ast
import json
import subprocess

root=Path(__file__).resolve().parents[1]
module=ast.parse((root/'backend/voice_research/comparison_review.py').read_text())
template=next(n.value for n in ast.walk(module) if isinstance(n,ast.Constant) and isinstance(n.value,str) and '<script>(async()' in n.value)
cases=[{'id':'case'+str(i),'text':'ụ́ fixture','review':{},'clips':{variant:{'sha256':variant+str(i),'seconds':1,'warnings':[],'audio':'fixture-only'} for variant in ('base','adapter')}} for i in range(3)]
script=template.split('<script>',1)[1].split('</script>',1)[0].replace('__DATA__',json.dumps(cases)).replace('__NONCE__',json.dumps('a'*24)).replace('__BACKUP__','{}')
result=subprocess.run(['node','scripts/test_comparison_drafts.mjs'],input=json.dumps({'script':script}),text=True,cwd=root,capture_output=True)
print(result.stdout,end='');print(result.stderr,end='')
raise SystemExit(result.returncode)
