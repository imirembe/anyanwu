"""Execute the actual source-review template with isolated JS fixtures."""
from pathlib import Path
import ast
import json
import subprocess

root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'backend/voice_research/source_expansion_review.py').read_text())
template=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and '<script>(async()' in n.value)
data=[{'id':'case'+str(i),'text':'ụ́ fixture','seconds':1,'audio_sha256':'fixture'+str(i),'audio':'fixture-only','review':{}} for i in range(14)]
script=template.split('<script>',1)[1].split('</script>',1)[0].replace('__DATA__',json.dumps(data)).replace('__NONCE__',json.dumps('a'*24)).replace('__BACKUP__','{}').replace('__COUNTS__',json.dumps({'accepted':0,'held':0,'incomplete':14}))
result=subprocess.run(['node','scripts/test_source_expansion_drafts.mjs'],input=json.dumps({'script':script}),text=True,capture_output=True,cwd=root)
print(result.stdout,end='');print(result.stderr,end='');raise SystemExit(result.returncode)
