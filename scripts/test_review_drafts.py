"""No browser or Google writes; the real form script runs in a Node fixture."""
from pathlib import Path
import ast
import json
import re
import subprocess

root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'backend/voice_research/cut_review.py').read_text())
template=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
script=re.search(r'<script>(.*?)</script>',template,re.S).group(1)
data=[{'id':f'case{i}','text':'Fixture only: Ọ bụ m.','seconds':1,'audio_sha256':f'hash{i}','audio':'data:audio/wav;base64,','review':{}} for i in range(7)]
script=script.replace('__DATA__',json.dumps(data)).replace('__NONCE__',json.dumps('A'*24)).replace('__HELD__','2').replace('__BACKUP__','{}')
result=subprocess.run(['node',str(root/'scripts/test_review_drafts.mjs')],input=json.dumps({'script':script}),capture_output=True,text=True,timeout=20)
print(result.stdout,end='')
if result.returncode:
 print(result.stderr)
 raise SystemExit(result.returncode)
