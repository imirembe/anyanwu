#!/usr/bin/env python3
"""Check Git's publishable file set without printing credential values.

This is a focused release guard, not a substitute for human source review.
Use --tracked in CI; the default also checks non-ignored, untracked files.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PRIVATE_PATHS=('AGENTS.md','Voice Research/private.ipynb','Configuration/Local.xcconfig',
               'backend/.env','backend/voice_research/brief.json','.build/private.json',
               'research/igbo_lab/data/private_review.jsonl')
RULES={
    'private-key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'openai-key':re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}'),
    'huggingface-token':re.compile(r'\bhf_[A-Za-z0-9]{20,}'),
    'github-token':re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}'),
    'aws-access-key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'personal-macos-path':re.compile(r'/Users/[A-Za-z0-9._-]+/'),
    'private-google-document':re.compile(r'https://(?:colab\.research\.google\.com/drive/|drive\.google\.com/(?:file/|drive/))'),
    'physical-device-id':re.compile(r'\b[0-9A-F]{8}-[0-9A-F]{16}\b'),
}
FORBIDDEN_SUFFIXES={'.p12','.pfx','.mobileprovision','.provisionprofile','.key','.pem',
                    '.sqlite','.sqlite3','.db','.wav','.mp3','.m4a','.mp4','.webm',
                    '.safetensors','.pt','.pth','.ckpt','.parquet','.arrow','.zip'}


def inspect_file(path,raw):
    """Return rule names/line numbers, never matched strings or file contents."""
    issues=[]
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:issues.append({'rule':'private-or-generated-file'})
    if path.name.startswith('.env') and not path.name.endswith('.example'):issues.append({'rule':'environment-file'})
    if len(raw)>10*1024**2:issues.append({'rule':'file-exceeds-10MiB'})
    if path.suffix.lower() in {'.png','.ttf'}:return issues
    try:text=raw.decode('utf-8')
    except UnicodeError:
        issues.append({'rule':'unreviewed-binary-file'});return issues
    for name,pattern in RULES.items():
        for match in pattern.finditer(text):issues.append({'rule':name,'line':text.count('\n',0,match.start())+1})
    if path.suffix=='.ipynb':
        try:
            nb=json.loads(text)
            if any(c.get('outputs') or c.get('execution_count') is not None for c in nb.get('cells',[]) if c.get('cell_type')=='code'):
                issues.append({'rule':'notebook-has-execution-output'})
        except (ValueError,TypeError):issues.append({'rule':'invalid-notebook'})
    return issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracked',action='store_true',help='Check tracked files only, for a clean CI checkout.')
    args=parser.parse_args()
    command=['git','ls-files','--cached','-z']
    if not args.tracked:command+=['--others','--exclude-standard']
    result=subprocess.run(command,cwd=ROOT,check=True,capture_output=True)
    paths=sorted(set(p.decode() for p in result.stdout.split(b'\0') if p))
    if not paths:raise SystemExit('No publishable files found.')
    issues=[];total=0
    for name in paths:
        path=ROOT/name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            issues.append({'path':name,'rule':'symlink-or-external-path'});continue
        if not path.is_file():
            issues.append({'path':name,'rule':'missing-file'});continue
        raw=path.read_bytes();total+=len(raw)
        issues.extend({'path':name,**item} for item in inspect_file(Path(name),raw))
    for name in PRIVATE_PATHS:
        check=subprocess.run(['git','check-ignore','--no-index','-q',name],cwd=ROOT)
        if check.returncode!=0:issues.append({'path':name,'rule':'private-path-not-ignored'})
    report={'files_checked':len(paths),'total_bytes':total,'passed':not issues,'issues':issues,
            'scope':'tracked' if args.tracked else 'tracked-and-nonignored-untracked'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if issues else 0


if __name__=='__main__':sys.exit(main())
