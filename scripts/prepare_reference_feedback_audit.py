"""Build a CPU-only audit of the two saved reference-comparison reviews."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
LAB=ROOT/'.build/VoiceResearch'
FOLDER=LAB/'auditions/conversational-reference-conditioning-v1'


def prepare():
    result=json.loads((FOLDER/'generation-result.json').read_text())
    allowed={}
    for row in result['clips']:
        allowed.setdefault(row['text_id'],{})[row['variant']]=row['audio_sha256']
    if len(allowed)!=2 or any(set(pair)!={'base','adapter'} for pair in allowed.values()):
        raise RuntimeError('Two exact reference comparison pairs required')
    validation=(FOLDER/'reference_review.py').read_text().split('\ndef restore_reference_listening',1)[0]
    old=(LAB/'auditions/conversational-voice-comparison-v1/feedback-audit-cell.py').read_text()
    audit=old[old.index('import importlib.util,platform,math'):]
    for before,after in [('conversational_feedback_audit','reference_conditioning_feedback_audit'),
                         ('conversational-voice-comparison-v1','conversational-reference-conditioning-v1'),
                         ('validate_conversational_comparison_feedback','validate_reference_feedback'),
                         ("len(value['records'])>4","len(value['records'])>2"),('len(rows)==4','len(rows)==2')]:
        if before not in audit:raise RuntimeError('Audit template changed: '+before)
        audit=audit.replace(before,after)
    marker="  report={'phase':'reference_conditioning_feedback_audit','saved_file_found':True"
    if audit.count(marker)!=1:raise RuntimeError('Audit completion template changed')
    audit=audit.replace(marker,"  if path.read_bytes()!=raw:raise RuntimeError('Feedback changed during audit; snapshot preserved, re-read latest')\n"+marker)
    cell='# Anyanwu verify reference-comparison feedback — read only, no form changes\n'+validation+'\nALLOWED='+repr(allowed)+'\n'+audit
    compile(cell,'reference-feedback-audit','exec')
    (FOLDER/'feedback-audit-cell.py').write_text(cell)
    receipt={'phase':'reference_feedback_audit_prepared','source_sha256':hashlib.sha256(cell.encode()).hexdigest(),
             'pairs':2,'audio_hashes':allowed,'ratings_submitted_by_agent':False,'new_synthesis':False,'training':False}
    (FOLDER/'feedback-audit-source-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
