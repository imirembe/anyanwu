"""Verify the six reviews against the twelve matched listening-file hashes."""
from pathlib import Path
import hashlib
import json
import uuid

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / '.build/VoiceResearch'
FOLDER = LAB / 'auditions/conversational-reference-validation-v2'


def prepare():
    result = json.loads((FOLDER / 'generation-result.json').read_text())
    plan = json.loads((FOLDER / 'plan.admin.json').read_text())
    allowed = {}
    for clip in result['clips']:
        pair = allowed.setdefault(clip['text_id'], {})
        if clip['variant'] in pair:
            raise RuntimeError('Duplicate listening clip')
        pair[clip['variant']] = clip['listening_audio_sha256']
    if (len(allowed) != 6 or set(allowed) != {text['id'] for text in plan['texts']}
            or any(set(pair) != {'base', 'adapter'} for pair in allowed.values())):
        raise RuntimeError('Six exact matched-volume pairs required')
    validation = (FOLDER / 'validation_review.admin.py').read_text().split('\ndef restore_reference_validation_listening', 1)[0]
    previous = (LAB / 'auditions/conversational-reference-conditioning-v1/feedback-audit-cell.py').read_text()
    audit = previous[previous.index('import importlib.util,platform,math'):]
    replacements = [
        ('reference_conditioning_feedback_audit', 'reference_validation_feedback_audit'),
        ('conversational-reference-conditioning-v1', 'conversational-reference-validation-v2'),
        ('/content/drive/MyDrive/Anyanwu Voice Research/Anyanwu Voice Research', '/content/drive/MyDrive/Anyanwu Voice Research'),
        ('validate_reference_feedback', 'validate_reference_validation_feedback'),
        ("len(value['records'])>2", "len(value['records'])>6"),
        ('len(rows)==2', 'len(rows)==6'),
        ('def publish(value):', 'def publish_validation_feedback_audit(value):'),
        ('publish(report)', 'publish_validation_feedback_audit(report)'),
    ]
    for before, after in replacements:
        if before not in audit:
            raise RuntimeError('Audit template changed: ' + before)
        audit = audit.replace(before, after)
    audit_id = str(uuid.uuid4())
    marker = "def publish_validation_feedback_audit(value):\n"
    audit = audit.replace(marker, marker + f" value={{**value,'audit_id':{audit_id!r}}}\n", 1)
    old = "snapshot=path.with_name('human-comparison.submitted-'+digest[:12]+'.json')"
    new = "complete=len(rows)==6 and not missing and not incomplete;snapshot=path.with_name(('human-comparison.submitted-' if complete else 'human-comparison.partial-')+digest[:12]+'.json')"
    if audit.count(old) != 1:
        raise RuntimeError('Snapshot template changed')
    audit = audit.replace(old, new)
    audit = audit.replace("not isinstance(stamp,(int,float)) or", "isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or")
    cell = '# Anyanwu verify SIX saved voice reviews; no form, audio or model changes\n' + validation + '\nALLOWED=' + repr(allowed) + '\n' + audit
    compile(cell, 'validation-feedback-audit', 'exec')
    (FOLDER / 'feedback-audit-cell.py').write_text(cell)
    receipt = {
        'audit_id': audit_id, 'source_sha256': hashlib.sha256(cell.encode()).hexdigest(),
        'pairs': 6, 'listening_hashes': 12, 'uses_matched_audio_hashes': True,
        'ratings_submitted_by_agent': False, 'new_synthesis': False, 'training': False,
    }
    (FOLDER / 'feedback-audit-source-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
