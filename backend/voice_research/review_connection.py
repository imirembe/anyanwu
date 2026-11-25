"""Reconnect a bounded Colab feedback callback without replacing the form."""
from pathlib import Path
import hashlib
import json
import re
import secrets
import shutil
import time

from backend.voice_research.review_rules import validate_cut_review


def register_review_connection(alignment, recover_existing_form=False):
    from google.colab import output
    from IPython.display import JSON
    folder=Path('/content/anyanwu-bbc-expansion-v1')
    drive=Path('/content/drive/MyDrive/Anyanwu Voice Research/datasets/bbc-expansion-v1')
    folder.mkdir(exist_ok=True)
    allowed={c['id']:c['candidate_audio_sha256'] for c in alignment['cases']
             if c.get('cloud_audio_path') and not c.get('timing_warnings')}
    if len(allowed)!=7:
        raise RuntimeError('Expected the seven fixed review cuts')
    # Normal rendering pins a newly generated nonce. Explicit connection recovery
    # adopts the still-open form's first VALID nonce through Colab's trusted API,
    # without clearing its unsaved choices. Later requests must match that nonce.
    nonce=None if recover_existing_form else secrets.token_urlsafe(18)
    state={'experiment':alignment['experiment_id'],'records':{}}
    filename=folder/'human-reviews.json'
    source=drive/'human-reviews.json'
    if not source.exists():
        source=filename
    if source.exists():
        if source.stat().st_size>64*1024:
            raise RuntimeError('Saved review file size cap')
        state=json.loads(source.read_text())
        if state.get('experiment')!=alignment['experiment_id'] or not isinstance(state.get('records'),dict) or any(k not in allowed or v['audio_sha256']!=allowed[k] for k,v in state['records'].items()):
            raise RuntimeError('Saved review belongs to a different audio snapshot')

    def save_review(payload):
        nonlocal nonce,state
        try:
            proposed=payload.get('nonce') if isinstance(payload,dict) else None
            expected=nonce
            if expected is None:
                if not isinstance(proposed,str) or not re.fullmatch(r'[A-Za-z0-9_-]{24}',proposed):
                    raise ValueError('Invalid existing-form session identifier')
                expected=proposed
            record=validate_cut_review(payload,allowed,expected)
            # Bind only after ID, choices, length and nonce validation succeed.
            nonce=expected
            record['saved_at_unix']=time.time()
            updated={**state,'records':{**state['records'],record['id']:record}}
            encoded=json.dumps(updated,ensure_ascii=False,indent=2)+'\n'
            if len(encoded.encode())>64*1024:
                raise ValueError('Review file size cap')
            if filename.exists():
                shutil.copyfile(filename,folder/'human-reviews.previous.json')
            temporary=folder/'human-reviews.tmp'
            temporary.write_text(encoded)
            temporary.replace(filename)
            state=updated
            durable=False
            warning=None
            try:
                if drive.is_dir():
                    destination=drive/'human-reviews.json'
                    if destination.exists():
                        shutil.copyfile(destination,drive/'human-reviews.previous.json')
                    temporary=drive/'human-reviews.tmp'
                    temporary.write_text(encoded)
                    digest=hashlib.sha256(encoded.encode()).hexdigest()
                    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest:
                        raise RuntimeError('Drive review checksum mismatch')
                    temporary.replace(destination)
                    durable=hashlib.sha256(destination.read_bytes()).hexdigest()==digest
            except OSError:
                warning='Drive write unavailable; runtime copy retained'
            result={'saved':True,'durable':durable,'status':record['status'],'review_count':len(state['records']),
                    'training_started':False,'warning':warning}
            report={'phase':'review_feedback_saved','id':record['id'],'status':record['status'],
                    'review_count':len(state['records']),'durable':durable,'training_started':False}
            (folder/'last-review-save.json').write_text(json.dumps(report)+'\n')
            return JSON(result)
        except Exception as error:
            return JSON({'saved':False,'error':str(error)[:180],'training_started':False})

    def check(value):
        # Read-only probe never binds a recovered form or creates feedback.
        ready=(value==nonce) if nonce is not None else (isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]{24}',value) is not None)
        return JSON({'ready':ready,'candidate_count':len(allowed),'feedback_records_changed':False,
                     'review_count':len(state['records']),'recovered_existing_form':recover_existing_form})

    output.register_callback('anyanwu.expanded_cuts.save',save_review)
    output.register_callback('anyanwu.expanded_cuts.check',check)
    return nonce,state
