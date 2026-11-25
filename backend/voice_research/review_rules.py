"""Validate explicit user listening feedback; never infer a human approval."""


def validate_cut_review(payload,allowed,nonce):
    keys={'id','nonce','words_match','complete_edges','clean_female_voice','notes'}
    if not isinstance(payload,dict) or set(payload)!=keys or payload['nonce']!=nonce or payload['id'] not in allowed:
        raise ValueError('Unknown or stale review request')
    choices=('unreviewed','yes','no')
    values=[payload[key] for key in ('words_match','complete_edges','clean_female_voice')]
    if any(value not in choices for value in values):
        raise ValueError('Invalid review choice')
    notes=payload['notes']
    if not isinstance(notes,str) or len(notes)>500:
        raise ValueError('Review notes exceed 500 characters')
    return {key:value for key,value in payload.items() if key!='nonce'} | {
        'audio_sha256':allowed[payload['id']], 'reviewer':'user',
        'status':'accepted_by_user_for_pilot' if all(v=='yes' for v in values) else 'held_by_user' if 'no' in values else 'pending',
        'scope':'Exact source cut/text and clean single female speech only; not model quality or general native certification'}
