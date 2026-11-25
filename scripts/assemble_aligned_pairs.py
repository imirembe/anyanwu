"""Merge the preserved timing passes without promoting them to verified training."""
from pathlib import Path
import base64
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'.build/VoiceResearch/training/anyanwu-female-v1'


def assemble():
    original=json.loads((FOLDER/'alignment-summary.json').read_text())
    context=json.loads((FOLDER/'alignment-context-summary.json').read_text())
    cases={c['id']:dict(c,alignment_pass='one_second_context') for c in original['cases']}
    for item in context['cases']:
        cases[item['id']]=dict(item,alignment_pass='two_second_context')
    candidates=[]
    held=[]
    for item in cases.values():
        item['source_group']=item['source_id']
        item['speaker_group_status']='Separate source group; individual speaker diarization not independently established'
        item['split']=None
        item['training_ready']=False
        if item.get('alignment_status')!='candidate_cut_created' or item.get('timing_warnings'):
            item['dataset_status']='held_for_boundary_repair'
            held.append(item)
        else:
            item['dataset_status']='unflagged_machine_alignment_candidate'
            candidates.append(item)
    report={'id':'bbc-caption-pairing-pilot-v1','user_caption_match_review':'confirmed_for_BBC_subtitles',
            'source_captions':'caption-text.json','candidates':candidates,'held':held,'candidates_count':len(candidates),
            'candidate_audio_seconds':round(sum(c['candidate_seconds'] for c in candidates),6),
            'held_count':len(held),'total_cut_count':len(cases),'trained':False,'optimizer_updates':0,
            'ready_training_pairs':0,'source_groups':len({c['source_id'] for c in candidates}),
            'split_policy':'Unassigned; never reuse the same source or near-duplicate caption across training and evaluation',
            'quality_status':'Machine timing candidates, not independently verified complete utterances or naturalness evidence',
            'limits':{'new_network_requests':0,'new_model_downloads':0,'maximum_captions':5,'cloud_only':True},
            'provenance':{'first_pass':'alignment-summary.json','context_repair':'alignment-context-summary.json','caption_plan':'alignment-plan.json','context_plan':'alignment-context-plan.json'}}
    (FOLDER/'pairing-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (FOLDER/'pairing-candidates.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in candidates),encoding='utf-8')
    return report


if __name__=='__main__':
    report=assemble()
    print(json.dumps({k:report[k] for k in ('candidates_count','candidate_audio_seconds','held_count','source_groups','trained','optimizer_updates')}))
