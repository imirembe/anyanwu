"""Materialise AI-transcribed subtitle text with user-review and frame provenance."""
from pathlib import Path
import hashlib
import json
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/".build/VoiceResearch/training/anyanwu-female-v1"


def assemble():
    decisions=json.loads((FOLDER/"caption-transcription-decisions.json").read_text())
    capture=json.loads((FOLDER/"caption-capture.json").read_text())
    repairs=json.loads((FOLDER/"caption-crop-repair.json").read_text())
    review=json.loads((FOLDER/"user-caption-review.json").read_text())
    plan=json.loads((FOLDER/"plan.json").read_text())
    entries=[]
    counts={}
    for item in decisions['entries']:
        source_id=item['source_id']
        expected=next(c for c in plan['clips'] if c['id']==source_id)
        first,last=item['first_seen'],item['last_seen']
        if not 12<=first<=last<=23:
            raise ValueError('Caption observations must stay in the actual pilot window.')
        repaired=next((r for r in repairs['sources'] if r['id']==source_id and any(f.get('source_second')==first and 'sha256' in f for f in r['frames'])),None)
        if repaired:
            image=next(f for f in repaired['frames'] if f['source_second']==first)
            cloud_image=f"/content/anyanwu-bbc-caption-crop-repair-v1/{source_id}/{image['png']}"
            image_hash=image['sha256']
        else:
            source=next(r for r in capture['sources'] if r['id']==source_id)
            image=next(a for a in source['artifacts'] if a['filename']==f'caption-{first:02d}.png')
            cloud_image=f"/content/anyanwu-bbc-caption-capture-v2/{source_id}/{image['filename']}"
            image_hash=image['sha256']
        lines=[unicodedata.normalize('NFC',line) for line in item['lines']]
        text=' '.join(lines)
        if item.get('join_hyphenated_line'):
            # The displayed word ga-ede wraps after a printed hyphen.
            text=text.replace('ga- ede','ga-ede')
        counts[source_id]=counts.get(source_id,0)+1
        clipped_left=first==12
        clipped_right=last==23
        entries.append({'id':source_id+f"-caption-{counts[source_id]:02d}",'source_id':source_id,'source_url':expected['source_url'],
                        'source_audio_sha256':expected['sha256'],'text_lines':lines,'text':text,
                        'observed_first_frame_seconds':first,'observed_last_frame_seconds':last,'sampling_interval_seconds':1,
                        'timing_type':'Approximate display observations, not word-aligned audio boundaries',
                        'potentially_clipped_at_sample_start':clipped_left,'potentially_clipped_at_sample_end':clipped_right,
                        'display_bounded_candidate':not (clipped_left or clipped_right),
                        'image_transcription_reviewer':'Codex AI visual review','image_transcription_confidence':item['confidence'],
                        'uncertainties':item.get('uncertainties',[]),'caption_image_cloud_path':cloud_image,'caption_image_sha256':image_hash,
                        'source_subtitle_speech_review':'User reports BBC subtitle/speech match; per-source/timestamp coverage was not separately itemised',
                        'training_ready':False,'remaining_work':'Exact complete-utterance audio segmentation; resolve any marked transcription ambiguity; preserve source folds.'})
    output={'id':'bbc-user-reviewed-caption-text-v1','user_caption_review':review,'method':decisions['method'],
            'entries':entries,'caption_entries':len(entries),'sources':len(counts),
            'display_bounded_candidates':sum(e['display_bounded_candidate'] for e in entries),
            'unambiguous_display_bounded_candidates':sum(e['display_bounded_candidate'] and not e['uncertainties'] for e in entries),
            'aligned_training_segments':0,'trained':False,'optimizer_updates':0,
            'caption_capture_body_bytes':repairs['combined_caption_stage_body_bytes'],
            'caption_capture_requests':repairs['combined_caption_stage_requests'],
            'note':'The source captions passed the user’s reported match review. ASR drafts were not promoted. Captured text is an AI visual derivative; observed frame times are not exact speech alignment.'}
    (FOLDER/'caption-text.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (FOLDER/'caption-text.txt').write_text('\n\n'.join(f"{e['id']} · observed {e['observed_first_frame_seconds']}–{e['observed_last_frame_seconds']}s\n{e['text']}" for e in entries)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':
    result=assemble()
    print(json.dumps({k:result[k] for k in ('caption_entries','sources','display_bounded_candidates','unambiguous_display_bounded_candidates','aligned_training_segments','trained')},ensure_ascii=False))
