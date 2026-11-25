"""Verify existing research inputs for a fixed-v1, two-reference diagnostic."""
from pathlib import Path
import hashlib
import json
import shutil


def checked_bytes(path, expected, cap):
    path = Path(path)
    if path.is_symlink() or not 0 < path.stat().st_size <= cap:
        raise RuntimeError('Reference input path/size guard')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Reference input checksum mismatch')
    return raw


def context_positions(prefix, prompt_text, prompt_speech, target_text, maximum, cap):
    """Actual generator embedding layout plus the entire reserved output budget."""
    positions = 3 + prefix + prompt_text + target_text + prompt_speech + maximum
    if min(prefix, prompt_text, prompt_speech, target_text, maximum) < 1 or positions > cap:
        raise RuntimeError('Full reference exceeds context capacity; no truncation')
    return positions


def match_reference_boundary(speech_tokens, mel_frames):
    """Publisher's 2:1 feature alignment; only allow tiny frame-edge rounding."""
    n = min(speech_tokens, mel_frames // 2)
    if n < 1 or speech_tokens - n > 2 or mel_frames - 2 * n > 4:
        raise RuntimeError('Reference boundary mismatch exceeds rounding allowance')
    return n


def restore_reference_inputs(plan, *, root, warm):
    # Imported by the local module; the notebook embeds this verified helper.
    from backend.voice_research.expanded_comparison_cloud import read_comparison_archive
    root = Path(root); warm = Path(warm); assets = root / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    prompt = 'ibo_616'
    selected = ['plan.json', 'training-result.json', 'adapter.safetensors',
                'data-result.json', 'speech-targets.json', f'data/{prompt}.wav', f'data/{prompt}.mp3']
    previous = read_comparison_archive(plan['archives']['previous'], selected)
    if (hashlib.sha256(previous['adapter.safetensors']).hexdigest() != plan['checkpoint_sha256']
            or json.loads(previous['training-result.json'])['optimizer_updates'] != 60):
        raise RuntimeError('Preserved v1 checkpoint differs')
    old_plan = json.loads(previous['plan.json'])
    reserved = {t['id'] for t in plan['texts']}
    if len(reserved) != 2 or reserved & {c['id'] for c in old_plan['clips']}:
        raise RuntimeError('Reserved text overlaps v1 adapter data')
    if reserved & set(plan['previously_reviewed_or_trained_ids']):
        raise RuntimeError('Reserved text overlaps previous listening/training')
    reference = plan['conditions'][1]
    review_raw = checked_bytes(plan['submitted_snapshot'], reference['submitted_transcript_snapshot_sha256'], 65536)
    target_raw = checked_bytes(plan['encoded_targets_path'], reference['encoded_targets_sha256'], 256*1024)
    records = json.loads(review_raw)['records']
    target = next(c for c in json.loads(target_raw) if c['id'] == reference['reference_id'])
    reviewed = records[reference['reference_id']]
    if (reviewed['words_match'] != 'yes' or reviewed['clean_complete_speech'] != 'yes'
            or reviewed['transcript'] != target['text']
            or hashlib.sha256(target['text'].encode()).hexdigest() != reference['transcript_sha256']
            or len(target['speech_ids']) != 611 or len(target['text_ids']) != 123):
        raise RuntimeError('Complete conversational targets differ from user review')
    audio = checked_bytes(plan['reference_audio_path'], reference['listening_audio_sha256'], 2*1024**2)
    source = next(c for c in json.loads(previous['data-result.json'])['clips'] if c['id'] == prompt)
    old_target = next(c for c in json.loads(previous['speech-targets.json']) if c['id'] == prompt)
    if (source['speaker_id'], source['gender'], source['adapter_split']) != ('5', 'Female', 'train'):
        raise RuntimeError('Original read reference differs')
    for suffix, key in (('.wav', 'audio_sha256'), ('.mp3', 'mp3_sha256')):
        if hashlib.sha256(previous[f'data/{prompt}{suffix}']).hexdigest() != source[key]:
            raise RuntimeError('Read reference checksum differs')
    conversational = {'id': reference['reference_id'], 'speaker_id': reference['speaker_id'],
        'gender': 'Female', 'source_group': 'silencio:IGB_005',
        'audio_sha256': reference['analysis_sha256'], 'wave24_sha256': reference['listening_audio_sha256'],
        'cloud_audio_path': str(assets/'data/conversation-ref-03.analysis.wav'),
        'cloud_audio24_path': str(assets/'data/conversation-ref-03.wav')}
    source = {**source, 'cloud_audio_path': str(assets/f'data/{prompt}.wav')}
    outputs = {'plan.json': previous['plan.json'], 'previous-adapter.safetensors': previous['adapter.safetensors'],
        'adapter.safetensors': previous['adapter.safetensors'],
        'speech-targets.json': json.dumps([old_target, target], ensure_ascii=False).encode(),
        'data-result.json': json.dumps({'clips': [source, conversational]}, ensure_ascii=False).encode(),
        f'data/{prompt}.wav': previous[f'data/{prompt}.wav'], f'data/{prompt}.mp3': previous[f'data/{prompt}.mp3'],
        'data/conversation-ref-03.wav': audio, 'submitted-transcripts.json': review_raw}
    for name, raw in outputs.items():
        path = assets/name; path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise RuntimeError('Different diagnostic input preserved')
        if not path.exists(): path.write_bytes(raw)
    cache = assets/'cache/hf/hub'; cache.parent.mkdir(parents=True, exist_ok=True)
    if (warm/'cache/hf/hub').is_dir() and not cache.exists():
        cache.symlink_to(warm/'cache/hf/hub', target_is_directory=True)
    return {'phase': 'reference_conditioning_inputs_verified', 'checkpoint_sha256': plan['checkpoint_sha256'],
        'reference_ids': [prompt, reference['reference_id']], 'full_conversation_speech_tokens': 611,
        'warm_cache': cache.is_symlink(), 'optimizer_updates_this_stage': 0}
