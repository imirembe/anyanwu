"""Word timing for user-reviewed complete responses; no cutting or transcript edits."""
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import socket
import time
import wave

from backend.voice_research.alignment_rules import alignment_text, caption_token_ranges


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def phrase_candidates(words, quiet, minimum_gap=.18):
    """Group by tentative acoustic gaps; never claim linguistic phrase boundaries."""
    if not words:
        return []
    groups, current = [], []
    for index, word in enumerate(words):
        current.append(word)
        following = words[index+1] if index+1 < len(words) else None
        gap = following['start_seconds']-word['end_seconds'] if following else None
        if following is None or gap >= minimum_gap:
            groups.append({'alignment_view_text': ' '.join(w['word'] for w in current),
                           'start_seconds': current[0]['start_seconds'], 'end_seconds': current[-1]['end_seconds'],
                           'word_count': len(current), 'gap_after_seconds': round(gap, 3) if gap is not None else None,
                           'quiet_interval_overlap': any(q['end_seconds'] > word['end_seconds'] and
                                                         q['start_seconds'] < following['start_seconds'] for q in quiet) if following else None,
                           'verified_phrase_boundary': False})
            current = []
    return groups


def main():
    root = Path('/content/anyanwu-conversational-alignment-v1')
    if (platform.system() != 'Linux' or os.environ.get('ANYANWU_CLOUD_ALIGNMENT') != '1'
            or os.environ.get('ANYANWU_ALIGNMENT_WORK') != str(root)):
        raise RuntimeError('Configured hosted offline alignment worker only')
    plan = json.loads((root/'plan.json').read_text())
    if (root/'alignment-summary.json').exists():
        raise RuntimeError('Existing alignment preserved; no repeat')
    reviewed = Path(plan['submitted_snapshot'])
    if reviewed.stat().st_size > 65536 or file_sha(reviewed) != plan['submitted_review_sha256']:
        raise RuntimeError('User-reviewed transcript snapshot changed')
    submitted = json.loads(reviewed.read_text())['records']
    source = Path('/content/anyanwu-conversational-transcripts-v1')
    model_paths = list((source/'cache/fairseq2').rglob('omniASR-CTC-300M-v2.pt'))
    if len(model_paths) != 1 or file_sha(model_paths[0]) != plan['asr_model_sha256']:
        raise RuntimeError('Cached ASR model missing/changed; no downloads allowed')

    def deny_network(*args, **kwargs):
        raise RuntimeError('Offline timing only; downloads disabled')
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    import torch
    import torchaudio.functional as AF
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    pipeline = ASRInferencePipeline(model_card='omniASR_CTC_300M_v2', device='cpu', dtype=torch.float32)
    encode = pipeline.tokenizer.create_raw_encoder(device=torch.device('cpu'))
    decode = pipeline.tokenizer.create_decoder(skip_special_tokens=True)
    vocab = pipeline.tokenizer.vocab_info
    cases = []
    started = time.monotonic()
    for entry in plan['cases']:
        record = submitted[entry['id']]
        if (record['words_match'] != 'yes' or record['clean_complete_speech'] != 'yes'
                or hashlib.sha256(record['transcript'].encode()).hexdigest() != entry['transcript_sha256']):
            raise RuntimeError('Expected unchanged user-accepted transcript')
        audio = source/(entry['id']+'.analysis.wav')
        if audio.is_symlink() or audio.stat().st_size > 2*1024**2 or file_sha(audio) != entry['analysis_sha256']:
            raise RuntimeError('Whole analysis waveform integrity failure')
        with wave.open(str(audio)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise RuntimeError('Analysis waveform format differs')
            seconds = wav.getnframes()/16000
        case = {'id': entry['id'], 'source_transcript': record['transcript'],
                'transcript_sha256': entry['transcript_sha256'], 'audio_sha256': record['audio_sha256'],
                'seconds': seconds, 'transcript_review': 'User marked exact and clean; spoken English preserved',
                'training_ready': False}
        try:
            text = alignment_text(record['transcript'])
            tokens = encode(text).to(torch.int64).reshape(-1)
            if not 1 <= len(tokens) <= 1500 or (tokens == 0).any().item() or (tokens == vocab.unk_idx).any().item():
                raise ValueError('Unsupported alignment token; no diacritic or word substitution fallback')
            ranges = caption_token_ranges(tokens.tolist(), text.split(),
                                           lambda value: decode(torch.tensor(value, dtype=torch.int64)).strip())
            captured = []

            def hook(module, arguments, output):
                logits, layout = output
                captured.append(logits[:, :int(layout.seq_lens[0])].detach().float().cpu().log_softmax(-1))

            handle = pipeline.model.register_forward_hook(hook)
            try:
                with torch.inference_mode():
                    pipeline.transcribe([str(audio)], batch_size=1)
            finally:
                handle.remove()
            if len(captured) != 1 or captured[0].shape[0] != 1:
                raise ValueError('Expected one emission for whole response')
            emission = captured.pop()
            ratio = seconds/emission.shape[1]
            with torch.inference_mode():
                paths, scores = AF.forced_align(emission.contiguous(), tokens.unsqueeze(0), blank=0)
                spans = AF.merge_tokens(paths[0], scores[0].exp(), blank=0)
            numeric = [(int(s.token), int(s.start), int(s.end), float(s.score)) for s in spans]
            if [s[0] for s in numeric] != tokens.tolist():
                raise ValueError('Forced alignment did not cover every user transcript token in order')
            previous = 0
            for _, start, end, score in numeric:
                if not previous <= start < end <= emission.shape[1] or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Invalid alignment path bounds')
                previous = end
            words = []
            for word, (a, b) in zip(text.split(), ranges):
                spans_word = numeric[a:b]
                words.append({'word': word, 'start_seconds': round(spans_word[0][1]*ratio, 4),
                              'end_seconds': round(spans_word[-1][2]*ratio, 4),
                              'mean_token_probability': round(sum(t[3] for t in spans_word)/len(spans_word), 6)})
            mean = sum(s[3] for s in numeric)/len(numeric)
            warnings = ['low_acoustic_alignment_score'] if mean < .1 else []
            if numeric[0][1] <= 1 or numeric[-1][2] >= emission.shape[1]-1:
                warnings.append('token_alignment_touches_recording_edge')
            groups = phrase_candidates(words, entry['quiet_candidates'])
            details = {**case, 'alignment_view': text,
                       'normalization': 'NFC/casefold/punctuation-only alignment view; original transcript preserved above',
                       'words': words, 'machine_phrase_candidates': groups, 'mean_token_probability': mean,
                       'warnings': warnings, 'linguistic_phrase_review': 'pending'}
            filename = entry['id']+'-timing.json'
            raw = (json.dumps(details, ensure_ascii=False, indent=2)+'\n').encode()
            (root/filename).write_bytes(raw)
            case.update(status='machine_timing_created', timing_file=filename, timing_sha256=hashlib.sha256(raw).hexdigest(),
                        words=len(words), phrase_candidates=len(groups), mean_token_probability=round(mean, 6),
                        warnings=warnings, phrase_preview=groups[:12])
            del emission
        except (ValueError, RuntimeError) as error:
            case.update(status='timing_held', error=str(error)[:300])
        cases.append(case)
        (root/'progress.json').write_text(json.dumps({'phase': 'conversational_asr_aligning', 'completed': len(cases), 'total': 2}))
    report = {'phase': 'conversational_asr_alignment_complete', 'experiment_id': plan['id'], 'cases': cases,
              'elapsed_seconds': round(time.monotonic()-started, 2), 'candidate_cuts': 0,
              'network_requests': 0, 'model_downloads': 0, 'optimizer_updates': 0,
              'source_text_or_audio_changed': False, 'training_ready': False,
              'interpretation': 'Target-constrained machine timings; not proof of text correctness, linguistic phrasing, emphasis or lexical tone. Whole source recordings retained.'}
    (root/'alignment-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':main()
