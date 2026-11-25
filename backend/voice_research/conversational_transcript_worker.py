"""Whole-response ASR drafts and energy-based pause candidates; cloud CPU only."""
from pathlib import Path
import array
import hashlib
import json
import math
import os
import platform
import sys
import time
import wave


def pause_candidates(path):
    """Quiet intervals are candidates, never linguistic/prosodic ground truth."""
    with wave.open(str(path)) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise RuntimeError('Expected whole mono16k analysis copy')
        raw = wav.readframes(wav.getnframes())
    pcm = array.array('h', raw)
    if sys.byteorder != 'little':
        pcm.byteswap()
    levels = []
    for start in range(0, len(pcm), 320):
        frame = pcm[start:start+320]
        rms = math.sqrt(sum(x*x for x in frame)/max(1, len(frame)))/32768
        levels.append(20*math.log10(max(rms, 1e-8)))
    threshold = max(-50, min(-32, sorted(levels)[int(.1*len(levels))] + 6))
    gaps, start = [], None
    for index, level in enumerate(levels + [0]):
        if level < threshold and start is None:
            start = index
        if level >= threshold and start is not None:
            if index - start >= 9:
                gaps.append({'start_seconds': round(start*.02, 3),
                             'end_seconds': round(min(index*.02, len(pcm)/16000), 3),
                             'position': 'leading' if start == 0 else 'trailing' if index == len(levels) else 'internal',
                             'verified': False})
            start = None
    return {'method': '20ms RMS quiet intervals >=180ms, adaptive threshold',
            'threshold_dbfs': round(threshold, 2), 'candidates': gaps,
            'limitations': 'Energy gaps may include soft speech/noise; not verified phrase boundaries, lexical tone or emphasis.'}


def main():
    if platform.system() != 'Linux' or os.environ.get('ANYANWU_CONVERSATIONAL_ASR') != '1':
        raise RuntimeError('Configured hosted CPU worker only')
    root = Path('/content/anyanwu-conversational-transcripts-v1')
    plan = json.loads((root/'plan.json').read_text())
    if (root/'draft-transcripts.json').exists():
        raise RuntimeError('Existing drafts preserved; no automatic repeat')
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pipeline = ASRInferencePipeline(model_card='omniASR_CTC_300M_v2', device='cpu', dtype=torch.float32)
    models = list((root/'cache/fairseq2').rglob('omniASR-CTC-300M-v2.pt'))
    if len(models) != 1 or models[0].stat().st_size != plan['asr_model_bytes']:
        raise RuntimeError('Expected one bounded ASR model')
    digest = hashlib.sha256()
    with models[0].open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            digest.update(chunk)
    if digest.hexdigest() != plan['asr_model_sha256']:
        raise RuntimeError('ASR model differs from prior observed research snapshot')
    started = time.monotonic()
    records = []
    for selected in plan['clips']:
        wav = root/(selected['id']+'.analysis.wav')
        if (wav.is_symlink() or wav.stat().st_size > 2*1024**2
                or hashlib.sha256(wav.read_bytes()).hexdigest() != selected['analysis_sha256']):
            raise RuntimeError('Analysis input integrity failure')
        with torch.inference_mode():
            output = pipeline.transcribe([str(wav)], batch_size=1)
        if len(output) != 1 or not isinstance(output[0], str) or not 0 < len(output[0].strip()) <= 6000:
            raise RuntimeError('Invalid bounded ASR draft')
        records.append({**selected, 'draft_transcript': output[0].strip(),
                        'draft_sha256': hashlib.sha256(output[0].strip().encode()).hexdigest(),
                        'pause_analysis': pause_candidates(wav), 'review_status': 'unreviewed',
                        'training_ready': False, 'transcription_method': 'omniASR_CTC_300M_v2 CPU float32'})
        (root/'progress.json').write_text(json.dumps({'phase': 'conversational_asr_transcribing',
                                                     'completed': len(records), 'total': 2}))
    from importlib.metadata import version
    report = {'phase': 'conversational_asr_drafts_complete', 'plan_id': plan['id'],
              'records': records, 'transcribed_clips': len(records),
              'source_speaker_id': 'IGB_005', 'input_seconds': sum(c['seconds'] for c in plan['clips']),
              'inference_seconds': round(time.monotonic()-started, 2),
              'packages': {n: version(n) for n in ('torch', 'torchaudio', 'fairseq2', 'omnilingual-asr', 'numpy', 'pyarrow')},
              'asr_model_sha256': digest.hexdigest(),
              'model_hash_provenance': 'Matches prior observed download; not an independently published checksum',
              'training_ready': False, 'optimizer_updates': 0,
              'limitations': 'Unverified source-derived machine drafts; no elicitation prompt supplied as transcript. Pause candidates require listening; no human emphasis/tone annotation inferred.'}
    (root/'draft-transcripts.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
