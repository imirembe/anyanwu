"""Constant-gain listening copies; preserve raw speech, timing and pitch."""
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import wave


def audio_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def choose_loudness_target(measurements,requested=-20.0,headroom=-1.5,max_gain=24.0):
    if not measurements:raise RuntimeError('No loudness measurements')
    for row in measurements:
        if not all(math.isfinite(row[k]) for k in ('integrated_lufs','true_peak_db')):
            raise RuntimeError('Silent/non-finite loudness measurement')
    target=min([requested]+[r['integrated_lufs']+min(max_gain,headroom-r['true_peak_db']) for r in measurements])
    if target < -35:raise RuntimeError('Common listening level would be too quiet; inspect inputs')
    return target,[target-r['integrated_lufs'] for r in measurements]


def measure_loudness(path):
    command=['ffmpeg','-nostdin','-hide_banner','-loglevel','info','-protocol_whitelist','file,pipe','-i',str(path),
             '-af','loudnorm=I=-20:TP=-1.5:LRA=11:print_format=json','-f','null','-']
    result=subprocess.run(command,capture_output=True,text=True,timeout=45)
    if result.returncode or len(result.stderr)>65536:raise RuntimeError('Bounded loudness measurement failed')
    matches=re.findall(r'\{\s*"input_i"[\s\S]*?\}',result.stderr)
    if len(matches)!=1:raise RuntimeError('Expected one loudness measurement')
    value=json.loads(matches[0])
    measured={'integrated_lufs':float(value['input_i']),'true_peak_db':float(value['input_tp'])}
    if not all(math.isfinite(v) for v in measured.values()):raise RuntimeError('Silent/non-finite measured audio')
    return measured


def pcm_frames(path):
    with wave.open(str(path),'rb') as reader:
        if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate())!=(1,2,24000):
            raise RuntimeError('Expected original mono 24kHz PCM')
        return reader.getnframes()


def normalize_listening_copies(root,plan,publish):
    root=Path(root);result_path=root/'generation-result.json';original=result_path.read_bytes();result=json.loads(original)
    expected=plan['outputs']
    if expected!=12 or result['completed_wavs']!=expected or len(result['clips'])!=expected or result['optimizer_updates_this_stage']!=0 or not result['source_checkpoints_unchanged']:
        raise RuntimeError('Twelve immutable raw clips required before loudness matching')
    parameters=plan['listening_loudness']
    finished=root/'listening-result.json'
    if finished.exists():
        receipt=json.loads(finished.read_text())
        if receipt['parameters']!=parameters or receipt['clips']!=expected:raise RuntimeError('Different listening-copy parameters preserved')
        for clip in result['clips']:
            if audio_sha(root/clip['audio_filename'])!=clip['audio_sha256'] or audio_sha(root/clip['listening_audio_filename'])!=clip['listening_audio_sha256']:
                raise RuntimeError('Existing raw/listening copy changed')
        return receipt
    if (root/'listening-attempt.json').exists():raise RuntimeError('Prior loudness attempt preserved; inspect before repair')
    (root/'listening-attempt.json').write_text('{"phase":"started","new_synthesis":false}\n')
    started=time.monotonic();initial_free=shutil.disk_usage(root).free
    def guard():
        free=shutil.disk_usage(root).free
        if time.monotonic()-started>300 or initial_free-free>64*1024**2 or free<5*1024**3:raise RuntimeError('Listening-copy time/disk guard')
    try:
        measurements=[];frame_counts=[]
        for i,clip in enumerate(result['clips']):
            guard();name=clip['audio_filename']
            if Path(name).name!=name or not re.fullmatch(r'[A-Za-z0-9_-]+\.wav',name):raise RuntimeError('Unsafe raw clip path')
            path=root/name
            if path.stat().st_size>2*1024**2 or audio_sha(path)!=clip['audio_sha256']:raise RuntimeError('Raw audio integrity failure')
            frames=pcm_frames(path)
            if not 2400<=frames<=20*24000:raise RuntimeError('Original duration guard')
            frame_counts.append(frames);measurements.append(measure_loudness(path))
            publish({'phase':'reference_validation_measuring_loudness','measured':i+1,'total':expected,'new_synthesis':False})
        target,gains=choose_loudness_target(measurements,parameters['requested_integrated_lufs'],parameters['true_peak_headroom_db'],parameters['maximum_gain_db'])
        matched=[]
        for i,(clip,before,gain,frames) in enumerate(zip(result['clips'],measurements,gains,frame_counts)):
            guard();raw=root/clip['audio_filename'];name='listen-'+clip['audio_filename'];destination=root/name;stage=root/(name+'.staging.wav')
            if destination.exists() or stage.exists():raise RuntimeError('Different/partial listening copy preserved')
            command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-protocol_whitelist','file,pipe','-i',str(raw),
                     '-af',f'volume={gain:.8f}dB','-ac','1','-ar','24000','-c:a','pcm_s16le',str(stage)]
            child=subprocess.run(command,capture_output=True,timeout=45)
            if child.returncode or stage.stat().st_size>2*1024**2 or pcm_frames(stage)!=frames:raise RuntimeError('Constant-gain copy failed timing/size check')
            after=measure_loudness(stage)
            if abs(after['integrated_lufs']-target)>.5 or after['true_peak_db']>-1.0:
                raise RuntimeError('Listening copy did not meet common loudness/headroom tolerance')
            stage.replace(destination)
            if audio_sha(raw)!=clip['audio_sha256']:raise RuntimeError('Raw WAV changed during matching')
            clip.update(listening_audio_filename=name,listening_audio_sha256=audio_sha(destination),listening_bytes=destination.stat().st_size,
                        listening_gain_db=round(gain,8),raw_loudness=before,listening_loudness=after,listening_frames=frames)
            matched.append(after['integrated_lufs'])
            publish({'phase':'reference_validation_matching_loudness','completed':i+1,'total':expected,'new_synthesis':False})
        if max(matched)-min(matched)>1.0:raise RuntimeError('Listening loudness spread exceeds one LU')
        raw_backup=root/'generation-result.raw.json'
        if raw_backup.exists() and raw_backup.read_bytes()!=original:raise RuntimeError('Different original generation receipt preserved')
        if not raw_backup.exists():raw_backup.write_bytes(original)
        receipt={'phase':'reference_validation_listening_copies_verified','clips':expected,'parameters':parameters,
                 'common_target_lufs':round(target,6),'maximum_spread_lu':round(max(matched)-min(matched),6),
                 'raw_wavs_unchanged':True,'frame_counts_unchanged':True,'constant_gain_only':True,'new_synthesis':False,
                 'listening_audio_bytes':sum(c['listening_bytes'] for c in result['clips'])}
        result['listening_copy_receipt']=receipt
        temporary=root/'generation-result.normalized.tmp';temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temporary.replace(result_path)
        finished.write_text(json.dumps(receipt,indent=2)+'\n');(root/'listening-attempt.json').write_text('{"phase":"complete","new_synthesis":false}\n')
        return receipt
    except Exception as error:
        (root/'listening-attempt.json').write_text(json.dumps({'phase':'stopped','error_type':type(error).__name__,'error':str(error)[:300],'new_synthesis':False},indent=2)+'\n')
        raise
