"""Measure whole-response TTS token lengths on CPU; no language-model training."""
from pathlib import Path
import hashlib
import json
import os
import platform
import urllib.request
import wave


def checksum(path, git=False):
    digest = hashlib.sha1() if git else hashlib.sha256()
    if git: digest.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path('/content/anyanwu-conversational-capacity-v1')
    if platform.system() != 'Linux' or os.environ.get('ANYANWU_CONVERSATIONAL_CAPACITY') != '1':
        raise RuntimeError('Configured hosted CPU worker only')
    plan = json.loads((root/'plan.json').read_text())
    if (root/'capacity-result.json').exists(): raise RuntimeError('Existing result preserved; no repeat')
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import Qwen2Tokenizer
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    snapshot = Path(plan['submitted_snapshot'])
    if snapshot.stat().st_size > 65536 or checksum(snapshot) != plan['submitted_review_sha256']:
        raise RuntimeError('User transcript snapshot differs')
    records = json.loads(snapshot.read_text())['records']
    files = {}
    for entry in plan['model']['files']:
        (root/'progress.json').write_text(json.dumps({'phase':'conversational_asr_capacity_asset','file':entry['rfilename']}))
        path = Path(hf_hub_download(plan['model']['repo'], entry['rfilename'], revision=plan['model']['revision'],
                                    token=False, cache_dir=str(root/'cache/hf/hub')))
        if path.stat().st_size != entry['size'] or checksum(path, not bool(entry.get('sha256'))) != (entry.get('sha256') or entry['git_blob_sha1']):
            raise RuntimeError('Pinned tokenizer asset integrity failure')
        files[entry['rfilename']] = path
    config = json.loads(files['CosyVoice-BlankEN/config.json'].read_text())
    if config.get('model_type') != 'qwen2' or config.get('max_position_embeddings') != 32768:
        raise RuntimeError('Pinned model capacity changed')
    text_dir = root/'text-tokenizer'; text_dir.mkdir(exist_ok=True)
    for name in ('vocab.json','merges.txt','tokenizer_config.json'):
        target = text_dir/name
        if not target.exists(): target.symlink_to(files['CosyVoice-BlankEN/'+name])
    tokenizer = Qwen2Tokenizer.from_pretrained(str(text_dir), local_files_only=True)
    prefix = tokenizer.encode('You are a helpful assistant.', add_special_tokens=False)+[151646]
    mel = root/'mel_filters.npz'
    if not mel.exists():
        with urllib.request.urlopen(plan['mel_filter']['url'], timeout=30) as response:
            raw = response.read(plan['mel_filter']['size']+1)
        if len(raw) != plan['mel_filter']['size']: raise RuntimeError('Mel filter size mismatch')
        mel.write_bytes(raw)
    if checksum(mel, True) != plan['mel_filter']['git_blob_sha1']: raise RuntimeError('Mel filter checksum differs')
    filters = torch.from_numpy(np.load(mel)['mel_128'])
    options = ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1; options.enable_cpu_mem_arena=False
    session = ort.InferenceSession(str(files['speech_tokenizer_v3.onnx']), options, providers=['CPUExecutionProvider'])
    cases = []
    for entry in plan['cases']:
        row = records[entry['id']]
        audio = Path('/content/anyanwu-conversational-transcripts-v1')/(entry['id']+'.analysis.wav')
        if (row['words_match'] != 'yes' or row['clean_complete_speech'] != 'yes'
                or hashlib.sha256(row['transcript'].encode()).hexdigest() != entry['transcript_sha256']
                or audio.is_symlink() or audio.stat().st_size > 2*1024**2 or checksum(audio) != entry['analysis_sha256']):
            raise RuntimeError('Reviewed whole-response text/audio integrity failure')
        with wave.open(str(audio)) as reader:
            if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate()) != (1,2,16000): raise RuntimeError('Whole analysis PCM differs')
            samples = np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(np.float32)/32768
        if not 20 <= len(samples)/16000 <= 30: raise RuntimeError('Whole response length changed')
        tensor = torch.from_numpy(samples.copy())[None]
        spectrum = torch.stft(tensor,400,160,window=torch.hann_window(400),return_complex=True)[...,:-1].abs()**2
        features = (filters@spectrum).clamp(min=1e-10).log10()
        features = ((torch.maximum(features,features.max()-8)+4)/4).numpy()
        inputs = session.get_inputs()
        speech = session.run(None,{inputs[0].name:features,inputs[1].name:np.array([features.shape[2]],dtype=np.int32)})[0].flatten().tolist()
        text = tokenizer.encode(row['transcript'],add_special_tokens=False)
        if tokenizer.decode(text) != row['transcript']: raise RuntimeError('Tokenizer changed exact corrected text')
        packed = len(prefix)+len(text)+len(speech)+3
        if not 1 <= len(speech) <= 800 or packed > 1280 or packed > config['max_position_embeddings']:
            raise RuntimeError('Declared whole-response capacity exceeded; no truncation or automatic cuts')
        cases.append({**entry,'source_group':'silencio:IGB_005','text':row['transcript'],'text_ids':text,'speech_ids':speech,
                      'text_tokens':len(text),'speech_tokens':len(speech),'packed_positions':packed,'prefix_tokens':len(prefix),
                      'seconds':len(samples)/16000,'speech_split':'train','source_audio_cut':False,'raw_text_changed':False})
        (root/'progress.json').write_text(json.dumps({'phase':'conversational_asr_capacity_encoded','completed':len(cases),'total':2}))
    targets = root/'speech-targets.json'; targets.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    from importlib.metadata import version
    result = {'phase':'conversational_asr_capacity_complete','experiment':plan['id'],'full_responses_encoded':2,
              'model_max_positions':config['max_position_embeddings'],'declared_position_cap':1280,'declared_speech_token_cap':800,
              'model_files_bytes':sum(f['size'] for f in plan['model']['files']),
              'cases':[{k:c[k] for k in ('id','text_tokens','speech_tokens','packed_positions','prefix_tokens','seconds','analysis_sha256','transcript_sha256')} for c in cases],
              'speech_targets_sha256':checksum(targets),'source_transcript_or_audio_changed':False,'optimizer_updates':0,
              'language_model_weights_downloaded':False,'laptop_media_or_models_bytes':0,
              'package_versions':{n:version(n) for n in ('torch','numpy','onnxruntime','transformers','huggingface_hub')}}
    (root/'capacity-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
