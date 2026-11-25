"""Cloud-only, five-update CPU speech LM LoRA experiment; no full corpus download."""
from pathlib import Path
import gc
import hashlib
import json
import os
import platform
import sys
import time
import wave

WORK = Path('/content/anyanwu-flash-lora-pilot-v1')


def guard():
    if platform.system() != 'Linux' or not Path('/content').is_dir() or os.environ.get('ANYANWU_FLASH_PILOT') != '1':
        raise RuntimeError('Hosted cloud worker only; no laptop install/model access')


def digest_file(path, algorithm='sha256', git=False):
    digest = hashlib.new(algorithm)
    if git:
        digest.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    guard()
    import numpy as np
    import torch
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    from transformers import Qwen2Config, Qwen2Model, Qwen2Tokenizer
    from flash_adapter import make_pilot_model, load_qwen_backbone
    import urllib.request

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    # Exercise adapter gradients on a tiny fixture before any large download.
    # These fixture operations are never counted as pretrained speech-model updates.
    tiny_config = Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                              _attn_implementation='eager')
    tiny_state = Qwen2Model(tiny_config).state_dict()
    tiny = load_qwen_backbone(tiny_config, tiny_state)
    fixture = make_pilot_model(tiny, torch.randn(210,32), torch.randn(210,32),
                               torch.randn(5,32), torch.randn(6,32), torch.randn(32), [8,9], ['fixture'])
    frozen_before = tiny.embed_tokens.weight.detach().clone()
    fixture_loss = fixture([10,11], [1,2,3], 0)
    fixture_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in fixture.named_parameters() if n.endswith('.B'))
    assert torch.equal(frozen_before,tiny.embed_tokens.weight) and tiny.embed_tokens.weight.grad is None
    del fixture, tiny, tiny_state, frozen_before, fixture_loss
    gc.collect()
    torch.manual_seed(7)
    plan = json.loads((WORK / 'plan.json').read_text())
    if plan['updates'] != 5 or len(plan['cases']) != 3 or plan['batch_size'] != 1:
        raise RuntimeError('Unexpected registered pilot size')
    started = time.monotonic()
    report = {'phase': 'starting', 'optimizer_updates': 0, 'trained': False,
              'experiment': plan['id'], 'quality_evaluated': False, 'speech_generated': False,
              'data_status': 'Provisional machine cuts; user-confirmed BBC source subtitles',
              'source_groups': [c['source_group'] for c in plan['cases']],
              'base_revision': plan['models'][0]['revision'], 'losses': []}

    def progress(phase, **extra):
        report.update(phase=phase, elapsed_seconds=round(time.monotonic()-started, 2), **extra)
        temporary = WORK / 'progress.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False))
        temporary.replace(WORK / 'progress.json')

    try:
        files = {}
        selected_bytes = sum(item['size'] for model in plan['models'] for item in model['files'])
        if selected_bytes > 4 * 1024**3:
            raise RuntimeError('Selected model bytes exceed four GiB')
        for model in plan['models']:
            for item in model['files']:
                progress('downloading_verified_model_file', current_file=item['rfilename'])
                filename = Path(hf_hub_download(repo_id=model['repo'], revision=model['revision'],
                                               filename=item['rfilename'], token=False,
                                               cache_dir=str(WORK/'cache/hf/hub')))
                if filename.stat().st_size != item['size']:
                    raise RuntimeError('Pinned model size mismatch: '+item['rfilename'])
                expected = item.get('sha256') or item['git_blob_sha1']
                actual = digest_file(filename, 'sha256' if item.get('sha256') else 'sha1', not bool(item.get('sha256')))
                if actual != expected:
                    raise RuntimeError('Pinned model checksum mismatch: '+item['rfilename'])
                files[item['rfilename']] = filename
        report['selected_model_bytes'] = selected_bytes
        progress('verifying_audio_and_tokenizer')
        melmeta = plan['mel_filter']
        melpath = WORK / 'mel_filters.npz'
        if not melpath.exists():
            with urllib.request.urlopen(melmeta['url'], timeout=30) as response:
                data = response.read(melmeta['size']+1)
            if len(data) != melmeta['size']:
                raise RuntimeError('Mel filter size mismatch')
            melpath.write_bytes(data)
        if digest_file(melpath, 'sha1', True) != melmeta['git_blob_sha1']:
            raise RuntimeError('Mel filter checksum mismatch')
        tokenizer_dir = WORK / 'text-tokenizer'
        tokenizer_dir.mkdir(exist_ok=True)
        for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json'):
            target = tokenizer_dir/name
            source = files['CosyVoice-BlankEN/'+name]
            if target.is_symlink() and target.resolve() != source.resolve():
                raise RuntimeError('Unexpected tokenizer link')
            if not target.exists():
                target.symlink_to(source)
        tokenizer = Qwen2Tokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        prefix = tokenizer.encode('You are a helpful assistant.', add_special_tokens=False) + [151646]
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 2
        settings.inter_op_num_threads = 1
        settings.enable_cpu_mem_arena = False
        session = ort.InferenceSession(str(files['speech_tokenizer_v3.onnx']), settings,
                                      providers=['CPUExecutionProvider'])
        mel_filters = torch.from_numpy(np.load(melpath)['mel_128'])
        cases = []
        for case in plan['cases']:
            progress('encoding_real_audio_targets', current_case=case['id'])
            audio = Path(case['cloud_audio_path'])
            if not audio.is_relative_to(Path('/content/anyanwu-caption-alignment-v1')) and not audio.is_relative_to(Path('/content/anyanwu-caption-alignment-v2-context')):
                raise RuntimeError('Audio path is outside the reviewed cloud cuts')
            if audio.stat().st_size != case['candidate_audio_bytes'] or digest_file(audio) != case['candidate_audio_sha256']:
                raise RuntimeError('Candidate audio checksum/size mismatch')
            with wave.open(str(audio),'rb') as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1,2,16000):
                    raise RuntimeError('Expected original mono 16k PCM16; no resampling')
                sample_count = wav.getnframes()
                if not 0 < sample_count <= 8*16000:
                    raise RuntimeError('Candidate sample-duration cap')
                samples = np.frombuffer(wav.readframes(sample_count), dtype='<i2').astype(np.float32)/32768
            speech = torch.from_numpy(samples.copy())[None]
            # OpenAI Whisper MIT log_mel_spectrogram, n_fft=400/hop=160/n_mels=128.
            stft = torch.stft(speech, 400, 160, window=torch.hann_window(400), return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2
            log_spec = (mel_filters @ magnitudes).clamp(min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max()-8.0)
            feature = ((log_spec+4.0)/4.0).numpy()
            inputs = session.get_inputs()
            ids = session.run(None,{inputs[0].name: feature,
                                    inputs[1].name: np.array([feature.shape[2]], dtype=np.int32)})[0].flatten().tolist()
            text_ids = tokenizer.encode(case['text'],add_special_tokens=False)
            if tokenizer.decode(text_ids) != case['text']:
                raise RuntimeError('Igbo Unicode text-tokenizer roundtrip failed')
            if not 0 < len(ids) <= 400 or len(prefix)+len(text_ids)+3+len(ids) > 512:
                raise RuntimeError('Speech or packed sequence length cap')
            cases.append({'id':case['id'],'source_group':case['source_group'],'text_ids':text_ids,
                          'speech_ids':ids,'sample_count':sample_count,'audio_sha256':case['candidate_audio_sha256']})
        del session, stft, feature, speech
        gc.collect()
        (WORK/'speech-targets.json').write_text(json.dumps(cases,ensure_ascii=False))
        report['cases'] = [{k:v for k,v in c.items() if k not in ('text_ids','speech_ids')} |
                           {'speech_tokens':len(c['speech_ids']),'text_tokens':len(c['text_ids'])} for c in cases]
        progress('loading_frozen_flash_backbone')
        checkpoint = torch.load(files['checkpoints/flash/flash_dpo/best.pt'], mmap=True,
                                map_location='cpu', weights_only=True)
        state = checkpoint['model']
        config = Qwen2Config(**json.loads(files['CosyVoice-BlankEN/config.json'].read_text()))
        config.use_cache = False
        config._attn_implementation = 'eager'
        prefix_key = 'q.llm.model.model.'
        backbone_state = {k[len(prefix_key):]:v.float() for k,v in state.items() if k.startswith(prefix_key)}
        backbone = load_qwen_backbone(config, backbone_state)
        speaker_map = json.loads(files['checkpoints/flash/flash_dpo/spk2id.json'].read_text())
        if len(speaker_map) != 4793 or speaker_map['waxal_ibo_5'] != 4773:
            raise RuntimeError('Speaker prior index mismatch')
        speech_weight = state['q.speech_embedding.weight'].float()
        decoder = state['q.llm_decoder.weight'].float()
        if speech_weight.shape != decoder.shape or speech_weight.shape[1] != 896:
            raise RuntimeError('Unexpected speech embeddings/decoder dimensions')
        model = make_pilot_model(backbone, speech_weight, decoder,
                                 state['lang_emb.weight'].float(), state['rate_emb.weight'].float(),
                                 state['spk_emb.weight'][4773].float(), prefix, report['source_groups'])
        if len(model.modules_adapted) != 48:
            raise RuntimeError('Expected 48 LoRA attention projections')
        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable)
        if not 0 < trainable_count <= 500000:
            raise RuntimeError('Trainable parameter cap')
        for case in cases:
            if any(type(i) is not int or not 0<=i<model.V for i in case['speech_ids']):
                raise RuntimeError('Acoustic token outside model vocabulary')
        report.update(trainable_parameters=trainable_count, lora_modules=len(model.modules_adapted),
                      speech_vocabulary=model.V, training_seconds=sum(c['sample_count'] for c in cases)/16000,
                      device='cpu', dtype='float32', source_vectors='Three separate provisional source vectors; female Igbo prior 4773')
        before = model.adapter_state()
        del checkpoint, state, backbone_state
        gc.collect()
        optimizer = torch.optim.AdamW(trainable, lr=plan['learning_rate'],weight_decay=0)
        model.train()
        train_start = time.monotonic()
        for update in range(plan['updates']):
            if time.monotonic()-train_start > 15*60:
                raise RuntimeError('Fifteen-minute training cap')
            index = update % len(cases)
            case = cases[index]
            progress('training_optimizer_step', current_case=case['id'], next_update=update+1)
            optimizer.zero_grad(set_to_none=True)
            loss = model(case['text_ids'], case['speech_ids'],index)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite speech loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
            optimizer.step()
            # Count an update only after optimizer.step actually returns.
            report['optimizer_updates'] += 1
            report['trained'] = True
            report['losses'].append({'update':update+1,'case':case['id'],
                                      'speech_cross_entropy':round(loss.item(),6),'gradient_norm':round(norm.item(),6),
                                      'interpretation':'In-sample debug loss; no quality/generalisation claim'})
            progress('optimizer_step_complete')
        after = model.adapter_state()
        changed = sum(not torch.equal(before[k],v) for k,v in after.items())
        delta = sum((after[k]-before[k]).double().square().sum().item() for k in after)**0.5
        if changed == 0 or delta == 0:
            raise RuntimeError('Optimizer steps produced no adapter change')
        output = WORK/'adapter.safetensors'
        save_file(after,str(output),metadata={'experiment':plan['id'],'base_revision':plan['models'][0]['revision'],
                  'license':'CC-BY-SA-4.0','steps':str(report['optimizer_updates']),
                  'status':'technical pilot; provisional cuts; quality not evaluated'})
        if output.stat().st_size > 16*1024**2:
            raise RuntimeError('Adapter checkpoint size cap')
        report.update(adapter_path=str(output),adapter_bytes=output.stat().st_size,adapter_sha256=digest_file(output),
                      changed_adapter_tensors=changed,adapter_l2_delta=delta,
                      training_wall_seconds=round(time.monotonic()-train_start,2),
                      limitations=plan['limitations'], attribution=plan['attribution'],
                      learning_rate=plan['learning_rate'],rank=4,alpha=8,seed=7,
                      base_checkpoint_sha256=plan['models'][0]['files'][0]['sha256'])
        progress('five_update_speech_lora_complete')
        (WORK/'training-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    except Exception as error:
        progress('flash_training_stopped', failed_phase=report['phase'],error_type=type(error).__name__,error=str(error)[:500])
        (WORK/'failure.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
