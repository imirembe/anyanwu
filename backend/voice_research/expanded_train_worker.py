"""Registered warm-start34/4 female Igbo adapter experiment, cloud GPU only."""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
import urllib.request
import wave

ROOT=Path('/content/anyanwu-waxal-female5-expanded-lora-v2')


def digest(path,kind='sha256',git=False):
    h=hashlib.new(kind)
    if git:h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    if platform.system()!='Linux' or not Path('/content').is_dir() or os.environ.get('ANYANWU_EXPANDED_TRAIN')!='1':
        raise RuntimeError('Hosted cloud worker only; no laptop weights/audio')
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import Qwen2Config,Qwen2Model,Qwen2Tokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file,load_file
    from flash_adapter import make_pilot_model,load_qwen_backbone
    plan=json.loads((ROOT/'plan.json').read_text())
    baseline_plan=json.loads((ROOT/'baseline-plan.json').read_text())
    if plan['id']!=ROOT.name or plan['registered_additional_updates']!=60 or len(plan['cases'])!=38:
        raise RuntimeError('Unexpected registered training plan')
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<14000*1024**2:
        raise RuntimeError('Existing GPU unavailable; no CPU fallback or resource purchase')
    torch.cuda.set_per_process_memory_fraction(.75,0)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(11);torch.cuda.manual_seed_all(11)
    started=time.monotonic()
    report={'phase':'starting','experiment':plan['id'],'optimizer_updates':0,'trained':False,'quality_evaluated':False,
            'speech_generated':False,'rejected_bbc_pairs_used':0,'loss_curve':[],'source_group':'waxal:ibo:5',
            'label_status':'14 new sources user-accepted; original24 publisher labels retain their original status'}
    def progress(phase,**extra):
        report.update(phase=phase,elapsed_seconds=round(time.monotonic()-started,2),**extra)
        temporary=ROOT/'progress.tmp';temporary.write_text(json.dumps(report,ensure_ascii=False));temporary.replace(ROOT/'progress.json')
    try:
        progress('checking_gpu_adapter_gradients')
        # Fixture gradients test compatibility; not counted as speech training updates.
        config=Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,_attn_implementation='eager')
        tiny=load_qwen_backbone(config,Qwen2Model(config).state_dict())
        fixture=make_pilot_model(tiny,torch.randn(210,32),torch.randn(210,32),torch.randn(5,32),torch.randn(6,32),torch.randn(32),[8,9],['fixture']).to('cuda')
        fixture([10,11],[1,2,3],0).backward()
        if not any(p.grad is not None and p.grad.abs().sum()>0 for n,p in fixture.named_parameters() if n.endswith('.B')):
            raise RuntimeError('GPU LoRA gradient fixture failed')
        del fixture,tiny;gc.collect();torch.cuda.empty_cache();torch.manual_seed(11)
        files={}
        selected=sum(f['size'] for m in baseline_plan['models'] for f in m['files'])
        if selected>4*1024**3:raise RuntimeError('Four GiB selected model cap')
        for model in baseline_plan['models']:
            for item in model['files']:
                progress('downloading_pinned_model_file',current_file=item['rfilename'])
                path=Path(hf_hub_download(repo_id=model['repo'],filename=item['rfilename'],revision=model['revision'],token=False,cache_dir=str(ROOT/'cache/hf/hub')))
                expected=item.get('sha256') or item['git_blob_sha1']
                if path.stat().st_size!=item['size'] or digest(path,'sha256' if item.get('sha256') else 'sha1',not bool(item.get('sha256')))!=expected:
                    raise RuntimeError('Pinned model file failed size/hash verification')
                files[item['rfilename']]=path
        report['selected_model_bytes']=selected
        melmeta=baseline_plan['mel_filter'];melpath=ROOT/'mel_filters.npz'
        if not melpath.exists():
            with urllib.request.urlopen(melmeta['url'],timeout=30) as response:data=response.read(melmeta['size']+1)
            if len(data)!=melmeta['size']:raise RuntimeError('Mel filter size mismatch')
            melpath.write_bytes(data)
        if digest(melpath,'sha1',True)!=melmeta['git_blob_sha1']:raise RuntimeError('Mel filter checksum mismatch')
        tokenizer_dir=ROOT/'text-tokenizer';tokenizer_dir.mkdir(exist_ok=True)
        for name in ('vocab.json','merges.txt','tokenizer_config.json'):
            target=tokenizer_dir/name
            if not target.exists():target.symlink_to(files['CosyVoice-BlankEN/'+name])
        tokenizer=Qwen2Tokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True)
        prefix=tokenizer.encode('You are a helpful assistant.',add_special_tokens=False)+[151646]
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.enable_cpu_mem_arena=False
        session=ort.InferenceSession(str(files['speech_tokenizer_v3.onnx']),options,providers=['CPUExecutionProvider'])
        filters=torch.from_numpy(np.load(melpath)['mel_128'])
        data=json.loads((ROOT/'data-result.json').read_text());cases=[]
        for clip in data['clips']:
            progress('encoding_whole_female_utterance',current_case=clip['id'])
            wav=Path(clip['cloud_audio_path'])
            if wav.parent!=ROOT/'data' or wav.stat().st_size!=clip['audio_bytes'] or digest(wav)!=clip['audio_sha256']:
                raise RuntimeError('Whole source utterance checksum/path mismatch')
            with wave.open(str(wav),'rb') as reader:
                if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate())!=(1,2,16000):raise RuntimeError('Unexpected whole waveform format')
                samples=np.frombuffer(reader.readframes(reader.getnframes()),dtype='<i2').astype(np.float32)/32768
            if not 16000<=len(samples)<=15*16000 or float(np.sqrt(np.mean(samples**2)))<.0001 or float(np.mean(np.abs(samples)>.999))>.01:
                raise RuntimeError('Source duration/energy/clipping guard failed')
            audio=torch.from_numpy(samples.copy())[None]
            spectrum=torch.stft(audio,400,160,window=torch.hann_window(400),return_complex=True)[...,:-1].abs()**2
            mel=(filters@spectrum).clamp(min=1e-10).log10();mel=torch.maximum(mel,mel.max()-8)
            features=((mel+4)/4).numpy();inputs=session.get_inputs()
            speech=session.run(None,{inputs[0].name:features,inputs[1].name:np.array([features.shape[2]],dtype=np.int32)})[0].flatten().tolist()
            text=tokenizer.encode(clip['text'],add_special_tokens=False)
            if tokenizer.decode(text)!=clip['text']:raise RuntimeError('Igbo Unicode tokenizer round trip changed source text')
            if not 1<=len(speech)<=400 or len(prefix)+len(text)+len(speech)+3>512:raise RuntimeError('Acoustic token/sequence cap')
            cases.append({'id':clip['id'],'split':clip['adapter_split'],'text_ids':text,'speech_ids':speech,'audio_sha256':clip['audio_sha256'],'seconds':clip['seconds']})
        del session,spectrum,features,audio;gc.collect()
        (ROOT/'speech-targets.json').write_text(json.dumps(cases,ensure_ascii=False))
        train=[c for c in cases if c['split']=='train'];validation=[c for c in cases if c['split']=='validation']
        if (len(train),len(validation))!=(34,4):raise RuntimeError('Registered34/4 split not met')
        progress('loading_frozen_female_igbo_base')
        checkpoint=torch.load(files['checkpoints/flash/flash_dpo/best.pt'],mmap=True,map_location='cpu',weights_only=True)
        state=checkpoint['model'];prefix_key='q.llm.model.model.'
        config=Qwen2Config(**json.loads(files['CosyVoice-BlankEN/config.json'].read_text()));config.use_cache=False;config._attn_implementation='eager'
        backbone=load_qwen_backbone(config,{k[len(prefix_key):]:v.float() for k,v in state.items() if k.startswith(prefix_key)})
        speaker_map=json.loads(files['checkpoints/flash/flash_dpo/spk2id.json'].read_text())
        if speaker_map['waxal_ibo_5']!=4773:raise RuntimeError('Female Igbo prior mapping changed')
        model=make_pilot_model(backbone,state['q.speech_embedding.weight'].float(),state['q.llm_decoder.weight'].float(),
              state['lang_emb.weight'].float(),state['rate_emb.weight'].float(),state['spk_emb.weight'][4773].float(),prefix,['waxal:ibo:5']).to('cuda')
        parameters=[p for p in model.parameters() if p.requires_grad]
        parameter_count=sum(p.numel() for p in parameters)
        if parameter_count!=271232 or len(model.modules_adapted)!=48:raise RuntimeError('Expected rank4 q/v adapter dimensions')
        if any(not 0<=i<model.V for c in cases for i in c['speech_ids']):raise RuntimeError('Acoustic token outside vocabulary')
        baseline_adapter=ROOT/'baseline/adapter.safetensors'
        if digest(baseline_adapter)!=plan['starting_adapter_sha256']:raise RuntimeError('Starting adapter hash changed')
        warm=load_file(str(baseline_adapter),device='cpu')
        named={name:p for name,p in model.named_parameters() if p.requires_grad}
        if set(warm)!=set(named) or any(warm[name].shape!=p.shape or not torch.isfinite(warm[name]).all() for name,p in named.items()):raise RuntimeError('Starting v1 adapter tensors are incompatible')
        with torch.no_grad():
            for name,p in named.items():p.copy_(warm[name].to(p.device))
        if any(not torch.equal(p.detach().cpu(),warm[name]) for name,p in named.items()):raise RuntimeError('Starting adapter reload differs')
        report.update(starting_adapter_sha256=plan['starting_adapter_sha256'],warm_start_equal=True,baseline_historical_updates=60,new_source_user_accepted=14)
        del warm,named
        before={k:v.cpu() for k,v in model.adapter_state().items()}
        del checkpoint,state,backbone;gc.collect()
        report.update(trainable_parameters=parameter_count,train_clips=34,validation_clips=4,
          train_audio_seconds=round(sum(c['seconds'] for c in train),6),validation_audio_seconds=round(sum(c['seconds'] for c in validation),6),
          device='cuda',gpu_name=torch.cuda.get_device_name(0),dtype='float32',data_audio_seconds=data['audio_seconds'])
        def evaluate():
            model.eval();losses=[]
            with torch.no_grad():
                for c in validation:losses.append(float(model(c['text_ids'],c['speech_ids'],0)))
            return sum(losses)/len(losses)
        progress('baseline_same_speaker_validation')
        report['baseline_validation_loss']=evaluate()
        optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],weight_decay=0)
        model.train();rng=random.Random(11);order=[];training_started=time.monotonic()
        for update in range(60):
            if time.monotonic()-training_started>900:raise RuntimeError('Fifteen-minute training cap')
            if not order:
                order=list(range(len(train)));rng.shuffle(order)
            c=train[order.pop()]
            progress('training_female_igbo_adapter',current_case=c['id'],next_update=update+1)
            optimizer.zero_grad(set_to_none=True)
            loss=model(c['text_ids'],c['speech_ids'],0)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite acoustic loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1,error_if_nonfinite=True);optimizer.step()
            report['optimizer_updates']+=1;report['trained']=True
            item={'update':update+1,'case':c['id'],'loss':round(loss.item(),6),'gradient_norm':round(norm.item(),6)}
            with (ROOT/'training-steps.jsonl').open('a') as log:log.write(json.dumps(item)+'\n')
            if update==0 or (update+1)%10==0:report['loss_curve'].append(item)
            if torch.cuda.max_memory_reserved()>12*1024**3:raise RuntimeError('Twelve GiB GPU memory cap')
            progress('female_igbo_optimizer_step_complete')
        report['training_wall_seconds']=round(time.monotonic()-training_started,2)
        progress('final_same_speaker_validation')
        report['final_validation_loss']=evaluate()
        after={k:v.cpu() for k,v in model.adapter_state().items()}
        changed=sum(not torch.equal(before[k],v) for k,v in after.items())
        delta=sum((after[k]-before[k]).double().square().sum().item() for k in after)**.5
        if changed==0 or not delta>0:raise RuntimeError('No adapter update detected')
        path=ROOT/'adapter.safetensors'
        save_file(after,str(path),metadata={'license':'CC-BY-SA-4.0','experiment':plan['id'],'steps':'60','status':'Research only;14 new source clips user-reviewed, original labels unchanged; generated quality pending'})
        if path.stat().st_size>16*1024**2:raise RuntimeError('Adapter checkpoint cap')
        reloaded=load_file(str(path),device='cpu')
        if set(reloaded)!=set(after) or any(not torch.equal(reloaded[k],v) or not torch.isfinite(v).all() for k,v in after.items()):raise RuntimeError('Adapter reload verification failed')
        base=files['checkpoints/flash/flash_dpo/best.pt']
        if digest(base)!=baseline_plan['models'][0]['files'][0]['sha256']:raise RuntimeError('Frozen base changed')
        if digest(baseline_adapter)!=plan['starting_adapter_sha256']:raise RuntimeError('Preserved starting adapter changed')
        report.update(phase='expanded_female_igbo_sixty_updates_complete',adapter_path=str(path),adapter_bytes=path.stat().st_size,
          adapter_sha256=digest(path),changed_adapter_tensors=changed,adapter_l2_delta=delta,
          peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),adapter_reloaded=True,base_hash_unchanged=True,starting_adapter_unchanged=digest(baseline_adapter)==plan['starting_adapter_sha256'],cumulative_adapter_updates=120,
          package_versions={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','onnxruntime','safetensors','huggingface_hub')},
          limitations=plan['limitations'],attribution=plan['attribution'],base_revision=baseline_plan['models'][0]['revision'],
          source_revision=baseline_plan['dataset_revision'],label_policy='Publisher transcripts unchanged; no synthetic text or rejected BBC pairs')
        progress('expanded_female_igbo_sixty_updates_complete')
        (ROOT/'training-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    except Exception as error:
        progress('female_igbo_training_stopped',failed_phase=report['phase'],error_type=type(error).__name__,error=str(error)[:400])
        raise


if __name__=='__main__':main()
