"""One registered30-update, two-source-group whole-response GPU research trial."""
from pathlib import Path
import gc
import hashlib
import json
import os
import platform
import time

ROOT=Path('/content/anyanwu-conversational-style-lora-v1')


def digest(path,git=False):
    h=hashlib.sha1() if git else hashlib.sha256()
    if git:h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    if platform.system()!='Linux' or os.environ.get('ANYANWU_CONVERSATIONAL_TRIAL')!='1':raise RuntimeError('Configured hosted GPU worker only')
    import torch
    from transformers import Qwen2Config,Qwen2Tokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file,save_file
    from flash_adapter import make_pilot_model,load_qwen_backbone
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<14000*1024**2:raise RuntimeError('Free hosted GPU unavailable; no CPU fallback')
    torch.cuda.set_per_process_memory_fraction(.75,0);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    plan=json.loads((ROOT/'plan.json').read_text());cases=json.loads((ROOT/'training-cases.json').read_text())
    if plan['updates']!=30 or plan['position_cap']!=1024 or len(plan['schedule'])!=30:raise RuntimeError('Registered trial changed')
    if (ROOT/'training-result.json').exists():raise RuntimeError('Completed trial preserved; no repeat')
    torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);started=time.monotonic()
    report={'phase':'starting','experiment':plan['id'],'optimizer_updates':0,'trained':False,'loss_curve':[],'quality_evaluated':False}
    def progress(phase,**extra):
        report.update(phase=phase,elapsed_seconds=round(time.monotonic()-started,2),**extra)
        temp=ROOT/'progress.tmp';temp.write_text(json.dumps(report,ensure_ascii=False));temp.replace(ROOT/'progress.json')
    try:
        files={}
        for model in plan['models']:
            for item in model['files']:
                progress('conversational_trial_download',current_file=item['rfilename'])
                path=Path(hf_hub_download(model['repo'],item['rfilename'],revision=model['revision'],token=False,cache_dir=str(ROOT/'cache/hf/hub')))
                if path.stat().st_size!=item['size'] or digest(path,not bool(item.get('sha256')))!=(item.get('sha256') or item['git_blob_sha1']):raise RuntimeError('Pinned model integrity failure')
                files[item['rfilename']]=path
        tokenizer_dir=ROOT/'text-tokenizer';tokenizer_dir.mkdir(exist_ok=True)
        for name in ('vocab.json','merges.txt','tokenizer_config.json'):
            target=tokenizer_dir/name
            if not target.exists():target.symlink_to(files['CosyVoice-BlankEN/'+name])
        tokenizer=Qwen2Tokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True)
        prefix=tokenizer.encode('You are a helpful assistant.',add_special_tokens=False)+[151646]
        for case in cases:
            if tokenizer.decode(case['text_ids'])!=case['text'] or len(prefix)+len(case['text_ids'])+len(case['speech_ids'])+3>1024:
                raise RuntimeError('Exact source text/sequence no longer matches encoding')
        config=Qwen2Config(**json.loads(files['CosyVoice-BlankEN/config.json'].read_text()))
        if config.max_position_embeddings<1024:raise RuntimeError('Model context insufficient')
        config.use_cache=False;config._attn_implementation='eager'
        checkpoint=torch.load(files['checkpoints/flash/flash_dpo/best.pt'],map_location='cpu',mmap=True,weights_only=True);state=checkpoint['model'];key='q.llm.model.model.'
        backbone=load_qwen_backbone(config,{k[len(key):]:v.float() for k,v in state.items() if k.startswith(key)})
        if json.loads(files['checkpoints/flash/flash_dpo/spk2id.json'].read_text())['waxal_ibo_5']!=4773:raise RuntimeError('Female prior mapping changed')
        model=make_pilot_model(backbone,state['q.speech_embedding.weight'].float(),state['q.llm_decoder.weight'].float(),state['lang_emb.weight'].float(),state['rate_emb.weight'].float(),state['spk_emb.weight'][4773].float(),prefix,['waxal:ibo:5','silencio:IGB_005']).to('cuda')
        baseline=ROOT/'baseline/adapter.safetensors'
        if digest(baseline)!=plan['starting_adapter_sha256']:raise RuntimeError('Starting adapter differs')
        warm=load_file(str(baseline),device='cpu');named={n:p for n,p in model.named_parameters() if p.requires_grad}
        if set(warm)!=set(named) or warm['source_vectors'].shape!=(1,896) or named['source_vectors'].shape!=(2,896):raise RuntimeError('Warm-start source groups incompatible')
        with torch.no_grad():
            for name,p in named.items():
                value=warm[name].repeat(2,1) if name=='source_vectors' else warm[name]
                if p.shape!=value.shape or not torch.isfinite(value).all():raise RuntimeError('Warm-start tensor mismatch')
                p.copy_(value.to(p.device))
        if any(not torch.equal(p.detach().cpu(),warm[n].repeat(2,1) if n=='source_vectors' else warm[n]) for n,p in named.items()):raise RuntimeError('Warm-start reload differs')
        frozen_source=named['source_vectors'][0].detach().cpu().clone()
        mask=torch.tensor([[0.],[1.]],device='cuda')
        named['source_vectors'].register_hook(lambda grad:grad*mask)
        before={k:v.cpu() for k,v in model.adapter_state().items()}
        if sum(p.numel() for p in named.values())!=272128 or len(model.modules_adapted)!=48:raise RuntimeError('Unexpected rank4 adapter shape')
        if any(any(i>=model.V for i in c['speech_ids']) for c in cases):raise RuntimeError('Acoustic target outside vocabulary')
        del state,checkpoint,backbone,warm;gc.collect()
        validation=[c for c in cases if c['split']=='validation'];by_id={c['id']:c for c in cases}
        def evaluate():
            model.eval()
            with torch.no_grad():return sum(float(model(c['text_ids'],c['speech_ids'],0)) for c in validation)/4
        progress('conversational_trial_baseline_diagnostic')
        report['baseline_read_speech_validation_loss']=evaluate()
        optimizer=torch.optim.AdamW(list(named.values()),lr=plan['learning_rate'],weight_decay=0)
        model.train();training_started=time.monotonic()
        for index,case_id in enumerate(plan['schedule']):
            if time.monotonic()-training_started>900:raise RuntimeError('15-minute optimizer work cap')
            case=by_id[case_id];optimizer.zero_grad(set_to_none=True)
            loss=model(case['text_ids'],case['speech_ids'],case['source_index'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite whole-response loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1,error_if_nonfinite=True);optimizer.step()
            report['optimizer_updates']+=1;report['trained']=True
            item={'update':index+1,'id':case_id,'source_group':case['source_group'],'loss':round(loss.item(),6),'gradient_norm':round(norm.item(),6)}
            report['loss_curve'].append(item)
            if torch.cuda.max_memory_reserved()>12*1024**3:raise RuntimeError('GPU memory budget reached')
            progress('conversational_trial_optimizer_step',current_case=case_id)
        report['final_read_speech_validation_loss']=evaluate()
        after={k:v.cpu() for k,v in model.adapter_state().items()}
        if not torch.equal(after['source_vectors'][0],frozen_source):raise RuntimeError('Protected read-speech source vector changed')
        delta=sum((after[k]-before[k]).double().square().sum().item() for k in after)**.5
        if not delta>0 or any(not torch.isfinite(v).all() for v in after.values()):raise RuntimeError('Adapter did not make finite changes')
        full=ROOT/'adapter-two-source.safetensors';export=ROOT/'adapter.safetensors'
        metadata={'experiment':plan['id'],'updates':'30','status':'Private research; source licence conditions retained; no merged redistribution licence asserted'}
        save_file(after,str(full),metadata=metadata)
        candidate={k:(v[1:2].clone() if k=='source_vectors' else v) for k,v in after.items()}
        save_file(candidate,str(export),metadata=metadata)
        for path,expected in ((full,after),(export,candidate)):
            loaded=load_file(str(path),device='cpu')
            if path.stat().st_size>16*1024**2 or set(loaded)!=set(expected) or any(not torch.equal(loaded[k],v) for k,v in expected.items()):raise RuntimeError('Adapter save/reload failed')
        if digest(baseline)!=plan['starting_adapter_sha256'] or digest(files['checkpoints/flash/flash_dpo/best.pt'])!=plan['base_checkpoint_sha256']:raise RuntimeError('Preserved base/baseline changed')
        report.update(phase='conversational_trial_thirty_updates_complete',adapter_sha256=digest(export),adapter_bytes=export.stat().st_size,
                      full_adapter_sha256=digest(full),full_adapter_bytes=full.stat().st_size,source_groups=2,anchor_source_vector_unchanged=True,
                      baseline_unchanged=True,base_unchanged=True,adapter_reloaded=True,adapter_l2_delta=delta,
                      historical_baseline_updates=60,cumulative_lora_updates=90,conversational_example_presentations=20,read_example_presentations=10,
                      whole_conversational_responses=2,train_pool=22,diagnostic_validation=4,optimizer_wall_seconds=round(time.monotonic()-training_started,2),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),raw_text_and_audio_unchanged=True,
                      limitations=plan['limitations'],comparison_pending=True)
        progress('conversational_trial_thirty_updates_complete')
        (ROOT/'training-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    except Exception as error:
        progress('conversational_trial_worker_stopped',error_type=type(error).__name__,error=str(error)[:350]);raise


if __name__=='__main__':main()
