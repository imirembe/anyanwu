"""Generate six one-take audio files from original and trained speech LMs."""
from pathlib import Path
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import wave

TRAIN=Path('/content/anyanwu-waxal-female5-lora-v1')
ROOT=Path('/content/anyanwu-waxal-female5-comparison-v1')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    if platform.system()!='Linux' or not Path('/content').is_dir() or os.environ.get('ANYANWU_COMPARE_VOICE')!='1':
        raise RuntimeError('Existing cloud synthesis worker only; no laptop audio or models')
    sys.path.insert(0,str(TRAIN))
    import numpy as np
    import onnxruntime as ort
    import torch
    import torchaudio.compliance.kaldi as kaldi
    from transformers import Qwen2Config,Qwen2Tokenizer
    from safetensors.torch import load_file
    from flash_adapter import make_pilot_model,load_qwen_backbone
    from flash_generation import generate_speech_tokens
    from cv3_runtime import DiT,PreLookaheadLayer,CausalConditionalCFM,CausalMaskedDiffWithDiT,CausalHiFTGenerator,CausalConvRNNF0Predictor,DictConfig,mel_spectrogram

    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('Existing GPU unavailable; no new resource or CPU fallback')
    torch.cuda.set_per_process_memory_fraction(.75,0)
    plan=json.loads((ROOT/'plan.json').read_text())
    training=json.loads((TRAIN/'plan.json').read_text())
    data=json.loads((TRAIN/'data-result.json').read_text())
    targets=json.loads((TRAIN/'speech-targets.json').read_text())
    acoustic=json.loads((ROOT/'acoustic-files.json').read_text())
    if sha(TRAIN/'adapter.safetensors')!=plan['adapter_sha256'] or sha(ROOT/'cv3_runtime.py')!=plan['runtime_sha256']:
        raise RuntimeError('Adapter or inference runtime checksum mismatch')
    if (ROOT/'generation-result.json').exists():raise RuntimeError('Completed comparison exists; no retakes')
    started=time.monotonic()
    report={'phase':'starting_comparison','experiment':plan['id'],'optimizer_updates_this_stage':0,'clips':[],
            'quality_evaluated':False,'source_adapter_training_updates':60,'seed':plan['seed']}
    def progress(phase,**values):
        report.update(phase=phase,elapsed_seconds=round(time.monotonic()-started,2),**values)
        temp=ROOT/'generation-progress.tmp';temp.write_text(json.dumps(report,ensure_ascii=False));temp.replace(ROOT/'generation-progress.json')
    def cached(repo,revision,name):
        return TRAIN/'cache/hf/hub'/('models--'+repo.replace('/','--'))/'snapshots'/revision/name
    try:
        source=next(c for c in data['clips'] if c['id']==plan['prompt_id'])
        prompt=next(c for c in targets if c['id']==plan['prompt_id'])
        if source['adapter_split']!='train' or source['gender']!='Female' or source['speaker_id']!='5':raise RuntimeError('Unexpected fixed source reference')
        wav16=Path(source['cloud_audio_path']);mp3=TRAIN/'data'/(source['id']+'.mp3')
        if sha(wav16)!=source['audio_sha256'] or sha(mp3)!=source['mp3_sha256']:raise RuntimeError('Prompt source integrity mismatch')
        wav24=ROOT/'prompt-24k.wav'
        if not wav24.exists():
            proc=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-protocol_whitelist','file,pipe','-i',str(mp3),'-ac','1','-ar','24000','-c:a','pcm_s16le',str(wav24)],capture_output=True,timeout=30)
            if proc.returncode:raise RuntimeError('Whole reference decoding failed')
        def load_pcm(path,rate):
            with wave.open(str(path),'rb') as handle:
                if (handle.getnchannels(),handle.getsampwidth(),handle.getframerate())!=(1,2,rate):raise RuntimeError('Unexpected reference PCM format')
                values=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float32)/32768
            if not 1<=len(values)/rate<=15:raise RuntimeError('Reference duration cap')
            return torch.from_numpy(values.copy())[None]
        speech16=load_pcm(wav16,16000);speech24=load_pcm(wav24,24000)
        pfeat=mel_spectrogram(speech24,n_fft=1920,num_mels=80,sampling_rate=24000,hop_size=480,win_size=1920,fmin=0,fmax=None,center=False).squeeze(0).transpose(0,1)[None]
        n=min(pfeat.shape[1]//2,len(prompt['speech_ids']))
        pfeat=pfeat[:,:2*n].contiguous();prompt_tokens=prompt['speech_ids'][:n]
        fbank=kaldi.fbank(speech16,num_mel_bins=80,dither=0,sample_frequency=16000)
        fbank=fbank-fbank.mean(0,keepdim=True)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(acoustic['campplus.onnx'],options,providers=['CPUExecutionProvider'])
        speaker=torch.tensor(session.run(None,{session.get_inputs()[0].name:fbank[None].numpy()})[0].reshape(1,-1))
        if speaker.shape!=(1,192) or not torch.isfinite(speaker).all():raise RuntimeError('Reference speaker embedding invalid')
        del session,fbank,speech16,speech24;gc.collect()
        token_receipt=ROOT/'token-result.json'
        if token_receipt.exists():
            tokens=json.loads(token_receipt.read_text())['outputs']
            if len(tokens)!=6 or any(sha(ROOT/t['filename'])!=t['sha256'] for t in tokens):raise RuntimeError('Existing generated token manifest changed')
        else:
            if list(ROOT.glob('tokens-*.json')):raise RuntimeError('Partial token generation exists; preserve before a declared repair')
            progress('loading_original_and_trained_speech_lm')
            base=cached(plan['base_repo'],plan['base_revision'],'checkpoints/flash/flash_dpo/best.pt')
            if sha(base)!=plan['base_checkpoint_sha256']:raise RuntimeError('Original Flash checkpoint changed')
            ck=torch.load(base,map_location='cpu',mmap=True,weights_only=True);state=ck['model']
            configpath=cached(plan['acoustic_repo'],plan['acoustic_revision'],'CosyVoice-BlankEN/config.json')
            config=Qwen2Config(**json.loads(configpath.read_text()));config.use_cache=True;config._attn_implementation='eager'
            prefix_key='q.llm.model.model.'
            backbone=load_qwen_backbone(config,{k[len(prefix_key):]:v.float() for k,v in state.items() if k.startswith(prefix_key)})
            tokenizer=Qwen2Tokenizer.from_pretrained(str(TRAIN/'text-tokenizer'),local_files_only=True)
            prefix=tokenizer.encode('You are a helpful assistant.',add_special_tokens=False)+[151646]
            model=make_pilot_model(backbone,state['q.speech_embedding.weight'].float(),state['q.llm_decoder.weight'].float(),state['lang_emb.weight'].float(),state['rate_emb.weight'].float(),state['spk_emb.weight'][4773].float(),prefix,['waxal:ibo:5']).to('cuda').eval()
            del ck,state,backbone;gc.collect()
            adapter=load_file(str(TRAIN/'adapter.safetensors'),device='cpu')
            parameters={name:p for name,p in model.named_parameters() if p.requires_grad}
            if set(adapter)!=set(parameters) or any(adapter[k].shape!=p.shape for k,p in parameters.items()):raise RuntimeError('Trained adapter names/shapes do not match')
            tokens=[]
            for variant in ('base','adapter'):
                if variant=='adapter':
                    with torch.inference_mode():
                        for name,p in parameters.items():p.copy_(adapter[name].to(p.device))
                for text in plan['texts']:
                    progress('generating_speech_tokens',variant=variant,text_id=text['id'],generated_token_sets=len(tokens))
                    text_ids=tokenizer.encode(text['text'],add_special_tokens=False)
                    if tokenizer.decode(text_ids)!=text['text']:raise RuntimeError('Requested Igbo text changed in tokenizer')
                    generated=generate_speech_tokens(model,prompt['text_ids'],prompt_tokens,text_ids,seed=plan['seed'],max_tokens=400)
                    filename='tokens-'+variant+'-'+text['id']+'.json'
                    (ROOT/filename).write_text(json.dumps(generated,ensure_ascii=False,indent=2)+'\n')
                    tokens.append({'variant':variant,'text_id':text['id'],'filename':filename,'sha256':sha(ROOT/filename),
                                   'count':generated['count'],'stop_reason':generated['stop_reason'],'warnings':generated['warnings']})
            (ROOT/'token-result.json').write_text(json.dumps({'outputs':tokens,'adapter_sha256':plan['adapter_sha256'],'seed':plan['seed'],'optimizer_updates':0},indent=2)+'\n')
            del model,adapter,parameters;gc.collect();torch.cuda.empty_cache()
        progress('loading_frozen_acoustic_decoder')
        cfm=DictConfig({'sigma_min':1e-6,'solver':'euler','t_scheduler':'cosine','training_cfg_rate':.2,'inference_cfg_rate':.7,'reg_loss_type':'l1'})
        estimator=DiT(dim=1024,depth=22,heads=16,dim_head=64,ff_mult=2,mel_dim=80,mu_dim=80,spk_dim=80,out_channels=80,static_chunk_size=50,num_decoding_left_chunks=-1)
        decoder=CausalConditionalCFM(in_channels=240,n_spks=1,spk_emb_dim=80,cfm_params=cfm,estimator=estimator)
        flow=CausalMaskedDiffWithDiT(input_size=80,output_size=80,spk_embed_dim=192,output_type='mel',vocab_size=6561,input_frame_rate=25,only_mask_loss=True,token_mel_ratio=2,pre_lookahead_len=3,pre_lookahead_layer=PreLookaheadLayer(in_channels=80,channels=1024,pre_lookahead_len=3),decoder=decoder)
        flow.load_state_dict(torch.load(acoustic['flow.pt'],map_location='cpu',mmap=True,weights_only=True),strict=True)
        flow=flow.to('cuda').eval()
        hift=CausalHiFTGenerator(in_channels=80,base_channels=512,nb_harmonics=8,sampling_rate=24000,nsf_alpha=.1,nsf_sigma=.003,nsf_voiced_threshold=10,upsample_rates=[8,5,3],upsample_kernel_sizes=[16,11,7],istft_params={'n_fft':16,'hop_len':4},resblock_kernel_sizes=[3,7,11],resblock_dilation_sizes=[[1,3,5]]*3,source_resblock_kernel_sizes=[7,7,11],source_resblock_dilation_sizes=[[1,3,5]]*3,lrelu_slope=.1,audio_limit=.99,conv_pre_look_right=4,f0_predictor=CausalConvRNNF0Predictor(num_class=1,in_channels=80,cond_channels=512))
        hift_state={k.replace('generator.',''):v for k,v in torch.load(acoustic['hift.pt'],map_location='cpu',mmap=True,weights_only=True).items()}
        hift.load_state_dict(hift_state,strict=True);hift=hift.to('cuda').eval();del hift_state,estimator,decoder
        pfeat=pfeat.to('cuda');speaker=speaker.to('cuda');ptok=torch.tensor([prompt_tokens],dtype=torch.int32,device='cuda')
        for item in tokens:
            path=ROOT/(item['variant']+'-'+item['text_id']+'.wav')
            if path.exists():raise RuntimeError('Waveform exists without completed manifest; no automatic retake')
            progress('decoding_generated_speech',variant=item['variant'],text_id=item['text_id'],completed_wavs=len(report['clips']))
            ids=json.loads((ROOT/item['filename']).read_text())['tokens']
            torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
            t=torch.tensor([ids],dtype=torch.int32,device='cuda')
            with torch.inference_mode():
                mel,_=flow.inference(token=t,token_len=torch.tensor([len(ids)],dtype=torch.int32,device='cuda'),prompt_token=ptok,
                  prompt_token_len=torch.tensor([len(prompt_tokens)],dtype=torch.int32,device='cuda'),prompt_feat=pfeat,
                  prompt_feat_len=torch.tensor([pfeat.shape[1]],dtype=torch.int32,device='cuda'),embedding=speaker,streaming=False,finalize=True)
                generated,_=hift.inference(speech_feat=mel,finalize=True)
            values=generated.squeeze().float().cpu().numpy()
            if values.ndim!=1 or not 2400<=len(values)<=20*24000 or not np.isfinite(values).all() or float(np.abs(values).max())>1.001:
                raise RuntimeError('Generated waveform finite/duration/range guard failed')
            with wave.open(str(path),'wb') as handle:
                handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(24000);handle.writeframes((np.clip(values,-1,1)*32767).astype('<i2').tobytes())
            if path.stat().st_size>2*1024**2:raise RuntimeError('Generated WAV byte cap')
            clip={**item,'audio_filename':path.name,'audio_path':str(path),'audio_sha256':sha(path),'bytes':path.stat().st_size,
                  'seconds':round(len(values)/24000,6),'sample_rate':24000,'peak':round(float(np.abs(values).max()),6),'rms':round(float(np.sqrt(np.mean(values**2))),6),
                  'requested_text':next(x['text'] for x in plan['texts'] if x['id']==item['text_id']),'transcript_verified':False}
            report['clips'].append(clip);progress('comparison_waveform_saved')
        report.update(phase='six_generated_comparison_clips_ready',completed_wavs=6,total_audio_bytes=sum(c['bytes'] for c in report['clips']),
                      total_audio_seconds=round(sum(c['seconds'] for c in report['clips']),6),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                      adapter_sha256=plan['adapter_sha256'],prompt_id=plan['prompt_id'],limitations=plan['limitations'],attribution=plan['attribution'])
        progress('six_generated_comparison_clips_ready')
        (ROOT/'generation-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    except Exception as error:
        progress('comparison_generation_stopped',failed_phase=report['phase'],error_type=type(error).__name__,error=str(error)[:450])
        raise


if __name__=='__main__':main()
