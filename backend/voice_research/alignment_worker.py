"""Offline CPU forced alignment using existing ASR weights and source audio."""
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import socket
import time
import wave

from backend.voice_research.alignment_rules import alignment_text,eligible_caption,evaluate_path,caption_token_ranges


def file_sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if platform.system()!="Linux" or not Path("/content").is_dir() or os.environ.get("ANYANWU_CLOUD_ALIGNMENT")!="1":
        raise RuntimeError("Use the configured cloud alignment environment; no laptop audio/models.")
    root=Path(os.environ.get("ANYANWU_ALIGNMENT_WORK","/content/anyanwu-caption-alignment-v1"))
    if str(root) not in {"/content/anyanwu-caption-alignment-v1","/content/anyanwu-caption-alignment-v2-context","/content/anyanwu-expanded-caption-alignment-v1","/content/anyanwu-expanded-caption-joint-alignment-v1"}:
        raise RuntimeError("Unexpected alignment work directory.")
    plan=json.loads((root/"plan.json").read_text())
    if (root/"alignment-summary.json").exists():
        raise RuntimeError("Alignment outputs already exist; no overwrite or automatic rerun.")
    model=Path("/content/anyanwu-cpu-transcripts-v1/cache/fairseq2/assets/f33741966d852362f4d416d5/omniASR-CTC-300M-v2.pt")
    if not model.is_file() or file_sha256(model)!=plan["asr_model_sha256"]:
        raise RuntimeError("Existing ASR model cache is missing or changed; no download fallback.")
    def deny_network(*args,**kwargs):
        raise RuntimeError("Alignment is offline; model/tokenizer downloads are disabled.")
    socket.socket.connect=deny_network
    socket.socket.connect_ex=deny_network
    socket.create_connection=deny_network
    import torch
    import torchaudio.functional as AF
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pipeline=ASRInferencePipeline(model_card="omniASR_CTC_300M_v2",device="cpu",dtype=torch.float32)
    encoder=pipeline.tokenizer.create_raw_encoder(device=torch.device("cpu"))
    decoder=pipeline.tokenizer.create_decoder(skip_special_tokens=True)
    vocab=pipeline.tokenizer.vocab_info
    info={key:getattr(vocab,key,None) for key in ("size","unk_idx","bos_idx","eos_idx","pad_idx")}
    # fairseq2 0.6 Wav2Vec2AsrModel uses torch ctc_loss's default blank=0.
    blank=0
    cases=[]
    excluded=[]
    for entry in plan["captions"]:
        if not eligible_caption(entry):
            excluded.append({"id":entry["id"],"reason":"caption clipped by sample edge or contains marked copied-text ambiguity"})
    selected=[e for e in plan["captions"] if eligible_caption(e) and (not plan.get("selected_ids") or e["id"] in plan["selected_ids"])]
    expanded=plan['id'] in {'anyanwu-expanded-caption-alignment-v1','anyanwu-expanded-caption-joint-alignment-v1'}
    if len(selected)>(10 if expanded else 5):
        raise RuntimeError("Declared caption alignment count cap exceeded.")
    started=time.monotonic()
    for source_id in sorted({e["source_id"] for e in selected}):
        source_cases=[e for e in selected if e["source_id"]==source_id]
        audio=(Path('/content/anyanwu-bbc-expansion-v1')/source_id/'speech-20s.wav') if expanded else (Path("/content/anyanwu-bbc-igbo-pilot-v1/sample-v1")/source_id/"speech-12s.wav")
        if source_id not in {'bbc-igbo-01','bbc-igbo-02','bbc-igbo-03','bbc-igbo-04','bbc-igbo-05'}:
            raise RuntimeError('Unexpected source identifier.')
        origin=24 if expanded else 12
        if audio.is_symlink() or not audio.is_file() or audio.stat().st_size>(700000 if expanded else 500000) or file_sha256(audio)!=source_cases[0]["source_audio_sha256"]:
            raise RuntimeError("Original source WAV failed integrity.")
        with wave.open(str(audio)) as reader:
            params=reader.getparams()
            if (params.nchannels,params.sampwidth,params.framerate)!=(1,2,16000):
                raise RuntimeError("Unexpected source PCM format.")
            pcm=reader.readframes(params.nframes)
            seconds=params.nframes/16000
        captured=[]
        def hook(module,arguments,output):
            logits,layout=output
            length=int(layout.seq_lens[0])
            captured.append(logits[:,:length].detach().float().cpu().log_softmax(-1))
        handle=pipeline.model.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                pipeline.transcribe([str(audio)],batch_size=1)
        finally:
            handle.remove()
        if len(captured)!=1 or captured[0].shape[0]!=1:
            raise RuntimeError("Expected one CTC emission for the original source.")
        emission=captured.pop()
        ratio=seconds/emission.shape[1]
        joint={}
        if plan.get('alignment_mode')=='ordered_source_captions':
            source_cases.sort(key=lambda e:e['observed_first_frame_seconds'])
            texts=[alignment_text(e['text']) for e in source_cases]
            all_ids=encoder(' '.join(texts)).to(torch.int64).reshape(-1)
            if not 1<=all_ids.numel()<=600 or (all_ids==blank).any().item() or (all_ids==info['unk_idx']).any().item():
                raise ValueError('Unsupported joint caption token sequence')
            ranges=caption_token_ranges(all_ids.tolist(),texts,lambda value:decoder(torch.tensor(value,dtype=torch.int64)).strip())
            margin=float(plan['search_margin_seconds'])
            lo=max(0,source_cases[0]['observed_first_frame_seconds']-origin-margin)
            hi=min(seconds,source_cases[-1]['observed_last_frame_seconds']-origin+margin)
            left=max(0,int(math.floor(lo/ratio)))
            right=min(emission.shape[1],int(math.ceil(hi/ratio)))
            with torch.inference_mode():
                paths,scores=AF.forced_align(emission[:,left:right].contiguous(),all_ids.unsqueeze(0),blank=blank)
                spans=AF.merge_tokens(paths[0],scores[0].exp(),blank=blank)
            numeric=[[int(s.token),int(s.start),int(s.end),float(s.score)] for s in spans]
            if [s[0] for s in numeric]!=all_ids.tolist():
                raise ValueError('Joint path did not cover the complete ordered target sequence')
            for e,(a,b) in zip(source_cases,ranges):
                joint[e['id']]={'ids':all_ids[a:b], 'numeric':numeric[a:b], 'left':left, 'right':right}
        for entry in source_cases:
            case={"id":entry["id"],"source_id":source_id,"text":entry["text"],"source_audio_sha256":entry["source_audio_sha256"],
                  "source_subtitle_review":"User confirmed BBC subtitle/speech match","training_ready":False,
                  "caption_components":entry.get('components',[entry['id']])}
            try:
                target_text=alignment_text(entry["text"])
                ids=joint[entry['id']]['ids'] if joint else encoder(target_text).to(torch.int64).reshape(-1)
                if not 1<=ids.numel()<=300 or (ids==blank).any().item() or (info["unk_idx"] is not None and (ids==info["unk_idx"]).any().item()):
                    raise ValueError("Caption has an unsupported/blank token; no diacritic stripping fallback.")
                roundtrip=decoder(ids).strip()
                if " ".join(roundtrip.split())!=target_text:
                    raise ValueError("Tokenizer round trip changed the caption alignment view.")
                margin=float(plan.get("search_margin_seconds",1.0))
                if margin not in (1.0,2.0):
                    raise ValueError("Unsupported declared search margin.")
                lo=max(0,entry["observed_first_frame_seconds"]-origin-margin)
                hi=min(seconds,entry["observed_last_frame_seconds"]-origin+margin)
                left=max(0,int(math.floor(lo/ratio)))
                right=min(emission.shape[1],int(math.ceil(hi/ratio)))
                if joint:
                    pack=joint[entry['id']]
                    numeric,left,right=pack['numeric'],pack['left'],pack['right']
                else:
                    log_probs=emission[:,left:right].contiguous()
                    with torch.inference_mode():
                        paths,scores=AF.forced_align(log_probs,ids.unsqueeze(0),blank=blank)
                        spans=AF.merge_tokens(paths[0],scores[0].exp(),blank=blank)
                    numeric=[[int(s.token),int(s.start),int(s.end),float(s.score)] for s in spans]
                result=evaluate_path(numeric,ids.tolist(),right-left,ratio,left*ratio)
                if joint and (result['relative_audio_start_seconds']<lo-ratio or result['relative_audio_end_seconds']>hi+ratio):
                    result['timing_warnings'].append('alignment_outside_caption_observation_context')
                start=max(0,round((result["relative_audio_start_seconds"]-0.08)*16000))
                end=min(params.nframes,round((result["relative_audio_end_seconds"]+0.08)*16000))
                if end<=start:
                    raise ValueError("Aligned audio cut has no duration.")
                dest=root/(entry["id"]+".wav")
                with wave.open(str(dest),"wb") as writer:
                    writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(16000)
                    writer.writeframes(pcm[start*2:end*2])
                case.update(result)
                case.update({"alignment_text":target_text,"alignment_status":"candidate_cut_created","cloud_audio_path":str(dest),
                            "audio_start_sample":start,"audio_end_sample":end,"audio_sample_rate":16000,
                            "candidate_seconds":round((end-start)/16000,6),"candidate_audio_sha256":file_sha256(dest),
                            "candidate_audio_bytes":dest.stat().st_size,"padding_seconds":0.08,
                            "nominal_source_start_seconds":round(origin+start/16000,6),"nominal_source_end_seconds":round(origin+end/16000,6),
                            "boundary_status":"Machine-aligned candidate; nominal absolute source times inherit pilot offset uncertainty. Review exact first/last words before training.",
                            "alignment_kind":"Target-constrained CTC; successful alignment is not proof of text correctness"})
                (root/(entry["id"]+"-token-spans.json")).write_text(json.dumps({"target_ids":ids.tolist(),"spans_token_start_end_probability":numeric,"frame_seconds":ratio,"window_left_frame":left},ensure_ascii=False)+"\n")
            except (RuntimeError,ValueError) as error:
                case["alignment_status"]="alignment_not_usable"
                case["error"]=str(error)[:240]
            cases.append(case)
            (root/"progress.json").write_text(json.dumps({"phase":"aligning_captions","completed":len(cases),"total":len(selected)}))
        del emission
    report={"phase":"caption_alignment_complete","experiment_id":plan["id"],"search_margin_seconds":plan.get("search_margin_seconds",1.0),"method":"Cached omniASR_CTC_300M_v2 emissions with torchaudio 2.8 CTC forced alignment",
            "device":"cpu","vocabulary":info,"ctc_blank":blank,"cases":cases,"excluded":excluded,
            "candidate_cuts":sum(c.get("alignment_status")=="candidate_cut_created" for c in cases),
            "flagged_cuts":sum(bool(c.get("timing_warnings")) for c in cases),"elapsed_seconds":round(time.monotonic()-started,2),
            "network_requests":0,"model_downloads":0,"trained":False,"optimizer_updates":0,"ready_training_pairs":0,
            "alignment_mode":plan.get('alignment_mode','independent_caption_windows')}
    (root/"alignment-summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
