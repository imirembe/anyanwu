"""Cloud CPU transcription of the five existing BBC pilot WAVs; no training."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time


def main():
    if platform.system()!="Linux" or not Path("/content").is_dir() or os.environ.get("ANYANWU_CLOUD_DRAFTS")!="1":
        raise RuntimeError("This worker runs only in the explicitly configured cloud CPU environment.")
    work=Path("/content/anyanwu-cpu-transcripts-v1")
    plan=json.loads((work/"plan.json").read_text())
    result_file=work/"draft-transcripts.json"
    if result_file.exists():
        raise RuntimeError("Draft outputs already exist; preserve them instead of regenerating.")
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pipeline=ASRInferencePipeline(model_card="omniASR_CTC_300M_v2",device="cpu",dtype=torch.float32)
    rows=[]
    start=time.monotonic()
    for clip in plan["clips"]:
        source_id=clip["id"]
        if source_id not in {f"bbc-igbo-{i:02d}" for i in range(1,6)}:
            raise RuntimeError("Unexpected source identifier.")
        wav=Path("/content/anyanwu-bbc-igbo-pilot-v1/sample-v1")/source_id/"speech-12s.wav"
        if wav.is_symlink() or not wav.is_file() or wav.stat().st_size>500000 or hashlib.sha256(wav.read_bytes()).hexdigest()!=clip["sha256"]:
            raise RuntimeError("Audio hash/size check failed.")
        with torch.inference_mode():
            output=pipeline.transcribe([str(wav)],batch_size=1)
        if len(output)!=1 or not isinstance(output[0],str) or len(output[0])>4000:
            raise RuntimeError("ASR returned an invalid bounded transcript.")
        rows.append({"id":source_id,"source_url":clip["source_url"],"audio_sha256":clip["sha256"],
                     "draft_transcript":output[0].strip(),"method":"omniASR_CTC_300M_v2 CPU float32",
                     "reviewer_type":"AI transcription only; not checked against speech or captions",
                     "language_verified":False,"speaker_verified":False,"alignment_verified":False,"training_ready":False})
        (work/"progress.json").write_text(json.dumps({"phase":"drafting_transcripts","completed":len(rows),"total":5}))
    versions={name:__import__("importlib.metadata",fromlist=["version"]).version(name) for name in ("torch","torchaudio","fairseq2","omnilingual-asr")}
    report={"phase":"cpu_transcript_drafts_complete","model":"omniASR_CTC_300M_v2","license":"Apache-2.0",
            "device":"cpu","package_versions":versions,"records":rows,"transcribed_clips":len(rows),
            "inference_seconds":round(time.monotonic()-start,2),"trained":False,"optimizer_updates":0,
            "training_ready":False,"note":"Source-derived ASR drafts only; preserve unchanged WAVs, review against speech and embedded captions before creating training targets."}
    result_file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
