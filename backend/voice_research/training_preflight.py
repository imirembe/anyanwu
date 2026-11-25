"""Cloud-only voice-data readiness and acoustic checks, not model training.

Uses existing pilot WAVs without network, installs, denoising or fabricated
transcripts. A successful integrity check never promotes a clip to a TTS target.
"""
from __future__ import annotations

from array import array
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import wave


def cloud_guard():
    if platform.system() != "Linux" or not Path("/content").is_dir() or importlib.util.find_spec("google.colab") is None:
        raise RuntimeError("Run this audit only on the hosted Colab VM; no local audio/model work.")


def acoustic_metrics(samples, rate):
    if not samples or rate <= 0:
        raise ValueError("Audio is empty or has an invalid rate.")
    rms = math.sqrt(sum(int(s)*int(s) for s in samples)/len(samples))/32768
    block = max(1, rate//50)
    energies = [math.sqrt(sum(int(s)*int(s) for s in samples[i:i+block])/len(samples[i:i+block]))/32768
                for i in range(0, len(samples), block)]
    return {"rms_dbfs":round(20*math.log10(max(rms, 1e-12)),2),
            "peak_dbfs":round(20*math.log10(max(max(abs(s) for s in samples)/32768, 1e-12)),2),
            "near_full_scale_fraction":round(sum(abs(s)>=32760 for s in samples)/len(samples),6),
            "low_energy_20ms_fraction":round(sum(e<10**(-45/20) for e in energies)/len(energies),4),
            "interpretation":"Signal statistics only; not speech recognition, speaker separation, music detection or naturalness scores."}


def read_audio(path, expected):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 500000:
        raise RuntimeError("Pilot WAV is missing, linked or too large.")
    body = path.read_bytes()
    actual = hashlib.sha256(body).hexdigest()
    if actual != expected:
        raise RuntimeError("Pilot WAV failed the recorded SHA256 check.")
    with wave.open(str(path), "rb") as reader:
        if (reader.getnchannels(),reader.getsampwidth(),reader.getframerate()) != (1,2,16000):
            raise RuntimeError("Pilot WAV must be mono 16-bit 16 kHz PCM.")
        count = reader.getnframes()
        if not 1 <= count <= 16000*20:
            raise RuntimeError("Pilot WAV is outside the bounded duration.")
        decoded = reader.readframes(count)
    if len(decoded) != count*2:
        raise RuntimeError("WAV data is truncated.")
    samples = array("h", decoded)
    if sys.byteorder != "little":
        samples.byteswap()
    return {"sha256":actual,"bytes":len(body),"seconds":round(count/16000,6),"sample_rate":16000,
            "channels":1,"metrics":acoustic_metrics(samples,16000)}


def missing_labels(row):
    missing=[]
    if not isinstance(row.get("transcript"),str) or not row["transcript"].strip():
        missing.append("verbatim_transcript")
    for key in ("igbo_speech_checked","transcript_audio_match_checked","single_speaker_checked",
                "music_overlap_checked","utterance_boundaries_checked"):
        if row.get(key) is not True:
            missing.append(key)
    if not row.get("speaker_id"):
        missing.append("speaker_id")
    if row.get("split") not in ("train","validation","test"):
        missing.append("split")
    if not row.get("reviewer_type") or not row.get("review_notes"):
        missing.append("review_provenance")
    return missing


def gpu_status():
    if not shutil.which("nvidia-smi"):
        return {"available":False,"detail":"No NVIDIA GPU in the current CPU runtime"}
    try:
        result=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
        lines=[line.strip() for line in result.stdout.splitlines() if line.strip()]
        usable=result.returncode==0 and len(lines)==1 and int(lines[0].rsplit(",",1)[1])>=14000
        return {"available":usable,"detail":lines[0][:160] if lines else "GPU unavailable", "required_vram_mib":14000}
    except (OSError,ValueError,subprocess.TimeoutExpired):
        return {"available":False,"detail":"GPU check failed; no automatic CPU training or paid fallback"}


def preflight(plan):
    cloud_guard()
    if shutil.disk_usage("/content").free < 5*1024**3:
        raise RuntimeError("Five GiB free cloud disk required.")
    if plan.get("permission") != "user_confirmed_training_permission":
        raise RuntimeError("This plan has no recorded source permission.")
    pilot=Path("/content/anyanwu-bbc-igbo-pilot-v1/sample-v1")
    rows=[]
    for expected in plan["clips"]:
        source_id=expected["id"]
        if source_id not in {f"bbc-igbo-{i:02d}" for i in range(1,6)}:
            raise RuntimeError("Unexpected pilot identifier.")
        row={"id":source_id,"source_url":expected["source_url"],"source_start_seconds":12,
             "audio_cloud_path":str(pilot/source_id/"speech-12s.wav"),"transcript":None,
             "speaker_id":None,"split":None,"reviewer_type":None,"review_notes":None,
             "training_ready":False,"source_role":"Unsegmented audit excerpt, not a complete verified utterance"}
        try:
            row["audio"]=read_audio(Path(row["audio_cloud_path"]),expected["sha256"])
            row["integrity"]="verified"
        except (RuntimeError,OSError,wave.Error) as error:
            row["integrity"]="failed"
            row["error"]=str(error)[:180] if isinstance(error,RuntimeError) else type(error).__name__
        row["missing_labels"]=missing_labels(row)
        rows.append(row)
    gpu=gpu_status()
    blockers=["Verbatim audio-matched transcripts and utterance/speaker review are missing.",
              "Only five audit excerpts are available; prepare the separate training/validation/test utterance set.",
              "No trainable base checkpoint and compatible pinned training implementation have been selected and validated."]
    if not gpu["available"]:
        blockers.append("No usable GPU is attached; Colab previously reported its free GPU quota exhausted.")
    if any(row["integrity"]!="verified" for row in rows):
        blockers.append("One or more original audio assets failed integrity checks.")
    report={"id":plan["id"],"phase":"training_preflight_completed","status":"not_ready_for_training",
            "trained":False,"optimizer_updates":0,"checkpoint_created":False,"gpu":gpu,
            "pilot_clips":len(rows),"audio_integrity_verified":sum(row["integrity"]=="verified" for row in rows),
            "pilot_seconds":round(sum(row.get("audio",{}).get("seconds",0) for row in rows),6),
            "verified_audio_text_pairs":0,"blockers":blockers,"records":rows,
            "resource_use":{"network_requests":0,"model_downloads":0,"media_copies":0,"paid_resources_started":False}}
    output=Path("/content/anyanwu-female-training-v1")
    output.mkdir(exist_ok=True)
    (output/"readiness.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output/"utterance-review.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    return report
