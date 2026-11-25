"""Cloud-only six-sentence Naija TTS audition. Never import GPU libraries on Mac."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import wave

MODEL = "PlotweaverAI/naija-tts"
REVISION = "6d3576c7f68143e10337ff6aab64e4b48f6c025f"
BASE_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
FILES = ["README.md", "ve.pt", "s3gen.pt", "conds.pt", "t3_mtl23ls_v2.safetensors",
         "grapheme_mtl_merged_expanded_v1.json"]


def ensure_mapping_link(directory: Path, mapping: Path):
    """Reuse a verified pinned mapping after failed startup, without replacing files."""
    target = directory / "Cangjie5_TC.json"
    expected = mapping.resolve(strict=True)
    if target.is_symlink() or target.exists():
        if not target.is_file() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(expected.read_bytes()).digest():
            raise RuntimeError("The cached auxiliary mapping differs from the pinned source; it was preserved.")
        return
    target.symlink_to(expected)


def main():
    if (platform.system() != "Linux" or os.environ.get("ANYANWU_CLOUD_AUDITION") != "1"
            or len(sys.argv) != 3):
        raise SystemExit("Cloud-only audition: run through the exported Colab notebook, never on the laptop.")
    # All heavy imports happen only after the explicit cloud guard.
    import numpy as np
    import torch
    import chatterbox.mtl_tts as multilingual
    from huggingface_hub import hf_hub_download, model_info, snapshot_download

    if not torch.cuda.is_available():
        raise SystemExit("An NVIDIA GPU is required; there is no CPU fallback.")
    package = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cases = package["cases"]
    if len(cases) != 6 or any(not isinstance(c["text"], str) or not 1 <= len(c["text"]) <= 420 for c in cases):
        raise SystemExit("Expected exactly six bounded published sentences.")
    output = Path(sys.argv[2])
    output.mkdir(exist_ok=False)
    # Check sizes before downloading only the named checkpoint files.
    info = model_info(MODEL, revision=REVISION, files_metadata=True, token=False)
    sizes = {s.rfilename: s.size for s in info.siblings}
    base = model_info("ResembleAI/chatterbox", revision=BASE_REVISION, files_metadata=True, token=False)
    mapping_size = next((s.size for s in base.siblings if s.rfilename == "Cangjie5_TC.json"), None)
    if any(type(sizes.get(f)) is not int for f in FILES) or type(mapping_size) is not int:
        raise SystemExit("Could not verify all checkpoint file sizes.")
    if sum(sizes[f] for f in FILES) + mapping_size > 4 * 1024**3:
        raise SystemExit("The selected model files exceed this cloud audition's 4 GiB weight budget.")
    directory = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=FILES, token=False,
                                       max_workers=2))
    mapping = hf_hub_download("ResembleAI/chatterbox", "Cangjie5_TC.json", revision=BASE_REVISION, token=False)
    ensure_mapping_link(directory, Path(mapping))
    multilingual.SUPPORTED_LANGUAGES["ig"] = "Igbo"
    model = multilingual.ChatterboxMultilingualTTS.from_local(
        str(directory), device="cuda")
    if model.conds is None:
        raise SystemExit("The checkpoint has no built-in voice; no reference recording will be substituted.")
    for sample in ("Anyanwụ", "ị ọ ụ ṅ", "à á ā ì í ù ú"):
        if (model.tokenizer.text_to_tokens(sample, language_id="ig") == 1).any().item():
            raise SystemExit("The tokenizer cannot represent the Igbo validation characters.")
    generated = []
    total = 0
    for number, case in enumerate(cases, 1):
        seed = 20261122 + number
        torch.manual_seed(seed)
        np.random.seed(seed)
        started = time.monotonic()
        with torch.inference_mode():
            audio = model.generate(case["text"], language_id="ig")
        audio = audio.detach().cpu().float().reshape(-1).numpy()
        rate = int(model.sr)
        if not 8000 <= rate <= 48000 or not 0 < audio.size <= rate * 30 or not np.isfinite(audio).all():
            raise SystemExit("A generated clip failed its 30-second/format limit; inspect the partial cloud output.")
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        name = f"generated-{number:02}.wav"
        path = output / name
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(pcm)
        total += path.stat().st_size
        if total > 24 * 1024**2 or path.stat().st_size > 4 * 1024**2:
            raise SystemExit("Generated audio exceeded the output budget; do not download this partial run.")
        generated.append({"id": case["id"], "text": case["text"], "filename": name,
                          "sample_rate": rate, "duration_seconds": round(audio.size / rate, 3),
                          "generation_seconds": round(time.monotonic() - started, 3), "seed": seed,
                          "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"Generated {number}/6: {audio.size/rate:.1f}s", flush=True)
    result = {"status": "generated_pending_listening", "model": MODEL, "revision": REVISION,
              "voice": "checkpoint built-in voice, not Anyanwu's final identity", "reference_audio_used": False,
              "evaluation_role": "Previously heard, user-reviewed reference comparison; not a held-out quality benchmark",
              "input_pack_sha256": hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
              "source_attribution": package["source_attribution"], "clips": generated}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Six outputs ready for listening. No model quality score has been assigned.", flush=True)


if __name__ == "__main__":
    main()
