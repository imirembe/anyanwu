# Anyanwụ self-hosted speech backend

The first milestone is **microphone → real ASR → labelled prepared Igbo reply → real TTS**. The iOS app's existing `/v1/turns` contract is retained. A conversation model and streaming transcription are separate next steps. This service uses existing baseline models. Separate research adapters are not bundled or automatically enabled.

## Selected evaluation baselines

| Stage | Baseline | Publisher's licence | Status |
|---|---|---|---|
| Speech recognition | Meta `omniASR_CTC_300M_v2`, `omnilingual-asr==0.2.0` | Apache 2.0, code and models | Adapter implemented; GPU inference and mixed-language quality unverified |
| Igbo speech output | `PlotweaverAI/naija-tts`, revision `6d3576c7f68143e10337ff6aab64e4b48f6c025f` | MIT according to the model card | Adapter implemented; native-speaker listening evaluation pending |

The [ASR language list](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py) includes `ibo_Latn` and `eng_Latn`. The selected CTC model is unconditioned; this does not establish code-switching accuracy. Its upstream per-input duration limit is handled with sequential 25-second chunks. Words crossing chunk boundaries need evaluation.

The [TTS model card](https://huggingface.co/PlotweaverAI/naija-tts/blob/6d3576c7f68143e10337ff6aab64e4b48f6c025f/README.md) lists Igbo support, inherits Chatterbox's watermark, and reports an imperfect, small evaluation. Do not infer production pronunciation or tone quality from its language label. We use the checkpoint's built-in voice and expose no voice-cloning controls. Its publisher licence is not an independent audit of training-data or speaker rights. Preserve publisher and base-model notices when redistributing models or images.

The libraries use different PyTorch versions, so ASR and TTS have separate images and caches. Both worker processes can share GPU 0. Model provenance and limitations are recorded in [model_sources.json](../inference/model_sources.json). OmniVoice was excluded from the commercial baseline because its [weight licence is non-commercial](https://huggingface.co/k2-fsa/OmniVoice#license), despite its permissive code licence.

## Cloud machine

Use an **x86-64 Linux NVIDIA GPU host** with Docker Compose v2, NVIDIA Container Toolkit, and a driver supporting CUDA 12.8. An initial sizing target is **24 GB VRAM, 32 GB system RAM, and 80 GB disk** for both images and model caches. This is a starting allocation, not a measured minimum or throughput guarantee. Benchmark VRAM, startup time, latency and concurrency before resizing or serving users.

Keep the repository and recordings on that cloud host. No training dataset is needed to run these baseline models. Do not build the GPU images or run their dependency installs on the laptop. Initial cloud image builds and model startup download several GB; the lightweight laptop API tests download neither weights nor datasets.

From the project root **on the cloud host**:

```sh
python3 backend/cloud/configure.py
docker compose --env-file backend/cloud/.env -f backend/cloud/compose.yaml config --quiet
docker compose --env-file backend/cloud/.env -f backend/cloud/compose.yaml build
docker compose --env-file backend/cloud/.env -f backend/cloud/compose.yaml up -d
docker compose --env-file backend/cloud/.env -f backend/cloud/compose.yaml logs --tail=40 asr tts
```

The configure script refuses macOS and hosts without a working NVIDIA GPU, creates a private `.env` with fresh tokens, and never overwrites or prints them. It enables model downloads on that host. Each worker otherwise refuses startup unless `ANYANWU_ALLOW_MODEL_DOWNLOAD=1`. Importing modules, launching the development gateway, and running the API tests never load model weights.

Workers load their model once during startup. Readiness becomes true after loading succeeds. First startup may take several minutes. A load failure exits the worker rather than returning pretend transcripts or audio. `/readyz` checks actual worker readiness; the gateway's existing `/health` capabilities describe configured stages, not measured model availability or linguistic quality.

The compose file publishes ports **8787 (gateway), 8001 (ASR), 8002 (TTS) only on the cloud host's `127.0.0.1`**. Every route requires its token. Worker calls use private loopback, with no remote HTTP exception added to the gateway's TLS policy. Do not publish these ports on `0.0.0.0` at the host level. For a physical iPhone or public deployment, add a valid HTTPS endpoint for the gateway, with user authentication and rate limits; keep model workers private. The generated shared gateway token is for private development, not a multi-user authentication system.

## Connect the existing simulator

Create an SSH tunnel from the laptop to the cloud host:

```sh
ssh -N -L 8787:127.0.0.1:8787 CLOUD_USER@CLOUD_HOST
```

In the existing Anyanwụ simulator, select **Settings → Service**, use `http://localhost:8787`, and enter `ANYANWU_ACCESS_TOKEN` from the cloud's private `.env`. The preferred On iPhone route does not need this backend. Do not paste tokens into chat or put the worker token in the app. No simulator needs to be created or reopened for backend work.

The text-generation model stays unconfigured. Replies remain `mode: preview` with the speech-test notice, even when transcription and playback use actual neural models. Once connected, words appear after recording stops; this version does not stream partial transcription while the user is speaking.

## Run a real voice test

Use a real recording already on the cloud host, at most 60 seconds and 6 MB. The smoke tool checks both workers and sends the same request as the app:

```sh
python3 -m backend.cloud.smoke .build/evaluation-clips/igbo.m4a
```

It prints the actual transcript, prepared reply and latency. It does not retain a new copy of the input or output audio. To deliberately save one generated reply for listening evaluation:

```sh
python3 -m backend.cloud.smoke .build/evaluation-clips/igbo.m4a --save-reply .build/evaluations/igbo-reply.mp3
```

Assess Igbo intelligibility, vowel distinctions, tone, names, numbers, dialects, truncation and voice consistency with fluent speakers. Do not use an ASR round-trip score as the sole test of TTS quality.

## Measure ASR before fine-tuning

Create a JSONL manifest alongside test clips in cloud storage. Use held-out, human-verified transcripts, separate speakers from any future training set, and keep usage/consent records with the dataset. Example row structure (provide actual matching recordings and text):

```json
{"id":"ig-001","language":"ig","audio":"clips/ig-001.wav","text":"Ndewo, kedu ka ị mere?"}
{"id":"en-001","language":"en","audio":"clips/en-001.m4a","text":"What is your name?"}
{"id":"mix-001","language":"mixed","audio":"clips/mix-001.wav","text":"Hello, kedu ka ị mere?"}
```

Run from the cloud project root, with the workers running:

```sh
python3 -m backend.evaluate_speech .build/evaluation-clips/manifest.jsonl --limit 100 --output .build/evaluations/asr-first.json
```

The tool processes one recording at a time, caps each clip, separates Igbo/English/mixed results, and writes aggregate word error rate, character error rate and request time. It preserves underdots and tone marks while normalizing Unicode, case, punctuation and spacing. Character errors are counted on NFC code points. It retains no hypotheses or extra audio copies. An existing report is never overwritten. These metrics are not yet measured for our models.

Fine-tuning follows this baseline evaluation: select commercially usable and consented speech/transcript data, build speaker-separated training/validation/test splits in cloud storage, identify actual failure patterns, then compare a tuned checkpoint against this fixed baseline. Raw user conversations are not automatically collected for training.

## Resource and retention behaviour

- One active request per model worker; excess requests get 503 and `Retry-After`. Health remains responsive. Run exactly one Uvicorn process per worker to avoid duplicating models.
- Uploads are bounded before multipart/JSON parsing, with a 15-second upload timeout. Audio is capped at 6 MB and decoded to mono 16 kHz, at most 60 seconds. A 250 ms codec-padding allowance is trimmed to 60 seconds. FFmpeg has a 20-second deadline and cannot fetch network playlists.
- iPhone M4A uses an unlinked seekable temporary file for decoding. In the cloud, this lives in bounded `/tmp` RAM storage and is closed on completion or failure. Multipart temporary files also close after the request. Input recordings are not placed in a persistent model volume.
- TTS has a 600-character input cap, short text chunks, and a 60-second output cap. Igbo diacritics are preserved. Cancellation holds the inference slot until native work actually finishes; it does not claim to interrupt an already-running GPU kernel.
- Only model caches are persistent. Request bodies and transcripts are not saved by the servers; access logging is disabled, and application failures log exception classes only. Container logs rotate. Explicit evaluation fixtures/reports remain until you delete them.
- CPU RAM and PID limits bound each container. GPU memory is shared and still requires measurement. There is no automatic paid scaling or restart loop. If a native inference call hangs, restart its cloud worker after inspecting the service; an HTTP timeout cannot kill a running GPU operation.

Stop the cloud services between experiments:

```sh
docker compose --env-file backend/cloud/.env -f backend/cloud/compose.yaml down
```

This retains model volumes for the next run. Container shutdown does not stop cloud GPU billing: stop the rented instance through its provider when finished; retained storage may still be billed.

## Verification on the laptop

```sh
python3 -m venv .build/backend-venv
.build/backend-venv/bin/python -m pip install -r backend/requirements-test.txt
.build/backend-venv/bin/python -B -m unittest discover -s backend -v
```

API tests inject explicit model fixtures and exercise the real request formats, Unicode handling, authentication, size limits, role separation, busy admission, cancellation and prepared-reply labels. FFmpeg tests perform real WAV/MP3/M4A conversion with synthetic tones, not language-quality tests. The GPU adapters are not substituted by fixtures in a deployed service.

Current verification: **47 tests pass**, including real media decoding, cloud configuration guards, and compatibility with the existing gateway. Both upstream CUDA image tags exist for amd64, and the compose file parses as YAML. Docker is unavailable in the current laptop shell, so Compose execution, full image builds, actual GPU inference, native-speaker quality and an end-to-end iPhone voice turn remain unverified. No cloud account or machine has been connected, and no paid infrastructure has been started.
