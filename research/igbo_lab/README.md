# Anyanwụ local Igbo model lab

The public repository includes tooling and small synthetic fixtures. Third-party corpora, curation records, private reviews and checkpoints are excluded. Data-dependent commands and historical experiment tests require separately prepared, licensed inputs; the examples below do not imply that those inputs are bundled.

This is the experimental **reply-model component of the voice assistant**. It trains a small causal Transformer from random weights on the user's Apple-silicon Mac. It accepts a speech transcript when testing replies. It does not record audio, supply speech recognition or an Igbo speaking voice, or replace the live iPhone assistant's replies.

The first corpus is 32 tiny, AI-authored demonstration records: short Igbo stories and English/Igbo dialogue examples with Igbo replies. They are marked `demo_only`, are unreviewed, and are included for software checks only. Lower loss or a memorized greeting on this corpus does not establish conversational ability, translation quality, dialect coverage or factual accuracy. Fluent speakers and a substantially better corpus are needed for those claims.

## Run

From `the repository root`:

```sh
python3 scripts/igbo_lab.py status
python3 scripts/igbo_lab.py setup
python3 scripts/igbo_lab.py check
python3 scripts/igbo_lab.py reply "What is your name?" --checkpoint previous
python3 scripts/igbo_lab.py compare
```

`setup` installs only pinned MLX, its Metal backend and SentencePiece, using prebuilt wheels and no global pip cache. The isolated environment, preparation output, checkpoints and temporary files all stay under `.build/IgboLab`. No pretrained model weights or external corpus download is performed. Source and seed data stay in `research/igbo_lab`.

`demo` prepares the synthetic corpus and runs at most 120 steps / 180 seconds. It refuses to replace a non-demo prepared corpus. Reports always label the model experimental. Model initialization and training happen locally; no cloud account or paid service is used.

The current workspace contains a reviewed public-data experiment, so `demo` intentionally refuses to overwrite it. `current` is the latest experimental candidate; `previous` preserves the measured comparison baseline. Both are unsuitable for live assistant replies.

## Reviewed text workflow: steps 1–4

1. **Bounded text-only import:** `python3 scripts/igbo_lab.py import-text --source public-igbo --rows 100`. The official Hub/dataset-viewer APIs share a 5 MiB HTTP-body budget, with at most 500 selected records and 2,000 scanned rows. Audio/media URLs, Parquet files, full datasets and weights are never fetched. Raw source text, declared license, source card, hashes and the observed Hub head are preserved. The viewer does not guarantee a pinned row snapshot; the importer checks the Hub head before/after and records that limitation. An existing sample is preserved instead of silently overwritten.
2. **Explicit Codex review:** `python3 scripts/igbo_lab.py review-sample research/igbo_lab/data/public_igbo_review_v1.jsonl`. Every decision targets an original record/content hash and records accept/edit/reject, reason and confidence. Low-confidence and unreviewed rows cannot enter the reviewed output. All current decisions are AI review, as requested by the user, and are not native-speaker validation. Generated source examples and Codex rewrites retain their stated origin; they are not relabelled authentic speech.
3. **Separate evaluation:** `data/voice_eval_v1.jsonl` contains 20 frozen transcript prompts: 8 English, 8 Igbo and 4 mixed. Exact prompt overlaps and source groups containing those overlaps are excluded. `python3 scripts/igbo_lab.py evaluate-transcripts --label baseline` records the old model before any candidate training. An existing baseline cannot be overwritten without the explicit `--replace-baseline` option, which begins a new comparison. `python3 scripts/igbo_lab.py prepare-reviewed` prepares the accepted sample with a review receipt and fixed evaluation hash, retaining the tokenizer for comparable scoring.
4. **Train once and compare:** `python3 scripts/igbo_lab.py train --resume --steps 120 --seconds 180`, then `python3 scripts/igbo_lab.py evaluate-transcripts --label candidate`, then `python3 scripts/igbo_lab.py compare`. Training refuses a real imported corpus without review metadata and refuses a mismatched/missing baseline for the frozen experiment. Evaluation scores reference reply tokens only, saves generated replies, and refuses comparisons across different cases, tokenizers or architectures. It does not measure microphone/ASR quality, Igbo speech output or native-speaker quality.

The initial sample has already been imported. Stored review decisions apply to that exact sample, not future publisher changes. Re-running preparation/training is an explicit new experiment; do not repeatedly tune on the frozen benchmark to chase its score.

### Africanvoice access

The Africanvoice source is separately supported:

```sh
python3 scripts/igbo_lab.py import-text --source africanvoice --token-prompt
```

Run this in an interactive local terminal after accepting that dataset's access conditions. The read-only Hugging Face token is entered without echo, sent to the official API hosts through curl's stdin, and never written to arguments, logs, sample files or the child environment. Token prompts are refused in noninteractive pipes/chat tools. A token can also be supplied through an existing `HF_TOKEN` environment variable; the importer never searches unrelated folders for credentials. Authentication failure stops the request and never falls back to full dataset downloads.

The user's authenticated 10-record import succeeded through the official [`/first-rows` JSON preview](https://huggingface.co/docs/dataset-viewer/first_rows), after `/rows` returned HTTP 501. It transferred 1,173,644 HTTP-body bytes, scanned 131 preview rows, and saved 61,466 bytes of source records with no audio. The source splits are 8 train, 1 validation and 1 test. The importer requests each split's preview once, uses at most 1 MiB per preview within the shared 5 MiB budget, and records the endpoint used. Truncated transcript/speaker cells are excluded. The earlier quota failure did not retain enough detail to establish its exact cause; the successful smaller request should not be used to infer that cause.

On a quota shortfall, the importer now checks all three splits and writes a content-addressed `imports/africanvoice-preview-*.json` diagnostic receipt. It records per-split row/selection counts, missing or non-text fields, empty values, duplicates and truncation. Eligible transcript records, hashed speaker groups and the source card are retained so they can be inspected locally; raw API JSON, media links, demographic values and raw speaker IDs are never saved. This receipt is explicitly **not** a training corpus, and source-head changes invalidate the attempt. Identical diagnostic retries reuse the same receipt. No incomplete training sample is installed. Run `--rows 10` for the diagnostic attempt (8 train, 1 validation, 1 test) and inspect the receipt before further retries.

The real preview reader was also checked against 100 public `0xnu/igbo` rows (28,153 bytes). Current checks cover missing/non-text speaker IDs, empty/truncated previews, full split diagnostics, source revision changes, receipt preservation, and omission of media links/raw identifiers.

### Africanvoice text review result

Codex read all 10 imported records: 8,379 whitespace-delimited words across 6 hashed speaker groups (4 train, 1 validation, 1 test). The groups are disjoint across the official splits. These are long monologues with speech annotations, dialect/orthographic variation, code-switches and uncertain wording; they are not assistant conversations. Matching the text to the audio has not been tested.

The explicit decisions in `data/africanvoice_review_v1.jsonl` retain 3 AI-edited excerpts (142 words), excluding the other 7 full records from the current clean language-text set. The retained excerpts select exact source spans, remove only bracketed pause/filler annotations, collapse whitespace and apply NFC. Source span offsets/hashes and every correction/rejection reason are recorded. No unknown words, tone marks, medical terms or missing speech were guessed. This is a conservative, text-only AI assessment of this small preview, not a verdict on the full dataset or its dialects.

```sh
python3 scripts/igbo_lab.py review-sample research/igbo_lab/data/africanvoice_review_v1.jsonl --source africanvoice
```

This saves the separate receipt and edited text under `.build/IgboLab/curation/africanvoice/`, preserving the measured public pilot's files. Review now saves an audit even when fewer than 12 records survive; the preparation step retains that minimum and additionally requires accepted examples in every official transcript split with disjoint speakers. `prepare-reviewed --source africanvoice` currently refuses before replacing the active corpus: only 3 records remain, all from the train split. No new training or repeated frozen-benchmark evaluation was run. The existing two checkpoints and public experiment are preserved. Passing these structural checks in a future sample would permit preparation, not establish language quality or production readiness.

The Africanvoice review stage passed 53 automated checks, including source-specific review isolation, small/all-rejected audit reports, preparation refusal, speaker separation and baseline preservation. Edited excerpts are explicitly labelled language-text derivatives, never audio-verified ASR targets or human/native-speaker validation.

Only transcript text and hashed speaker grouping are retained; audio links and demographic fields are dropped. The current importer samples the source's official train/dev/dev_test splits with an 80/10/10-style allocation (80/10/10 records for the default 100-row request), retaining their assignments. The corpus loader refuses identical speaker groups crossing splits. This source's publisher prohibits TTS/voice-cloning/adaptation uses, so this importer is for the text/ASR side only. Transcript wording can be reviewed from text; matching a transcript against its recording is not verified without listening. Sources and observed terms are recorded in [SOURCES.md](SOURCES.md).

### Aya expansion and immutable combined corpus

The next source is [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset), observed Hub head `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`, declared Apache 2.0. The publisher describes human annotations and explicitly permits academic/commercial use. We select only `language_code=ibo` and `annotation_type=original-annotations`; re-annotations and other languages are excluded. Source cards and attribution are preserved. Annotator identifiers and demographic values are discarded.

The viewer's filter/search requests timed out, so acquisition uses a bounded mixed-language row scan and client-side selection. Three independent attempts, starting at rows 0, 2,000 and 4,000, each scanned at most 2,000 rows and stayed within their own 5 MiB response budget. They transferred 2,527,368 / 2,805,288 / 2,585,518 bytes and retained 13 / 9 / 6 eligible pairs respectively. Only 38,572 bytes of candidate JSONL were retained, plus cards/manifests. No full dataset, Parquet, audio or model weights were fetched.

Later batches use `--offset` and separate `imports/<source>/batches/offset-NNNNNNN/` directories. Existing samples are refused **before** token entry or network work. `review-sample` and `prepare-reviewed` accept the matching `--offset`. Transcript previews support offset zero only; a nonzero offset is refused rather than pretending the preview can paginate.

Codex explicitly reviewed all 28 Aya records. Fifteen edited pairs were retained and 13 excluded. Decisions are in `data/aya_igbo_review_0000000_v1.jsonl`, `data/aya_igbo_review_0002000_v1.jsonl` and `data/aya_igbo_review_0004000_v1.jsonl`, with original content hashes, correction/rejection reasons and supporting sources where factual verification was needed. Long or malformed answers were shortened; uncertain medical, biographical, climate and cultural claims were excluded. Retained riddle/proverb variants remain AI-reviewed source examples, not independently validated cultural references. All edited answers are labelled AI derivatives.

```sh
python3 scripts/igbo_lab.py assemble-reviewed public-igbo africanvoice aya-igbo aya-igbo@2000 aya-igbo@4000
```

This produces `.build/IgboLab/curation/assemblies/4c1418fc13cf7bf2/`: **58 records, 47 train / 4 validation / 7 test**. It combines 40 earlier public pairs, 3 Africanvoice excerpts and 15 Aya pairs. It preserves source attribution/restrictions and original group assignments. Normalized exact prompts shared across splits exclude lower-priority source groups; duplicate text/prompts/records are removed. Frozen evaluation prompts are excluded. Semantic paraphrases still need manual review.

Assembly requires matching review/source hashes and the same frozen evaluation hash. Its content-addressed output is immutable and transactionally installed; identical assembly calls reuse it. Cleanup removes only owned abandoned assembly staging, preserving completed assemblies, raw imports, reviews and checkpoints. The active token corpus and both measured checkpoints remain unchanged, and the frozen 20-case benchmark has not been rerun.

An audit with the existing 458-token vocabulary counts 2,711 train / 161 validation / 401 test tokens, including per-document EOS. Every conversation pair fits the 128-token context; one **training-only plain-text** excerpt is 285 tokens and will require bounded language-model windows rather than silent truncation. No token files were prepared and no new training was performed. `training_ready` in the assembly manifest means structural preparation checks passed; 58 records and AI review do not establish a usable assistant. More natural conversation diversity and speech evaluation remain necessary.

The expansion stage passed **65 lab checks** and **7 macOS resource checks**. Evidence is in `.build/IgboLab/corpus_expansion_verification.json` and `corpus_expansion_token_audit.json`. The following experiment subsequently prepared and trained this assembly; the expansion receipts remain historical evidence of the earlier stage.

### Registered expanded-corpus experiment

```sh
python3 scripts/igbo_lab.py run-experiment .build/IgboLab/curation/assemblies/4c1418fc13cf7bf2
```

This experiment is **complete**. Invoking the same assembly again returns the saved result without preparing tokens, training or evaluating again. Interrupted/failed runs are reported for inspection instead of retrained automatically. The active prepared corpus is now bound to this experiment; a standalone `train` call refuses to extend a completed run.

Before training, the command revalidated the immutable assembly, source/review hashes, baseline weights/optimizer/tokenizer and frozen cases. It recorded `experiments/expanded-4c1418fc13cf7bf2-v1/plan.json`: the same 865,152-parameter architecture and 458-unit tokenizer, **120 updates / 180-second cap / batch 2**, AdamW at 0.0005, weight decay 0.01, gradient clipping at 1.0, fixed seeds, and continuation from the preserved **previous** baseline at step 125. Source-code hashes bind the run to that implementation. The previous public candidate's scores were not used to choose additional steps or change the fixed questions.

The actual run completed 120 updates in **1.75 seconds** of measured training/evaluation-window time. Training reported **60.3 MiB MLX peak allocation** and **84.0 MiB peak process memory**; these overlap and must not be added. The prepared splits contain 2,711 / 161 / 401 tokens (47 / 4 / 7 records). The full longer plain-text excerpt remains in the memory-mapped token stream and is learned through bounded 128-token windows; it was not truncated. Every conversation pair fits that context.

On the expanded corpus, fixed-seed sampled-window validation loss fell **6.2803 → 3.6713** and test loss fell **6.4192 → 3.0453**. These are small-corpus continuation measurements, not full-corpus likelihood or speech accuracy. On the unchanged 20-case transcript benchmark, reference-reply loss rose **1.8097 → 3.0156**; lower is better. The new candidate scores below the earlier public-only candidate's 3.5948, but still regresses against the original baseline. Codex reviewed all 20 new generated replies: **0 complete suitable replies and 0 useful partial replies**. The baseline remains unsuitable as well. No voice quality or native-speaker validation was established.

`checkpoints/previous` preserves baseline model SHA `bcdd58169df662d7efb35bb4b9fa27bdeb31121b38474cbccaa63fd32b9c30b3`; `current` now contains expanded candidate `df03c2846bb6aea99dfd6a15aeb53440215726f8aef4f046e93cb0589ac973d3`, step 245. Only two completed checkpoint slots remain. The older public-only candidate's weights were retired by that policy; its reports and original reviewed corpus remain available, with the old metadata also captured in `before.json`.

The original `.build/IgboLab/evaluations/` reports remain unchanged as the first public pilot. New `evaluations/baseline.json`, `candidate.json`, `comparison.json`, `generation_review.json`, `prepared_corpus.json`, `run.json` and `result.json` live under the experiment directory. The baseline reference scores were reused only after verifying identical weights, tokenizer, architecture and case hashes. `status` reports the latest experiment directory. To inspect its comparison:

```sh
python3 scripts/igbo_lab.py compare --report-directory .build/IgboLab/experiments/expanded-4c1418fc13cf7bf2-v1/evaluations
```

All **72 lab checks** and **7 macOS resource checks** passed. This completed the registered run; no additional steps or parameter changes were made after seeing its result. `.build/IgboLab/expanded_experiment_verification.json` records hashes, resource usage and the AI generation review. The pipeline works locally, but this small research model is not suitable for live assistant replies. Broader reviewed language/conversation coverage is needed before planning a separate experiment. The iPhone voice flow remains separate from these research weights.

### Published-source expansion after the controlled run

The user selected published sources only; no new AI-authored practice conversations were added. The fixed-filter alias `aya-originals` avoids scanning unrelated languages. Every row is still checked locally for Igbo `original-annotations`; the filter has no unfiltered fallback. Requests fetch 50 matches each with a 512 KiB page cap and the existing shared 5 MiB body budget. A filter request may take 55 seconds and retry once on timeout; retry bytes/requests remain counted. Other endpoints retain their shorter timeout.

```sh
python3 scripts/igbo_lab.py import-text --source aya-originals --rows 500
python3 scripts/igbo_lab.py review-sample research/igbo_lab/data/aya_originals_published_review_v1.jsonl --source aya-originals
python3 scripts/igbo_lab.py assemble-reviewed aya-originals africanvoice
```

The first command has already completed: 500 unique pairs, 542 filtered matches processed, 344,116 response-body bytes, 15 requests and no audio. The viewer reported 707 matching rows. Existing imports cannot be overwritten. A future offset counts **filtered matches**, not original multilingual rows; the observed continuation position is 542, while each record retains its actual original `source_row_index`. Source ordering/revision stability must be rechecked and duplicate IDs removed when combining future batches.

The versioned review contains 114 decisions: 72 retained, 42 excluded and 386 still pending. Twenty-eight decisions reuse exact ID/content-hash matches from the earlier Aya reviews with explicit file/hash provenance; they are not new independent assessments. The new sample's original source card, text and hashes remain unchanged. Vocabulary and proverb fragments use `content_use: "language-reference"` with an explicit `edit`/`text`; they have no role markers. Actual source questions/replies use `assistant-pair`. The compiler refuses ambiguous conversions and role-marker injection, and applies frozen-prompt protection before converting references. Optional `family_group`/`family_reason` keep reviewed semantic variants together and preserve their original `source_group`; official transcript speaker/split groups cannot be replaced.

The new immutable assembly `55d5ff6b7e546915` contains 75 records: 29 source Q&A pairs and 46 plain references, including the three preserved Africanvoice excerpts. It has 1,000 whitespace-delimited words and 7,056 UTF-8 text bytes, split 58 train / 10 validation / 7 test. The older 58-record experiment, active token corpus, tokenizer, both checkpoints and frozen evaluation remain unchanged. A compact pending queue references the 386 unreviewed raw records under `.build/IgboLab/curation/aya-originals/pending_queue.jsonl`; pending records cannot enter an assembly.

All **81 lab checks** and **7 macOS resource check groups** passed. The lab occupies about 269 MiB with about 31.8 GiB free on the Mac. The evidence receipt is `.build/IgboLab/published_expansion_verification.json`. No model training, benchmark rerun, audio download, phone build or simulator action occurred in this data stage. `training_ready` in a curation manifest describes structural checks only. This fragment-heavy set lacks everyday dialogue and does not establish fluent or usable voice replies; acquire and review suitable published conversations before declaring another training experiment. See [SOURCES.md](SOURCES.md) for source terms and unimported leads.

### Everyday request-understanding data

The next published source is [INJONGO](https://aclanthology.org/2025.acl-long.464/): native-speaker elicited requests with intent labels. `injongo-igbo` imports only the `ibo` subset and eight everyday classes: alarms, calendar updates, calls, music, shopping-list updates, text messages, time and weather. These are user utterances, not conversations with assistant responses. Keep this source separate from reply-model training.

```sh
python3 scripts/igbo_lab.py import-text --source injongo-igbo --rows 160
python3 scripts/igbo_lab.py review-sample research/igbo_lab/data/injongo_igbo_requests_review_v1.jsonl --source injongo-igbo
python3 scripts/igbo_lab.py assemble-reviewed injongo-igbo
```

These commands have completed. The raw sample has 160 records: 14 train, 2 validation and 4 test examples per selected intent. It used 79,020 response-body bytes across 28 requests. The source's filtered endpoint failed; the successful importer uses explicit small `/rows` slices from the observed alphabetical class-block layout, verifying full split row counts, original row positions and every label before saving. It accepts 80–480 records in increments of 80, with no later-offset mode, automatic alternative subset or full-dataset fallback. Source layout, revision or license-label changes stop the import. Original samples are never overwritten. Slot fields, media links and demographic fields are not retained.

All 160 requests received explicit AI review. **155 remain**, with 5 excluded; 123 are unchanged and 32 have NFC/outer-space or specified quote repairs. Keep all original wording and per-record reasons. Review preserves the source intent and example ID alongside the official split and actual row index; source example IDs repeat across intents, so they must never be used alone as global identities. Training/validation/test now contain **111/16/28** records. The eight task classes occur in each split; the entire set is 1,417 whitespace words. It is a small convenience sample with a 28-request test set, not a reliable accuracy estimate by itself. Some publisher labels are broad topic categories rather than executable actions; real action routing still needs clarification and capability checks.

The immutable snapshot is `.build/IgboLab/curation/assemblies/90cbf2af9a0dfd46/`. It does not replace the 75-record assembly, tokenizer, active corpus or either checkpoint. `prepare-reviewed --source injongo-igbo` refuses to replace the reply corpus, and `run-experiment` refuses request-only corpora with no assistant reply targets before preparing tokens or registering training. A later intent-understanding model needs its own declared experiment and evaluation policy. No new responses, translations, audio or model weights were generated/downloaded; no model was trained or evaluated and no iPhone/simulator operation ran in this stage.

The Hub's Apache 2.0 label and the paper's CC BY 4.0 data statement are both preserved with author/source attribution. See [SOURCES.md](SOURCES.md), the saved publisher card, and `curation/injongo-igbo/ATTRIBUTION.txt`. All **89 lab checks** and **7 macOS resource groups** passed. Evidence and source screening decisions are in `.build/IgboLab/request_data_verification.json`. The next development component is local request understanding; everyday multi-turn replies and English/mixed speech still need suitable published examples and independent speech evaluation.

### First local intent prototype

`intent-prototype-v1` is a working **statistical request classifier**, separate from the neural next-word/reply model. It learns word/character TF-IDF weights and compares a speech transcript with the nearest training example per task. NFC normalization retains Igbo underdots and tone marks. It can propose one of eight request categories or return `unknown`; it generates no response and executes no phone action.

```sh
python3 scripts/igbo_lab.py intent-experiment
python3 scripts/igbo_lab.py classify-intent "<speech transcript>"
```

The experiment has already completed. Repeating the first command reads its recorded results, without fitting or evaluating again. The second rebuilds the small deterministic index in RAM from the frozen training snapshot and does not save the supplied transcript. There are **no additional persisted weights/checkpoints**; both existing neural checkpoints, their corpus/tokenizer and original experiments remain unchanged. `status` exposes the separate intent result pointer.

The declared plan, source hashes, query snapshot, calibration grid and results are under `.build/IgboLab/intent_experiments/intent-prototype-v1/`. The 155 published INJONGO requests contribute 111 train / 16 validation / 28 test examples. Explicit AI decisions in `data/intent_background_review_v1.jsonl` identify 29 existing published Aya questions outside those eight task categories, preserving their 20/5/4 source splits. Their original assistant answers are omitted from this classification snapshot, and no new questions/dialogues were authored. This gives **131 training / 21 validation / 32 test** examples with disjoint groups and unique normalized prompts. The frozen 20-case reply benchmark was not evaluated again.

Only training rows determine vocabulary, IDF and neighbours. Validation selects from the fixed similarity/margin grid using this predeclared order: fewest accepted errors, most correctly accepted requests, then lower thresholds. The selected gate is cosine similarity ≥0.15, margin ≥0.10 over both the next known category and background, and feature coverage ≥0.35. Exact ties abstain. These numbers are **not probabilities**. The 32-case test was then scored once; no model, feature or threshold change followed its results.

| Held-out outcome | Count |
| --- | ---: |
| Known request correctly accepted | 22 / 28 |
| Known request abstained | 5 / 28 |
| Known request assigned the wrong category | 1 / 28 |
| Background question correctly rejected | 3 / 4 |
| Background question incorrectly accepted | 1 / 4 |
| Overall correct, including background rejection | 25 / 32 (78.1%) |

Of the 24 accepted predictions, 22 were correct (91.7%). Before rejection, the nearest-known category matched 25/28 task requests; report the rejected cases too, rather than using that figure alone. The wrong known request was a timed call classified as an alarm; a geography question was classified as a shopping-list update. Message requests also had low coverage. These errors rule out automatic action routing. The CLI smoke check reproduced the timed-call error, with `action_executed: false` and `transcript_saved: false`.

This small run fitted in 0.0596 seconds, with about 31.9 MiB process peak memory. Core test scoring averaged 1.348 ms per query; that excludes index rebuilding, CLI startup, disk guards and speech processing. It derived 4,464 features and 21,015 sparse coefficients. The plan/snapshot/calibration/reports occupy 162,487 bytes, and the index is rebuilt when needed. The run enforces a 60-second / 512 MiB process budget plus the existing disk quota/reserve, with bounded feature/document/query counts. A failed/incomplete run is preserved and is not automatically retried. Changed implementation or saved-artifact hashes refuse inference under the old calibration.

All **99 software checks** and **7 macOS resource check groups** passed. The evidence receipt is `.build/IgboLab/intent_prototype_verification.json`. This establishes an inexpensive local prototype, not a reliable bilingual voice assistant: the four unknown test cases are cultural/factual questions, not representative unseen action types, and English/mixed speech and microphone/ASR input remain unvalidated. The next data work should cover published English requests, realistic mixed speech and unrepresented task requests; a later model needs a distinct plan and fresh evaluation, without retuning this run against its inspected test cases.

### First public-data result

The successful import transferred 573,140 bytes and yielded 66 distinct candidate pairs after scanning 2,000 rows. Codex kept 40: 14 unchanged, 26 edited; 26 were rejected. The prepared corpus contains 33 training, 3 validation and 4 test records. It is a small mixed/synthetic-source pilot, not a natural-conversation corpus.

One 120-step continuation took 7.02 seconds, with about 60.3 MiB MLX peak allocation and 75.6 MiB process peak memory. Source validation loss fell from 4.77 to 2.41, while the frozen conversation reference-reply loss **rose from 1.81 to 3.59**. Lower loss is better: this candidate regressed on that conversation test. Codex's separate review found no complete, suitable replies among the 20 generated candidate answers. The baseline also remains unsuitable, with mostly fragments/off-topic answers and two partial acknowledgments. These are AI assessments of a tiny transcript benchmark, not validated speech results.

The original baseline is retained in `checkpoints/previous`, the candidate in `checkpoints/current`. Atomic checkpoint saves preserve the pinned comparison baseline instead of retiring it at each periodic candidate save; still only two completed slots are kept. Generation can inspect either with `reply --checkpoint current` or `reply --checkpoint previous`. Both remain separate from the live iPhone assistant.

Evidence lives under `.build/IgboLab/`: `imports/public-igbo/manifest.json`, `curation/review_report.json`, `evaluations/baseline.json`, `evaluations/candidate.json`, `evaluations/comparison.json` and `evaluations/generation_review.json`. The expanded checks cover response limits, private token handling, review gates, evaluation isolation, transactional import failure, balanced transcript sampling, and baseline preservation. Test counts/results are recorded in the generated pilot report.

## Storage and memory

| Resource | Pilot policy |
| --- | --- |
| Entire generated lab, including Python environment | 2 GiB maximum, checked before operations and periodically during training |
| Free space remaining on the Mac | At least 20 GiB; refuse an operation that would breach the reserve |
| Imported JSONL text | At most 32 MiB, 20,000 records, 32 KiB of text per record |
| Vocabulary | 512–8,192 SentencePiece units, including byte fallback |
| Model | At most 5 million parameters; default 4 layers, width 128, 4 heads, 128-token context |
| Training batch | 1–4 sequences; default 2 |
| MLX allocation guideline | 2 GiB, with a 128 MiB reusable cache |
| Observed memory | Check MLX peak and process peak during training; stop if MLX peak exceeds 2 GiB or process peak exceeds 3 GiB |
| Run duration | At most 2,000 steps / 10 minutes per command; default 120 steps / 3 minutes |
| Checkpoints | Current and previous; preserve the comparison baseline when pinned, plus one bounded staging copy while saving |

MLX's memory-limit API is a **guideline**, not a hard operating-system limit on total process RAM. Small model/context/batch limits constrain expected allocation, and runtime measurements are reported separately. Initial evaluation and a running operation can finish before a stop is observed. Disk checks likewise do not reserve space against writes from another application; checks repeat during training and before saving.

Token files use memory mapping instead of loading a whole corpus into Python. The tokenizer is trained on the training split only, uses identity normalization after explicit NFC normalization, and preserves Igbo underdots and tone marks. Byte fallback represents unseen names and characters. Existing tokenizers are reused when importing additional data, keeping token IDs stable for continued training.

## Import your own text

Place a small UTF-8 JSONL file inside the Anyanwụ project. Each record needs:

```json
{"id":"story-001-part-01","group":"story-001","source":"our-reviewed-stories-v1","license":"your-actual-permission-or-license","text":"Your reviewed Igbo passage here.","review":{"type":"AI","reviewer":"actual-reviewer-name","decision":"accept","confidence":"high","reason":"Record the actual language/meaning checks performed."}}
```

Keep all pieces from a book, story, recording session or source document in the **same group**. A deterministic source/group hash assigns train, validation and test splits. An explicit `split` of `train`, `validation` or `test` is also accepted, but one group cannot span splits. Exact duplicate normalized records are removed globally before tokenization. Split long material into records under 32 KiB, preserving its original group. Training needs more than 128 tokens. Short validation/test splits are scored at their actual shorter context; text is never repeated within a sequence just to pad it. Real-text training requires accepted review metadata for every record; use `AI` or `human` to describe the actual review, not a claimed credential.

For conversational material, use the format below inside `text`:

```text
<|user|>Accurate Igbo, English or mixed-speech transcript.<|assistant|>Reviewed Igbo reply.<|end|>
```

The `source` and `license` fields record provenance; the program does not determine whether a source's terms allow training or commercial use. Review those permissions before importing real books, academic texts, speech transcripts or datasets. Do not import private user conversations by default. No app recordings or saved conversations are read by this lab.

```sh
python3 scripts/igbo_lab.py prepare research/my-reviewed-corpus.jsonl --vocabulary 2048 --rebuild-tokenizer
python3 scripts/igbo_lab.py train --steps 300 --seconds 180
python3 scripts/igbo_lab.py train --resume --steps 100 --seconds 120
```

`--resume` restores model weights and AdamW state from this lab's current checkpoint. It permits continued training on newly prepared text only when the tokenizer is unchanged. Random sampling restarts from a step-derived seed; resumed runs are not promised to be bit-for-bit equivalent to uninterrupted training. To experiment with a new tokenizer, explicitly prepare with `--rebuild-tokenizer` and start fresh training without `--resume`. The two checkpoint slots retain their own tokenizer copies and SHA256 digests.

The first real-corpus example deliberately rebuilds the tiny demo tokenizer. For later additions to that corpus, omit `--rebuild-tokenizer` to keep token IDs stable, then resume. Keep the original corpus files as source material; the prepared corpus is a replaceable generated snapshot.

SIGINT / SIGTERM request a graceful stop at the next step boundary and then save. A 50-step checkpoint interval and atomic directory promotion protect a previously completed checkpoint. An abrupt process kill can lose work since the last checkpoint. A failed or oversized preparation preserves the prior completed corpus and original inputs.

## Cleanup and evaluation

```sh
python3 scripts/igbo_lab.py clean
python3 scripts/igbo_lab.py status
```

`clean` removes only the explicit temporary/staging allowlist while no other lab command is active. It preserves original data, imported samples, review receipts, the prepared corpus, the Python environment, current/previous checkpoints, iPhone assets and all other projects. Checkpoint rotation retires only the lab's own generated checkpoints, while retaining a pinned comparison baseline.

`check` verifies quota/reserve refusal, Unicode marks, byte fallback, frozen token IDs, duplicate removal, source-group holdouts, path confinement, cleanup preservation and causal attention using a tiny real MLX model. Training reports initial/final validation loss, final test loss, elapsed time and observed memory. The small synthetic holdout is a pipeline check; it is not a real-world Igbo benchmark. Keep a separate real evaluation set untouched during development.

The next data milestone is a small, permission-reviewed collection of natural conversations and passages, with fluent-speaker evaluation. A trained text model remains one part of the voice pipeline: speech recognition supplies its input and an independently validated Igbo voice produces audio.

## First observed run

On this Apple M3 / 16 GB Mac, the initial 865,152-parameter demo completed 120 steps in 8.45 seconds. MLX reported about 60.3 MiB peak allocation; the process reported about 78.0 MiB peak resident memory. These overlapping measurements are not added together. The lab, including its isolated environment and two checkpoint slots, occupied about 268 MiB. A three-step resume restored the optimizer successfully, and a subsequent SIGTERM check saved a checkpoint with the `interrupted` reason and exactly two retained slots. All 12 automated checks passed. Raw evidence is generated in `.build/IgboLab/verification.json`.

The sample question “What is your name?” produced the incomplete text “Aha m ” after the short run. This demonstrates that generation loads the learned weights, and also that this tiny, unreviewed corpus has not produced a usable voice-assistant reply model. The measurements are a small smoke-run result, not a forecast for a larger model or real corpus.

Technical references: [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html), [MLX memory-limit semantics](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html), [SentencePiece](https://github.com/google/sentencepiece).


### Version-2 published request data (unscored)

The next data stage is complete, **without fitting another model**. All 368 new source records received explicit Codex AI decisions; 345 remain and 23 were excluded. No new practice dialogues, translations or assistant answers were authored. Review is not human/native-speaker validation.

| New source batch | Raw | Retained | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| INJONGO additional Igbo, same eight tasks | 160 | 152 | 108 | 16 | 28 |
| INJONGO other Igbo actions, local `unknown` | 80 | 72 | 51 | 6 | 15 |
| CLINC original English, eight tasks | 96 | 94 | 63 | 15 | 16 |
| Orinode scripted Igbo-English, local `unknown` | 32 | 27 | 27 | 0 | 0 |
| **Total** | **368** | **345** | **249** | **37** | **59** |

The source selections were declared in `.build/IgboLab/request_v2_selection_plan.json`, with a separate amendment for the discovered mixed-text source. Existing source samples and the initial plan were preserved. Aliases `injongo-igbo-v2` and `injongo-igbo-other` use the pinned INJONGO release. The extension starts at within-class train/validation/test positions 14/2/4, skipping the first sample. Other-action source labels remain intact: balance, flight/hotel booking, PIN changes, recipes, restaurant reservations, transfers and translation. Their local `unknown` mapping is explicit AI annotation, never a silently changed publisher label.

`clinc-english` reads the official numeric ClassLabel map, uses two-row class-position lookups and small row slices, and checks every label, original row index and split size. Larger filter queries intermittently returned HTTP 500; unsuccessful attempts installed nothing. The successful import transferred 143,866 body bytes over 37 requests. Derived English example IDs are explicitly marked local, rather than presented as publisher IDs. No slot labels are supplied by CLINC.

`orinode-mixed --rows 32 --offset 5000` is restricted to the first 32 numbered lines of the pinned Igbo file. Its successful import transferred 17,372 body bytes over five requests. The publisher claims hand-written fictional customer-service scenarios; individual authorship is not independently verified. Review stripped line numbers and excluded five rows with uncertain meaning, Hausa contamination or only isolated English loans. All 27 retained rows are **out-of-scope**, in **one train-only file group**. They are scripted text, not spontaneous speech. There are no supported mixed-action examples, independent mixed test cases, or English unknown cases in this expansion. A future evaluation must address those gaps; do not claim full bilingual/mixed-speech understanding from these counts.

New Igbo imports retain original-text slot metadata, explicit UTF-8 boundary status and a raw-text hash. Across the 224 retained Igbo records, 126 have nonempty mechanically valid spans, 32 have empty spans and 66 have unusable spans. Only **90 nonempty sets remain aligned with unchanged reviewed text**. All semantic labels remain unverified; even NFC/whitespace edits invalidate direct use of original offsets with edited text. No guessed realignment or fabricated English/mixed slots was added.

The immutable assembly is `.build/IgboLab/curation/assemblies/ca0b1ecfd890e4cf/`. The mapped request snapshot is `.build/IgboLab/request_data_snapshots/88f830477b4f9807/`, containing `dataset.jsonl`, the **59-case unscored** `test.jsonl`, and a manifest with hashes, coverage gaps and review policy. The test contains 43 Igbo and 16 English requests, preserving official source folds. Exact normalized duplicates across the new retained data, all v1 queries/raw samples and the frozen reply prompts were checked; none remained. Unique rows do not establish semantic independence or speaker generalization. The old v1 test set stays separate; do not move it into training.

All original source cards, import/review receipts, attribution notes and decisions accompany the assembly. See [SOURCES.md](SOURCES.md) for terms and collection limitations. `prepare-reviewed` rejects these request-only sources as reply targets; the mixed batch also has no standalone validation/test coverage. The active neural corpus, tokenizer, both checkpoint slots and all 95 protected earlier artifacts are unchanged. No fitting, calibration, test scoring, app build or phone/simulator action occurred. A later intent/details prototype needs its own declared plan; keep the completed v1 implementation and inference path frozen.

Verification: **105 lab checks and seven macOS resource groups passed**. The lab occupies about **272.9 MiB**, with **30.14 GiB free** at the final check. The mapped snapshot is about 621 KiB; raw new source JSONL totals about 503 KiB. `.build/IgboLab/request_v2_verification.json` records source hashes, preserved artifacts, resource use and coverage limits. No full dataset or model-weight downloads occurred.


### Version-2 intent and raw details

`request-understanding-v2` is now a completed **offline intent-and-details prototype**, separate from the existing neural reply model and the frozen v1 implementation. It keeps the v1 TF-IDF word/character features, nearest-neighbour algorithm and validation-only threshold-selection objective. Only the dataset and resulting calibration change. Rules in `details.py` extract tentative contact, time, date and geographic references from the transcript. These are raw text spans, not resolved contacts/clock values/locations or permission to act. The prototype does not generate Igbo replies or evaluate speech.

The experiment uses **222 training / 37 validation / 59 test** Igbo/English requests. The 27 scripted mixed-language examples are stored in `quarantine.jsonl` and excluded from all fitting, calibration and scoring: their all-unknown labels could teach the model to associate mixed language with rejection. Supported mixed tasks and English unknown rejection remain untested. Every source fold stays intact; none of the inspected v1 validation/test inputs enters the candidate training data.

Three explicit training-only spot checks found semantically wrong source slots despite valid UTF-8 boundaries: a name annotation selected a partial name, and a language annotation selected surrounding text instead of the language. `.build/IgboLab/request_source_slot_audit_v2.json` records those cases; it is not a full audit or an estimated error rate. **No publisher slot labels are used to train or score this experiment.** `data/request_details_v2_review.jsonl` instead records 96 explicit Codex AI reviews of source substrings: 37 validation requests with 23 target spans, and 59 test requests with 35 spans. Empty lists mean no target in the declared four-field scope, not no other entities. Original source text, including spelling and unresolved relations, is preserved. The same AI did development and reference review; this is not independent native-speaker annotation or an external blind benchmark.

The plan was registered before fitting. Training alone determines TF-IDF vocabulary, IDF and neighbours. Validation selected cosine similarity ≥0.45, positive margin (zero ties abstain), and feature coverage ≥0.35 using the unchanged minimum-accepted-error objective. On validation, 18/31 supported requests were correctly accepted, 13 abstained, and all six unknown requests rejected. Preparation was frozen before scoring the new test once. No feature, rule, label or threshold change followed those test results.

| Same fresh 59-case test | Frozen v1 | Version 2 |
| --- | ---: | ---: |
| Supported requests correctly accepted | 23 / 44 | 26 / 44 |
| Supported requests abstained | 19 / 44 | 18 / 44 |
| Supported requests assigned wrong task | 2 / 44 | 0 / 44 |
| Unsupported requests correctly rejected | 13 / 15 | 15 / 15 |
| Unsupported requests falsely accepted | 2 / 15 | 0 / 15 |
| Overall correct, including rejection | 36 / 59 (61.0%) | 41 / 59 (69.5%) |

All 26 candidate acceptances happened to be correct in this small sample; this does not establish perfect accuracy or robust unknown rejection. Its closed-set nearest-known label matched 37/44 supported tasks before the abstention gate, but 18 requests were rejected and must remain in the reported outcomes. English supported acceptance improved from 1/16 to **6/16**; ten English requests still abstained. Igbo supported acceptance fell from 22/28 to **20/28**, while unsupported Igbo rejection improved from 13/15 to 15/15. Overall Igbo correctness stayed 35/43. These results show a tradeoff, not uniformly better Igbo understanding.

For details, exact type/value/UTF-8 boundaries matched **25/35** reference spans when given the true intent, with four extra/incorrect spans (precision 86.2%, recall 71.4%, F1 78.1%). End to end, using predicted intents, **18/35** matched with two extra/incorrect spans (precision 90.0%, recall 51.4%, F1 65.5%). Only **12/28** test requests containing target spans had their complete scoped detail set recovered end to end. Do not use the many empty-reference records to imply stronger entity coverage. Locations, dayparts and non-clock temporal wording remain weak; message bodies, events, music references and shopping items have no parser yet.

Use the commands below to inspect the completed run or try an offline transcript. Repeating preparation/evaluation reads the recorded artifacts; it does not refit or score the test again. Failed/incomplete experiments are not retried automatically. Inference verifies implementation/artifact hashes and rebuilds the small index in RAM, saving no transcript and executing no action. It needs only Python's standard library; there are no new downloaded models or dependencies.

```sh
python3 scripts/igbo_lab.py request-experiment --phase prepare
python3 scripts/igbo_lab.py request-experiment --phase evaluate
python3 scripts/igbo_lab.py understand-request "Biko Deegara Benita ozi, sị ya kpọ m"
```

The last example is a published training utterance used for a CLI mechanics check. It produces a `text` candidate and the raw contact reference `Benita`; it is not held-out evidence. Every result is explicitly tentative, leaves entity/operation resolution false, and requires review. Byte offsets refer to NFC-normalized text and include a normalization/hash reference. Text inside a message payload is not turned into a scheduled call/alarm by the parser.

Plans, frozen data, span references, quarantine, calibration and results are under `.build/IgboLab/request_experiments/request-understanding-v2/`. The original data snapshot remains unchanged as historical preparation evidence; **its 59 test cases have now been spent once** in this experiment. The original 32-case v1 test, 20-case reply benchmark, active neural corpus/tokenizer, both neural checkpoint slots and all 124 protected prior artifacts remain unchanged. The separate pointer is `latest_request_experiment.json`; the v1 pointer remains v1. Do not retune this implementation against the inspected 59 cases; later revisions need separate modules, a distinct plan and fresh evaluation.

Fitting the index took 0.233 seconds, with 7,213 features / 35,986 derived coefficients. The measured test phase took 21.395 seconds including the frozen baseline and repeated resource checks; candidate scoring averaged 11.760 ms per request. These are this run's measurements, not a phone/ASR/CLI latency guarantee. Peak process memory was about 35.9 MiB and overlaps feature storage. No extra intent weights/checkpoints are persisted; the complete experiment artifacts occupy about 726 KiB. The app, backend providers, phones and simulators were not changed.

Final version-2 verification: **116 lab checks, seven macOS resource groups and three CLI mechanics cases passed**. The lab occupies **273.6 MiB**, with **29.98 GiB free** at the recorded check. Evidence is `.build/IgboLab/request_understanding_v2_verification.json`. Both neural checkpoint slots and all 124 protected older artifacts were hash-verified unchanged; no weights, full datasets or speech assets were downloaded.


### Version-3 fresh bilingual data

A new data stage adds **426 AI-reviewed published requests**, separate from the completed version-2 experiment. It improves English coverage and adds the English unknown examples that were absent before. All **448 originals** received explicit decisions: 426 retained, 22 excluded, none pending. No new request, translation, assistant answer or artificial language mixture was authored. This stage performs no model fitting, calibration or test scoring.

| New batch | Raw | Retained | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLINC English supported tasks | 192 | 185 | 126 | 28 | 31 |
| CLINC original OOS, reviewed as local unknown | 96 | 92 | 61 | 15 | 16 |
| INJONGO Igbo supported tasks | 80 | 75 | 52 | 8 | 15 |
| INJONGO Igbo other actions, local unknown | 80 | 74 | 50 | 8 | 16 |
| **Total** | **448** | **426** | **289** | **59** | **78** |

The source-selection plan was saved before import as `.build/IgboLab/request_v3_data_selection_plan.json`. English supported slices start at per-class train/validation/test positions 8/2/2, skipping the earlier raw sample. Verified class positions from the pinned release avoid repeating class-filter lookups. Igbo supported slices begin at 28/4/8 and other-action slices at 7/1/2. Every returned row still verifies its source label, original row position, official split and split size. New Igbo aliases omit publisher slot fields after the semantic problems found in version 2; no offset repair is guessed.

The initial English-negative plan selected eight other action classes. Repeated class-filter HTTP 500s and a later row HTTP 429 installed no sample. An explicit amendment, `request_v3_original_oos_amendment.json`, replaced that selection with **original CLINC OOS text** from pinned `plus` row slices: train offset 15000/length 64, validation 3000/16, test 4500/16. The importer verifies the official numeric ClassLabel map and requires every returned row to have source label `oos`; there is no class filter or full-dataset fallback in this path. The earlier selection plan and failed-attempt evidence remain preserved. This is a deliberate source-selection change, not evidence that the original eight negative categories were acquired in English.

Original source `oos` does not automatically mean unknown to Anyanwụ. Review excluded opening-time and implicit-call questions that would conflict with the broad task labels in the retained Igbo source. The English negatives include general questions, news/statistics, conditional bank warnings, inventory counts and unsupported device actions. Igbo negatives retain the eight banking/booking/recipe/translation publisher categories. Their negative domains differ across languages; later evaluation should report each language and task/rejection outcome separately. Typo/quotation repairs are explicit AI edits (46 retained records overall), not human-authored replacements. All review is Codex AI, not native-speaker validation or fact-checked answers.

The immutable assembly is `.build/IgboLab/curation/assemblies/ff63346dbe231b23/`. The mapped snapshot is `.build/IgboLab/request_data_snapshots/3c9229ae583c4506/`, containing `dataset.jsonl`, **78 unscored test cases**, and a manifest with source hashes, coverage and limits. There are 277 English and 149 Igbo records, with supported/unknown examples in both languages and every official fold. Test contains 47 English (31 supported / 16 unknown) and 31 Igbo (15 supported / 16 unknown). All eight supported classes occur in each overall fold. One near-paraphrase of an inspected v1 test was explicitly excluded. Comparison against 1,239 protected earlier raw/curated/benchmark text keys found no exact normalized overlap; none occurred inside the new accepted data either. Repeated source templates remain a limitation: unique rows are not proof of semantic independence.

A bounded search of primary published sources did **not** admit a new suitable mixed-command corpus. Gated AfriSwitch/AfriSwitchCare are noncommercial, evaluation-only speech benchmarks; other screened leads contained model preference/translation material, synthetic data or no Igbo. No access requests, author contacts, purchases or audio downloads occurred. The existing Orinode customer-service file remains a quarantined training group. One new INJONGO training request has an English calendar phrase in an Igbo question; this is insertional date wording, not evidence of full English-clause switching or spontaneous speech. Do not present this dataset as validated mixed-command coverage. Screening decisions and URLs are in `request_v3_mixed_source_screening.json` and [SOURCES.md](SOURCES.md).

Successful acquisition transferred 326,570 HTTP-body bytes across 98 counted requests, including bounded retries; each attempt enforced the existing 5 MiB shared body cap. The successful four raw JSONL files total about 493 KiB; the mapped snapshot, test copy and manifest total about 586 KiB. Failed attempts saved no incomplete corpus and never fetched media, full dataset text/Parquet files or pretrained weights. Source cards, attribution notes, original labels/splits/indices and per-record decisions remain available alongside each batch.

Useful commands (imports already exist and are preserved):

```sh
python3 scripts/igbo_lab.py status
python3 scripts/igbo_lab.py assemble-reviewed clinc-english-v3 clinc-original-oos injongo-igbo-v3 injongo-igbo-other-v3
```

`status` has a separate `latest_request_data` entry, not a replacement model. The existing fitted prototype and its implementation hashes remain version 2. All 135 protected earlier artifacts, the active neural corpus/tokenizer and both checkpoint slots are unchanged. There is no new detail-reference gold or new accuracy result in this stage. A later version-3 experiment needs a declared plan and fresh AI detail-span review if it changes the extractor; do not retune the frozen v2 rules using the spent 59-case test or connect this unscored data directly to phone actions.

Final version-3 data verification: **121 lab checks and seven macOS resource groups passed**, with the status data-pointer smoke check passing. The lab occupies **276.2 MiB**, with **26.02 GiB free** at the recorded check. `.build/IgboLab/request_v3_data_verification.json` records import/review hashes, source-selection amendments, resource use, coverage limits and all 135 protected artifacts verified unchanged. The latest fitted request model remains v2; the new 78-case data test is unscored.
