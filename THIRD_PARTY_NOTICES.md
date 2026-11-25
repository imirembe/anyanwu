# Third-party notices

The root MIT license covers original Anyanwụ software and documentation.
It does not relicense third-party assets, libraries, model weights, datasets,
or generated research packages. Preserve their notices when using them.

## Bundled fonts

Noto Sans and Noto Serif are distributed under the SIL Open Font License 1.1.
Their copyright and license texts are included in
`Sources/Anyanwu/Resources/Fonts/NotoSans-OFL.txt` and
`Sources/Anyanwu/Resources/Fonts/NotoSerif-OFL.txt`.
`FontSources.json` records source URLs, sizes and SHA256 hashes.

## Optional on-device speech model

Meta Omnilingual ASR and the Soniqo/aufklarer Core ML conversion are external
model assets. They are not bundled in this repository. The app includes the
publisher's Apache 2.0 license text and attribution in
`Sources/Anyanwu/Resources/AppleSpeech/`. Downloads are pinned and verified by
the app; the software license does not establish speech quality.

## Optional cloud research dependencies

Research code refers to external models and source implementations. In
particular, generated CosyVoice inference packages incorporate code from
[QwenAudio/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) under Apache 2.0
and [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) under MIT.
The runtime builder records upstream commits, source hashes, changes and
license texts in the generated research archive. These generated packages
and their embedded upstream implementations are not part of this release.

The Dynamic Accent Flash checkpoint is separately licensed CC BY-SA 4.0.
The WAXAL Media Trust Igbo TTS subset and Silencio conversational speech have
their own attribution and usage terms. The research modules retain source
checks and provenance; no model weights, speech recordings, transcripts of
private conversations, datasets, adapters or reviewer feedback are distributed
by this repository. A model's license is separate from its code and data rights.

OpenAI and other hosted services are optional external services. Their API
access, billing and data policies are separate from this repository's license.

## Synthetic text fixtures

`research/igbo_lab/data/demo.jsonl` retains its existing CC0-1.0 declaration.
It contains AI-written smoke examples, not human speech or verified fluent
Igbo. `voice_eval_v1.jsonl` contains AI-reviewed text fixtures covered by the
root software license. Neither fixture is a native-speaker speech benchmark.

Python and JavaScript dependencies are installed separately and retain their
upstream licenses. See the requirements files for the pinned packages.
