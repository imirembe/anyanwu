# Contributing to Anyanwụ

Contributions to Igbo speech, accessibility, iOS, evaluation and documentation
are welcome. Start with a GitHub issue for a substantial change. Describe the
problem, proposed behavior and how it can be tested.

## Development

Use the setup and checks in the README. Keep generated files under `.build/`.
Choose an existing simulator or a physical device in Xcode; do not create,
erase or shut down someone else's simulator as part of a script or test.

Keep speech recognition, response generation, tool execution and speech output
separate. Clearly label prepared replies and model failures. Preserve Igbo
underdots and tone marks, Dynamic Type, and the distinction between interface
language and response language.

## Changes and verification

Keep pull requests focused and explain the behavior change, checks run and
remaining limitations. Use mocked providers for API checks. Record actual
results rather than claiming fluency, accent quality or task success from
software tests alone.

Igbo spelling, dialect and delivery feedback is valuable. Explain the target
variety, text, listening conditions and the words or timing that need work.
Do not describe AI review as native-speaker validation.

## Data and credentials

Never commit API keys, tokens, recordings, private transcripts, model weights,
checkpoints, personal Google Drive links or reviewer records. Private research
notebooks and device-specific helpers are intentionally excluded. Do not add
user conversations to training data automatically.

For a proposed dataset or model, link its original source, exact version,
license, relevant speaker permissions and limitations. Research access does
not imply redistribution or synthesis permission. See THIRD_PARTY_NOTICES.md.

Original software contributions are made under the repository's MIT license.
Keep attribution and separate licenses for third-party material.
