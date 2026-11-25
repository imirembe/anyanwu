# Anyanwụ

An open-source Igbo voice-assistant project by IKENAA LTD. It includes a native
iPhone app, a small Python gateway, a browser voice trial, and separate speech
and language research tools.

The intended conversation policy is simple: speak Igbo, English, or both;
Anyanwụ replies in Igbo. Interface text can be switched between Igbo and English.
Model support and speech quality vary. This is an experimental project, not a
production assistant or a guarantee of fluent speech.

## What is included

- SwiftUI app for iOS 18+, with voice capture, conversations, history controls,
  Igbo-friendly Noto fonts, and Siri/App Shortcuts entry points.
- Optional on-device speech recognition with pinned, verified model downloads
  to a physical iPhone. Models are not bundled with the app or repository.
- Stateless gateway with separate speech-recognition, response and speech-output
  providers. Prepared replies remain clearly labelled as previews.
- Isolated OpenAI browser voice trial with server-side credentials, Igbo reply
  instructions, web research, and source links.
- Research tooling for bounded acquisition, transcript review, evaluation,
  resource limits, cloud experiments and reproducible provenance.

Bookings, purchases, messaging and general phone control are not connected.
An App Shortcut opens Anyanwụ; it does not add Igbo support to Siri itself.
The on-device route does not supply a validated Igbo speaking voice or reply model.

## Run the iPhone app

Use macOS with Xcode supporting your selected iOS SDK. Open
`Anyanwu.xcodeproj`, select the `Anyanwu` scheme and an existing device, then Run.
The project uses Xcode filesystem-synchronised groups; Xcode 16+ is required.

For a physical iPhone, copy `Configuration/Local.xcconfig.example` to
`Configuration/Local.xcconfig` and set your Apple Developer team. Choose a bundle
identifier available to your team in Xcode. The local file is ignored by Git.
Simulator builds do not require a development team.

To regenerate the checked-in Xcode project:

```sh
python3 scripts/create_project.py
```

Use the app's Settings to select its on-device or service route. Apple speech
capabilities are checked at runtime; prepared text is not presented as generated
intelligence. Real Igbo/English/mixed-speech performance needs device testing.

## Run the development gateway

The basic gateway uses the Python standard library and downloads no models:

```sh
cp backend/.env.example backend/.env
python3 backend/server.py
```

It binds to `127.0.0.1:8787` by default. Configure your own ASR, text and TTS
endpoints in the ignored `.env` file. Without a reply model, responses remain
labelled previews. A recording request requires an ASR provider.

See [the API contract](docs/API.md) and
[the optional cloud backend](backend/cloud/README.md). GPU images and model
downloads belong on a separately configured cloud host.

## Try the browser voice interface

```sh
python3 scripts/run_openai_voice_test.py --preview --port 8789
```

Preview opens a key-free interface with voice disabled. To use the hosted trial:

```sh
python3 scripts/run_openai_voice_test.py
```

Enter your own OpenAI API key at the private Terminal prompt. On macOS the
launcher stores it in Anyanwụ's dedicated Keychain entry and reuses it on later
starts. It never embeds the key in the webpage. `--no-save-key`, `--replace-key`
and `--forget-key` control that entry. The launcher installs its small pinned
dependencies under `.build/`; it does not install speech weights.

Open the localhost address printed by the launcher. The trial has a two-minute
session timer and explicit Stop controls. API access and usage billing are
separate from this project's license and from any ChatGPT subscription.

## Tests

Portable backend checks use explicit provider fixtures. They do not contact paid
APIs, download model weights, or establish Igbo speech quality.

```sh
python3 -m venv .build/backend-venv
.build/backend-venv/bin/python -m pip install -r backend/requirements-test.txt
mkdir -p .build
.build/backend-venv/bin/python -B -m unittest discover -s backend -v
node --test backend/voice_test/controller.test.mjs backend/voice_test/research.test.mjs
python3 scripts/check_public_release.py
```

On macOS, `python3 scripts/check_resources.py` checks storage, context and
transport behavior without opening a simulator. `scripts/check_apple.py` checks
the Apple speech infrastructure using a small tokenizer and synthetic fixtures;
it does not download model weights. Historical research tests can require
locally prepared inputs that are intentionally not distributed here.

## Research and data

[The text lab](research/igbo_lab/README.md) and `backend/voice_research/` contain
experimental tools. The tiny bundled text fixtures are AI-written/reviewed
smoke material, not native-speaker gold or a fluent assistant model.

Speech recordings, third-party corpora, trained adapters, checkpoints, individual
reviewer feedback, account-specific notebooks and private Drive links are not
part of this software release. Generated artifacts belong under `.build/`.
Reproducing an experiment requires its licensed inputs and a separately
registered plan. Keep research outputs separate from the app's live defaults.

## Privacy and storage

The native app uses paged SQLite conversation history and bounded temporary
audio. Voice recordings are temporary; app conversations do not automatically
become training data. Service mode sends the relevant input and recent context
to the configured gateway. The isolated browser trial keeps captions and
research cards in bounded memory and does not save recordings or transcripts.

Provider data practices apply when using an external service. Keep development
servers on loopback; see [SECURITY.md](SECURITY.md) before deployment.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and language-review guidance.
Original software is released under the [MIT license](LICENSE).
[Third-party notices](THIRD_PARTY_NOTICES.md) describe separately licensed fonts,
external models, research sources and synthetic fixtures.
