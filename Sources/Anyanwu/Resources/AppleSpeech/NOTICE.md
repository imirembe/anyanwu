# On-device speech recognition

Anyanwụ's optional speech download uses Meta Omnilingual ASR CTC-300M,
converted to Core ML with INT8-palettized weights by aufklarer/Soniqo.

Upstream: https://github.com/facebookresearch/omnilingual-asr
Conversion: https://huggingface.co/aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s
Pinned revision: ce99f14bbc768d8dc6ffd3235a21cd9751ceb139
Publisher's declared licence: Apache License 2.0, inherited from upstream.
The licence text is included in Apache-2.0.txt.

Model weights and training datasets are not bundled with this app. A user may
download the pinned speech-recognition package directly to a physical iPhone.
Each downloaded file has a fixed size and SHA256 digest verified by the app.
This is a community conversion; Igbo, English, code-switching, dialect quality,
device latency, memory use and battery impact require evaluation on real phones.

This recognizer does not provide an Igbo voice or an Igbo conversation model.
Prepared replies remain labelled Preview. Apple speech synthesis is used only
when the device explicitly exposes an Igbo voice.
