# Gateway contract

`GET /health` returns:

```json
{
  "service": "anyanwu",
  "version": "0.1.0",
  "capabilities": { "text": false, "transcription": false, "speech": false }
}
```

`POST /v1/turns` accepts:

```json
{
  "conversationId": "a-client-generated-uuid",
  "inputLanguages": ["ig", "en"],
  "responseLanguage": "ig",
  "interfaceLanguage": "ig",
  "input": { "type": "text", "text": "Ndewo" },
  "messages": []
}
```

For recordings, `input` is `{ "type": "audio", "audioBase64": "…", "mimeType": "audio/mp4" }`. Input languages are `ig` and `en`; `responseLanguage` is always `ig`. `interfaceLanguage` can be `ig` or `en` and only changes service notices. The legacy `language` hints `ig` and `ig-en` are accepted for old clients but no longer allow bilingual replies. Optional context contains up to 12 `{ "role": "user|assistant", "content": "…" }` messages, each at most 8,000 characters, with a combined 24,000-character limit. The gateway is stateless; context is supplied with each request.

Successful response:

```json
{
  "transcript": null,
  "reply": "Ndewo! Kedu ka ị mere?",
  "mode": "preview",
  "responseLanguage": "ig",
  "audioBase64": null,
  "audioMimeType": null,
  "notice": null
}
```

`mode` must be `preview` for prepared replies or `live` for actual model output. Audio requests without a configured ASR provider return HTTP 503. Errors use `{ "error": "A user-readable message" }` with an appropriate HTTP status. Audio is limited to 6 MB; text messages to 8,000 characters.
