# Security

Please do not post credentials, private audio or personal transcripts in
public issues. Use GitHub's private vulnerability reporting feature when it
is available for this repository. Otherwise, report the general issue without
private details and request a private follow-up channel.

The gateway and browser voice trial are development services. Keep their
default loopback binding. A public deployment needs HTTPS, per-user
authentication, rate limits and a review of its configured providers.

Store API credentials in the operating system's credential store or an
appropriate deployment secret store. Keep local `.env` and signing files
outside version control. If a credential is exposed, revoke it at its issuer;
removing a file from Git history does not revoke a credential.

Keep changes to audio upload limits, provider URL validation, origin checks,
temporary-file cleanup and credential handling covered by meaningful tests.
