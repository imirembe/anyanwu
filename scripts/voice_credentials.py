"""Anyanwụ's OpenAI credential: explicit macOS Keychain backend, never a file."""
from __future__ import annotations

import getpass
import sys
import warnings

SERVICE = "com.ikenaa.anyanwu.voice-trial"
ACCOUNT = "openai-api-key"


class CredentialError(Exception):
    """A safe, user-facing error that never contains the credential."""


class MacKeychain:
    def __init__(self):
        if sys.platform != "darwin":
            raise CredentialError("Saved keys require macOS Keychain. Use --no-save-key for a temporary key on this system.")
        try:
            # Explicit selection avoids plugin discovery and plaintext/fallback backends.
            from keyring.backends.macOS import Keyring
            if Keyring.priority <= 0:
                raise RuntimeError("Native Keychain API unavailable")
            self.backend = Keyring()
            self.backend.keychain = None
        except Exception:
            raise CredentialError("macOS Keychain support is unavailable. Rerun the launcher to install its pinned dependencies, or use --no-save-key.") from None

    def get(self) -> str | None:
        try:
            return self.backend.get_password(SERVICE, ACCOUNT)
        except Exception:
            raise CredentialError("Could not access Anyanwụ's saved key. Unlock or allow access to macOS Keychain, then retry; --no-save-key uses a temporary key.") from None

    def set(self, key: str) -> None:
        try:
            self.backend.set_password(SERVICE, ACCOUNT, key)
        except Exception:
            raise CredentialError("The key could not be saved to macOS Keychain. Unlock or allow Keychain access and retry; no plaintext file was written.") from None

    def delete(self) -> None:
        try:
            self.backend.delete_password(SERVICE, ACCOUNT)
        except Exception:
            raise CredentialError("The stored key could not be removed from macOS Keychain. Unlock or allow access and retry.") from None


def validate_key(value: str) -> str:
    if not isinstance(value, str):
        raise CredentialError("The stored key is invalid. Run with --replace-key to replace it privately.")
    value = value.strip()
    if not value or len(value) > 8192 or any(c.isspace() for c in value):
        raise CredentialError("Enter only a nonempty API key without spaces; use --replace-key if the stored key is invalid.")
    return value


def private_prompt(*, saving: bool) -> str:
    if not sys.stdin.isatty():
        raise CredentialError("Run this command in your own Terminal to enter the key privately. Later launches can reuse the saved Keychain entry.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            label = "OpenAI API key (hidden; saved to macOS Keychain): " if saving else "OpenAI API key (hidden; this run only): "
            return validate_key(getpass.getpass(label))
        except getpass.GetPassWarning:
            raise CredentialError("This terminal cannot hide input. Use a normal macOS Terminal window.") from None


def load_key(*, environment, replace=False, temporary=False, store_factory=MacKeychain, prompt=private_prompt):
    """Return a key/source without logging it. Environment overrides stay temporary."""
    # Remove it before starting further child processes; never persist an implicit
    # environment override or replace an existing saved key with it.
    env_key = environment.pop("OPENAI_API_KEY", "")
    if env_key and not replace:
        return validate_key(env_key), "environment"
    if temporary:
        return validate_key(prompt(saving=False)), "temporary"
    store = store_factory()
    if not replace:
        key = store.get()
        if key is not None:
            return validate_key(key), "keychain"
    key = validate_key(prompt(saving=True))
    store.set(key)
    return key, "saved"


def forget_key(store_factory=MacKeychain) -> bool:
    store = store_factory()
    if store.get() is None:
        return False
    store.delete()
    return True
