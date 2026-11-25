#!/usr/bin/env python3
"""Check native Keychain with a disposable fake entry, never the real API key."""
import secrets
import sys
import uuid

sys.dont_write_bytecode = True
import voice_credentials as credentials


def main():
    # A fresh namespace ensures the real app credential and other items are untouched.
    credentials.SERVICE += ".test." + uuid.uuid4().hex
    credentials.ACCOUNT = "disposable-fixture"
    store = credentials.MacKeychain()
    created = False
    try:
        if store.get() is not None:
            raise RuntimeError("Unexpected fixture collision; no changes made.")
        value = "FAKE_TEST_ONLY_" + secrets.token_hex(16)
        store.set(value)
        created = True
        if store.get() != value:
            raise RuntimeError("Keychain round trip failed.")
        value = "FAKE_REPLACEMENT_" + secrets.token_hex(16)
        store.set(value)
        if store.get() != value:
            raise RuntimeError("Keychain replacement failed.")
        store.delete()
        created = False
        if store.get() is not None:
            raise RuntimeError("Keychain deletion did not complete.")
        print("Native Keychain save/read/replace/remove checks passed. Disposable entry removed; no real API key was accessed.")
    finally:
        if created:
            store.delete()


if __name__ == "__main__":
    try:
        main()
    except credentials.CredentialError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
