"""No real keys, Keychain entries, server sockets or paid API calls."""
from contextlib import redirect_stdout, redirect_stderr
import io
import sys
import types
import unittest
from unittest.mock import Mock, patch

import run_openai_voice_test as launcher
import voice_credentials as credentials

FAKE = "FAKE_TEST_CREDENTIAL_123"


class Store:
    def __init__(self, value=None):
        self.value = value
        self.sets = []

    def get(self): return self.value
    def set(self, key): self.value = key; self.sets.append(key)
    def delete(self): self.value = None


class CredentialsTests(unittest.TestCase):
    def test_first_launch_saves_once_and_next_launch_does_not_prompt(self):
        store, prompt = Store(), Mock(return_value=FAKE)
        result = credentials.load_key(environment={}, store_factory=lambda: store, prompt=prompt)
        self.assertEqual(result, (FAKE, "saved"))
        self.assertEqual(store.sets, [FAKE])
        result = credentials.load_key(environment={}, store_factory=lambda: store, prompt=prompt)
        self.assertEqual(result, (FAKE, "keychain"))
        prompt.assert_called_once_with(saving=True)
        self.assertEqual(store.sets, [FAKE])

    def test_environment_override_is_not_saved_or_sent_to_keychain(self):
        factory = Mock(side_effect=AssertionError("Keychain must not be touched"))
        environment = {"OPENAI_API_KEY": FAKE}
        self.assertEqual(credentials.load_key(environment=environment, store_factory=factory), (FAKE, "environment"))
        self.assertNotIn("OPENAI_API_KEY", environment)
        factory.assert_not_called()

    def test_temporary_mode_never_reads_or_writes_keychain(self):
        factory, prompt = Mock(side_effect=AssertionError()), Mock(return_value=FAKE)
        self.assertEqual(credentials.load_key(environment={}, temporary=True, store_factory=factory, prompt=prompt), (FAKE, "temporary"))
        factory.assert_not_called(); prompt.assert_called_once_with(saving=False)

    def test_replace_uses_new_private_input_and_cancellation_preserves_old_key(self):
        store = Store(FAKE)
        with self.assertRaises(KeyboardInterrupt):
            credentials.load_key(environment={}, replace=True, store_factory=lambda: store, prompt=Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(store.value, FAKE)
        result = credentials.load_key(environment={"OPENAI_API_KEY": "env_override"}, replace=True,
                                      store_factory=lambda: store, prompt=Mock(return_value="NEW_FAKE_VALUE"))
        self.assertEqual(result, ("NEW_FAKE_VALUE", "saved"))
        self.assertEqual(store.sets, ["NEW_FAKE_VALUE"])

    def test_invalid_input_is_not_saved_and_does_not_appear_in_errors(self):
        store = Store(FAKE)
        for value in ("", "secret with spaces", "x" * 9000):
            with self.assertRaises(credentials.CredentialError) as error:
                credentials.load_key(environment={}, replace=True, store_factory=lambda: store, prompt=Mock(return_value=value))
            if value: self.assertNotIn(value, str(error.exception))
        self.assertEqual(store.sets, [])

    def test_forget_removes_only_the_dedicated_entry_and_handles_absence(self):
        store = Store(FAKE)
        self.assertTrue(credentials.forget_key(lambda: store))
        self.assertFalse(credentials.forget_key(lambda: store))

    def test_adapter_uses_explicit_macos_backend_and_dedicated_namespace(self):
        backend = Mock()
        module = types.ModuleType("keyring.backends.macOS")
        module.Keyring = Mock(return_value=backend)
        module.Keyring.priority = 5
        with patch.dict(sys.modules, {"keyring.backends.macOS": module}), patch.object(credentials.sys, "platform", "darwin"):
            store = credentials.MacKeychain()
        store.get(); store.set(FAKE); store.delete()
        backend.get_password.assert_called_once_with(credentials.SERVICE, credentials.ACCOUNT)
        backend.set_password.assert_called_once_with(credentials.SERVICE, credentials.ACCOUNT, FAKE)
        backend.delete_password.assert_called_once_with(credentials.SERVICE, credentials.ACCOUNT)
        self.assertIsNone(backend.keychain)

    def test_backend_failure_is_sanitized_and_does_not_prompt_or_fallback(self):
        store = credentials.MacKeychain.__new__(credentials.MacKeychain)
        store.backend = Mock()
        store.backend.get_password.side_effect = RuntimeError(FAKE)
        prompt = Mock()
        with self.assertRaises(credentials.CredentialError) as error:
            credentials.load_key(environment={}, store_factory=lambda: store, prompt=prompt)
        self.assertNotIn(FAKE, str(error.exception)); prompt.assert_not_called()
        store.backend.set_password.side_effect = RuntimeError(FAKE)
        with self.assertRaises(credentials.CredentialError) as error: store.set(FAKE)
        self.assertNotIn(FAKE, str(error.exception))

    def test_prompt_refuses_echoing_fallback_or_noninteractive_first_setup(self):
        with patch.object(credentials.sys.stdin, "isatty", return_value=False):
            with self.assertRaises(credentials.CredentialError): credentials.private_prompt(saving=True)
        with patch.object(credentials.sys.stdin, "isatty", return_value=True), \
             patch.object(credentials.getpass, "getpass", side_effect=credentials.getpass.GetPassWarning("cannot hide")):
            with self.assertRaises(credentials.CredentialError): credentials.private_prompt(saving=True)


class LauncherTests(unittest.TestCase):
    def test_installed_dependencies_skip_pip_and_network(self):
        with patch.object(launcher.sys, "prefix", str(launcher.VENV)), \
             patch.object(launcher, "dependencies_ready", return_value=True), \
             patch.object(launcher.subprocess, "run") as run:
            launcher.prepare_environment([])
        run.assert_not_called()

    def test_missing_dependency_installs_without_inheriting_key(self):
        with patch.object(launcher.sys, "prefix", str(launcher.VENV)), \
             patch.object(launcher, "dependencies_ready", return_value=False), \
             patch.dict(launcher.os.environ, {"OPENAI_API_KEY": FAKE}), \
             patch.object(launcher.subprocess, "run") as run:
            launcher.prepare_environment([])
        args, kwargs = run.call_args
        self.assertNotIn(FAKE, " ".join(args[0]))
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])

    def test_reexec_preserves_flags_without_adding_secret_arguments(self):
        with patch.object(launcher.sys, "prefix", "/fake/system-python"), \
             patch.object(launcher.Path, "exists", return_value=True), \
             patch.object(launcher.os, "execv") as execute:
            launcher.prepare_environment(["--replace-key", "--port", "8799"])
        self.assertEqual(execute.call_args.args[1][-3:], ["--replace-key", "--port", "8799"])
        self.assertNotIn(FAKE, str(execute.call_args))

    def test_preview_never_accesses_keychain_or_dependencies(self):
        fake_server = types.ModuleType("openai_voice_test"); fake_server.serve = Mock()
        with patch.dict(sys.modules, {"openai_voice_test": fake_server}), \
             patch.object(launcher.sys, "argv", ["run_openai_voice_test.py", "--preview"]), \
             patch.object(launcher, "prepare_environment") as prepare, patch.object(launcher, "load_key") as load:
            launcher.main()
        prepare.assert_not_called(); load.assert_not_called(); fake_server.serve.assert_called_once_with("", 8788)

    def test_launcher_uses_saved_key_without_printing_it(self):
        fake_server = types.ModuleType("openai_voice_test"); fake_server.serve = Mock()
        output = io.StringIO()
        with patch.dict(sys.modules, {"openai_voice_test": fake_server}), \
             patch.object(launcher.sys, "argv", ["run_openai_voice_test.py"]), \
             patch.object(launcher, "prepare_environment"), \
             patch.object(launcher, "load_key", return_value=(FAKE, "keychain")), redirect_stdout(output):
            launcher.main()
        self.assertNotIn(FAKE, output.getvalue())
        self.assertIn("saved API key", output.getvalue())
        fake_server.serve.assert_called_once_with(FAKE, 8788)

    def test_forget_command_exits_without_starting_server(self):
        fake_server = types.ModuleType("openai_voice_test"); fake_server.serve = Mock()
        with patch.dict(sys.modules, {"openai_voice_test": fake_server}), \
             patch.object(launcher.sys, "argv", ["run_openai_voice_test.py", "--forget-key"]), \
             patch.object(launcher, "prepare_environment"), patch.object(launcher, "forget_key", return_value=True), \
             patch.object(launcher, "load_key") as load, redirect_stdout(io.StringIO()):
            launcher.main()
        load.assert_not_called(); fake_server.serve.assert_not_called()


if __name__ == "__main__": unittest.main()
