import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from openai_voice_test import OpenAITransport, Trial, TrialError, session_payload
from test_openai_voice_test import Transport, REQUEST_ID, eventually
from voice_profiles import VoiceProfiles, VOICE_DIRECTION, VOICE_DESIGN, COMPARISON_TEXT, design_payload


class VoiceTests(unittest.TestCase):
    def test_same_model_and_web_tools_with_only_voice_direction_changed(self):
        profiles = VoiceProfiles()
        original = session_payload("v=0", *profiles.select("marin-original"))
        owerri = session_payload("v=0", *profiles.select("owerri-marin"))
        self.assertEqual(original["session"]["audio"], owerri["session"]["audio"])
        self.assertNotIn(VOICE_DIRECTION, original["session"]["instructions"])
        self.assertIn(VOICE_DIRECTION, owerri["session"]["instructions"])
        self.assertEqual(original["session"]["delegation"], owerri["session"]["delegation"])
        self.assertEqual(design_payload()["script_hint"], COMPARISON_TEXT)
        self.assertNotIn("audio_sample", design_payload())

    def test_metadata_roundtrip_contains_only_public_voice_design_fields(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".build") as temp:
            path = Path(temp) / "voice.json"
            profiles = VoiceProfiles(path)
            self.assertTrue(profiles.install("voice_example123"))
            record = json.loads(path.read_text())
            self.assertEqual(set(record), {"id", "type", "prompt", "script_hint"})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            restored = VoiceProfiles(path)
            self.assertTrue(restored.persisted)
            self.assertEqual(restored.select("owerri-designed"), ({"id": "voice_example123"}, VOICE_DESIGN + " " + VOICE_DIRECTION))
            self.assertEqual(len(restored.catalog()), 3)
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_invalid_or_oversized_metadata_and_voice_ids_are_not_used(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".build") as temp:
            path = Path(temp) / "voice.json"
            for value in ("x" * 9000, "not json", json.dumps({"id": "../../secret", "type": "prompt", "prompt": VOICE_DESIGN}),
                          json.dumps({"id": "voice_example", "type": "audio_sample", "prompt": VOICE_DESIGN})):
                path.write_text(value)
                self.assertIsNone(VoiceProfiles(path).voice_id)
        profiles = VoiceProfiles()
        with self.assertRaises(ValueError): profiles.select("voice_external")
        with self.assertRaises(ValueError): profiles.install("invalid_id")

    def test_invalid_voice_cannot_create_a_session(self):
        transport = Transport()
        trial = Trial(transport)
        with self.assertRaises(TrialError): trial.create(REQUEST_ID, "v=0", "voice_unregistered")
        self.assertEqual(transport.creations, 0)
        self.assertEqual(trial.attempts, 0)
        trial.create(REQUEST_ID, "v=0")
        self.assertEqual(transport.voice, "marin")
        self.assertIn("Owere-Igbo", transport.direction)
        with self.assertRaises(TrialError): trial.design_voice()
        trial.command(REQUEST_ID, "stop"); eventually(lambda: trial.call.confirmed)

    def test_creation_occurs_once_and_reuses_the_saved_voice(self):
        transport = Transport()
        calls = []
        transport.design_voice = lambda: calls.append(1) or "voice_example"
        trial = Trial(transport)
        first = trial.design_voice()
        self.assertFalse(first["metadata_saved"])
        self.assertEqual(first["voices"][-1]["id"], "owerri-designed")
        trial.design_voice()
        self.assertEqual(calls, [1])
        trial.create(REQUEST_ID, "v=0", "owerri-designed")
        self.assertEqual(transport.voice, {"id": "voice_example"})
        trial.command(REQUEST_ID, "stop"); eventually(lambda: trial.call.confirmed)

    def test_ambiguous_creation_is_not_retried_and_original_voice_is_available(self):
        transport = Transport()
        def fail(): raise TrialError(502, "Unconfirmed", True)
        transport.design_voice = fail
        trial = Trial(transport)
        with self.assertRaises(TrialError): trial.design_voice()
        with self.assertRaises(TrialError): trial.design_voice()
        self.assertEqual(trial.voice_config()["voice_design_state"], "unconfirmed")
        trial.create(REQUEST_ID, "v=0", "marin-original")
        self.assertEqual(transport.direction, "")
        trial.command(REQUEST_ID, "stop"); eventually(lambda: trial.call.confirmed)

    def test_created_voice_remains_usable_if_local_metadata_cannot_be_saved(self):
        profiles = VoiceProfiles(Path("unused"))
        with patch("pathlib.Path.mkdir", side_effect=OSError("disk")):
            self.assertFalse(profiles.install("voice_example"))
        self.assertFalse(profiles.persisted)
        self.assertEqual(profiles.select("owerri-designed")[0], {"id": "voice_example"})

    def test_upstream_voice_creation_uses_fixed_text_without_key_in_body(self):
        transport = OpenAITransport("PRIVATE_KEY_FIXTURE")
        response = io.BytesIO(json.dumps({"type": "prompt", "id": "voice_example"}).encode())
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value = response
            self.assertEqual(transport.design_voice(), "voice_example")
            request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/audio/voices")
        self.assertEqual(json.loads(request.data), design_payload())
        self.assertNotIn("PRIVATE_KEY_FIXTURE", request.data.decode())
        self.assertEqual(request.headers["Authorization"], "Bearer PRIVATE_KEY_FIXTURE")

    def test_voice_creation_access_failure_has_no_sensitive_error_details(self):
        transport = OpenAITransport("PRIVATE_KEY_FIXTURE")
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.side_effect = urllib.error.HTTPError("https://api.openai.com", 403, "secret", {}, io.BytesIO(b"private"))
            with self.assertRaises(TrialError) as error: transport.design_voice()
        self.assertEqual(error.exception.status, 403)
        self.assertNotIn("secret", str(error.exception))


if __name__ == "__main__": unittest.main()
