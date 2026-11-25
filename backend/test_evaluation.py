import json
from pathlib import Path
import tempfile
import unittest

from backend.evaluate_speech import distance, evaluate, normalize


class EvaluationTests(unittest.TestCase):
    def test_metrics_preserve_igbo_underdots_and_normalize_combining_forms(self):
        self.assertEqual(normalize("Ị\u0301 nọ ebe a!"), normalize("I\u0323\u0301 nọ ebe a."))
        self.assertNotEqual(normalize("ị"), normalize("i"))
        self.assertNotEqual(normalize("á"), normalize("a"))
        self.assertEqual(distance("a b c".split(), "a x c d".split()), 2)

    def test_real_reference_comparison_is_grouped_and_sample_count_is_bounded(self):
        class ProviderFixture:
            calls = 0

            def transcribe(self, audio, mime):
                self.calls += 1
                return "Ndewo ụwa" if self.calls == 1 else "hello earth"

        Path(".build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=".build", prefix="evaluation-check-") as directory:
            directory = Path(directory)
            (directory / "test.wav").write_bytes(b"fixture")
            rows = [{"id": "ig-1", "language": "ig", "text": "Ndewo ụwa", "audio": "test.wav"},
                    {"id": "en-1", "language": "en", "text": "hello world", "audio": "test.wav"},
                    {"id": "mixed-1", "language": "mixed", "text": "hello ụwa", "audio": "test.wav"}]
            manifest = directory / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            provider = ProviderFixture()
            report = evaluate(manifest, provider, 2)
            self.assertEqual(provider.calls, 2)
            self.assertEqual(report["samples"], 2)
            self.assertEqual(report["groups"]["ig"]["WER"], 0)
            self.assertEqual(report["groups"]["en"]["WER"], 0.5)
            self.assertIsNone(report["groups"]["mixed"]["WER"])

    def test_evaluation_refuses_audio_paths_outside_its_dataset_directory(self):
        Path(".build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=".build", prefix="evaluation-check-") as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(json.dumps({"id": "1", "language": "ig", "text": "Ndewo", "audio": "../private.wav"}))
            with self.assertRaisesRegex(ValueError, "manifest directory"):
                evaluate(manifest, object(), 1)
