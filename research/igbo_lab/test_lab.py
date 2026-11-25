import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import patch

from . import corpus, resources


class CorpusAndBudgetChecks(unittest.TestCase):
    def setUp(self):
        temporary = resources.LAB / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="checks-", dir=temporary)
        self.project = Path(self.temporary.name)
        self.lab = self.project / ".build/IgboLab"
        self.lab.mkdir(parents=True)
        self.patches = [patch.object(resources, "PROJECT", self.project), patch.object(resources, "LAB", self.lab),
                        patch.object(corpus, "LAB", self.lab)]
        for change in self.patches:
            change.start()

    def tearDown(self):
        for change in reversed(self.patches):
            change.stop()
        self.temporary.cleanup()

    def records(self):
        return [dict(id=str(index), group=split, source="fixture", license="CC0-1.0", split=split,
                     demo_only=True, text=f"{split}: Ị́ ị̀ ọ́ ọ̀ ụ́ ụ̀ Ṅ́. Aha m bụ Anyanwụ. Kedu ka ị mere?")
                for index, split in enumerate(("train", "validation", "test"))]

    def write(self, records):
        path = self.project / "source.jsonl"
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        return path

    def test_igbo_marks_survive_normalization(self):
        text = "Ị́ ị̀ ọ́ ọ̀ ụ́ ụ̀ Ṅ́"
        self.assertEqual(corpus.normalized(unicodedata.normalize("NFD", text)), unicodedata.normalize("NFC", text))

    def test_duplicates_do_not_leak_between_splits(self):
        rows = self.records()
        rows.append({**rows[0], "id": "duplicate", "group": "other-book", "split": "test"})
        loaded = corpus.read_records(self.write(rows))
        self.assertEqual(len(loaded), 3)
        self.assertEqual(sum(record["text"] == rows[0]["text"] for record in loaded), 1)

    def test_book_group_cannot_cross_holdouts(self):
        rows = self.records(); rows[1]["group"] = rows[0]["group"]
        with self.assertRaises(resources.LabError):
            corpus.read_records(self.write(rows))

    def test_missing_source_or_license_is_rejected(self):
        for field in ("source", "license", "group", "id"):
            rows = self.records(); rows[0].pop(field)
            with self.assertRaises(resources.LabError):
                corpus.read_records(self.write(rows))

    def test_oversized_document_is_rejected(self):
        rows = self.records(); rows[0]["text"] = "ị" * resources.DOCUMENT_LIMIT
        with self.assertRaises(resources.LabError):
            corpus.read_records(self.write(rows))

    def test_disk_reserve_is_enforced(self):
        with patch.object(resources.shutil, "disk_usage", return_value=SimpleNamespace(free=resources.FREE_RESERVE - 1)):
            with self.assertRaises(resources.LabError):
                resources.ensure_capacity()

    def test_workspace_quota_is_enforced(self):
        with patch.object(resources, "directory_bytes", return_value=resources.WORKSPACE_LIMIT):
            with self.assertRaises(resources.LabError):
                resources.ensure_capacity(1)

    def test_cleanup_preserves_sources_and_checkpoints(self):
        original = self.project / "original.txt"; original.write_text("preserve")
        checkpoint = self.lab / "checkpoints/current"; checkpoint.mkdir(parents=True)
        (checkpoint / "weights").write_text("preserve")
        stage = self.lab / "corpus-staging"; stage.mkdir()
        resources.remove_generated(stage)
        with self.assertRaises(resources.LabError):
            resources.remove_generated(checkpoint)
        with self.assertRaises(resources.LabError):
            resources.remove_generated(original)
        self.assertEqual(original.read_text(), "preserve")
        self.assertTrue((checkpoint / "weights").exists())

    def test_output_symlink_escape_is_rejected(self):
        target = self.project / "outside"; target.mkdir()
        link = self.lab / "corpus-staging"; link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(resources.LabError):
            resources.remove_generated(link)
        self.assertTrue(target.exists())

    def test_tokenizer_byte_fallback_and_frozen_vocabulary(self):
        import sentencepiece as spm
        path = self.write(self.records())
        with patch("builtins.print"):
            first = corpus.prepare(path)
        tokenizer = spm.SentencePieceProcessor(model_file=str(self.lab / "corpus/tokenizer.model"))
        for text in ("ị́ ọ̀ ụ́ ṅ", "NewName𐐷123"):
            text = unicodedata.normalize("NFC", text)
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        rows = self.records(); rows.append({**rows[0], "id": "more", "text": "A ọhụrụ sample with extra words."})
        with patch("builtins.print"):
            second = corpus.prepare(self.write(rows))
        self.assertTrue(second["tokenizer_reused"])
        self.assertEqual(first["tokenizer_sha256"], second["tokenizer_sha256"])
        self.assertNotEqual(first["corpus_sha256"], second["corpus_sha256"])


class CausalModelChecks(unittest.TestCase):
    def test_model_cannot_see_future_tokens(self):
        import mlx.core as mx
        from .model import IgboLanguageModel, ModelConfig
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(resources.MLX_MEMORY_GUIDELINE)
        mx.set_cache_limit(resources.MLX_CACHE_LIMIT)
        model = IgboLanguageModel(ModelConfig(vocabulary=512, dimensions=64, layers=1, context=32))
        first = model(mx.array([[1, 2, 3, 4]]))
        changed_future = model(mx.array([[1, 2, 300, 400]]))
        self.assertTrue(mx.allclose(first[:, :2], changed_future[:, :2], atol=1e-5).item())
        self.assertFalse(mx.allclose(first[:, -1], changed_future[:, -1]).item())
        mx.clear_cache()

    def test_large_model_configuration_is_rejected(self):
        from .model import ModelConfig
        with self.assertRaises(resources.LabError):
            ModelConfig(vocabulary=100_000, dimensions=1024, layers=24).validate()


if __name__ == "__main__":
    unittest.main()
