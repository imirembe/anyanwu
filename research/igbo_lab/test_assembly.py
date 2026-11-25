import hashlib
import json
import unittest
from unittest.mock import patch

from . import assembly, importer, resources, review, test_pipeline


class AssemblyChecks(unittest.TestCase):
    setUp = test_pipeline.ReviewPipelineChecks.setUp
    tearDown = test_pipeline.ReviewPipelineChecks.tearDown
    candidates = test_pipeline.ReviewPipelineChecks.candidates
    fixtures = test_pipeline.ReviewPipelineChecks.fixtures
    write_rows = test_pipeline.ReviewPipelineChecks.write_rows

    def reviewed(self, repeated_prompt=False, crossing_group=False):
        rows, decisions, path, _, cases = self.fixtures()
        if repeated_prompt or crossing_group:
            rows[0]["split"] = "train"; rows[1]["split"] = "test"
            if crossing_group:
                rows[1]["group"] = rows[0]["group"]
            if repeated_prompt:
                for decision in decisions[:2]:
                    decision.update({"decision":"edit", "input":"A shared question?"})
            self.write_rows(path, decisions)
            raw = self.lab / "imports/public-igbo/candidates.jsonl"
            self.write_rows(raw, rows)
            (raw.parent / "manifest.json").write_text(json.dumps({"candidates_sha256":hashlib.sha256(raw.read_bytes()).hexdigest()}))
        with patch("builtins.print"):
            review.compile_review("public-igbo", path, cases)
        return rows, cases

    def test_later_import_batch_preserves_original_sample(self):
        with patch("builtins.print"):
            importer.import_text("aya-igbo", rows=3, client=test_pipeline.FakeHubClient(source="aya-igbo"))
        original = (self.lab / "imports/aya-igbo/candidates.jsonl").read_bytes()
        with patch("builtins.print"):
            result = importer.import_text("aya-igbo", rows=3, offset=2000, client=test_pipeline.FakeHubClient(source="aya-igbo"))
        later = review.load_jsonl(importer.sample_directory("aya-igbo", 2000) / "candidates.jsonl")
        self.assertTrue(all(row["source_row_index"] >= 2000 for row in later))
        self.assertEqual(result["batch_start_offset"], 2000)
        self.assertEqual((self.lab / "imports/aya-igbo/candidates.jsonl").read_bytes(), original)

    def test_invalid_or_unsupported_offsets_fail_before_token_prompt(self):
        with patch.object(importer, "private_token") as token:
            for source, offset in (("aya-igbo", -1), ("aya-igbo", 1_000_001), ("africanvoice", 2000)):
                with self.assertRaises(resources.LabError):
                    importer.import_text(source, rows=10, offset=offset, token_prompt=True)
            token.assert_not_called()

    def test_assembly_preserves_active_corpus_and_is_reused_without_overwriting(self):
        _, cases = self.reviewed()
        corpus = self.lab / "corpus"; corpus.mkdir()
        marker = corpus / "measured.txt"; marker.write_text("preserved")
        with patch("builtins.print"):
            first = assembly.assemble_reviewed(["public-igbo"], cases)
            second = assembly.assemble_reviewed(["public-igbo"], cases)
        self.assertEqual(first, second)
        self.assertEqual(first["records"], 15)
        self.assertFalse(first["training_performed"])
        self.assertEqual(marker.read_text(), "preserved")
        self.assertEqual(len(list((self.lab / "curation/assemblies").iterdir())), 1)
        self.assertFalse((self.lab / "curation/assembly-staging").exists())

    def test_repeated_prompt_across_splits_never_leaks_into_training(self):
        rows, cases = self.reviewed(repeated_prompt=True)
        with patch("builtins.print"):
            manifest = assembly.assemble_reviewed(["public-igbo"], cases)
        path = self.lab / "curation/assemblies" / manifest["assembly_sha256"][:16] / "records.jsonl"
        combined = review.load_jsonl(path)
        shared = [row for row in combined if assembly.record_prompt(row) == "a shared question"]
        self.assertEqual([row["split"] for row in shared], ["test"])
        self.assertNotIn(rows[0]["id"], {row["id"] for row in combined})
        self.assertTrue(manifest["excluded_records"])

    def test_source_group_cannot_span_assembly_splits(self):
        _, cases = self.reviewed(crossing_group=True)
        with self.assertRaisesRegex(resources.LabError, "source/speaker group spans"):
            assembly.assemble_reviewed(["public-igbo"], cases)

    def test_changed_original_sample_or_review_is_refused(self):
        _, cases = self.reviewed()
        raw = self.lab / "imports/public-igbo/candidates.jsonl"
        original = raw.read_bytes(); raw.write_bytes(original + b"\n")
        with self.assertRaisesRegex(resources.LabError, "original source sample changed"):
            assembly.assemble_reviewed(["public-igbo"], cases)
        raw.write_bytes(original)
        path = self.lab / "curation/reviewed.jsonl"; path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(resources.LabError, "reviewed batch changed"):
            assembly.assemble_reviewed(["public-igbo"], cases)

    def test_manifest_write_failure_removes_staging_without_installing_assembly(self):
        _, cases = self.reviewed()
        with patch.object(assembly, "write_json", side_effect=OSError("fixture failure")):
            with self.assertRaises(OSError):
                assembly.assemble_reviewed(["public-igbo"], cases)
        self.assertFalse((self.lab / "curation/assembly-staging").exists())
        self.assertFalse((self.lab / "curation/assemblies").exists())
        self.assertTrue((self.lab / "curation/reviewed.jsonl").exists())

    def test_duplicate_selectors_and_changed_evaluation_are_refused(self):
        _, cases = self.reviewed()
        with self.assertRaisesRegex(resources.LabError, "each reviewed batch once"):
            assembly.assemble_reviewed(["public-igbo", "public-igbo@0"], cases)
        rows = review.load_jsonl(cases); rows[0]["prompt"] += " Changed."
        self.write_rows(cases, rows)
        with self.assertRaisesRegex(resources.LabError, "same frozen evaluation"):
            assembly.assemble_reviewed(["public-igbo"], cases)
