from array import array
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import importer, review, evaluation, resources, assembly, experiments
from . import test_lab


def response(value, status=200, returncode=0):
    body = json.dumps(value).encode() if not isinstance(value, bytes) else value
    return SimpleNamespace(stdout=body + f"\n{status}".encode(), stderr=b"", returncode=returncode)


class ClientChecks(unittest.TestCase):
    def test_audio_and_non_hub_hosts_are_refused(self):
        with patch.object(importer.subprocess, "run") as run:
            with self.assertRaises(resources.LabError):
                importer.BoundedHubClient().get("https://audio.example/clip.wav")
            with self.assertRaises(resources.LabError):
                importer.BoundedHubClient().get("http://huggingface.co/api/datasets/x")
            run.assert_not_called()

    def test_token_uses_stdin_not_arguments_or_child_environment(self):
        token = "hf_" + "x" * 32
        with patch.dict(importer.os.environ, {"HF_TOKEN": token}), patch.object(importer.subprocess, "run", return_value=response({"ok": True})) as run:
            self.assertEqual(importer.BoundedHubClient(token).get("https://huggingface.co/api/datasets/x"), {"ok": True})
            self.assertNotIn(token, " ".join(run.call_args.args[0]))
            self.assertIn(token.encode(), run.call_args.kwargs["input"])
            self.assertNotIn("HF_TOKEN", run.call_args.kwargs["env"])

    def test_gated_access_is_reported_without_exposing_token(self):
        token = "hf_" + "x" * 32
        with patch.object(importer.subprocess, "run", return_value=response({"error": "access required"}, 401)):
            with self.assertRaises(importer.HubAccessRequired) as caught:
                importer.BoundedHubClient(token).get("https://datasets-server.huggingface.co/rows")
        self.assertNotIn(token, str(caught.exception))

    def test_shared_transfer_budget_stops_further_requests(self):
        value = {"text": "x" * 580}
        client = importer.BoundedHubClient(budget=2048)
        with patch.object(importer.subprocess, "run", return_value=response(value)) as run:
            client.get("https://huggingface.co/a"); client.get("https://huggingface.co/b")
            with self.assertRaises(resources.LabError):
                client.get("https://huggingface.co/c")
            self.assertEqual(run.call_count, 2)

    def test_transient_retry_is_bounded_and_counted(self):
        client = importer.BoundedHubClient()
        with patch.object(importer.subprocess, "run", side_effect=[response({"error": "temporary"}, 502), response({"ok": True})]) as run:
            self.assertEqual(client.get("https://huggingface.co/a"), {"ok": True})
            self.assertEqual(run.call_count, 2)
            self.assertEqual(client.requests, 2)
            self.assertGreater(client.received, len(json.dumps({"ok": True}).encode()))

    def test_oversize_response_never_falls_back_to_dataset_download(self):
        with patch.object(importer.subprocess, "run", return_value=response(b"", 0, 63)) as run:
            with self.assertRaises(resources.LabError):
                importer.BoundedHubClient().get("https://datasets-server.huggingface.co/rows")
            self.assertEqual(run.call_count, 1)

    def test_501_is_typed_not_retried_and_never_echoes_response_body(self):
        token = "hf_" + "x" * 32
        with patch.object(importer.subprocess, "run", return_value=response({"error":token}, 501)) as run:
            with self.assertRaises(importer.HubHTTPError) as caught:
                importer.BoundedHubClient(token).get("https://datasets-server.huggingface.co/rows")
        self.assertEqual(caught.exception.status, 501)
        self.assertNotIn(token, str(caught.exception))
        self.assertEqual(run.call_count, 1)

    def test_preview_allowance_stays_within_shared_budget(self):
        client = importer.BoundedHubClient(budget=700_000)
        with patch.object(importer.subprocess, "run", return_value=response({"rows":[]})) as run:
            client.get("https://datasets-server.huggingface.co/first-rows")
        args = run.call_args.args[0]
        self.assertEqual(args[args.index("--max-filesize") + 1], "700000")
        self.assertEqual(client.remaining, 700_000 - client.received)

    def test_timeout_identifies_endpoint_without_echoing_credentials(self):
        with patch.object(importer.subprocess, "run", return_value=response(b"", 0, 28)):
            with self.assertRaisesRegex(resources.LabError, "/filter timed out"):
                importer.BoundedHubClient().get("https://datasets-server.huggingface.co/filter")

    def test_timeout_retry_counts_attempts_within_one_shared_budget(self):
        client = importer.BoundedHubClient()
        with patch.object(importer.subprocess,"run",side_effect=[response(b"",0,28),response({"rows":[]})]) as run:
            self.assertEqual(client.get("https://datasets-server.huggingface.co/filter"), {"rows":[]})
        self.assertEqual(client.requests,2)
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("--max-time")+1],"55")


class ReviewPipelineChecks(unittest.TestCase):
    def setUp(self):
        test_lab.CorpusAndBudgetChecks.setUp(self)
        for module in (importer, review, evaluation, assembly, experiments):
            change = patch.object(module, "LAB", self.lab)
            change.start(); self.patches.append(change)

    def tearDown(self):
        test_lab.CorpusAndBudgetChecks.tearDown(self)

    def candidates(self, count=15):
        spec = importer.SOURCES["public-igbo"]
        return [importer.text_record(spec, "a" * 40, "train", index,
                {"type": "conversation", "input": f"Question {index}?", "output": "Ndewo. Kedu ka ị mere?"}) for index in range(count)]

    def fixtures(self):
        rows = self.candidates()
        folder = self.lab / "imports/public-igbo"; folder.mkdir(parents=True)
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (folder / "candidates.jsonl").write_bytes(payload)
        (folder / "manifest.json").write_text(json.dumps({"candidates_sha256": hashlib.sha256(payload).hexdigest()}))
        decisions = [{"id": row["id"], "content_sha256": row["content_sha256"], "decision": "accept", "reviewer": "Codex",
                      "review_type": "AI", "reason": "Fixture decision only.", "confidence": "high"} for row in rows]
        decision_path = self.project / "decisions.jsonl"
        cases = [{"id": f"case-{index}", "group": f"group-{index}", "prompt": f"Unseen prompt {index}", "expected": "Ndewo.",
                  "language": ("en", "ig", "mixed")[index % 3], "reviewer": "Codex", "review_type": "AI"} for index in range(6)]
        case_path = self.project / "cases.jsonl"
        self.write_rows(decision_path, decisions); self.write_rows(case_path, cases)
        return rows, decisions, decision_path, cases, case_path

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def transcript_review_fixtures(self, count=15):
        cases_path = self.project / "cases.jsonl"
        if not cases_path.exists():
            _, _, _, _, cases_path = self.fixtures()
        rows = []
        for index in range(count):
            split = "train" if index < count-2 else "dev" if index == count-2 else "dev_test"
            rows.append(importer.text_record(importer.SOURCES["africanvoice"], "a"*40, split, index,
                                             {"transcript":f"Fixture transcript {index}.", "speaker_id":f"fixture-speaker-{index}"}))
        folder = self.lab / "imports/africanvoice"; folder.mkdir(parents=True)
        self.write_rows(folder / "candidates.jsonl", rows)
        payload = (folder / "candidates.jsonl").read_bytes()
        (folder / "manifest.json").write_text(json.dumps({"candidates_sha256":hashlib.sha256(payload).hexdigest()}))
        decisions = [{"id":row["id"], "content_sha256":row["content_sha256"], "decision":"accept", "reviewer":"fixture Codex",
                      "review_type":"AI", "confidence":"medium", "reason":"Fixture review only."} for row in rows]
        path = self.project / "transcript-decisions.jsonl"; self.write_rows(path, decisions)
        return rows, decisions, path, cases_path

    def test_small_transcript_review_is_saved_without_replacing_public_corpus(self):
        _, _, path, _, cases = self.fixtures()
        with patch("builtins.print"):
            review.compile_review("public-igbo", path, cases)
        public_bytes = (self.lab / "curation/reviewed.jsonl").read_bytes()
        rows, decisions, path, cases = self.transcript_review_fixtures(count=10)
        for index, decision in enumerate(decisions):
            decision["decision"] = "edit" if index < 3 else "reject"
            if index < 3:
                decision["text"] = rows[index]["content"]["text"] + " Edited."
        self.write_rows(path, decisions)
        with patch("builtins.print"):
            report = review.compile_review("africanvoice", path, cases)
        self.assertEqual(report["accepted_records"], 3)
        self.assertFalse(report["training_ready"])
        self.assertEqual(report["accepted_counts_by_split"], {"train":3,"validation":0,"test":0})
        self.assertEqual((self.lab / "curation/reviewed.jsonl").read_bytes(), public_bytes)
        edited = review.load_jsonl(self.lab / "curation/africanvoice/reviewed.jsonl")
        self.assertTrue(all("not an audio-verified ASR target" in record["text_derivation"] for record in edited))
        corpus = self.lab / "corpus"; corpus.mkdir()
        marker = corpus / "preserved.txt"; marker.write_text("prior measured corpus")
        with self.assertRaisesRegex(resources.LabError, "not ready for training"):
            review.prepare_reviewed(cases, source="africanvoice")
        self.assertEqual(marker.read_text(), "prior measured corpus")

    def test_all_rejected_transcript_audit_is_saved_and_not_called_human_validated(self):
        _, decisions, path, cases = self.transcript_review_fixtures()
        for decision in decisions:
            decision["decision"] = "reject"
        self.write_rows(path, decisions)
        with patch("builtins.print"):
            report = review.compile_review("africanvoice", path, cases)
        self.assertEqual(report["accepted_records"], 0)
        self.assertFalse(report["human_validated"])
        self.assertFalse(report["training_ready"])
        self.assertEqual((self.lab / "curation/africanvoice/reviewed.jsonl").read_bytes(), b"")

    def test_transcript_preparation_requires_official_holdouts_and_disjoint_speakers(self):
        rows, _, _, _ = self.transcript_review_fixtures()
        self.assertEqual(review.training_readiness(rows, "africanvoice"), [])
        rows[-1]["group"] = rows[0]["group"]
        self.assertIn("An accepted speaker group spans official splits.", review.training_readiness(rows, "africanvoice"))
        for row in rows:
            row["split"] = "train"
        issues = review.training_readiness(rows, "africanvoice")
        self.assertIn("No accepted records remain in the official validation split.", issues)
        self.assertIn("No accepted records remain in the official test split.", issues)

    def test_wrong_source_review_receipt_cannot_prepare_a_corpus(self):
        folder = self.lab / "curation/africanvoice"; folder.mkdir(parents=True)
        (folder / "review_report.json").write_text(json.dumps({"source":"public-igbo"}))
        with self.assertRaisesRegex(resources.LabError, "does not match the selected source"):
            review.prepare_reviewed(self.project / "unused-cases.jsonl", source="africanvoice")

    def test_aya_retains_only_igbo_original_annotations_and_drops_identifiers(self):
        spec = importer.SOURCES["aya-igbo"]
        row = {"inputs":"Kedu?", "targets":"Ị́ ị̀ ọ́ ụ̀.", "language_code":"ibo", "annotation_type":"original-annotations",
               "user_id":"private-annotator-id", "gender":"private-demographics"}
        record = importer.text_record(spec, "a"*40, "train", 7, row)
        self.assertEqual(record["content"]["output"], row["targets"])
        self.assertEqual(record["source_annotation_type"], "original-annotations")
        self.assertNotIn("private-", json.dumps(record))
        self.assertNotIn("split", record)
        self.assertIsNone(importer.text_record(spec, "a"*40, "train", 7, {**row,"language_code":"eng"}))
        self.assertIsNone(importer.text_record(spec, "a"*40, "train", 7, {**row,"annotation_type":"re-annotations"}))

    def test_aya_bounded_scan_imports_only_original_igbo_pairs(self):
        client = FakeHubClient(source="aya-igbo")
        with patch("builtins.print"):
            report = importer.import_text("aya-igbo", rows=60, client=client)
        rows = review.load_jsonl(self.lab / "imports/aya-igbo/candidates.jsonl")
        self.assertEqual(len(rows), 60)
        self.assertGreater(report["rows_scanned"], 60)
        self.assertLessEqual(report["rows_scanned"], 2000)
        self.assertTrue(all(row["source_language_code"] == "ibo" for row in rows))
        self.assertFalse(any("demographics" in url or "/filter?" in url or "/search?" in url for url in client.urls))
        self.assertNotIn("private-annotator-id", (self.lab / "imports/aya-igbo/candidates.jsonl").read_text())

    def test_existing_sample_refuses_before_prompting_or_network_transfer(self):
        folder = self.lab / "imports/aya-igbo"; folder.mkdir(parents=True)
        (folder / "original.txt").write_text("preserve")
        with patch.object(importer,"private_token") as token, patch.object(importer.BoundedHubClient,"get") as get:
            with self.assertRaisesRegex(resources.LabError, "already exists"):
                importer.import_text("aya-igbo", rows=60, token_prompt=True)
            token.assert_not_called(); get.assert_not_called()
        self.assertEqual((folder / "original.txt").read_text(), "preserve")

    def test_filtered_aya_batch_uses_exact_filter_and_original_row_indices(self):
        client = FakeHubClient(source="aya-originals")
        with patch("builtins.print"):
            report = importer.import_text("aya-originals", rows=55, client=client, offset=200)
        from urllib.parse import parse_qs, urlparse
        urls = [url for url in client.urls if "/filter?" in url]
        self.assertEqual([int(parse_qs(urlparse(url).query)["offset"][0]) for url in urls], [200,250])
        self.assertTrue(all(parse_qs(urlparse(url).query)["where"][0] == importer.SOURCES["aya-originals"]["viewer_where"] for url in urls))
        self.assertFalse(any("/rows?" in url for url in client.urls))
        self.assertEqual(report["indexed_matching_rows_by_split"], {"train":707})
        records = review.load_jsonl(importer.sample_directory("aya-originals",200)/"candidates.jsonl")
        self.assertEqual(records[0]["source_row_index"], 12000)
        self.assertEqual(len(records), 55)

    def test_server_filtered_rows_are_still_checked_locally(self):
        client = FakeHubClient(source="aya-originals")
        get = client.get
        def changed(url, **kwargs):
            page = get(url, **kwargs)
            if "/filter?" in url:
                page["rows"][0]["row"]["language_code"] = "eng"
                page["rows"][1]["row"]["annotation_type"] = "re-annotations"
                page["rows"][2].pop("row_idx")
            return page
        client.get = changed
        with patch("builtins.print"):
            importer.import_text("aya-originals",rows=3,client=client)
        rows=review.load_jsonl(self.lab/"imports/aya-originals/candidates.jsonl")
        self.assertEqual([row["source_row_index"] for row in rows], [10030,10040,10050])

    def test_failed_filter_never_falls_back_to_unfiltered_download(self):
        client = FakeHubClient(source="aya-originals",rows_status=501)
        with self.assertRaises(importer.HubHTTPError):
            importer.import_text("aya-originals",rows=10,client=client)
        self.assertFalse(any("/rows?" in url or "/first-rows?" in url for url in client.urls))
        self.assertFalse((self.lab/"imports/aya-originals").exists())

    def test_pending_decisions_cannot_enter_reviewed_corpus(self):
        rows, decisions, decision_path, cases, case_path = self.fixtures()
        self.write_rows(decision_path, decisions[:12])
        with patch("builtins.print"):
            report = review.compile_review("public-igbo", decision_path, case_path)
        self.assertEqual(report["accepted_records"], 12)
        self.assertEqual(report["pending_records"], 3)
        self.assertFalse(report["human_validated"])

    def test_published_reference_has_plain_text_and_explicit_derivation(self):
        rows, decisions, path, _, cases = self.fixtures()
        decisions[0].update(decision="edit", content_use="language-reference", text="Ị́ ị̀ ọ́ ụ̀. Published reference.")
        self.write_rows(path, decisions)
        with patch("builtins.print"):
            report = review.compile_review("public-igbo", path, cases)
        record = review.load_jsonl(self.lab / "curation/reviewed.jsonl")[0]
        self.assertEqual(record["text"], "Ị́ ị̀ ọ́ ụ̀. Published reference.")
        self.assertNotIn("<|user|>", record["text"])
        self.assertEqual(record["source_content_sha256"], rows[0]["content_sha256"])
        self.assertIn("not a user/assistant conversation", record["text_derivation"])
        self.assertEqual(report["accepted_counts_by_use"], {"assistant-pair":14,"language-reference":1})

    def test_reference_conversion_requires_explicit_edit_and_unambiguous_text(self):
        _, decisions, path, _, cases = self.fixtures()
        for fields in ({"decision":"accept", "text":"Reference."},
                       {"decision":"edit"},
                       {"decision":"edit", "text":"Reference.", "input":"Ignored prompt"},
                       {"decision":"edit", "text":"<|user|>Disguised conversation"}):
            changed = [{**decisions[0], "content_use":"language-reference", **fields}]
            self.write_rows(path, changed)
            with self.assertRaises(resources.LabError):
                review.compile_review("public-igbo", path, cases)

    def test_semantic_variants_keep_one_family_and_original_source_groups(self):
        rows, decisions, path, _, cases = self.fixtures()
        for decision in decisions[:2]:
            decision.update(family_group="same-proverb", family_reason="Fixture: two phrasings of the same proverb.")
        self.write_rows(path, decisions)
        with patch("builtins.print"):
            review.compile_review("public-igbo", path, cases)
        records = review.load_jsonl(self.lab / "curation/reviewed.jsonl")
        self.assertEqual(records[0]["group"], records[1]["group"])
        self.assertEqual(records[0]["source_group"], rows[0]["group"])
        self.assertEqual(records[1]["source_group"], rows[1]["group"])
        self.assertEqual(assembly.group_split(records[0]["source"] + "/" + records[0]["group"]),
                         assembly.group_split(records[1]["source"] + "/" + records[1]["group"]))

    def test_semantic_group_never_replaces_official_speaker_holdout(self):
        _, decisions, path, cases = self.transcript_review_fixtures()
        decisions[0].update(family_group="not-a-speaker", family_reason="Invalid override.")
        self.write_rows(path, decisions)
        with self.assertRaisesRegex(resources.LabError, "never replace an official"):
            review.compile_review("africanvoice", path, cases)

    def test_reference_conversion_cannot_hide_a_frozen_evaluation_prompt(self):
        rows, decisions, path, cases, case_path = self.fixtures()
        decisions[0].update(decision="edit", content_use="language-reference", text="Reference with no question.",
                            family_group="different-label", family_reason="Cannot bypass prompt protection.")
        cases[0]["prompt"] = rows[0]["content"]["input"]
        self.write_rows(path, decisions); self.write_rows(case_path, cases)
        with patch("builtins.print"):
            report = review.compile_review("public-igbo", path, case_path)
        self.assertEqual(report["accepted_records"], 14)
        self.assertEqual(report["rejected_records"][0]["id"], rows[0]["id"])

    def test_changed_original_content_invalidates_review(self):
        rows, decisions, decision_path, cases, case_path = self.fixtures()
        decisions[0]["content_sha256"] = "wrong"
        self.write_rows(decision_path, decisions)
        with self.assertRaises(resources.LabError):
            review.compile_review("public-igbo", decision_path, case_path)

    def test_exact_evaluation_prompt_is_excluded(self):
        rows, decisions, decision_path, cases, case_path = self.fixtures()
        cases[0]["prompt"] = rows[0]["content"]["input"]
        self.write_rows(case_path, cases)
        with patch("builtins.print"):
            report = review.compile_review("public-igbo", decision_path, case_path)
        self.assertEqual(report["accepted_records"], 14)
        self.assertTrue(any("Evaluation" in item["reason"] for item in report["rejected_records"]))

    def test_low_confidence_is_excluded_even_when_marked_accept(self):
        rows, decisions, decision_path, cases, case_path = self.fixtures()
        decisions[0]["confidence"] = "low"
        self.write_rows(decision_path, decisions)
        with patch("builtins.print"):
            report = review.compile_review("public-igbo", decision_path, case_path)
        self.assertEqual(report["accepted_records"], 14)

    def test_edit_cannot_be_disguised_as_accept(self):
        rows, decisions, decision_path, cases, case_path = self.fixtures()
        decisions[0]["output"] = "A different answer."
        self.write_rows(decision_path, decisions)
        with self.assertRaises(resources.LabError):
            review.compile_review("public-igbo", decision_path, case_path)

    def test_transcript_records_drop_audio_urls_and_hash_speaker_group(self):
        record = importer.text_record(importer.SOURCES["africanvoice"], "a" * 40, "train", 0,
                                      {"transcript": "Ndewo.", "speaker_id": "speaker-private", "audio_path": {"src": "https://audio.example/clip.wav"}})
        self.assertEqual(record["content"]["text"], "Ndewo.")
        self.assertFalse(record["audio_downloaded"])
        self.assertNotIn("audio.example", json.dumps(record))
        self.assertNotIn("speaker-private", json.dumps(record))

    def test_transcript_preserves_official_holdout_assignment(self):
        record = importer.text_record(importer.SOURCES["africanvoice"], "a" * 40, "dev", 0,
                                      {"transcript":"Ndewo.","speaker_id":"fixture-speaker"})
        self.assertEqual(record["split"], "validation")

    def test_transcript_import_samples_each_official_split(self):
        with patch("builtins.print"):
            report = importer.import_text("africanvoice", rows=10, client=FakeHubClient(source="africanvoice"))
        self.assertEqual(report["sample_counts_by_source_split"], {"train":8,"dev":1,"dev_test":1})
        rows=review.load_jsonl(self.lab / "imports/africanvoice/candidates.jsonl")
        self.assertEqual({row["split"] for row in rows},{"train","validation","test"})

    def test_501_uses_one_preview_per_split_without_media_requests(self):
        client = FakeHubClient(source="africanvoice", rows_status=501)
        with patch("builtins.print"):
            report = importer.import_text("africanvoice", rows=100, client=client)
        self.assertEqual(report["sample_counts_by_source_split"], {"train":80,"dev":10,"dev_test":10})
        self.assertEqual(set(report["viewer_endpoints_by_source_split"].values()), {"first-rows"})
        self.assertEqual(sum("/rows?" in url for url in client.urls), 1)
        self.assertEqual(sum("/first-rows?" in url for url in client.urls), 3)
        self.assertFalse(any("audio.example" in url or ".parquet" in url for url in client.urls))
        payload = (self.lab / "imports/africanvoice/candidates.jsonl").read_text()
        self.assertNotIn("audio.example", payload)
        self.assertNotIn("-speaker-", payload)

    def test_preview_is_not_used_to_retry_access_denial_or_other_errors(self):
        for status in (401, 403, 400, 500):
            client = FakeHubClient(source="africanvoice", rows_status=status)
            with self.assertRaises(resources.LabError):
                importer.import_text("africanvoice", rows=10, client=client)
            self.assertFalse(any("/first-rows?" in url for url in client.urls))

    def test_preview_unavailable_installs_nothing_and_requests_no_files(self):
        client = FakeHubClient(source="africanvoice", rows_status=501, preview_status=501)
        with patch("builtins.print"), self.assertRaisesRegex(resources.LabError, "preview returned HTTP 501"):
            importer.import_text("africanvoice", rows=10, client=client)
        self.assertFalse((self.lab / "imports/africanvoice").exists())
        self.assertFalse(any("/resolve/" in url for url in client.urls))

    def test_preview_quota_shortfall_does_not_loop_or_install_partial_data(self):
        client = FakeHubClient(source="africanvoice", rows_status=501, preview_rows=2)
        with patch("builtins.print"), self.assertRaisesRegex(resources.LabError, "split quotas"):
            importer.import_text("africanvoice", rows=10, client=client)
        self.assertEqual(sum("/first-rows?" in url for url in client.urls), 3)
        self.assertFalse((self.lab / "imports/africanvoice").exists())
        files = list((self.lab / "imports").glob("africanvoice-preview-*.json"))
        self.assertEqual(len(files), 1)
        diagnostic = json.loads(files[0].read_text())
        self.assertEqual(set(diagnostic["splits"]), {"train", "dev", "dev_test"})
        self.assertEqual(diagnostic["splits"]["train"]["selected"], 2)
        self.assertEqual(diagnostic["splits"]["train"]["requested"], 8)
        self.assertEqual(len(diagnostic["eligible_preview_records"]), 6)
        self.assertFalse(diagnostic["training_sample_installed"])
        self.assertNotIn("audio.example", files[0].read_text())
        self.assertNotIn("-speaker-", files[0].read_text())

    def test_truncated_transcripts_are_excluded_but_truncated_audio_is_ignored(self):
        for column in ("transcript", "speaker_id"):
            client = FakeHubClient(source="africanvoice", rows_status=501, truncated_column=column)
            with patch("builtins.print"), self.assertRaisesRegex(resources.LabError, "truncated text"):
                importer.import_text("africanvoice", rows=10, client=client)
        with patch("builtins.print"):
            report = importer.import_text("africanvoice", rows=10, client=FakeHubClient(source="africanvoice", rows_status=501, truncated_column="audio_path"))
        self.assertEqual(report["records"], 10)

    def test_preview_wrong_split_is_rejected(self):
        client = FakeHubClient(source="africanvoice", rows_status=501, preview_wrong_split=True)
        with patch("builtins.print"), self.assertRaisesRegex(resources.LabError, "unexpected dataset/subset/split"):
            importer.import_text("africanvoice", rows=10, client=client)
        self.assertFalse((self.lab / "imports/africanvoice").exists())

    def test_non_text_speaker_is_diagnosed_without_storing_original_value(self):
        client = FakeHubClient(source="africanvoice", rows_status=501)
        get = client.get
        def changed(url, **kwargs):
            page = get(url, **kwargs)
            if "/first-rows?" in url:
                for entry in page["rows"]:
                    entry["row"]["speaker_id"] = 987654321
            return page
        client.get = changed
        with patch("builtins.print"), self.assertRaisesRegex(resources.LabError, "Inspect it before retrying"):
            importer.import_text("africanvoice", rows=10, client=client)
        path = next((self.lab / "imports").glob("africanvoice-preview-*.json"))
        diagnostic = json.loads(path.read_text())
        for stats in diagnostic["splits"].values():
            self.assertEqual(stats["excluded"], {"non_text_speaker_id":100})
            self.assertEqual(stats["field_types"]["speaker_id"], {"int":100})
        self.assertNotIn("987654321", path.read_text())
        self.assertEqual(diagnostic["eligible_preview_records"], [])

    def test_empty_previews_report_all_splits_without_recommending_same_retry(self):
        client = FakeHubClient(source="africanvoice", rows_status=501, preview_rows=0)
        with patch("builtins.print"), self.assertRaisesRegex(resources.LabError, "smaller --rows value may not help"):
            importer.import_text("africanvoice", rows=10, client=client)
        path = next((self.lab / "imports").glob("africanvoice-preview-*.json"))
        diagnostic = json.loads(path.read_text())
        self.assertEqual(set(diagnostic["splits"]), {"train", "dev", "dev_test"})
        self.assertTrue(all(stats["rows_returned"] == 0 for stats in diagnostic["splits"].values()))

    def test_truncated_preview_diagnostic_does_not_retain_partial_text(self):
        client = FakeHubClient(source="africanvoice", rows_status=501, truncated_column="transcript")
        with patch("builtins.print"), self.assertRaises(resources.LabError):
            importer.import_text("africanvoice", rows=10, client=client)
        path = next((self.lab / "imports").glob("africanvoice-preview-*.json"))
        diagnostic = json.loads(path.read_text())
        self.assertEqual(diagnostic["eligible_preview_records"], [])
        self.assertEqual(diagnostic["splits"]["train"]["excluded"], {"truncated_transcript":100})
        self.assertNotIn("train text", path.read_text())

    def test_diagnostic_preserves_original_receipt_and_deduplicates_identical_retry(self):
        for unused in range(2):
            with patch("builtins.print"), self.assertRaises(resources.LabError):
                importer.import_text("africanvoice", rows=10, client=FakeHubClient(source="africanvoice", rows_status=501, preview_rows=2))
        self.assertEqual(len(list((self.lab / "imports").glob("africanvoice-preview-*.json"))), 1)

    def test_changed_revision_never_saves_failed_preview_as_current_source(self):
        client = FakeHubClient(source="africanvoice", rows_status=501, preview_rows=2, changed_revision=True)
        with patch("builtins.print"), self.assertRaisesRegex(resources.LabError, "changed during import"):
            importer.import_text("africanvoice", rows=10, client=client)
        self.assertEqual(list((self.lab / "imports").glob("africanvoice-preview-*.json")), [])

    def test_whitespace_speaker_is_rejected(self):
        row = {"transcript":"Ndewo.", "speaker_id":"  "}
        self.assertEqual(importer.transcript_issue(row), "empty_speaker_id")
        self.assertIsNone(importer.text_record(importer.SOURCES["africanvoice"], "a"*40, "train", 0, row))

    def test_real_text_training_requires_explicit_review_metadata(self):
        from . import corpus, runner
        rows = test_lab.CorpusAndBudgetChecks.records(self)
        for row in rows:
            row["demo_only"] = False
        path = self.project / "real.jsonl"; self.write_rows(path, rows)
        with patch("builtins.print"), patch.object(runner,"LAB",self.lab):
            corpus.prepare(path)
            with self.assertRaises(resources.LabError):
                runner.read_manifest()
            for row in rows:
                row["review"]={"type":"AI","reviewer":"fixture Codex","decision":"accept","confidence":"high","reason":"fixture review"}
            self.write_rows(path, rows)
            corpus.prepare(path)
            self.assertEqual(runner.read_manifest()["review_summary"]["ai_reviewed"],3)

    def test_mismatched_tokenizers_cannot_be_compared(self):
        folder = self.lab / "evaluations"; folder.mkdir()
        for label, token in (("baseline", "a"), ("candidate", "b")):
            (folder / f"{label}.json").write_text(json.dumps({"evaluation_sha256": "same", "tokenizer_sha256": token, "model_config": {}}))
        with self.assertRaises(resources.LabError):
            evaluation.compare_evaluations()

    def test_existing_baseline_requires_explicit_replacement(self):
        folder = self.lab / "evaluations"; folder.mkdir()
        (folder / "baseline.json").write_text("{}")
        with self.assertRaises(resources.LabError):
            evaluation.evaluate_checkpoint(self.project / "cases.jsonl", "baseline")

    def test_short_holdout_is_evaluated_without_repeating_its_text(self):
        from . import runner
        path = self.lab / "short.tokens"
        with path.open("wb") as handle:
            array("H", [10, 11, 12, 13]).tofile(handle)
        with patch.object(runner, "LAB", self.lab), runner.TokenFile(path) as tokens:
            self.assertEqual(tokens.batch(2, 128, random.Random(1), allow_short=True).tolist(), [[10,11,12,13],[10,11,12,13]])
            with self.assertRaises(resources.LabError):
                tokens.batch(2, 128, random.Random(1))

    def test_import_commits_only_text_and_preserves_source_card(self):
        client = FakeHubClient()
        with patch("builtins.print"):
            result = importer.import_text("public-igbo", rows=2, client=client)
        path = self.lab / "imports/public-igbo"
        self.assertEqual(result["records"], 2)
        self.assertFalse(result["audio_downloaded"])
        self.assertNotIn("audio.example", (path / "candidates.jsonl").read_text())
        self.assertTrue((path / "source-card.md").exists())
        self.assertFalse((path.parent / "import-staging").exists())

    def test_revision_change_aborts_without_installing_sample(self):
        with self.assertRaises(resources.LabError):
            importer.import_text("public-igbo", rows=2, client=FakeHubClient(changed_revision=True))
        self.assertFalse((self.lab / "imports/public-igbo").exists())

    def test_gated_sample_never_attempts_media_or_full_dataset_fallback(self):
        client = FakeHubClient(gated=True)
        with self.assertRaises(importer.HubAccessRequired):
            importer.import_text("public-igbo", rows=2, client=client)
        self.assertFalse(any("/rows?" in url for url in client.urls))
        self.assertFalse((self.lab / "imports/public-igbo").exists())

    def test_failed_manifest_write_removes_import_staging(self):
        with patch.object(importer, "write_json", side_effect=OSError("fixture write failure")):
            with self.assertRaises(OSError):
                importer.import_text("public-igbo", rows=2, client=FakeHubClient())
        self.assertFalse((self.lab / "imports/public-igbo").exists())
        self.assertFalse((self.lab / "imports/import-staging").exists())

    def test_comparison_baseline_survives_periodic_candidate_saving(self):
        import mlx.core as mx
        from . import runner
        root = self.lab / "checkpoints"; root.mkdir()
        for slot, data in (("current", b"older candidate"), ("previous", b"baseline")):
            folder = root / slot; folder.mkdir()
            (folder / "model.safetensors").write_bytes(data)
            (folder / "checkpoint.json").write_text(json.dumps({"files":{"model.safetensors":hashlib.sha256(data).hexdigest()}}))
        evaluation_dir = self.lab / "evaluations"; evaluation_dir.mkdir()
        (evaluation_dir / "baseline.json").write_text(json.dumps({"model_sha256": hashlib.sha256(b"baseline").hexdigest()}))
        corpus = self.lab / "corpus"; corpus.mkdir()
        (corpus / "tokenizer.model").write_bytes(b"tokenizer fixture")
        model = SimpleNamespace(parameters=lambda:{"weight":mx.array([1.0])}, configuration=lambda:{"fixture":True},
                                save_weights=lambda path:Path(path).write_bytes(b"new candidate"))
        optimizer = SimpleNamespace(state={"step":mx.array(1)})
        manifest = {"demo_only":False,"tokenizer_sha256":"fixture","corpus_sha256":"fixture"}
        with patch.object(runner, "LAB", self.lab):
            runner.save_checkpoint(model, optimizer, manifest, 1, {}, "fixture")
        self.assertEqual((root / "previous/model.safetensors").read_bytes(), b"baseline")
        self.assertEqual((root / "current/model.safetensors").read_bytes(), b"new candidate")
        self.assertFalse((root / "retired-checkpoint").exists())
        self.assertEqual(sorted(path.name for path in root.iterdir()), ["current","previous"])


class FakeHubClient:
    def __init__(self, changed_revision=False, gated=False, source="public-igbo", rows_status=None,
                 preview_status=None, preview_rows=100, truncated_column=None, preview_wrong_split=False):
        self.remaining = importer.MAX_TRANSFER
        self.received = 0
        self.requests = 0
        self.urls = []
        self.metadata_calls = 0
        self.changed_revision, self.gated = changed_revision, gated
        self.spec = importer.SOURCES[source]
        self.rows_status, self.preview_status = rows_status, preview_status
        self.preview_rows, self.truncated_column = preview_rows, truncated_column
        self.preview_wrong_split = preview_wrong_split

    def get(self, url, *, json_response=True):
        self.urls.append(url); self.requests += 1
        self.received += 100; self.remaining -= 100
        if "/api/datasets/" in url:
            self.metadata_calls += 1
            revision = "b" * 40 if self.changed_revision and self.metadata_calls > 1 else "a" * 40
            return {"sha":revision,"cardData":{"license":self.spec["license"]}}
        if "/raw/" in url:
            return "# Fixture dataset\nApache 2.0, test-only content."
        if "/splits?" in url:
            if self.gated:
                raise importer.HubAccessRequired("fixture access required")
            return {"splits":[{"config":self.spec["config"],"split":split} for split in self.spec["splits"]]}
        if "/rows?" in url or "/first-rows?" in url or "/filter?" in url:
            preview = "/first-rows?" in url
            status = self.preview_status if preview else self.rows_status
            if status in {401,403}:
                raise importer.HubAccessRequired("fixture access required")
            if status:
                raise importer.HubHTTPError("/first-rows" if preview else "/filter" if "/filter?" in url else "/rows", status)
            if self.spec["kind"] == "instruction":
                from urllib.parse import parse_qs, urlparse
                query=parse_qs(urlparse(url).query); start=int(query["offset"][0]); length=int(query["length"][0])
                if "/filter?" in url:
                    return {"num_rows_total":707,"rows":[{"row_idx":10000+10*index,"row":{"inputs":f"Original question {index}?","targets":"Ndewo.",
                            "language_code":"ibo","annotation_type":"original-annotations","user_id":"private-annotator-id"}}
                            for index in range(start,min(start+length,707))]}
                return {"rows":[{"row_idx":index,"row":{"inputs":f"Original question {index}?","targets":"Ndewo.",
                                "language_code":"ibo" if index%3==0 else "eng",
                                "annotation_type":"original-annotations" if index%6!=0 else "re-annotations",
                                "user_id":"private-annotator-id"}} for index in range(start,min(start+length,10000))]}
            if self.spec["kind"] == "transcript":
                from urllib.parse import parse_qs, urlparse
                query=parse_qs(urlparse(url).query)
                split=query["split"][0]; length=self.preview_rows if preview else int(query["length"][0])
                return {"split":"wrong-split" if self.preview_wrong_split else split,
                        "rows":[{"row_idx":index,"truncated_cells":[self.truncated_column] if self.truncated_column else [],
                                 "row":{"transcript":f"{split} text {index}.","speaker_id":f"{split}-speaker-{index}",
                                                            "audio_path":{"src":"https://audio.example/never.wav"}}} for index in range(length)]}
            return {"rows":[{"row_idx":index,"row":{"type":"conversation","input":f"Question {index}?","output":"Ndewo.",
                                                         "audio_path":{"src":"https://audio.example/never.wav"}}} for index in range(2)]}
        raise AssertionError("Unexpected fixture API request")
