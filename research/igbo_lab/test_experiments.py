import hashlib
import json
import os
import unittest
from unittest.mock import patch

from . import assembly, corpus, evaluation, experiments, resources, runner, test_assembly, test_pipeline


class ExperimentChecks(unittest.TestCase):
    setUp = test_pipeline.ReviewPipelineChecks.setUp
    tearDown = test_pipeline.ReviewPipelineChecks.tearDown
    candidates = test_pipeline.ReviewPipelineChecks.candidates
    fixtures = test_pipeline.ReviewPipelineChecks.fixtures
    write_rows = test_pipeline.ReviewPipelineChecks.write_rows
    reviewed = test_assembly.AssemblyChecks.reviewed

    def fixture(self):
        _, cases = self.reviewed()
        with patch("builtins.print"):
            manifest = assembly.assemble_reviewed(["public-igbo"], cases)
        folder = self.lab / "curation/assemblies" / manifest["assembly_sha256"][:16]
        active = self.lab / "corpus"; active.mkdir()
        (active / "tokenizer.model").write_bytes(b"tokenizer")
        (active / "manifest.json").write_text(json.dumps({"fixture":"preserved original preparation"}))
        files = {"model.safetensors":b"baseline", "optimizer.safetensors":b"optimizer", "tokenizer.model":b"tokenizer"}
        config = {"vocabulary":512,"dimensions":128,"layers":4,"heads":4,"context":128}
        checkpoint = {"step":125,"config":config,"tokenizer_sha256":hashlib.sha256(b"tokenizer").hexdigest(),
                      "files":{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
        for slot in ("current", "previous"):
            root = self.lab / "checkpoints" / slot; root.mkdir(parents=True)
            for name,data in files.items():
                (root / name).write_bytes(data)
            (root / "checkpoint.json").write_text(json.dumps(checkpoint))
        baseline = {"model_sha256":checkpoint["files"]["model.safetensors"],"evaluation_sha256":manifest["evaluation_sha256"],
                    "tokenizer_sha256":checkpoint["tokenizer_sha256"],"model_config":config}
        root = self.lab / "evaluations"; root.mkdir()
        (root / "baseline.json").write_text(json.dumps(baseline))
        return folder, cases, checkpoint

    def test_changed_baseline_is_refused_before_mutating_corpus(self):
        folder, cases, _ = self.fixture()
        active = self.lab / "corpus/manifest.json"; original = active.read_bytes()
        (self.lab / "checkpoints/previous/optimizer.safetensors").write_bytes(b"changed")
        with patch("builtins.print"), self.assertRaisesRegex(resources.LabError, "baseline checkpoint failed"):
            experiments.run_experiment(folder, cases)
        self.assertEqual(active.read_bytes(), original)
        self.assertFalse((self.lab / "experiments").exists())

    def test_failed_preparation_registers_plan_and_cannot_trigger_automatic_retry(self):
        folder, cases, _ = self.fixture()
        with patch.object(corpus,"prepare",side_effect=resources.LabError("fixture preparation failure")) as prepare, patch.object(runner,"train") as train, patch("builtins.print"):
            with self.assertRaisesRegex(resources.LabError, "fixture preparation failure"):
                experiments.run_experiment(folder, cases)
            with self.assertRaisesRegex(resources.LabError, "already started"):
                experiments.run_experiment(folder, cases)
            self.assertEqual(prepare.call_count, 1)
            train.assert_not_called()
        root = next((self.lab / "experiments").iterdir())
        plan = json.loads((root / "plan.json").read_text())
        self.assertEqual(plan["training"], {"steps":120,"seconds":180,"batch_size":2,"resume_slot":"previous","resume":True,"cpu":False})
        self.assertEqual(json.loads((root / "run.json").read_text())["status"], "failed")
        self.assertEqual((self.lab / "checkpoints/previous/model.safetensors").read_bytes(), b"baseline")

    def test_completed_experiment_is_read_only_when_invoked_again(self):
        folder, cases, _ = self.fixture()
        root = self.lab / "experiments" / ("expanded-" + folder.name + "-v1"); root.mkdir(parents=True)
        (root / "run.json").write_text(json.dumps({"status":"complete"}))
        (root / "result.json").write_text(json.dumps({"fixture":"existing result"}))
        with patch.object(experiments,"preflight") as preflight, patch.object(runner,"train") as train, patch("builtins.print"):
            result = experiments.run_experiment(folder, cases)
            preflight.assert_not_called(); train.assert_not_called()
        self.assertEqual(result, {"fixture":"existing result"})

    def test_changed_frozen_cases_are_refused_before_registering_run(self):
        folder, cases, _ = self.fixture()
        rows = json.loads(cases.read_text().splitlines()[0]); rows["prompt"] += " changed"
        content = cases.read_text().splitlines(); content[0] = json.dumps(rows); cases.write_text("\n".join(content)+"\n")
        with self.assertRaisesRegex(resources.LabError, "preserve the assembly's frozen"):
            experiments.run_experiment(folder, cases)
        self.assertFalse((self.lab / "experiments").exists())

    def test_closed_or_foreign_experiment_cannot_train_again(self):
        _, _, checkpoint = self.fixture()
        identifier = "expanded-" + "a"*16 + "-v1"
        root = self.lab / "experiments" / identifier; root.mkdir(parents=True)
        manifest = {"demo_only":False,"review_summary":{},"tokenizer_sha256":checkpoint["tokenizer_sha256"],"controlled_experiment":identifier}
        (self.lab / "corpus/manifest.json").write_text(json.dumps(manifest))
        for status, pid in (("complete",os.getpid()), ("training",os.getpid()+1)):
            (root / "run.json").write_text(json.dumps({"status":status,"owner_pid":pid}))
            with patch.object(runner,"LAB",self.lab), self.assertRaisesRegex(resources.LabError, "single controlled experiment"):
                runner.read_manifest()

    def test_mutated_plan_or_token_files_are_refused(self):
        _, _, checkpoint = self.fixture()
        identifier = "expanded-" + "a"*16 + "-v1"
        root = self.lab / "experiments" / identifier; root.mkdir(parents=True)
        (root / "run.json").write_text(json.dumps({"status":"training","owner_pid":os.getpid()}))
        (root / "plan.json").write_text("{}")
        manifest = {"demo_only":False,"review_summary":{},"tokenizer_sha256":checkpoint["tokenizer_sha256"],
                    "controlled_experiment":identifier,"controlled_plan_sha256":"wrong"}
        (self.lab / "corpus/manifest.json").write_text(json.dumps(manifest))
        with patch.object(runner,"LAB",self.lab), self.assertRaisesRegex(resources.LabError, "plan changed"):
            runner.read_manifest()
        manifest["controlled_plan_sha256"] = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
        manifest["token_files_sha256"] = {"train.tokens":"wrong"}
        (self.lab / "corpus/train.tokens").write_bytes(b"tokens")
        (self.lab / "corpus/manifest.json").write_text(json.dumps(manifest))
        with patch.object(runner,"LAB",self.lab), self.assertRaisesRegex(resources.LabError, "token files changed"):
            runner.read_manifest()

    def test_comparison_reports_stay_separate_from_original_pilot(self):
        _, _, _ = self.fixture()
        original = (self.lab / "evaluations/baseline.json").read_bytes()
        root = self.lab / "experiments/fixture/evaluations"; root.mkdir(parents=True)
        for label, loss in (("baseline",2.0),("candidate",3.0)):
            (root / (label+".json")).write_text(json.dumps({"evaluation_sha256":"cases","tokenizer_sha256":"tokenizer","model_config":{},
                "model_sha256":label,"mean_reference_reply_loss":loss,"mean_by_input_language":{"en":loss,"ig":loss,"mixed":loss}}))
        with patch("builtins.print"):
            comparison = evaluation.compare_evaluations(root)
        self.assertEqual(comparison["candidate_minus_baseline"], 1.0)
        self.assertTrue((root / "comparison.json").exists())
        self.assertEqual((self.lab / "evaluations/baseline.json").read_bytes(), original)
