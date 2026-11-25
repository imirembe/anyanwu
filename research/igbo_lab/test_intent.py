"""Software-only fixtures: none of this synthetic text enters research corpora."""
import json
from pathlib import Path
import unicodedata
import unittest
from unittest.mock import patch

from . import intent, resources, test_lab


def fixture_rows():
    return [{"id":f"{split}-{label}","label":label,"text":f"fixture {label} marker", "split":split,
             "group":f"{split}-{label}","source":"software-fixture","source_content_sha256":"fixture","review_type":"AI"}
            for split in ("train","validation","test") for label in (*intent.LABELS,"unknown")]


class IntentChecks(unittest.TestCase):
    def setUp(self):
        test_lab.CorpusAndBudgetChecks.setUp(self)
        change=patch.object(intent,"LAB",self.lab);change.start();self.patches.append(change)

    tearDown=test_lab.CorpusAndBudgetChecks.tearDown

    def test_unicode_marks_survive_and_canonical_equivalents_match(self):
        marked="Ị́ ị̀ ọ́ ọ̀ ụ́ ụ̀ Ṅ́"
        self.assertEqual(intent.features(marked),intent.features(unicodedata.normalize("NFD",marked)))
        self.assertIn("w:ị́",intent.features(marked))
        self.assertNotEqual(intent.features("ọ"),intent.features("o"))
        self.assertNotEqual(intent.features("ụ́"),intent.features("ụ̀"))

    def test_idf_and_neighbours_fit_training_only(self):
        rows=fixture_rows()
        model=intent.IntentIndex([r for r in rows if r["split"]=="train"])
        self.assertNotIn("w:validationonly",model.idf)
        model.score("validationonly")
        self.assertNotIn("w:validationonly",model.idf)
        with self.assertRaisesRegex(resources.LabError,"training rows only"):
            intent.IntentIndex(rows)
        for label in intent.LABELS:
            score=model.score(f"fixture {label} marker")
            self.assertEqual(intent.decide(score,{"minimum_similarity":0.1,"minimum_margin":0,"minimum_feature_coverage":0.1}),label)

    def test_unknown_background_empty_input_and_input_limit(self):
        model=intent.IntentIndex([r for r in fixture_rows() if r["split"]=="train"])
        thresholds={"minimum_similarity":0.15,"minimum_margin":0.03,"minimum_feature_coverage":0.35}
        self.assertEqual(intent.decide(model.score("fixture unknown marker"),thresholds),"unknown")
        self.assertEqual(intent.decide(model.score(""),thresholds),"unknown")
        self.assertEqual(intent.decide(model.score("𐐷"*20),thresholds),"unknown")
        for text in ("a"*2049,"bad\x00text"):
            with self.assertRaises(resources.LabError):model.score(text)

    def test_validation_calibration_rejects_test_rows_and_prefers_no_accepted_errors(self):
        rows=[{"label":"alarm","split":"validation"},{"label":"unknown","split":"validation"}]
        scores=[{"nearest_known_label":"alarm","similarity":0.6,"margin":0.3,"feature_coverage":1},
                {"nearest_known_label":"alarm","similarity":0.2,"margin":0.02,"feature_coverage":1}]
        chosen=intent.calibrate(rows,scores)["selected"]
        self.assertEqual(chosen["metrics"]["accepted_errors"],0)
        self.assertEqual(chosen["metrics"]["known_correct"],1)
        with self.assertRaisesRegex(resources.LabError,"validation-only"):
            intent.calibrate([{**r,"split":"test"} for r in rows],scores)

    def test_ambiguous_equal_scores_abstain_even_at_zero_margin_threshold(self):
        score={"nearest_known_label":"alarm","similarity":0.8,"margin":0.0,"feature_coverage":1}
        self.assertEqual(intent.decide(score,{"minimum_similarity":0.15,"minimum_margin":0.0,"minimum_feature_coverage":0.35}),"unknown")

    def test_source_group_and_duplicate_queries_are_refused(self):
        request=self.lab/"curation/assemblies/requests";background=self.lab/"curation/assemblies/background"
        request.mkdir(parents=True);background.mkdir()
        # Eight known labels plus unknown in each split, all software-only fixtures.
        rows=fixture_rows()
        request_rows=[{**r,"text":r['text']+" "+r['split'],"source_intent":r["label"],"source_config":"ibo",
                       "source":"masakhane/InjongoIntent","review":{"type":"AI"}} for r in rows if r["label"]!="unknown"]
        background_rows=[{**r,"text":"<|user|>"+r['text']+" "+r['split']+"<|assistant|>fixture<|end|>","review":{"type":"AI"}} for r in rows if r["label"]=="unknown"]
        decisions=[{**{k:r[k] for k in ("id","source","group","split","source_content_sha256")},"expected_intent":"unknown",
                    "review_type":"AI","confidence":"medium","reason":"Fixture unknown review."} for r in background_rows]
        (request/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in request_rows))
        (background/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in background_rows))
        review=self.project/"background.jsonl";review.write_text("".join(json.dumps(r)+"\n" for r in decisions))
        with patch.object(intent,"assembly_identity",return_value=(None,{"assembly_sha256":"fixture"})):
            actual,_=intent.build_dataset(request,background,review)
            self.assertEqual(len(actual),27)
            request_rows[8]["group"]=request_rows[0]["group"]
            (request/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in request_rows))
            with self.assertRaisesRegex(resources.LabError,"crosses official splits"):
                intent.build_dataset(request,background,review)
            request_rows[8]["group"]="other-group"
            request_rows[8]["text"]=request_rows[0]["text"]
            (request/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in request_rows))
            with self.assertRaisesRegex(resources.LabError,"Duplicate"):
                intent.build_dataset(request,background,review)

    def run_fixture(self,fail=False):
        request=self.lab/"request";background=self.lab/"background"
        review=self.project/"review.jsonl";review.write_text("{}\n")
        cases=self.project/"cases.jsonl";cases.write_text("{}\n")
        manifest={"parents":[{"sample":"fixture"}]}
        changes=[patch.object(intent,"assembly_identity",return_value=(request,manifest)),
                 patch.object(intent,"assemble_reviewed",return_value=manifest),
                 patch.object(intent,"build_dataset",return_value=(fixture_rows(),{"fixture":"fixed source"})),
                 patch.object(intent,"FROZEN_CASES",cases)]
        for change in changes:change.start();self.patches.append(change)
        if fail:
            with patch.object(intent,"IntentIndex",side_effect=resources.LabError("fixture fit failed")),self.assertRaises(resources.LabError):
                intent.run_intent_experiment(request,background,review)
        else:
            with patch("builtins.print"):intent.run_intent_experiment(request,background,review)
        return request,background,review

    def test_completed_experiment_reuses_results_without_refitting_or_new_weights(self):
        request,background,review=self.run_fixture()
        folder=intent.folder_for(intent.NAME)
        hashes={p.name:intent.sha(p) for p in folder.iterdir() if p.is_file()}
        with patch.object(intent,"IntentIndex") as fit,patch("builtins.print"):
            result=intent.run_intent_experiment(request,background,review)
            fit.assert_not_called()
        self.assertFalse(result["persisted_intent_weights"])
        self.assertEqual(hashes,{p.name:intent.sha(p) for p in folder.iterdir() if p.is_file()})
        self.assertFalse(list(folder.rglob("*.safetensors")))

    def test_failed_experiment_does_not_retry_automatically(self):
        request,background,review=self.run_fixture(fail=True)
        with patch.object(intent,"IntentIndex") as fit,self.assertRaisesRegex(resources.LabError,"not retried automatically"):
            intent.run_intent_experiment(request,background,review)
        fit.assert_not_called()

    def test_saved_artifact_tampering_refused_and_lookup_saves_no_transcript(self):
        self.run_fixture()
        folder=intent.folder_for(intent.NAME)
        before={p.name:intent.sha(p) for p in folder.iterdir() if p.is_file()}
        with patch("builtins.print"):
            result=intent.classify_intent("fixture alarm marker")
        self.assertEqual(result["intent"],"alarm")
        self.assertFalse(result["action_executed"])
        self.assertFalse(result["transcript_saved"])
        self.assertEqual(before,{p.name:intent.sha(p) for p in folder.iterdir() if p.is_file()})
        path=folder/"calibration.json";path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaisesRegex(resources.LabError,"changed after completion"):
            intent.classify_intent("fixture alarm marker")

    def test_output_path_cannot_escape_the_lab(self):
        for name in ("../outside","/absolute","x/y",""):
            with self.assertRaises(resources.LabError):intent.folder_for(name)


if __name__=="__main__":unittest.main()
