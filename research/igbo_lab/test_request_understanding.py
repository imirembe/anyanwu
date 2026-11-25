"""Software fixtures only; these sentences never enter published corpora."""
import copy
import hashlib
import json
import unicodedata
import unittest
from unittest.mock import patch

from . import details, intent, request_understanding as requests, resources, test_lab


def fixtures():
    rows=[{"id":f"{split}-{label}","label":label,"text":f"fixture {label} {split}","split":split,
           "source":"software-fixture","group":f"{split}-{label}","source_language_code":"ibo"}
          for split in ("train","validation","test") for label in (*intent.LABELS,"unknown")]
    rows.append({"id":"mixed-quarantine","label":"unknown","text":"mixed software fixture only","split":"train",
                 "source":"software-fixture","group":"mixed-file","source_language_code":"ig-en"})
    return rows


class RequestUnderstandingChecks(unittest.TestCase):
    def setUp(self):
        test_lab.CorpusAndBudgetChecks.setUp(self)
        for module in (requests,intent):
            change=patch.object(module,"LAB",self.lab);change.start();self.patches.append(change)
        change=patch.object(requests,"implementations",return_value={"fixture":"fixed implementation"})
        change.start();self.patches.append(change)

    tearDown=test_lab.CorpusAndBudgetChecks.tearDown

    def write_rows(self,path,rows):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))

    def review_fixture(self,rows):
        return [{"id":r["id"],"split":r["split"],"label":r["label"],
                 "text_sha256":details.offset_reference(r["text"])["transcript_sha256"],
                 "review_type":"AI","reviewer":"software fixture","confidence":"medium",
                 "reason":"Fixture empty four-field reference.","spans":[]}
                for r in rows if r["split"]!="train"]

    def prepare_fixture(self,fail=False):
        rows,quarantine=requests.validate_rows(fixtures())
        baseline=self.lab/"baseline";baseline.mkdir()
        self.write_rows(baseline/"dataset.jsonl",rows)
        thresholds={"minimum_similarity":0.15,"minimum_margin":0.03,"minimum_feature_coverage":0.35}
        calibration={"selected":{"thresholds":thresholds}}
        for name,value in (("plan.json",{}),("calibration.json",calibration),("result.json",{})):
            (baseline/name).write_text(json.dumps(value))
        review=self.project/"review.jsonl";self.write_rows(review,self.review_fixture(rows))
        for change in (patch.object(requests,"load_snapshot",return_value=(rows,quarantine,{"dataset_sha256":"fixture-source","test_sha256":"fixture-test","assembly_sha256":"fixture-assembly"})),
                       patch.object(intent,"load_completed",return_value=(baseline,{},calibration,{}))):
            change.start();self.patches.append(change)
        if fail:
            with patch.object(intent,"IntentIndex",side_effect=resources.LabError("fixture failure")),self.assertRaises(resources.LabError):
                requests.prepare_experiment(self.lab/"fixture",review)
        else:
            with patch("builtins.print"):requests.prepare_experiment(self.lab/"fixture",review)
        return requests.folder_for(requests.NAME)

    def test_unicode_spans_use_nfc_utf8_and_unseen_contact_is_raw(self):
        text="Kpọ Ada n’ekwentị"
        for original in (text,unicodedata.normalize("NFD",text)):
            normalized=details.transcript(original).encode()
            found=details.extract_details(original,"make_call")
            self.assertEqual([(s["field"],s["value"]) for s in found],[("contact_reference","Ada")])
            for s in found:self.assertEqual(normalized[s["start_byte"]:s["limit_byte"]].decode(),s["value"])
        self.assertNotEqual(details.offset_reference("ọ"),details.offset_reference("o"))

    def test_message_payload_and_unknown_topic_do_not_become_scheduled_actions(self):
        text="text ada and tell her to call obi at 5pm tomorrow"
        self.assertEqual([(s["field"],s["value"]) for s in details.extract_details(text,"text")],[("contact_reference","ada")])
        self.assertEqual(details.extract_details("transfer to Ada tomorrow at 5pm","unknown"),[])
        for text in ("a"*2049,"bad\x00text"):
            with self.assertRaises(resources.LabError):details.extract_details(text,"alarm")

    def test_multiple_clocks_dates_and_geographic_reference_remain_raw(self):
        found=details.extract_details("set an alarm for 2pm and 3pm tomorrow","alarm")
        self.assertEqual([s["value"] for s in found],["2pm","3pm","tomorrow"])
        found=details.extract_details("Tinye mgbịrịgba n'elekere anọ nke ụtụtụ echi","alarm")
        self.assertEqual([s["value"] for s in found],["elekere anọ nke ụtụtụ","echi"])
        found=details.extract_details("weather in Lagos tomorrow","weather")
        self.assertEqual([s["value"] for s in found],["Lagos","tomorrow"])
        self.assertEqual(details.extract_details("alarm for 132pm","alarm"),[])

    def test_data_quarantine_split_and_duplicate_guards(self):
        rows,quarantine=requests.validate_rows(fixtures())
        self.assertEqual(len(rows),27);self.assertEqual(len(quarantine),1)
        for mutation in ("duplicate","group","mixed","missing-label"):
            rows=copy.deepcopy(fixtures())
            if mutation=="duplicate":rows[1]["text"]=rows[0]["text"]
            elif mutation=="group":rows[9]["group"]=rows[0]["group"]
            elif mutation=="mixed":rows[-1]["label"]="alarm"
            else:rows[0]["label"]="unknown"
            with self.assertRaises(resources.LabError):requests.validate_rows(rows)

    def test_bad_detail_reference_and_missing_review_refused(self):
        rows,_=requests.validate_rows(fixtures());rows[0]["text"]="fixture train replacement"
        review=self.review_fixture(rows);path=self.project/"gold.jsonl"
        self.write_rows(path,review[:-1])
        with self.assertRaisesRegex(resources.LabError,"every validation/test"):requests.load_detail_review(path,rows)
        review[0]["spans"]=[{"field":"time_reference","value":"bad","start_byte":0,"limit_byte":2}]
        self.write_rows(path,review)
        with self.assertRaises(resources.LabError):requests.load_detail_review(path,rows)
        review=self.review_fixture(rows);review[0]["review_type"]="human";self.write_rows(path,review)
        with self.assertRaises(resources.LabError):requests.load_detail_review(path,rows)

    def test_prepare_fits_only_train_and_never_scores_test(self):
        real=intent.IntentIndex;seen=[]
        def fit(rows):
            seen.extend(rows);self.assertTrue(all(r["split"]=="train" for r in rows))
            index=real(rows);original=index.score
            def score(text):
                self.assertNotIn("test",text)
                return original(text)
            index.score=score;return index
        with patch.object(intent,"IntentIndex",side_effect=fit):folder=self.prepare_fixture()
        self.assertEqual(len(seen),9)
        self.assertNotIn("mixed",str(seen))
        self.assertEqual(requests.read_json(folder/"run.json")["status"],"prepared")
        self.assertFalse((folder/"result.json").exists())
        self.assertFalse(requests.read_json(folder/"calibration.json")["test_scored"])

    def test_evaluation_is_once_and_completed_reuse_does_not_fit(self):
        folder=self.prepare_fixture()
        with patch("builtins.print"):result=requests.evaluate_experiment()
        self.assertEqual(result["candidate"]["overall"]["examples"],9)
        self.assertTrue(result["test_scored_once"])
        before={p.name:requests.sha(p) for p in folder.iterdir() if p.is_file()}
        with patch.object(intent,"IntentIndex") as fit,patch("builtins.print"):
            requests.evaluate_experiment();fit.assert_not_called()
        self.assertEqual(before,{p.name:requests.sha(p) for p in folder.iterdir() if p.is_file()})
        self.assertFalse(list(folder.rglob("*.safetensors")))

    def test_tampering_or_failed_preparation_does_not_retry(self):
        folder=self.prepare_fixture(fail=True)
        with patch.object(intent,"IntentIndex") as fit,self.assertRaises(resources.LabError):requests.evaluate_experiment()
        fit.assert_not_called()
        with self.assertRaises(resources.LabError):requests.prepare_experiment()
        # A failed state cannot be made reusable just by adding a result file.
        (folder/"result.json").write_text("{}")
        with self.assertRaises(resources.LabError):requests.load_run(requests.NAME)

    def test_lookup_executes_nothing_saves_no_query_and_detects_artifact_edits(self):
        folder=self.prepare_fixture();before={p.name:requests.sha(p) for p in folder.iterdir() if p.is_file()}
        with patch("builtins.print"):result=requests.understand_request("fixture alarm train")
        self.assertFalse(result["action_executed"]);self.assertFalse(result["transcript_saved"])
        self.assertFalse(result["entities_resolved"])
        self.assertEqual(before,{p.name:requests.sha(p) for p in folder.iterdir() if p.is_file()})
        path=folder/"calibration.json";path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaises(resources.LabError):requests.understand_request("fixture alarm train")

    def test_span_metrics_count_extra_missing_and_wrong_boundaries(self):
        row={"id":"fixture"};gold={"fixture":{"spans":[{"field":"contact_reference","value":"Ada","start_byte":1,"limit_byte":4}]}}
        predicted=[[{"field":"contact_reference","value":"Ad","start_byte":1,"limit_byte":3}]]
        result=details.detail_metrics([row],predicted,gold)
        self.assertEqual({k:result["micro"][k] for k in ("tp","fp","fn")},{"tp":0,"fp":1,"fn":1})
        self.assertEqual(result["exact_records_with_gold_spans"],0)

    def test_path_and_runtime_guards(self):
        for name in ("../bad","/bad","bad/name",""):
            with self.assertRaises(resources.LabError):requests.folder_for(name)
        with patch.object(requests.time,"perf_counter",return_value=61),self.assertRaises(resources.LabError):requests.runtime_guard(0)


if __name__=="__main__":unittest.main()
