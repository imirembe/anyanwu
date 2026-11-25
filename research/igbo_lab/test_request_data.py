import hashlib
import json
import re
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from . import assembly, english_data, english_oos, experiments, importer, request_data, resources, review, test_pipeline


class IntentClient(test_pipeline.FakeHubClient):
    def __init__(self,source="injongo-igbo"):
        super().__init__(source=source)

    def get(self,url,**kwargs):
        if "/rows?" not in url:
            return super().get(url,**kwargs)
        self.urls.append(url); self.requests+=1; self.received+=100; self.remaining-=100
        q=parse_qs(urlparse(url).query)
        split=q["split"][0]; length=int(q["length"][0]);start=int(q["offset"][0])
        intent=request_data.INTENT_ORDER[start//request_data.BLOCK_ROWS[split]]
        within=start%request_data.BLOCK_ROWS[split]
        prefix="dev" if split=="validation" else split
        return {"num_rows_total":40*request_data.BLOCK_ROWS[split],
                "rows":[{"row_idx":start+i,"truncated_cells":[],
                         "row":{"text":f"Ị́ ị̀ ọ́ ụ̀. Fixture {intent} {split} {within+i}.","intent":intent,
                                "example_id":f"{prefix}-{within+i:08d}",
                                "spans":[{"private":"unused"}],"speaker_id":"private-speaker", "gender":"unused",
                                "audio_path":"https://audio.example/never.wav"}} for i in range(length)]}


class EnglishClient(test_pipeline.FakeHubClient):
    def __init__(self,source="clinc-english"):
        super().__init__(source=source)
        self.labels=list(reversed(self.spec["intents"])) + ["w2","rollover_401k"]

    def get(self,url,**kwargs):
        if "/api/datasets/" in url:
            value=super().get(url,**kwargs);value["cardData"]["license"]=[self.spec["license"]]
            return value
        if "/info?" in url:
            self.urls.append(url);self.requests+=1;self.received+=100;self.remaining-=100
            return {"dataset_info":{"features":{"intent":{"_type":"ClassLabel","names":self.labels}}},"partial":False}
        if "/filter?" not in url and "/rows?" not in url:return super().get(url,**kwargs)
        self.urls.append(url);self.requests+=1;self.received+=100;self.remaining-=100
        q=parse_qs(urlparse(url).query);n=int(q["length"][0]);split=q["split"][0]
        block_rows={"train":100,"validation":20,"test":30}[split]
        label=int(q["where"][0].split("=")[1]) if "where" in q else int(q["offset"][0])//block_rows
        within=0 if "where" in q else int(q["offset"][0])%block_rows
        return {"features":[{"name":"intent","type":{"names":self.labels}}],"partial":False,"num_rows_total":100 if "/filter?" in url else {"train":15250,"validation":3100,"test":5500}[split],
                "rows":[{"row_idx":label*block_rows+within+i,"truncated_cells":[],"row":{"text":f"English fixture {self.labels[label]} {split} {within+i}.","intent":label}} for i in range(n)]}


class OOSClient(test_pipeline.FakeHubClient):
    def __init__(self):
        super().__init__(source="clinc-original-oos")
        self.labels=["oos"]+[f"fixture{i}" for i in range(7)]

    def get(self,url,**kwargs):
        if "/info?" in url:
            self.urls.append(url);self.requests+=1;self.received+=100;self.remaining-=100
            return {"dataset_info":{"features":{"intent":{"_type":"ClassLabel","names":self.labels}}}}
        if "/rows?" not in url:return super().get(url,**kwargs)
        self.urls.append(url);self.requests+=1;self.received+=100;self.remaining-=100
        q=parse_qs(urlparse(url).query);split=q["split"][0];start=int(q["offset"][0]);n=int(q["length"][0])
        return {"features":[{"name":"intent","type":{"names":self.labels}}],"num_rows_total":english_oos.LAYOUT[split][2],
                "rows":[{"row_idx":start+i,"truncated_cells":[],"row":{"intent":0,"text":f"Original OOS fixture {split} {i}."}} for i in range(n)]}


class RequestDataChecks(unittest.TestCase):
    fixtures=test_pipeline.ReviewPipelineChecks.fixtures
    candidates=test_pipeline.ReviewPipelineChecks.candidates
    write_rows=test_pipeline.ReviewPipelineChecks.write_rows

    def setUp(self):
        test_pipeline.ReviewPipelineChecks.setUp(self)
        for module in (request_data,english_data,english_oos):
            change=patch.object(module,"LAB",self.lab);change.start();self.patches.append(change)

    tearDown=test_pipeline.ReviewPipelineChecks.tearDown

    def sample(self,client=None):
        with patch("builtins.print"):
            report=importer.import_text("injongo-igbo",80,client=client or IntentClient())
        folder=self.lab/"imports/injongo-igbo"
        return report,review.load_jsonl(folder/"candidates.jsonl")

    def test_balanced_igbo_requests_preserve_official_splits_and_drop_unused_fields(self):
        client=IntentClient();report,rows=self.sample(client)
        self.assertEqual(report["sample_counts_by_source_split"],{"train":56,"validation":8,"test":16})
        for counts in report["sample_counts_by_split_intent"].values():self.assertEqual(len(set(counts.values())),1)
        for r in rows:
            self.assertEqual(r["source_language_code"],"ibo")
            self.assertEqual(r["source_config"],"ibo")
            self.assertNotIn("input",r["content"])
        saved=(self.lab/"imports/injongo-igbo/candidates.jsonl").read_text()
        for omitted in ("audio.example","private-speaker",'"spans"','"gender"'):self.assertNotIn(omitted,saved)
        self.assertTrue(all("config=ibo" in u for u in client.urls if "/rows?" in u))
        self.assertEqual(client.requests,28)
        self.assertFalse(report["audio_downloaded"])

    def test_bad_quota_or_offset_refused_before_token_or_network(self):
        with patch.object(importer,"private_token") as token,patch.object(importer,"BoundedHubClient") as client:
            for n in (1,79,81,500):
                with self.assertRaises(resources.LabError):importer.import_text("injongo-igbo",n,token_prompt=True)
            with self.assertRaises(resources.LabError):importer.import_text("injongo-igbo",80,offset=1,token_prompt=True)
            token.assert_not_called();client.assert_not_called()

    def test_truncation_wrong_filter_label_and_missing_identity_install_nothing(self):
        for mutation in ("truncated","label","index","id","subset","partial","short"):
            client=IntentClient();original=client.get
            def changed(url,**kwargs):
                value=original(url,**kwargs)
                if "/rows?" in url:
                    if mutation=="truncated":value["rows"][0]["truncated_cells"]=["text"]
                    elif mutation=="label":value["rows"][0]["row"]["intent"]="text"
                    elif mutation=="index":value["rows"][0]["row_idx"]=True
                    elif mutation=="id":value["rows"][0]["row"]["example_id"]="test-00000000"
                    elif mutation=="subset":value["config"]="eng"
                    elif mutation=="partial":value["partial"]=True
                    else:value["rows"].pop()
                return value
            client.get=changed
            with self.assertRaises(resources.LabError):self.sample(client)
            self.assertFalse((self.lab/"imports/injongo-igbo").exists())
            self.assertFalse(any("/filter?" in u or "/first-rows?" in u or ".parquet" in u for u in client.urls))

    def test_source_head_change_and_partial_transaction_never_replace_originals(self):
        client=IntentClient();client.changed_revision=True
        with self.assertRaisesRegex(resources.LabError,"changed during import"):self.sample(client)
        with patch.object(request_data,"write_json",side_effect=OSError("fixture failure")),self.assertRaises(OSError):self.sample()
        self.assertFalse((self.lab/"imports/import-staging").exists())
        self.assertFalse((self.lab/"imports/injongo-igbo").exists())
        self.sample()
        original=(self.lab/"imports/injongo-igbo/candidates.jsonl").read_bytes()
        with patch.object(importer,"private_token") as token,self.assertRaises(resources.LabError):self.sample()
        token.assert_not_called()
        self.assertEqual((self.lab/"imports/injongo-igbo/candidates.jsonl").read_bytes(),original)

    def test_duplicate_utterances_keep_distinct_ids_and_shared_group_across_splits(self):
        spec=importer.SOURCES["injongo-igbo"]
        text="Ị chọrọ oge?"
        rows=[importer.text_record(spec,"a"*40,split,0,{"text":text,"intent":"time","example_id":prefix+"-00000000"})
              for split,prefix in (("train","train"),("validation","dev"),("test","test"))]
        self.assertEqual(len({r["id"] for r in rows}),3)
        self.assertEqual(len({r["group"] for r in rows}),1)
        self.assertIn("An accepted utterance group spans official splits.",review.training_readiness(rows,"injongo-igbo"))

    def test_review_preserves_source_labels_without_making_assistant_replies(self):
        _,rows=self.sample()
        _,_,_,_,cases=self.fixtures()
        decisions=[{"id":r["id"],"content_sha256":r["content_sha256"],"decision":"accept","content_use":"language-reference",
                    "review_type":"AI","reviewer":"Fixture Codex","reason":"Fixture source request check.","confidence":"medium"} for r in rows]
        path=self.project/"request-review.jsonl";self.write_rows(path,decisions)
        with patch("builtins.print"):report=review.compile_review("injongo-igbo",path,cases)
        kept=review.load_jsonl(self.lab/"curation/injongo-igbo/reviewed.jsonl")
        self.assertEqual(report["accepted_counts_by_split"],{"train":56,"validation":8,"test":16})
        self.assertEqual(kept[0]["source_intent"],rows[0]["source_intent"])
        self.assertEqual(kept[0]["source_example_id"],rows[0]["source_example_id"])
        self.assertEqual(kept[0]["text"],rows[0]["content"]["text"])
        self.assertNotIn("<|assistant|>",kept[0]["text"])
        self.assertIn("no assistant reply target",kept[0]["text_derivation"])

    def test_user_request_cannot_leak_a_frozen_reply_benchmark_prompt(self):
        _,rows=self.sample()
        _,_,_,cases,case_path=self.fixtures()
        cases[0]["prompt"]=rows[0]["content"]["text"];self.write_rows(case_path,cases)
        decision={"id":rows[0]["id"],"content_sha256":rows[0]["content_sha256"],"decision":"edit","text":"Changed visible text.",
                  "review_type":"AI","reviewer":"Fixture Codex","reason":"Fixture edit cannot hide original overlap.","confidence":"medium"}
        path=self.project/"request-review.jsonl";self.write_rows(path,[decision])
        with patch("builtins.print"):report=review.compile_review("injongo-igbo",path,case_path)
        self.assertEqual(report["accepted_records"],0)
        self.assertIn("Evaluation prompt overlap",report["rejected_records"][0]["reason"])

    def test_request_only_corpus_cannot_replace_reply_corpus_or_start_reply_training(self):
        _,rows=self.sample()
        _,_,_,_,cases=self.fixtures()
        decisions=[{"id":r["id"],"content_sha256":r["content_sha256"],"decision":"accept","review_type":"AI",
                    "reviewer":"Fixture Codex","reason":"Fixture source request check.","confidence":"medium"} for r in rows]
        path=self.project/"request-review.jsonl";self.write_rows(path,decisions)
        with patch("builtins.print"):
            review.compile_review("injongo-igbo",path,cases)
            receipt=assembly.assemble_reviewed(["injongo-igbo"],cases)
        corpus=self.lab/"corpus";corpus.mkdir();marker=corpus/"preserved.txt";marker.write_text("existing tokens")
        with self.assertRaisesRegex(resources.LabError,"not assistant reply targets"):
            review.prepare_reviewed(cases,source="injongo-igbo")
        folder=self.lab/"curation/assemblies"/receipt["assembly_sha256"][:16]
        with self.assertRaisesRegex(resources.LabError,"no assistant reply targets"):
            experiments.run_experiment(folder,cases)
        self.assertEqual(marker.read_text(),"existing tokens")
        self.assertFalse((self.lab/"experiments").exists())

    def test_new_igbo_slice_skips_v1_and_keeps_original_row_positions(self):
        client=IntentClient("injongo-igbo-v2")
        # The fixture release is deliberately pinned instead of bypassing the revision check.
        with patch.dict(importer.SOURCES["injongo-igbo-v2"],{"required_revision":"a"*40}),patch("builtins.print"):
            report=importer.import_text("injongo-igbo-v2",160,client=client)
        rows=review.load_jsonl(self.lab/"imports/injongo-igbo-v2/candidates.jsonl")
        for split,start in (("train",14),("validation",2),("test",4)):
            first=next(r for r in rows if r["split"]==split)
            self.assertEqual(first["source_row_index"],start)
        self.assertEqual(report["slot_status_counts"],{"invalid-byte-boundary-or-label":160})
        self.assertFalse((self.lab/"imports/injongo-igbo").exists())

    def test_source_slots_validate_utf8_boundaries_and_edits_invalidate_direct_offsets(self):
        text="ọgwụ taa"
        start=len("ọgwụ ".encode())
        good={"text":text,"spans":[{"start_byte":start,"limit_byte":len(text.encode()),"label":"DATE"}]}
        meta=importer.slot_metadata(good)
        self.assertTrue(meta["source_slot_alignment_valid"])
        self.assertEqual(meta["source_slots"][0]["value"],"taa")
        self.assertFalse(meta["source_slot_semantics_reviewed"])
        bad={"text":text,"spans":[{"start_byte":1,"limit_byte":2,"label":"DATE"}]}
        self.assertFalse(importer.slot_metadata(bad)["source_slot_alignment_valid"])
        _,rows=self.sample()
        raw=self.lab/"imports/injongo-igbo/candidates.jsonl"
        rows[0].update(meta)
        self.write_rows(raw,rows)
        (raw.parent/"manifest.json").write_text(json.dumps({"candidates_sha256":hashlib.sha256(raw.read_bytes()).hexdigest()}))
        _,_,_,_,cases=self.fixtures()
        d={"id":rows[0]["id"],"content_sha256":rows[0]["content_sha256"],"decision":"edit","text":"A changed fixture request.",
           "review_type":"AI","reviewer":"Fixture Codex","reason":"Fixture edit.","confidence":"medium"}
        p=self.project/"slots-review.jsonl";self.write_rows(p,[d])
        with patch("builtins.print"):review.compile_review("injongo-igbo",p,cases)
        kept=review.load_jsonl(self.lab/"curation/injongo-igbo/reviewed.jsonl")[0]
        self.assertFalse(kept["slots_usable_with_reviewed_text"])
        self.assertEqual(kept["source_slot_text_sha256"],meta["source_slot_text_sha256"])

    def test_english_class_ids_are_decoded_from_official_map_with_real_source_indices(self):
        client=EnglishClient()
        with patch("builtins.print"):report=importer.import_text("clinc-english",96,client=client)
        rows=review.load_jsonl(self.lab/"imports/clinc-english/candidates.jsonl")
        self.assertEqual(report["sample_counts_by_source_split"],{"train":64,"validation":16,"test":16})
        for r in rows:
            self.assertEqual(r["source_intent"],client.labels[r["source_intent_id"]])
            self.assertEqual(r["source_language_code"],"eng")
            self.assertEqual(r["source_config"],"plus")
            self.assertTrue(r["source_example_id_kind"].startswith("derived-from-official"))
        self.assertEqual(client.requests,37)
        self.assertFalse(any(".parquet" in u or "/first-rows?" in u for u in client.urls))

    def test_english_bad_labels_truncation_or_page_map_install_nothing(self):
        for mutation in ("map","id","truncated","index","partial","holdout-label","holdout-index","holdout-size"):
            client=EnglishClient();get=client.get
            def changed(url,**kwargs):
                value=get(url,**kwargs)
                if "/filter?" in url:
                    if mutation=="map":value["features"][0]["type"]["names"]=list(reversed(client.labels))
                    elif mutation=="id":value["rows"][0]["row"]["intent"]=-1
                    elif mutation=="truncated":value["rows"][0]["truncated_cells"]=["text"]
                    elif mutation=="index":value["rows"][0]["row_idx"]=True
                    elif mutation=="partial":value["partial"]=True
                if "/rows?" in url:
                    if mutation=="holdout-label":value["rows"][0]["row"]["intent"]=-1
                    elif mutation=="holdout-index":value["rows"][0]["row_idx"]+=1
                    elif mutation=="holdout-size":value["num_rows_total"]-=1
                return value
            client.get=changed
            with self.assertRaises(resources.LabError):importer.import_text("clinc-english",96,client=client)
            self.assertFalse((self.lab/"imports/clinc-english").exists())
        with patch.object(importer,"private_token") as token,self.assertRaises(resources.LabError):
            importer.import_text("clinc-english",80,token_prompt=True)
        token.assert_not_called()

    def test_mixed_text_is_one_published_file_group_in_train_only(self):
        client=test_pipeline.FakeHubClient(source="orinode-mixed");get=client.get
        def changed(url,**kwargs):
            if "/rows?" not in url:return get(url,**kwargs)
            client.urls.append(url);client.requests+=1;client.received+=100;client.remaining-=100
            return {"num_rows_total":15000,"partial":False,"rows":[{"row_idx":5000+i,"truncated_cells":[],
                "row":{"text":f"{i+1}. Published-source fixture line {i}."}} for i in range(32)]}
        client.get=changed
        with patch.dict(importer.SOURCES["orinode-mixed"],{"required_revision":"a"*40}),patch("builtins.print"):
            report=importer.import_text("orinode-mixed",32,offset=5000,client=client)
        rows=review.load_jsonl(self.lab/"imports/orinode-mixed/batches/offset-0005000/candidates.jsonl")
        self.assertEqual(report["records"],32)
        self.assertEqual({r["split"] for r in rows},{"train"})
        self.assertEqual(len({r["group"] for r in rows}),1)
        self.assertTrue(all(r["source_file"]=="igbo_english_5000.txt" for r in rows))
        self.assertEqual([r["source_row_index"] for r in rows],list(range(5000,5032)))
        self.assertFalse(any("audio" in u or ".txt" in u or ".parquet" in u for u in client.urls))
        issues=review.training_readiness(rows,"orinode-mixed")
        self.assertIn("No accepted records remain in the official validation split.",issues)
        self.assertIn("No accepted records remain in the official test split.",issues)

    def test_mixed_text_wrong_section_number_or_unbounded_request_is_refused(self):
        spec=importer.SOURCES["orinode-mixed"]
        for index,text in ((4999,"1. Fixture."),(5000,"2. Fixture."),(5032,"33. Fixture.")):
            self.assertIsNone(importer.text_record(spec,"a"*40,"train",index,{"text":text}))
        with patch.object(importer,"private_token") as token,patch.object(importer,"BoundedHubClient") as client:
            for n,offset in ((32,0),(31,5000),(500,5000),(32,10000)):
                with self.assertRaises(resources.LabError):importer.import_text("orinode-mixed",n,offset=offset,token_prompt=True)
            token.assert_not_called();client.assert_not_called()
        with self.assertRaisesRegex(resources.LabError,"not assistant reply targets"):
            review.prepare_reviewed(self.project/"unused-cases.jsonl",source="orinode-mixed",offset=5000)

    def test_new_english_known_slice_skips_old_rows_and_keeps_other_action_labels(self):
        client=EnglishClient("clinc-english-v3")
        positions={label:client.labels.index(label) for label in client.spec["intents"]}
        with patch.dict(importer.SOURCES["clinc-english-v3"],{"required_revision":"a"*40,"class_block_positions":positions}),patch("builtins.print"):
            report=importer.import_text("clinc-english-v3",192,client=client)
        rows=review.load_jsonl(self.lab/"imports/clinc-english-v3/candidates.jsonl")
        self.assertEqual(report["sample_counts_by_source_split"],{"train":128,"validation":32,"test":32})
        for split,offset,size in (("train",8,100),("validation",2,20),("test",2,30)):
            first=next(r for r in rows if r["split"]==split)
            self.assertEqual(first["source_row_index"]%size,offset)
        self.assertFalse(any("/filter?" in u for u in client.urls))
        other=EnglishClient("clinc-english-other")
        with patch.dict(importer.SOURCES["clinc-english-other"],{"required_revision":"a"*40}),patch("builtins.print"):
            importer.import_text("clinc-english-other",96,client=other)
        rows=review.load_jsonl(self.lab/"imports/clinc-english-other/candidates.jsonl")
        self.assertEqual({r["source_intent"] for r in rows},set(other.spec["intents"]))
        self.assertNotIn("unknown",{r["source_intent"] for r in rows})

    def test_new_igbo_slices_skip_both_old_stages_and_invalid_english_plan_refused_early(self):
        for source,offsets in (("injongo-igbo-v3",{"train":28,"validation":4,"test":8}),
                               ("injongo-igbo-other-v3",{"train":7,"validation":1,"test":2})):
            client=IntentClient(source)
            with patch.dict(importer.SOURCES[source],{"required_revision":"a"*40}),patch("builtins.print"):
                importer.import_text(source,80,client=client)
            rows=review.load_jsonl(self.lab/f"imports/{source}/candidates.jsonl")
            for split,start in offsets.items():
                r=next(r for r in rows if r["split"]==split)
                self.assertEqual(r["source_row_index"]%request_data.BLOCK_ROWS[split],start)
            self.assertFalse(any("source_slots" in r for r in rows))
        with patch.dict(importer.SOURCES["clinc-english-v3"],{"within_class_start":{"validation":19}}),patch.object(importer,"private_token") as token:
            with self.assertRaises(resources.LabError):importer.import_text("clinc-english-v3",192,token_prompt=True)
            token.assert_not_called()

    def test_original_english_oos_uses_explicit_rows_and_preserves_original_label(self):
        client=OOSClient()
        with patch.dict(importer.SOURCES["clinc-original-oos"],{"required_revision":"a"*40}),patch("builtins.print"):
            report=importer.import_text("clinc-original-oos",96,client=client)
        rows=review.load_jsonl(self.lab/"imports/clinc-original-oos/candidates.jsonl")
        self.assertEqual(report["sample_counts_by_source_split"],{"train":64,"validation":16,"test":16})
        self.assertEqual({r["source_intent"] for r in rows},{"oos"})
        self.assertTrue(all(r["source_language_code"]=="eng" and r["source_intent_id"]==0 for r in rows))
        self.assertEqual(client.requests,8)
        self.assertFalse(any("/filter?" in u or ".parquet" in u for u in client.urls))

    def test_original_oos_partial_wrong_class_or_changed_layout_installs_nothing(self):
        for mutation in ("short","label","index","map","total","truncated"):
            client=OOSClient();get=client.get
            def changed(url,**kwargs):
                page=get(url,**kwargs)
                if "/rows?" in url:
                    if mutation=="short":page["rows"].pop()
                    elif mutation=="label":page["rows"][0]["row"]["intent"]=1
                    elif mutation=="index":page["rows"][0]["row_idx"]=True
                    elif mutation=="map":page["features"][0]["type"]["names"]=list(reversed(client.labels))
                    elif mutation=="total":page["num_rows_total"]-=1
                    else:page["rows"][0]["truncated_cells"]=["text"]
                return page
            client.get=changed
            with patch.dict(importer.SOURCES["clinc-original-oos"],{"required_revision":"a"*40}),self.assertRaises(resources.LabError):
                importer.import_text("clinc-original-oos",96,client=client)
            self.assertFalse((self.lab/"imports/clinc-original-oos").exists())

    def test_original_oos_bad_quota_and_transaction_failure_preserve_inputs(self):
        with patch.object(importer,"private_token") as token,self.assertRaises(resources.LabError):
            importer.import_text("clinc-original-oos",95,token_prompt=True)
        token.assert_not_called()
        with patch.dict(importer.SOURCES["clinc-original-oos"],{"required_revision":"a"*40}),patch.object(english_oos,"write_json",side_effect=OSError("fixture failure")),patch("builtins.print"),self.assertRaises(OSError):
            importer.import_text("clinc-original-oos",96,client=OOSClient())
        self.assertFalse((self.lab/"imports/clinc-original-oos").exists())
        self.assertFalse((self.lab/"imports/import-staging").exists())


if __name__=="__main__":unittest.main()
