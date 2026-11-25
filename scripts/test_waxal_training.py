"""Meaningful source/split/guard checks with no local speech/model imports."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.voice_research.waxal_training_data import acquire_waxal_pairs


class FemaleTrainingTests(unittest.TestCase):
    def test_laptop_stops_before_network_or_media(self):
        with patch('backend.voice_research.waxal_training_data.platform.system',return_value='Darwin'):
            with self.assertRaisesRegex(RuntimeError,'no laptop'):
                acquire_waxal_pairs({}, {},lambda _:None)

    def test_publisher_labels_and_disjoint_adapter_validation(self):
        plan=json.loads((ROOT/'.build/VoiceResearch/training/waxal-female5-lora-v1/plan.json').read_text())
        clips=plan['clips'];train={c['id'] for c in clips if c['adapter_split']=='train'};validation={c['id'] for c in clips if c['adapter_split']=='validation'}
        self.assertEqual((len(clips),len(train),len(validation)),(24,20,4))
        self.assertFalse(train&validation)
        self.assertTrue(all(c['source']=='google/WaxalNLP' and c['speaker_id']=='5' and c['gender']=='Female' and not c['truncated_cells'] for c in clips))
        self.assertTrue(all(c['source_split']=='train' and not c['independent_native_review'] for c in clips))
        self.assertEqual(plan['updates'],60)

    def test_source_texts_unchanged_and_rejected_reviews_preserved(self):
        prior=ROOT/'.build/VoiceResearch/training/anyanwu-female-v1'
        original=json.loads((prior/'waxal-female-short-rows.json').read_text())
        plan=json.loads((ROOT/'.build/VoiceResearch/training/waxal-female5-lora-v1/plan.json').read_text())
        source={r['row']['id']:r['row']['text'] for r in original['rows']}
        self.assertTrue(all(c['text']==source[c['id']] for c in plan['clips']))
        rejected=json.loads((prior/'expansion-v1/submitted-human-feedback-summary.json').read_text())
        self.assertEqual((rejected['accepted'],rejected['held']),(0,7))
        self.assertTrue(all(not c['id'].startswith('bbc') for c in plan['clips']))


if __name__=='__main__':unittest.main()
