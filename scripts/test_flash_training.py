"""Meaningful loss alignment, selection and cloud guard checks, with no torch/model IO."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.voice_research.flash_adapter import speech_targets
from backend.voice_research.flash_train_worker import guard


class SpeechPilotTests(unittest.TestCase):
    def test_task_predicts_first_acoustic_token_and_last_input_predicts_eos(self):
        # SOS, two prefix, three text, speaker, TASK, four speech inputs.
        y=speech_targets(2,3,[4,6,2,1],10)
        self.assertEqual(len(y),12)
        self.assertEqual(y[:7],[-100]*7)
        self.assertEqual(y[7:],[4,6,2,1,11])

    def test_invalid_targets_rejected(self):
        for ids in ([],[-1],[10],[1.5],[True]):
            with self.assertRaises(ValueError):
                speech_targets(2,3,ids,10)

    def test_laptop_worker_stops_before_importing_torch(self):
        with self.assertRaises(RuntimeError):
            guard()

    def test_only_unflagged_candidates_and_preserve_provisional_status(self):
        p=json.loads((ROOT/'.build/VoiceResearch/training/anyanwu-female-v1/flash-training-plan.json').read_text())
        self.assertEqual(len(p['cases']),3)
        self.assertEqual(len({c['source_group'] for c in p['cases']}),3)
        self.assertTrue(all(not c['training_ready'] and c['split'] is None and not c['timing_warnings'] for c in p['cases']))
        self.assertEqual(p['updates'],5)
        self.assertLess(sum(f['size'] for m in p['models'] for f in m['files']),4*1024**3)


if __name__=='__main__':
    unittest.main()
