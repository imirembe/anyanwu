from array import array
import unittest
from unittest.mock import patch

from backend.voice_research.training_preflight import acoustic_metrics, cloud_guard, missing_labels


class VoiceTrainingPreflightTests(unittest.TestCase):
    def test_local_runtime_refused_before_media_or_model_access(self):
        with patch("backend.voice_research.training_preflight.platform.system",return_value="Darwin"):
            with self.assertRaisesRegex(RuntimeError,"hosted Colab"):
                cloud_guard()

    def test_silent_signal_is_not_classified_as_verified_speech(self):
        metrics=acoustic_metrics(array("h",[0]*1600),16000)
        self.assertEqual(metrics["low_energy_20ms_fraction"],1)
        self.assertEqual(metrics["near_full_scale_fraction"],0)
        self.assertIn("not speech recognition",metrics["interpretation"])

    def test_existing_training_ready_flag_does_not_replace_missing_transcript(self):
        row={"training_ready":True,"transcript":"  ","speaker_id":"source-A","split":"train",
             "igbo_speech_checked":True,"transcript_audio_match_checked":True,"single_speaker_checked":True,
             "music_overlap_checked":True,"utterance_boundaries_checked":True,"reviewer_type":"AI","review_notes":"Provisional research review"}
        self.assertEqual(missing_labels(row),["verbatim_transcript"])
        row["transcript"]="Ndewo."
        row["transcript_audio_match_checked"]=False
        self.assertIn("transcript_audio_match_checked",missing_labels(row))

    def test_peak_metric_distinguishes_full_scale_samples(self):
        metrics=acoustic_metrics(array("h",[32767,-32768,100,-100]),16000)
        self.assertEqual(metrics["near_full_scale_fraction"],0.5)


if __name__=="__main__":
    unittest.main()
