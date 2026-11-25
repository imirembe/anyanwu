import unittest
import unicodedata
from unittest.mock import patch
from backend.voice_research.alignment_rules import alignment_text,eligible_caption,evaluate_path,caption_token_ranges
from backend.voice_research.alignment_setup import run_alignment


class CaptionAlignmentTests(unittest.TestCase):
    def test_local_worker_stops_before_audio_or_models(self):
        with patch('backend.voice_research.alignment_setup.platform.system',return_value='Darwin'):
            with self.assertRaisesRegex(RuntimeError,'Colab'):
                run_alignment({},'',lambda _:None)

    def test_igbo_dots_and_tones_survive_alignment_normalisation(self):
        text="Ọ BỤ M; ị̀ ọ́ ụ̄ ṅ ‘usher’ n’ụlọ 100."
        expected="ọ bụ m ị̀ ọ́ ụ̄ ṅ usher n'ụlọ 100"
        self.assertEqual(alignment_text(text),unicodedata.normalize('NFC',expected))
        self.assertIn('na-eme',alignment_text('na-eme'))

    def test_clipped_or_uncertain_captions_are_not_selected(self):
        base={'display_bounded_candidate':True,'image_transcription_confidence':'high','uncertainties':[]}
        self.assertTrue(eligible_caption(base))
        self.assertFalse(eligible_caption({**base,'display_bounded_candidate':False}))
        self.assertFalse(eligible_caption({**base,'uncertainties':['one letter']}))

    def test_path_coverage_and_timing_are_validated(self):
        result=evaluate_path([[4,2,3,.8],[8,6,8,.9]],[4,8],12,.02,1)
        self.assertEqual(result['relative_audio_start_seconds'],1.04)
        self.assertEqual(result['relative_audio_end_seconds'],1.16)
        self.assertIn('implausible_aligned_duration',result['timing_warnings'])
        with self.assertRaises(ValueError):
            evaluate_path([[4,2,3,.8]],[4,8],12,.02,1)

    def test_low_timing_score_does_not_invalidate_user_source_review(self):
        result=evaluate_path([[4,0,15,.02]],[4],15,.02,0)
        self.assertIn('low_acoustic_alignment_score',result['timing_warnings'])
        self.assertIn('not a verdict',result['interpretation'])

    def test_joint_boundaries_preserve_igbo_text_and_caption_order(self):
        text='Ndewo. Ọ bụ m.'
        ids=list(text)
        ranges=caption_token_ranges(ids,['Ndewo.','Ọ bụ m.'],lambda v:''.join(v))
        self.assertEqual([''.join(ids[a:b]).strip() for a,b in ranges],['Ndewo.','Ọ bụ m.'])
        self.assertLessEqual(ranges[0][1],ranges[1][0])
        with self.assertRaises(ValueError):
            caption_token_ranges(ids,['Changed','text'],lambda v:''.join(v))

    def test_joint_boundary_must_not_cut_through_one_token(self):
        with self.assertRaisesRegex(ValueError,'exact token boundary'):
            caption_token_ranges(['one-token'],['Ndewo.','Ọ bụ m.'],lambda v:'Ndewo. Ọ bụ m.' if v else '')


if __name__=='__main__':
    unittest.main()
