import unittest
from backend.voice_research.review_rules import validate_cut_review


class CutReviewTests(unittest.TestCase):
    def payload(self):
        return {'id':'case1','nonce':'fixture','words_match':'unreviewed','complete_edges':'unreviewed',
                'clean_female_voice':'unreviewed','notes':''}

    def test_unchecked_choices_never_become_approval(self):
        result=validate_cut_review(self.payload(),{'case1':'hash'},'fixture')
        self.assertEqual(result['status'],'pending')
        self.assertNotIn('nonce',result)

    def test_every_dimension_is_required(self):
        payload={**self.payload(),'words_match':'yes','complete_edges':'yes','clean_female_voice':'yes','notes':'Ọ bụ m.'}
        self.assertEqual(validate_cut_review(payload,{'case1':'hash'},'fixture')['status'],'accepted_by_user_for_pilot')
        self.assertEqual(validate_cut_review({**payload,'clean_female_voice':'no'},{'case1':'hash'},'fixture')['status'],'held_by_user')

    def test_rejects_stale_unknown_or_unbounded_input(self):
        for patch in ({'nonce':'old'},{'id':'other'},{'notes':'a'*501},{'words_match':'maybe'},{'extra':'field'}):
            with self.assertRaises(ValueError):
                validate_cut_review({**self.payload(),**patch},{'case1':'hash'},'fixture')


if __name__=='__main__':
    unittest.main()
