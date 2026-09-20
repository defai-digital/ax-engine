"""Template for published QA mutation controls; no model is executed."""
import copy
import unittest

import verify_evidence as v


class EvidenceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.evidence = v.load()

    def test_all_original_responses_are_reconstructed(self):
        result = v.verify(self.summary, self.evidence)
        self.assertTrue(result['collection_valid'])
        self.assertEqual(result['requests'], 210)
        self.assertEqual(result['mtp_certification'], v.GATES)
        self.assertFalse(result['qualification'])
        self.assertFalse(result['release_ready'])

    def test_raw_answer_and_checker_mutations_are_rejected(self):
        for kind in ('raw', 'checker', 'budget', 'missing_case'):
            evidence = copy.deepcopy(self.evidence)
            cases = evidence['records']['output/result.json']['modes']['required']['cases']
            if kind == 'raw':
                cases[0]['response']['response']['choices'][0]['text'] = 'changed'
            elif kind == 'checker':
                cases[0]['closed_checker']['passed'] = not cases[0]['closed_checker']['passed']
            elif kind == 'budget':
                cases[0]['response']['response']['usage']['completion_tokens'] = 257
            else:
                cases.pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                v.verify(self.summary, evidence)

    def test_source_and_cleanup_mutations_are_rejected(self):
        for kind in ('source', 'cleanup', 'environment'):
            evidence = copy.deepcopy(self.evidence)
            raw = evidence['records']['output/result.json']
            if kind == 'source':
                raw['source_commit'] = 'old'
            elif kind == 'cleanup':
                evidence['records']['transfer-receipt.json']['absent_groups'] = []
            else:
                raw['environment_overrides']['AX_MLX_FLASH_NEXT_SELECTED_EXPERTS'] = '1'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                v.verify(self.summary, evidence)

    def test_quality_regrading_and_release_promotion_are_rejected(self):
        for kind in ('quality', 'release', 'certification'):
            summary = copy.deepcopy(self.summary)
            if kind == 'quality':
                summary['original_qa_passed'] = not summary['original_qa_passed']
            elif kind == 'release':
                summary['release_ready'] = True
            else:
                summary['mtp_certification']['MTP-S'] = 'passed'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                v.verify(summary, self.evidence)


if __name__ == '__main__':
    unittest.main(verbosity=2)
