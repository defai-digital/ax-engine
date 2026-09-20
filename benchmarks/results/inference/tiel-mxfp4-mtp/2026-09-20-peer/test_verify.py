"""Boundary regressions for the benchmark's phase accounting."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('peer_verify', Path(__file__).with_name('verify.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AccountingTests(unittest.TestCase):
    def row(self):
        return dict(output_tokens=list(range(10)), emissions=[(1.0, 3), (3.0, 7)],
                    ttft_s=1.0, completion_s=3.0, completion_tok_s=10 / 3,
                    first_batch_tokens=3, decode_tokens=7, decode_s=2.0,
                    decode_tok_s=3.5, api_return_s=3.1)

    def test_batched_first_emission(self):
        module.verify_row(self.row(), 10)

    def test_decode_cannot_count_first_batch(self):
        row = self.row()
        row['decode_tok_s'] = 5.0
        with self.assertRaises(AssertionError):
            module.verify_row(row, 10)

    def test_reject_incomplete_count(self):
        with self.assertRaises(AssertionError):
            module.verify_row(self.row(), 11)

    def test_even_median_uses_both_middle_samples(self):
        rows = []
        for value in (1, 2, 3, 7, 8, 9):
            row = self.row()
            row['completion_tok_s'] = value
            rows.append(row)
        runs = [dict(model='tiel', arm='ax', cases=[dict(case='test', trials=rows[:3])]),
                dict(model='tiel', arm='ax', cases=[dict(case='test', trials=rows[3:])])]
        self.assertEqual(module.summarize(runs)['tiel', 'test', 'ax']['completion_tok_s'], 5)


if __name__ == '__main__':
    unittest.main()
