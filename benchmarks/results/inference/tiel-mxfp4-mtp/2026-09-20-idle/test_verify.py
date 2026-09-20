import copy
import json
from pathlib import Path
import unittest
from verify import verify

class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(Path(__file__).with_name("trials.json").read_text())

    def mutated(self):
        data = copy.deepcopy(self.source)
        return data, data["hosts"]["m4"]["runs"][0]["result"]

    def test_rejects_changed_output_even_if_native_result_matches(self):
        data, result = self.mutated()
        row = result["cases"][0]["trials"][0]
        row["output_tokens"][0] += 1
        row["native"]["output_tokens"][0] += 1
        with self.assertRaises(AssertionError): verify(data)

    def test_rejects_prefix_reuse(self):
        data, result = self.mutated()
        result["cases"][0]["trials"][0]["native"]["route"]["crossover_decisions"]["core_prefix_reuse_disabled"] = 0
        with self.assertRaises(AssertionError): verify(data)

    def test_rejects_hidden_shortened_idle(self):
        data, result = self.mutated()
        result["cases"][0]["trials"][0]["idle_elapsed_s"] = 0.5
        with self.assertRaises(AssertionError): verify(data)

    def test_rejects_missing_counterbalanced_block(self):
        data, _ = self.mutated()
        data["hosts"]["m4"]["runs"].pop()
        with self.assertRaises(AssertionError): verify(data)

if __name__ == "__main__": unittest.main()
