"""Tests for the offline Flash Next Optuna studies and their safety gates."""
from importlib.util import find_spec
import json
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from . import tune_flash_next_serving as tune


class ServingEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        root = Path(temp.name)
        runpy.run_path(str(tune.UNPACK))["unpack"](tune.SERVING_BUNDLE, root)
        cls.folder = root / "mixed-result"
        cls.workload = tune.audit.read_json(root / "workload.json")
        cls.prompts = {row.id: row for row in tune.load_corpus(root / "mixed-corpus.jsonl")}
        cls.baseline = tune.solo_tokens(root / "solo-reference.json", cls.workload, cls.prompts)
        cls.metadata = tune.audit.read_json(cls.folder / "receipt.json")["arms"]["off-1"]

    def score(self):
        return tune.score_arm(self.folder, "off-1", "off", 8192, self.metadata,
                              self.workload, self.prompts, self.baseline, 512, 8192)

    def replace_temporarily(self, file, transform):
        original = file.read_bytes()
        self.addCleanup(file.write_bytes, original)
        file.write_text(transform(original))

    def test_audits_both_orders_and_recomputes_serving_metrics(self):
        (first, reverse), sources = tune.measured_arms(tune.SERVING_BUNDLE)
        self.assertEqual(sources["requests_per_arm"], 220)
        self.assertEqual(tuple(first), tune.POLICIES)
        self.assertTrue(all(row.values is not None for data in (first, reverse) for row in data.values()))
        self.assertEqual(self.score(), first["off"])
        self.assertLess(first["off"].values[0], first["long"].values[0])

    def test_reverse_boots_are_distinct_from_an_independent_holdout_corpus(self):
        data, provenance = tune.measured_arms(tune.SERVING_BUNDLE)
        with self.assertRaisesRegex(ValueError, "distinct traffic"):
            tune.add_holdouts(data, provenance, [tune.SERVING_BUNDLE])

    def test_memory_pressure_rejects_trial(self):
        file = self.folder / "off-1-samples.jsonl"
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        rows[-1]["pressure"] = 2
        self.replace_temporarily(file, lambda _: "\n".join(map(json.dumps, rows)))
        with self.assertRaisesRegex(ValueError, "pressure"):
            self.score()

    def test_divergent_output_and_changed_throughput_are_rejected(self):
        file = self.folder / "off-1.json"
        original = file.read_bytes()
        run = json.loads(original)
        measured = next(row for row in run["observations"] if row["phase"] == "measured")
        measured["output_token_ids"][0] ^= 1
        self.replace_temporarily(file, lambda _: json.dumps(run))
        with self.assertRaisesRegex(ValueError, "tokens differ"):
            self.score()
        run = json.loads(original)
        run["summary"]["output_token_throughput_tok_s"] *= 2
        file.write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError, "client timeline"):
            self.score()


class SearchTests(unittest.TestCase):
    def test_frontier_respects_maximize_and_rejects_ineligible_rows(self):
        m = tune.Measurement
        rows = {"baseline": m((2, 10)), "lower_latency": m((1, 8)), "dominated": m((2, 9)),
                "rejected": m(None, "pressure")}
        self.assertEqual(tune.frontier(rows, ("minimize", "maximize")), ["baseline", "lower_latency"])

    def test_holdout_regression_cannot_be_shortlisted(self):
        m = tune.Measurement
        train = {"off": m((10, 10)), "candidate": m((5, 12))}
        holdout = {"off": m((10, 10)), "candidate": m((6, 9))}
        report = tune.describe(train, [holdout], ("minimize", "maximize"), "off")
        self.assertEqual(report["training_frontier"], ["candidate"])
        self.assertEqual(report["shortlisted_candidates"], [])

    def test_projection_leaves_unobserved_long_first_token_unscored(self):
        (median, low, high), _ = tune.repayment_arms()
        for data in (median, low, high):
            self.assertEqual(len(data), 101)
            self.assertEqual(data[0].values[1], 0)
            self.assertIsNone(data[100].values)

    @unittest.skipUnless(find_spec("optuna"), "Optuna is an optional offline tooling dependency")
    def test_actual_optuna_runs_exhaustive_and_sampled_studies_deterministically(self):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        serving, _ = tune.measured_arms(tune.SERVING_BUNDLE)
        result = tune.compare_searches(optuna, serving, ("minimize", "minimize", "maximize"), 3, 42)
        for strategy in ("fixed_grid", "optuna_tpe"):
            self.assertEqual(result[strategy]["training_frontier"], ["off"])
            self.assertEqual(len(result[strategy]["evaluated"]), 3)
        projected, _ = tune.repayment_arms()
        first = tune.compare_searches(optuna, projected, ("minimize", "minimize"), 24, 42)
        second = tune.compare_searches(optuna, projected, ("minimize", "minimize"), 24, 42)
        self.assertEqual(first, second)
        for strategy in ("fixed_grid", "optuna_tpe"):
            observations = first[strategy]["evaluated"]
            self.assertEqual(len({row["setting"] for row in observations}), 24)
            self.assertTrue(any(row["rejection"] for row in observations))
            self.assertEqual(first[strategy]["training_frontier"], [0])
