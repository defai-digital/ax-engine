#!/usr/bin/env python3
"""Offline unit tests for embedding QA probes (no live server / weights)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

from embedding_probes import (  # noqa: E402
    EmbedReport,
    average_precision,
    cosine,
    default_batch_threshold,
    extract_vectors,
    hit_at_k,
    infer_model_kind,
    infer_pooling,
    is_finite_vector,
    l2_norm,
    mrr_at_k,
    probe_api_shape,
    probe_empty_rejected,
    probe_l2_normalized,
    probe_pair_classification,
    probe_retrieval,
    probe_semantic_order,
    rank_by_cosine,
    resolve_eos_id,
)


class EmbeddingMathTests(unittest.TestCase):
    def test_cosine_identical(self) -> None:
        v = [0.0, 3.0, 4.0]
        self.assertAlmostEqual(cosine(v, v), 1.0, places=6)

    def test_cosine_orthogonal(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_l2_norm(self) -> None:
        self.assertAlmostEqual(l2_norm([3.0, 4.0]), 5.0, places=6)

    def test_is_finite(self) -> None:
        self.assertTrue(is_finite_vector([1.0, -0.5]))
        self.assertFalse(is_finite_vector([1.0, float("nan")]))
        self.assertFalse(is_finite_vector([float("inf")]))

    def test_hit_and_mrr(self) -> None:
        self.assertEqual(hit_at_k([2, 0, 1], gold=2, k=1), 1.0)
        self.assertEqual(hit_at_k([0, 2, 1], gold=2, k=1), 0.0)
        self.assertAlmostEqual(mrr_at_k([0, 2, 1], gold=2, k=4), 0.5)

    def test_average_precision_perfect(self) -> None:
        # Higher scores for positives first → AP = 1
        scores = [0.9, 0.8, 0.1, 0.05]
        labels = [True, True, False, False]
        self.assertAlmostEqual(average_precision(scores, labels), 1.0)

    def test_rank_by_cosine(self) -> None:
        q = [1.0, 0.0]
        docs = [[0.0, 1.0], [0.9, 0.1], [0.1, 0.9]]
        ranked = rank_by_cosine(q, docs)
        self.assertEqual(ranked[0], 1)


class EmbeddingInferTests(unittest.TestCase):
    def test_infer_pooling(self) -> None:
        self.assertEqual(infer_pooling("qwen3-embedding-0.6b"), "last")
        self.assertEqual(infer_pooling("embeddinggemma-300m"), "mean")

    def test_infer_model_kind(self) -> None:
        self.assertEqual(infer_model_kind("qwen3-embedding"), "qwen3")
        self.assertEqual(infer_model_kind("embeddinggemma"), "embeddinggemma")

    def test_default_batch_threshold_looser_for_qwen8b(self) -> None:
        self.assertEqual(
            default_batch_threshold("Qwen3-Embedding-8B-4bit-DWQ"), 0.996
        )
        self.assertEqual(default_batch_threshold("Qwen3-Embedding-0.6B-8bit"), 0.999)

    def test_resolve_eos_prefers_im_end_over_legacy_s(self) -> None:
        """Regression: Qwen configs use <|im_end|>; </s> must not win first."""

        class FakeTok:
            def token_to_id(self, s: str):
                return {
                    "</s>": 128247,
                    "<|endoftext|>": 151643,
                    "<|im_end|>": 151645,
                }.get(s)

        # Without config path, priority list still prefers im_end over </s>
        self.assertEqual(resolve_eos_id(FakeTok()), 151645)

    def test_resolve_eos_from_tokenizer_config(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        class FakeTok:
            def token_to_id(self, s: str):
                return {"<|im_end|>": 99, "</s>": 1, "<|endoftext|>": 2}.get(s)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tokenizer.json").write_text("{}")
            (root / "tokenizer_config.json").write_text(
                json.dumps({"eos_token": {"content": "<|im_end|>"}})
            )
            self.assertEqual(
                resolve_eos_id(FakeTok(), str(root / "tokenizer.json")), 99
            )


class EmbeddingProbeMockTests(unittest.TestCase):
    def test_extract_vectors_sorts_by_index(self) -> None:
        body = {
            "object": "list",
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ],
        }
        vecs = extract_vectors(body)
        self.assertEqual(vecs[0], [1.0, 0.0])
        self.assertEqual(vecs[1], [0.0, 1.0])

    def test_probe_api_shape_ok(self) -> None:
        raw = [1.0] + [0.0] * 7
        n = math.sqrt(sum(x * x for x in raw))
        unit = [x / n for x in raw]
        body = {
            "object": "list",
            "data": [
                {"index": 0, "embedding": unit},
                {"index": 1, "embedding": unit},
            ],
        }
        with mock.patch(
            "embedding_probes._post_embeddings", return_value=(200, body)
        ):
            result, vecs = probe_api_shape(
                "http://x", "m", [[1], [2]], pooling="last", timeout=1
            )
        self.assertTrue(result.passed, result.detail)
        self.assertEqual(len(vecs), 2)

    def test_probe_l2_normalized(self) -> None:
        unit = [0.0, 1.0]
        bad = [2.0, 0.0]
        self.assertTrue(probe_l2_normalized([unit, unit]).passed)
        self.assertFalse(probe_l2_normalized([bad]).passed)

    def test_empty_rejected(self) -> None:
        with mock.patch(
            "embedding_probes._post_embeddings", return_value=(400, {"error": "empty"})
        ):
            r = probe_empty_rejected("http://x", "m", pooling="last", timeout=1)
        self.assertTrue(r.passed)

    def test_semantic_order_majority(self) -> None:
        class Tok:
            pass

        def fake_encode(tokenizer, texts, model_kind="qwen3"):
            return [[i + 1] for i in range(len(texts))]

        a = [1.0, 0.0]
        p = [0.9, 0.1]
        n = [0.0, 1.0]

        def fake_post(base_url, model, input_ids, pooling="last", normalize=True, timeout=60.0):
            body = {
                "object": "list",
                "data": [
                    {"index": 0, "embedding": a},
                    {"index": 1, "embedding": p},
                    {"index": 2, "embedding": n},
                ],
            }
            return 200, body

        with mock.patch("embedding_probes.encode_texts", side_effect=fake_encode):
            with mock.patch("embedding_probes._post_embeddings", side_effect=fake_post):
                r, metrics = probe_semantic_order(
                    "http://x",
                    "m",
                    Tok(),
                    model_kind="qwen3",
                    pooling="last",
                    timeout=1,
                    margin=0.02,
                )
        self.assertTrue(r.passed)
        self.assertEqual(r.name, "sts_triple_ranking")
        self.assertIn("sts_triple_accuracy", metrics)
        self.assertGreaterEqual(metrics["sts_triple_accuracy"], 0.99)

    def test_pair_classification_mock(self) -> None:
        class Tok:
            pass

        # Alternate high/low cosine based on call order matching labels.
        # PAIR_CLASSIFICATION alternates True/False roughly — return high for True pairs.
        call = {"i": 0}

        def fake_encode(tokenizer, texts, model_kind="qwen3"):
            return [[1], [2]]

        def fake_post(*a, **k):
            from embedding_bank import PAIR_CLASSIFICATION as pairs

            idx = call["i"]
            call["i"] += 1
            label = pairs[idx % len(pairs)][2]
            # high cosine for paraphrase, low otherwise
            if label:
                b = [0.99, 0.01]
            else:
                b = [0.0, 1.0]
            return 200, {
                "object": "list",
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": b},
                ],
            }

        with mock.patch("embedding_probes.encode_texts", side_effect=fake_encode):
            with mock.patch("embedding_probes._post_embeddings", side_effect=fake_post):
                r, m = probe_pair_classification(
                    "http://x",
                    "m",
                    Tok(),
                    model_kind="qwen3",
                    pooling="last",
                    timeout=1,
                    min_ap=0.9,
                )
        self.assertTrue(r.passed, r.detail)
        self.assertIn("pair_classification_ap", m)
        self.assertGreaterEqual(m["pair_classification_ap"], 0.9)

    def test_retrieval_mock(self) -> None:
        class Tok:
            pass

        # Return vectors so doc0 is always closest to query
        def fake_encode(tokenizer, texts, model_kind="qwen3"):
            return [[i] for i in range(len(texts))]

        def fake_post(base_url, model, input_ids, pooling="last", normalize=True, timeout=60.0):
            # query + N docs; query and gold-aligned doc share direction
            n = len(input_ids)
            data = [{"index": 0, "embedding": [1.0, 0.0]}]
            for i in range(1, n):
                # make first doc closest
                if i == 1:
                    data.append({"index": i, "embedding": [0.95, 0.05]})
                else:
                    data.append({"index": i, "embedding": [0.0, 1.0]})
            return 200, {"object": "list", "data": data}

        with mock.patch("embedding_probes.encode_texts", side_effect=fake_encode):
            with mock.patch("embedding_probes._post_embeddings", side_effect=fake_post):
                # Override cases so gold is always 0 (first doc)
                with mock.patch(
                    "embedding_probes.RETRIEVAL_CASES",
                    [
                        {
                            "query": "q",
                            "docs": ["a", "b", "c"],
                            "gold": 0,
                        }
                    ],
                ):
                    r, m = probe_retrieval(
                        "http://x",
                        "m",
                        Tok(),
                        model_kind="qwen3",
                        pooling="last",
                        timeout=1,
                        min_hit_at_1=1.0,
                    )
        self.assertTrue(r.passed, r.detail)
        self.assertEqual(m["retrieval_hit_at_1"], 1.0)

    def test_report_schema_v2(self) -> None:
        report = EmbedReport(
            base_url="u", model="m", pooling="last", dim=8, tier="standard"
        )
        from embedding_probes import EmbedProbeResult

        report.results = [EmbedProbeResult("api_shape", True)]
        report.metrics = {"retrieval_hit_at_1": 1.0}
        d = report.as_dict()
        self.assertEqual(d["schema_version"], 2)
        self.assertEqual(d["kind"], "embedding_probes")
        self.assertTrue(d["hard_passed"])
        self.assertIn("market_alignment", d)
        self.assertFalse(d["market_alignment"]["mteb_full"])
        self.assertTrue(d["market_alignment"]["mteb_shaped_tasks"])


if __name__ == "__main__":
    unittest.main()
