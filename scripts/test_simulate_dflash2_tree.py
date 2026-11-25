#!/usr/bin/env python3
"""Unit tests for the DFlash2 tree-verify simulator."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("simulate_dflash2_tree.py")
MODULE_SPEC = importlib.util.spec_from_file_location("simulate_dflash2_tree", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
sim = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = sim
MODULE_SPEC.loader.exec_module(sim)


def draft(anchor, candidates, unary=None, edge0=None, edges=None, chain=None, request=None):
    """A draft record. Defaults: descending logits, no edge scores, and the
    chain of each position's first candidate."""
    positions = len(candidates)
    top_k = len(candidates[0])
    record = {
        "kind": "draft",
        "anchor": anchor,
        "top_k": top_k,
        "positions": positions,
        "candidates": candidates,
        "unary": unary or [[float(top_k - j) for j in range(top_k)] for _ in range(positions)],
        "edge0": edge0 or [0.0] * top_k,
        "edges": edges or [[[0.0] * top_k for _ in range(top_k)] for _ in range(positions - 1)],
        "chain": chain or [row[0] for row in candidates],
    }
    if request is not None:
        record["request"] = request
    return record


def commit(tokens, accepted=None, window=None, matched=None, request=None):
    record = {
        "kind": "commit",
        "tokens": tokens,
        "accepted_drafts": len(tokens) - 1 if accepted is None else accepted,
    }
    if window is not None:
        record["window"] = window
    if matched is not None:
        record["matched_drafts"] = matched
    if request is not None:
        record["request"] = request
    return record


def start(request=None):
    record = {"kind": "request_start"}
    if request is not None:
        record["request"] = request
    return record


def write_trace(directory: Path, records, name: str = "trace.jsonl") -> Path:
    path = Path(directory) / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def load(records):
    with tempfile.TemporaryDirectory() as tmp:
        return sim.load_traces([write_trace(Path(tmp), records)])


def round_of(record, index=0):
    return sim.Round(draft=sim.parse_draft(record, "test"), index=index)


# Truth after anchor 1 is 10, 21, 30, 40. The chain drafts 10, 20, 30: its
# second token is wrong, and the right one (21) is the second candidate.
SIBLING_DRAFT = draft(1, [[10, 11], [20, 21], [30, 31]])
SIBLING_TRUTH = [10, 21, 30, 40]


def dense_trace(tokens, chains, first_anchor=1):
    """One request with a draft at every output position (accept cap 0):
    `chains[i]` is the chain drafted after `tokens[i - 1]`."""
    records = [start()]
    anchor = first_anchor
    for index, (token, chain) in enumerate(zip(tokens, chains, strict=True)):
        candidates = [[drafted, drafted + 1000] for drafted in chain]
        # What the engine's verify matched before the cap cut the round.
        matched = 0
        while matched < len(chain) and tokens[index + matched : index + matched + 1] == [
            chain[matched]
        ]:
            matched += 1
        records += [
            draft(anchor, candidates),
            commit([token], accepted=0, window=len(chain) + 1, matched=matched),
        ]
        anchor = token
    return records


class LoadTraceTests(unittest.TestCase):
    def test_requests_tokens_and_round_anchors(self):
        requests = load(
            [
                start(),
                draft(1, [[10, 11], [20, 21]]),
                commit([10, 20, 5], window=3, matched=2),
                draft(5, [[6, 7], [8, 9]]),
                commit([9], window=3, matched=0),
                start(),
                draft(3, [[4, 5], [6, 7]]),
                commit([4, 6, 2]),
            ]
        )
        self.assertEqual([r.tokens for r in requests], [[10, 20, 5, 9], [4, 6, 2]])
        self.assertEqual([rnd.index for rnd in requests[0].rounds], [0, 3])
        self.assertEqual(requests[0].rounds[0].window, 3)
        self.assertEqual(requests[0].rounds[0].matched_drafts, 2)
        self.assertEqual(requests[0].truth(3), [9])
        self.assertFalse(requests[0].is_dense())

    def test_interleaved_requests_split_by_request_id(self):
        requests = load(
            [
                start(request=7),
                start(request=8),
                draft(1, [[10, 11]], request=7),
                draft(2, [[30, 31]], request=8),
                commit([30, 3], request=8),
                commit([10, 4], request=7),
                draft(4, [[5, 6]], request=7),
                commit([5], request=7),
            ]
        )
        self.assertEqual([r.tokens for r in requests], [[10, 4, 5], [30, 3]])
        self.assertEqual([len(r.rounds) for r in requests], [2, 1])

    def test_anchor_mismatch_fails_loudly(self):
        records = [
            start(),
            draft(1, [[10, 11]]),
            commit([10, 5]),
            draft(6, [[7, 8]]),
            commit([7]),
        ]
        with self.assertRaisesRegex(sim.TraceError, "anchor 6 is not the last committed token 5"):
            load(records)

    def test_malformed_records_fail(self):
        bad_shape = draft(1, [[10, 11], [20, 21]])
        bad_shape["edges"] = []
        with self.assertRaisesRegex(sim.TraceError, "shapes"):
            load([start(), bad_shape, commit([10])])
        with self.assertRaisesRegex(sim.TraceError, "no commit"):
            load([start(), draft(1, [[10, 11]]), draft(1, [[10, 11]])])
        with self.assertRaisesRegex(sim.TraceError, "unknown record kind"):
            load([{"kind": "other"}])

    def test_commit_without_draft_keeps_the_token_stream(self):
        requests = load([start(), commit([10, 5]), draft(5, [[6, 7]]), commit([6])])
        self.assertEqual(requests[0].tokens, [10, 5, 6])
        self.assertEqual(requests[0].undrafted_commits, 1)
        self.assertEqual([rnd.index for rnd in requests[0].rounds], [2])

    def test_null_scores_read_as_minus_infinity(self):
        record = draft(1, [[10, 11]], unary=[[None, 2.0]])
        rnd = round_of(record)
        q, order = rnd.children(0, 0)
        self.assertEqual(q, [0.0, 1.0])
        self.assertEqual(order, [1, 0])


class PolicyTests(unittest.TestCase):
    def test_chain_reproduces_hand_computed_acceptance(self):
        record = draft(1, [[10, 11], [20, 21], [30, 31], [40, 41]])
        parsed = sim.parse_draft(record, "test")
        # Drafts 10, 20, 30, 40 against truth 10, 20, 99: two accepted.
        truth = [10, 20, 99, 40, 50]
        self.assertEqual(sim.chain_accepted(parsed, 5, truth), 2)
        # A two-row window verifies one draft only.
        self.assertEqual(sim.chain_accepted(parsed, 2, truth), 1)
        self.assertEqual(sim.chain_accepted(parsed, 1, truth), 0)
        # Every draft right: capped by the window, then by the block.
        self.assertEqual(sim.chain_accepted(parsed, 4, [10, 20, 30, 40, 50]), 3)
        self.assertEqual(sim.chain_accepted(parsed, 9, [10, 20, 30, 40, 50]), 4)
        rnd = round_of(record)
        chain = sim.Policy("chain", 5)
        self.assertEqual(sim.policy_emitted(chain, rnd, truth, sim.Calibration()), 3)
        # The request ends inside the accepted prefix: nothing past its tokens.
        self.assertEqual(sim.policy_emitted(chain, rnd, [10, 20], sim.Calibration()), 2)

    def test_tree_with_the_right_sibling_accepts_deeper_than_the_chain(self):
        rnd = round_of(SIBLING_DRAFT)
        self.assertEqual(sim.chain_accepted(rnd.draft, 4, SIBLING_TRUTH), 1)
        # Eight rows hold the sibling 21 under 10 but not yet its child 30
        # (the wrong branches 11 and 20 outrank it); nine rows hold both.
        nodes = sim.grow_tree(rnd, rows=8, children=2, estimate=sim.identity_estimate)
        self.assertEqual(sim.tree_accepted(rnd, nodes, SIBLING_TRUTH), 2)
        nodes = sim.grow_tree(rnd, rows=9, children=2, estimate=sim.identity_estimate)
        self.assertEqual(sim.tree_accepted(rnd, nodes, SIBLING_TRUTH), 3)
        tree = sim.Policy("tree", 9, 2, "q")
        self.assertEqual(sim.policy_emitted(tree, rnd, SIBLING_TRUTH, sim.Calibration()), 4)
        chain = sim.Policy("chain", 4)
        self.assertEqual(sim.policy_emitted(chain, rnd, SIBLING_TRUTH, sim.Calibration()), 2)
        # One child per node is the unary-argmax chain again.
        nodes = sim.grow_tree(rnd, rows=8, children=1, estimate=sim.identity_estimate)
        self.assertEqual(len(nodes), 4)
        self.assertEqual(sim.tree_accepted(rnd, nodes, SIBLING_TRUTH), 1)
        # The oracle ignores the row budget and stops at a token no candidate has.
        self.assertEqual(sim.oracle_accepted(rnd.draft, SIBLING_TRUTH), 3)
        self.assertEqual(sim.oracle_accepted(rnd.draft, [10, 99, 30]), 1)

    def test_edges_pick_the_child_of_the_parent_candidate(self):
        # After candidate 0 at position 0 the edge prefers token 21, after
        # candidate 1 it prefers token 20.
        record = draft(
            1,
            [[10, 11], [20, 21]],
            unary=[[2.0, 1.0], [0.0, 0.0]],
            edges=[[[0.0, 5.0], [5.0, 0.0]]],
        )
        rnd = round_of(record)
        q_after_first, order_first = rnd.children(1, 0)
        q_after_second, order_second = rnd.children(1, 1)
        self.assertEqual((order_first, order_second), ([1, 0], [0, 1]))
        self.assertGreater(q_after_first[1], 0.99)
        self.assertGreater(q_after_second[0], 0.99)
        nodes = sim.grow_tree(rnd, rows=3, children=1, estimate=sim.identity_estimate)
        self.assertEqual(sim.tree_accepted(rnd, nodes, [10, 21]), 2)
        self.assertEqual(sim.tree_accepted(rnd, nodes, [10, 20]), 1)
        # The first position reads edge0.
        record = draft(1, [[10, 11]], unary=[[1.0, 1.0]], edge0=[0.0, 3.0])
        self.assertEqual(round_of(record).children(0, 0)[1], [1, 0])

    def test_row_budget_and_tree_structure(self):
        candidates = [[100 * p + j for j in range(4)] for p in range(1, 6)]
        unary = [[1.0, 0.7, 0.4, 0.1] for _ in candidates]
        rnd = round_of(draft(1, candidates, unary=unary))
        for rows in (1, 2, 5, 8, 12, 16):
            for children in (1, 2, 3, 4):
                nodes = sim.grow_tree(rnd, rows, children, sim.identity_estimate)
                with self.subTest(rows=rows, children=children):
                    # The anchor row plus at most `rows - 1` draft rows; the
                    # budget is filled whenever enough candidates exist.
                    self.assertLessEqual(len(nodes), rows)
                    if children > 1 or rows <= 6:
                        self.assertEqual(len(nodes), rows)
                    self.assertEqual(nodes[0], (0, -1, -1, 1.0))
                    seen = set()
                    fanout: dict[int, int] = {}
                    for index, (depth, candidate, parent, score) in enumerate(nodes[1:], 1):
                        # Every node's parent was placed before it, one level up.
                        self.assertLess(parent, index)
                        self.assertEqual(depth, nodes[parent][0] + 1)
                        self.assertLessEqual(depth, rnd.draft.positions)
                        self.assertLessEqual(score, nodes[parent][3])
                        self.assertNotIn((parent, candidate), seen)
                        seen.add((parent, candidate))
                        fanout[parent] = fanout.get(parent, 0) + 1
                    self.assertTrue(all(count <= children for count in fanout.values()))
                    # Best-first: rows are placed in non-increasing score order.
                    scores = [node[3] for node in nodes[1:]]
                    self.assertEqual(scores, sorted(scores, reverse=True))

    def test_softmax_and_estimates_are_monotone(self):
        logits = [3.0, -1.0, 0.5, 2.0, float("-inf")]
        q = sim.softmax(logits)
        self.assertAlmostEqual(sum(q), 1.0)
        self.assertEqual(q[4], 0.0)
        by_logit = sorted(range(len(logits)), key=lambda j: logits[j])
        self.assertEqual(by_logit, sorted(range(len(q)), key=lambda j: q[j]))
        self.assertEqual(sim.softmax([float("-inf")] * 4), [0.25] * 4)
        self.assertEqual(sim.softmax([1000.0, 0.0]), [1.0, 0.0])

        calibration = sim.Calibration(slope=0.6, intercepts=(0.4, -0.8))
        grid = [0.0, 1e-9, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0]
        for depth in (1, 2, 5):
            values = [calibration.probability(value, depth) for value in grid]
            self.assertEqual(values, sorted(values))
            self.assertTrue(all(0.0 < value < 1.0 for value in values))
        # Depths past the fitted range reuse the last intercept.
        self.assertEqual(calibration.probability(0.3, 5), calibration.probability(0.3, 2))
        # Siblings keep their order and never sum past one.
        siblings = [0.5, 0.3, 0.15, 0.05]
        estimates = sim.Calibration(slope=0.2, intercepts=(2.0,)).estimate(siblings, 1)
        self.assertLessEqual(sum(estimates), 1.0 + 1e-12)
        self.assertEqual(list(estimates), sorted(estimates, reverse=True))
        self.assertEqual(list(sim.identity_estimate(siblings, 3)), siblings)


class CalibrationTests(unittest.TestCase):
    def test_no_samples_keeps_the_identity(self):
        fitted = sim.fit_calibration({}, positions=3)
        self.assertEqual((fitted.slope, fitted.intercepts), (1.0, (0.0, 0.0, 0.0)))
        self.assertAlmostEqual(fitted.probability(0.3, 2), 0.3)

    def test_fit_moves_towards_the_observed_rate(self):
        # Candidates with q = 0.5 are right 90% of the time at depth 1 and
        # 10% of the time at depth 2.
        bin_index = 0
        samples = {(1, bin_index): [1000.0, 0.0, 900.0], (2, bin_index): [1000.0, 0.0, 100.0]}
        fitted = sim.fit_calibration(samples, positions=2)
        self.assertAlmostEqual(fitted.probability(0.5, 1), 0.9, delta=0.01)
        self.assertAlmostEqual(fitted.probability(0.5, 2), 0.1, delta=0.01)

    def test_cross_fit_leaves_the_scored_request_out(self):
        # Request A's top candidate is always right; request B's candidates
        # never hold the target's token.
        right = [start(request=1)]
        wrong = [start(request=2)]
        anchor = 1
        for step in range(20):
            token = 100 + step
            right += [
                draft(anchor, [[token, 9000]], unary=[[1.0, 0.0]], request=1),
                commit([token], request=1),
            ]
            wrong += [
                draft(anchor, [[9000, 9001]], unary=[[1.0, 0.0]], request=2),
                commit([token], request=2),
            ]
            anchor = token
        requests = load(right + wrong)
        fits, pooled = sim.cross_fit(requests, positions=1)
        q_top = sim.softmax([1.0, 0.0])[0]
        # A is scored by B's fit (nothing right) and B by A's.
        self.assertLess(fits[0].probability(q_top, 1), 0.3)
        self.assertGreater(fits[1].probability(q_top, 1), 0.7)
        self.assertAlmostEqual(pooled.probability(q_top, 1), 0.5, delta=0.1)
        single, _ = sim.cross_fit(requests[:1], positions=1)
        self.assertGreater(single[0].probability(q_top, 1), 0.7)


class ModeTests(unittest.TestCase):
    # Output after anchor 1; chains drafted after each position's anchor.
    TOKENS = [10, 20, 30, 40, 50, 60]
    CHAINS = [
        [10, 20, 99],  # after 1: two right, then 30 is the target's own token
        [20, 30, 40],
        [30, 40, 50],
        [40, 77, 60],  # after 30: one right, then 50 is the target's own token
        [50, 60, 70],
        [61, 62, 63],  # after 50: nothing right, the target emits 60
    ]

    def test_exact_mode_round_count(self):
        requests = load(dense_trace(self.TOKENS, self.CHAINS))
        self.assertTrue(requests[0].is_dense())
        policies = [sim.Policy("chain", 4), sim.Policy("chain", 2), sim.Policy("chain", 1)]
        calibrations = [sim.Calibration()]
        results = sim.evaluate_exact(requests, policies, calibrations)
        # Full window: rounds at positions 0 (emits 3), 3 (emits 2), 5 (emits 1).
        self.assertEqual(results[0], [(3, 6)])
        # One draft per round: positions 0, 2, 4 emit two tokens each.
        self.assertEqual(results[1], [(3, 6)])
        # No drafts: one round per token.
        self.assertEqual(results[2], [(6, 6)])
        # Paired scores all six recorded rounds: 3 + 4 + 4 + 2 + 2 + 1 tokens
        # (the round after 40 would emit three, but only two tokens remain).
        paired = sim.evaluate_paired(requests, policies[:1], calibrations)
        self.assertEqual(paired[0], [(6, 16)])

    def test_exact_mode_refuses_a_sparse_trace(self):
        sparse = load(
            [
                start(),
                draft(1, [[10, 11], [20, 21]]),
                commit([10, 20, 30]),
                draft(30, [[40, 41], [50, 51]]),
                commit([40, 50, 60]),
            ]
        )
        self.assertFalse(sparse[0].is_dense())
        with self.assertRaisesRegex(sim.TraceError, "AX_MLX_DFLASH2_TRACE_ACCEPT_CAP=0"):
            sim.evaluate_exact(sparse, [sim.Policy("chain", 3)], [sim.Calibration()])
        # Paired mode still scores it.
        report = sim.build_report(sparse, sim.default_grid(2), "paired")
        self.assertFalse(report["dense"])
        self.assertNotIn("exact", report)
        with self.assertRaises(sim.TraceError):
            sim.build_report(sparse, sim.default_grid(2), "both")

    def test_report_ratios_and_projected_throughput(self):
        requests = load(dense_trace(self.TOKENS, self.CHAINS))
        report = sim.build_report(
            requests, sim.default_grid(3), "both", costs={4: 40.0, 8: 60.0}, per_request=True
        )
        self.assertEqual(report["baseline"], "chain(4)")
        self.assertEqual((report["requests"], report["rounds"], report["tokens"]), (1, 6, 6))
        # The last two rounds have less ground truth than their window.
        self.assertEqual(report["engine_check"], {"rounds_compared": 4, "rounds_agreed": 4})
        exact = {row["policy"]: row for row in report["exact"]["policies"]}
        chain = exact["chain(4)"]
        self.assertEqual((chain["rounds"], chain["tokens"]), (3, 6))
        self.assertAlmostEqual(chain["tokens_per_round"], 2.0)
        self.assertAlmostEqual(chain["vs_chain"], 1.0)
        # 6 tokens in 3 rounds of 40 ms.
        self.assertAlmostEqual(chain["tok_per_s"], 50.0)
        oracle = exact["oracle"]
        self.assertGreaterEqual(oracle["tokens_per_round"], chain["tokens_per_round"])
        for row in report["exact"]["policies"]:
            self.assertGreaterEqual(oracle["tokens_per_round"], row["tokens_per_round"])
            self.assertEqual(row["tokens"], 6)
        tree = exact["tree(8,c2,q)"]
        self.assertAlmostEqual(tree["round_ms"], 60.0)
        self.assertAlmostEqual(tree["tok_per_s"], 1000.0 * 6 / (tree["rounds"] * 60.0))
        self.assertAlmostEqual(tree["tok_per_s_vs_chain"], tree["tok_per_s"] / 50.0)
        paired = {row["policy"]: row for row in report["paired"]["policies"]}
        self.assertNotIn("tok_per_s", paired["chain(4)"])
        self.assertAlmostEqual(paired["chain(4)"]["tokens_per_round"], 16 / 6)
        self.assertEqual(len(report["exact"]["per_request"]), 1)
        text = sim.format_report(report)
        self.assertIn("exact: requests replayed round by round", text)
        self.assertIn("tree(16,c4,cal)", text)

    def test_default_grid(self):
        names = [policy.name for policy in sim.default_grid(7)]
        self.assertEqual(names[:5], ["chain(4)", "chain(5)", "chain(6)", "chain(7)", "chain(8)"])
        self.assertEqual(len([name for name in names if name.startswith("tree(")]), 18)
        self.assertEqual(names[-1], "oracle")
        self.assertEqual([p.name for p in sim.default_grid(1, (), (), ())], ["chain(2)", "oracle"])

    def test_round_cost_interpolates_between_points(self):
        costs = {8: 45.9, 12: 47.0, 16: 48.0}
        self.assertEqual(sim.round_ms(costs, 12), (47.0, False))
        ms, outside = sim.round_ms(costs, 10)
        self.assertAlmostEqual(ms, 46.45)
        self.assertFalse(outside)
        ms, outside = sim.round_ms(costs, 14)
        self.assertAlmostEqual(ms, 47.5)
        self.assertFalse(outside)
        # Outside the points the nearest segment is extended and flagged.
        ms, outside = sim.round_ms(costs, 4)
        self.assertAlmostEqual(ms, 45.9 - 1.1)
        self.assertTrue(outside)
        ms, outside = sim.round_ms(costs, 20)
        self.assertAlmostEqual(ms, 49.0)
        self.assertTrue(outside)
        self.assertEqual(sim.round_ms({8: 45.0}, 16), (45.0, True))
        self.assertEqual(sim.parse_cost("12=47"), (12, 47.0))


class CliTests(unittest.TestCase):
    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = sim.main(argv)
        return status, out.getvalue(), err.getvalue()

    def test_json_and_table_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_trace(Path(tmp), dense_trace(ModeTests.TOKENS, ModeTests.CHAINS))
            status, out, _ = self.run_main(
                [str(path), "--mode", "both", "--cost", "4=40", "--cost", "8=60", "--json"]
            )
            self.assertEqual(status, 0)
            report = json.loads(out)
            self.assertEqual(report["mode"], "both")
            self.assertEqual(report["cost_ms"], {"4": 40.0, "8": 60.0})
            self.assertEqual(len(report["exact"]["policies"]), len(report["paired"]["policies"]))
            status, out, _ = self.run_main([str(path), "--per-request"])
            self.assertEqual(status, 0)
            self.assertIn("paired: every recorded round scored at its own anchor", out)
            self.assertIn("paired trace.jsonl#1", out)
            self.assertNotIn("exact:", out)

    def test_sparse_exact_and_bad_anchor_exit_non_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            sparse = write_trace(
                Path(tmp), [start(), draft(1, [[10, 11], [20, 21]]), commit([10, 20, 30])]
            )
            status, out, err = self.run_main([str(sparse), "--mode", "exact"])
            self.assertEqual((status, out), (2, ""))
            self.assertIn("AX_MLX_DFLASH2_TRACE_ACCEPT_CAP=0", err)
            broken = write_trace(
                Path(tmp),
                [start(), draft(1, [[10, 11]]), commit([10]), draft(3, [[4, 5]]), commit([4])],
                name="broken.jsonl",
            )
            status, _, err = self.run_main([str(broken)])
            self.assertEqual(status, 2)
            self.assertIn("broken.jsonl:4", err)


if __name__ == "__main__":
    unittest.main()
