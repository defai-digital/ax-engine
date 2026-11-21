"""Regression controls for the Flash Next mixed-stream evidence replay."""

from types import SimpleNamespace
import unittest

if __package__:
    from . import audit_flash_next_mixed_serving as audit
else:
    import audit_flash_next_mixed_serving as audit


def mixed_streams():
    prompts = {}
    observations = []
    traces = []
    for ident, widths, prefill_ids, decode_ids, gaps in (
        (1, [512], [2], [2, 4, 8], [170, 200]),
        (2, audit.WIDTHS, [3, 4, 5, 6, 7], [7, 9, 10], [45, 20]),
    ):
        name = f"p{ident}"
        count = sum(widths)
        tokens = [ident] * count
        prompts[name] = SimpleNamespace(input_tokens=tokens, arrival_s=0, category=str(count))
        observations.append({
            "phase": "measured", "prompt_id": name, "request_id": ident, "ok": True,
            "prompt_token_ids": tokens, "scheduled_at_s": 0, "started_at_s": 0.1,
            "queue_delay_ms": 100, "ttft_ms": 20, "e2e_latency_ms": 500,
            "input_tokens": count, "category": str(count), "output_tokens": 3,
            "output_token_ids": [1, 2, 3], "stream_step_interval_ms": gaps,
        })
        traces.append({
            "request_id": ident, "initial_cursor": 0, "cursor": count, "decode_step_ids": decode_ids,
            "prefill": [
                {"step_id": step, "width": width, "cursor": sum(widths[:index + 1]),
                 "scheduled": 2 if step == 4 else 1, "split_end": count if ident == 2 else None}
                for index, (step, width) in enumerate(zip(prefill_ids, widths, strict=True))
            ],
        })
    return {"observations": observations}, traces, prompts


class MixedServingAuditTests(unittest.TestCase):
    def test_prefill_between_and_on_decoding_steps(self):
        data, _, shared = audit.streams(*mixed_streams(), 2, True, 3, "pair")
        self.assertEqual(shared, 1)
        self.assertEqual(data["p1"]["peer_prefill_gaps_at_least_100ms"], 2)
        self.assertEqual(data["p1"]["long_gap_peer_profiles"], {"2048/split": 2})
        self.assertEqual(data["p1"]["arrival_to_first_token_ms"], 120)
        self.assertEqual(data["p1"]["client_ttft_ms"], 20)
        self.assertFalse(data["p2"]["peer_prefill"])

    def test_previous_step_does_not_double_count_peer_prefill(self):
        artifact, traces, prompts = mixed_streams()
        traces[0]["decode_step_ids"] = [2, 7, 8]
        data, _, _ = audit.streams(artifact, traces, prompts, 2, True, 3, "pair")
        self.assertEqual(data["p1"]["gaps_at_least_100ms"], 2)
        self.assertEqual(data["p1"]["peer_prefill_gaps_at_least_100ms"], 1)

    def test_missing_route_and_gap_cannot_pass(self):
        artifact, traces, prompts = mixed_streams()
        traces[1]["prefill"][1]["split_end"] = None
        with self.assertRaisesRegex(ValueError, "prefill cuts"):
            audit.streams(artifact, traces, prompts, 2, True, 3, "pair")
        artifact, traces, prompts = mixed_streams()
        artifact["observations"][0]["stream_step_interval_ms"].pop()
        with self.assertRaisesRegex(ValueError, "gap alignment"):
            audit.streams(artifact, traces, prompts, 2, True, 3, "pair")

    def test_paired_delta_retains_the_maximum_p95_tradeoff(self):
        data, _, _ = audit.streams(*mixed_streams(), 2, True, 3, "pair")
        candidate = {ident: dict(row) for ident, row in data.items()}
        candidate["p1"]["per_request_gap_max_ms"] -= 2
        candidate["p1"]["per_request_gap_p95_ms"] += 1
        candidate["p1"]["peer_prefill"] = False
        delta = audit.compare(candidate, data)["512"]
        self.assertEqual(delta["requests"], 1)
        self.assertEqual(delta["lower_max_gap_with_higher_gap_p95"], 1)
        self.assertEqual(delta["per_request_gap_max_ms"]["p95"], -2)
