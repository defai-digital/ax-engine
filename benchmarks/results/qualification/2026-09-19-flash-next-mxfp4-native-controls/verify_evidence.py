"""Recompute bounded native controls offline; this does not execute or qualify a model."""
import gzip
import hashlib
import json
from pathlib import Path

import review_same_state
import state_control
import validate_pair

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load():
    summary = json.loads((ROOT / "summary.json").read_text())
    for name, row in summary["artifacts"].items():
        require(Path(name).name == name, "Invalid artifact name")
        raw = (ROOT / name).read_bytes()
        require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                "Changed artifact: " + name)
    raw = (ROOT / "native-controls.json.gz").read_bytes()
    require(raw[4:8] == bytes(4), "Nondeterministic gzip timestamp")
    return summary, json.loads(gzip.decompress(raw))


def verify(summary, evidence):
    require(evidence["source_commit"] == summary["source_commit"] ==
            "5f583018032c406cd3016c6c1c016612dd8b2fbd", "Wrong source")
    require(evidence["binary_sha256"] == summary["binary_sha256"] ==
            "f06165d173ce260bf3f24b4a10e1e3bb5a4bf518d53e29466c6dd4cc2b416629", "Wrong binary")
    require(evidence["wheel_sha256"] == summary["wheel_sha256"] ==
            "06c65e8069431258e77490be0eefd9067adcb262cdea156e0c126c0525f0cd14", "Wrong wheel")
    for item in (summary, evidence, evidence["root_review"]):
        require(item["qualification"] is False and item["release_ready"] is False,
                "Bounded controls cannot qualify a release")
    gates = {name: "not_assessed" for name in ("MTP-S", "MTP-P", "MTP-D")}
    require(summary["mtp_certification"] == evidence["root_review"]["mtp_certification"] == gates,
            "Unjustified MTP gate promotion")
    records = evidence["records"]
    manifest = records["trained-head-prompts.json"]
    require(len(manifest["requests"]) == 104 and manifest["max_new_tokens"] == 16,
            "Changed original cohort")
    states, runner = (records["output/" + name + ".json"] for name in ("state", "runner"))
    state_control.state_control(states)
    state_control.state_control(runner, runner=True)
    require(states["compared_generated_tokens"] == summary["state_compared_generated_tokens"] == 4
            and states["compared_state_steps"] == summary["state_compared_steps"] == 2
            and len(states["state_arrays"]) == summary["state_arrays"] == 109,
            "Changed state coverage")
    require(len(runner["direct_ids"]) == summary["runner_compared_tokens"] == 3
            and runner["prefill_state_compared"] is summary["runner_prefill_state_compared"] is False
            and runner["decode_state_compared"] is summary["runner_decode_state_compared"] is False,
            "Overstated runner coverage")
    reports = {}
    for name in ("real_head", "permuted_head"):
        raw = records["output/" + name + ".json"]
        report = review_same_state.review(raw, manifest, records["output/" + name + ".log"],
                                          [248046, 248044])
        stored = records["output/" + name + "-same-state.json"]
        require(all(stored[key] == value for key, value in report.items()), "Changed trace report")
        require(report["invalid_acceptances"] == summary["invalid_same_state_acceptances"][name] == 0,
                "Invalid same-state acceptance")
        expected = summary["head_summary"][name]
        require(expected["proposed"] == report["rate_proposals"]
                and expected["accepted"] == report["rate_acceptances"]
                and expected["requests"] == len(report["requests"]) == 104
                and expected["excluded_short_requests"] == raw["too_short_requests"] == 55
                and expected["contributing_requests"] == 49
                and expected["acceptance_rate"] == raw["acceptance_rate"], "Changed head summary")
        reports[name] = report
    pair = validate_pair.validate_pair(records["output/real_head.json"],
                                       records["output/permuted_head.json"], manifest, [248046, 248044])
    require(pair == evidence["root_review"]["original_pair"] and pair["passed"] is True
            and pair["greedy_identity"] is summary["paired_greedy_identity"] is True,
            "Changed original paired identity")
    return dict(bounded_native_controls_passed=True, same_state_invalid_acceptances=0,
                requests_per_head=104, real_accepted=reports["real_head"]["rate_acceptances"],
                real_proposed=reports["real_head"]["rate_proposals"],
                permuted_accepted=reports["permuted_head"]["rate_acceptances"],
                permuted_proposed=reports["permuted_head"]["rate_proposals"],
                mtp_certification=gates, qualification=False, release_ready=False)


if __name__ == "__main__":
    print(json.dumps(verify(*load()), indent=2))
