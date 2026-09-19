"""Check existing canonical verifier traces without regrading cross-route identity."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

PROMPTS_SHA = "392dcbf78d6e39224e5917217752042567e81a45ddc8a1c9c9b186cb1ba6ab88"
EVENT = "FLASH_NEXT_VERIFY_DIAGNOSTIC "
BOUNDARY = re.compile(
    r"Flash Next MTP oracle request: session_proposed=(\d+) session_accepted=(\d+) "
    r"agreement_samples=(\d+) agreement_matches=(\d+)$"
)


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value):
    require(type(value) is int and value >= 0, "Invalid nonnegative integer")
    return value


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def groups(log):
    completed, events = [], []
    for line in log.splitlines():
        if EVENT in line:
            require(line.count(EVENT) == 1, "Ambiguous verifier event")
            events.append(parse(line.split(EVENT, 1)[1]))
        elif "Flash Next MTP oracle request:" in line:
            match = BOUNDARY.search(line)
            require(match is not None, "Malformed request boundary")
            completed.append((events, tuple(map(int, match.groups()))))
            events = []
    require(not events, "Unterminated request trace")
    return completed


def review(raw, manifest, log, terminal_ids):
    require(raw["qualification"] is False and
            raw["route"] == "flash_next_mtp_trained_head_oracle" and
            raw["target_schedule"] == "canonical_singleton", "Wrong trace scope")
    require(raw["terminal_ids"] == terminal_ids, "Changed terminal IDs")
    prompts, requests, traces = manifest["requests"], raw["requests"], groups(log)
    require(len(prompts) == len(requests) == len(traces) > 0, "Incomplete trace cohort")
    require(len({p["id"] for p in prompts}) == len(prompts), "Duplicate request IDs")
    budget = integer(manifest["max_new_tokens"])
    proposed = accepted = excluded = all_proposed = all_accepted = 0
    request_reports = []
    for prompt, item, (events, counters) in zip(prompts, requests, traces, strict=True):
        require(item["id"] == prompt["id"] and item["prompt_ids"] == prompt["prompt_ids"],
                "Changed request identity or order")
        tokens = item["generated_ids"]
        require(tokens == item["greedy_tokens"] and 0 < len(tokens) <= budget,
                "Invalid output or output aliases")
        for token in tokens:
            integer(token)
        terminal = [i for i, token in enumerate(tokens) if token in terminal_ids]
        require(terminal == ([len(tokens) - 1] if terminal else []), "Output after EOS")
        require(bool(terminal) or len(tokens) == budget, "Incomplete nonterminal output")
        require(item["stopped_at_terminal"] is bool(terminal) and
                item["terminal_position"] == (terminal[0] if terminal else None), "Wrong EOS accounting")
        short = (terminal[0] if terminal else len(tokens)) < 2
        require(item["too_short"] is short, "Wrong short exclusion")
        count, good = integer(item["proposed"]), integer(item["accepted"])
        agreement = item["draft_vs_primary_top1_agreement"]
        require(all(type(value) is bool for value in agreement), "Invalid agreement samples")
        require(counters == (count, good, len(agreement), sum(agreement)) and
                len(events) == count, "Request counters or event coverage disagree")
        cursor = actual_accepted = 0
        for event in events:
            require(event["schema"] == "ax-engine.flash-next.verify-diagnostic.v1" and
                    event["qualification"] is False and event["performance_claim"] is False and
                    event["target_schedule"] == "canonical_singleton", "Wrong event scope")
            require(integer(event["position"]) == len(prompt["prompt_ids"]) + cursor and
                    integer(event["remaining"]) == budget - cursor > 1, "Wrong event frontier or budget")
            require(cursor < len(tokens) and integer(event["primary"]) == tokens[cursor]
                    and tokens[cursor] not in terminal_ids, "Wrong committed primary")
            decision = event["singleton"]
            require(decision["source"] == "canonical_primary" and event["batched_row0"] is None
                    and event["batched_singleton_argmax_equal"] is None, "Not a same-state canonical decision")
            target, draft = integer(decision["token"]), integer(event["draft"])
            margin = decision["margin"]
            require(type(margin) in (int, float) and math.isfinite(margin) and margin >= 0,
                    "Invalid target margin")
            accepted_here = draft == target and target not in terminal_ids
            require(type(event["accepted"]) is bool and event["accepted"] == accepted_here,
                    "Invalid same-state acceptance or rejection")
            consumed = 2 if accepted_here else 1
            require(integer(event["committed_len"]) == consumed, "Wrong committed length")
            require(cursor + 1 < len(tokens) and tokens[cursor + 1] == target,
                    "Verifier decision was not emitted")
            actual_accepted += int(accepted_here)
            cursor += consumed
        require(actual_accepted == good <= count, "Accepted counter mismatch")
        require(cursor == len(tokens) or
                (cursor + 1 == len(tokens) and
                 (tokens[-1] in terminal_ids or cursor == budget - 1)),
                "Unexplained output beyond verifier coverage")
        all_proposed += count
        all_accepted += good
        excluded += int(short)
        if not short:
            require(count > 0, "Non-short request has no proposals")
            proposed += count
            accepted += good
        request_reports.append(dict(id=item["id"], proposed=count, accepted=good,
                                    excluded_short=short, verified_events=len(events)))
    require(integer(raw["proposed"]) == proposed > 0 and integer(raw["accepted"]) == accepted
            and integer(raw["too_short_requests"]) == excluded, "Aggregate counters mismatch")
    require(math.isclose(raw["acceptance_rate"], accepted / proposed, rel_tol=1e-12, abs_tol=1e-12),
            "Aggregate acceptance rate mismatch")
    return dict(schema="ax.flash_next.same_state_trace_review.v1", bounded_acceptance_passed=True,
                invalid_acceptances=0, requests=request_reports, all_proposals=all_proposed,
                all_acceptances=all_accepted, rate_proposals=proposed, rate_acceptances=accepted,
                accepted_path_observed=all_accepted > 0,
                rejection_path_observed=all_proposed > all_accepted,
                acceptance_coverage_gap=None if all_accepted else "No accepted draft was observed",
                qualification=False, release_ready=False,
                mtp_certification={g: "not_assessed" for g in ("MTP-S", "MTP-P", "MTP-D")},
                scope="Bounded canonical acceptance trace only; rollback, installed routing and full safety qualification remain separate",
                rejected_draft_scope="Rejected draft logits are not independently recomputed",
                cross_route_identity="Retained in original result; neither inferred nor regraded here")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "result", "log", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--log-sha256", required=True)
    args = parser.parse_args()
    data = {}
    for name, pin in (("manifest", PROMPTS_SHA), ("result", args.result_sha256), ("log", args.log_sha256)):
        data[name] = getattr(args, name).read_bytes()
        require(hashlib.sha256(data[name]).hexdigest() == pin, "Changed input: " + name)
    manifest = parse(data["manifest"])
    require(len(manifest["requests"]) == 104 and manifest["max_new_tokens"] == 16,
            "Changed original cohort")
    result = review(parse(data["result"]), manifest, data["log"].decode(), [248046, 248044])
    result["input_hashes"] = {k: hashlib.sha256(v).hexdigest() for k, v in data.items()}
    result["reader_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
