#!/usr/bin/env python3
"""Compare a fixed grid and Optuna on audited Flash Next scheduling evidence."""

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from math import isclose, isfinite
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory

if __package__:
    from . import audit_flash_next_mixed_serving as audit
    from .bench_ax_serving import load_corpus
else:
    import audit_flash_next_mixed_serving as audit
    from bench_ax_serving import load_corpus


RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
SERVING_BUNDLE = RESULTS / "flash-next-mxfp4-selective-prefill-m5-20261121"
UNPACK = RESULTS / "flash-next-mxfp4-mixed-serving-m5-20261121" / "replay.py"
REPAY = RESULTS / "flash-next-prefill-repayment-sweep-m5-20261122" / "replay.py"
POLICIES = ("off", "all", "long")
ARM_ORDER = ("off-1", "all-1", "long-1", "long-2", "all-2", "off-2")
SERVING_METRICS = ("short_stream_gap_p95_ms", "long_arrival_ttft_p95_ms", "output_tokens_per_second")
REPAYMENT_METRICS = ("short_runner_gap_p95_us", "long_first_token_added_runner_us")


@dataclass(frozen=True)
class Measurement:
    values: tuple[float, ...] | None
    reason: str | None = None


def measured_arms(bundle):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = runpy.run_path(str(UNPACK))["unpack"](bundle, root)
        folder = root / "mixed-result"
        receipt = audit.read_json(folder / "receipt.json")
        build = audit.read_json(root / "build.json")
        workload = audit.read_json(root / "workload.json")
        protocol = audit.read_json(root / "protocol.json")
        corpus = root / "mixed-corpus.jsonl"
        audit.require(source["partial"] is False and receipt["status"] == "complete"
                      and tuple(source["arm_order"]) == ARM_ORDER
                      and tuple(receipt["arms"]) == ARM_ORDER
                      and source["binary_sha256"] == receipt["binary_sha256"] == build["server_sha256"]
                      and receipt["workload"] == workload and workload["pack_revision"] == audit.PACK
                      and audit.digest(Path(__file__).with_name("bench_ax_serving.py").read_bytes())
                      == workload["benchmark_sha256"]
                      and audit.digest(corpus.read_bytes()) == workload["corpus_sha256"],
                      "Incomplete or unbound serving campaign")
        inventory = audit.read_json(root / "mxfp4-inventory.json")
        audit.require(inventory["revision"] == audit.PACK and len(inventory["members"]) == 49,
                      "Incorrect publisher inventory")
        parsed = load_corpus(corpus)
        prompts = {row.id: row for row in parsed}
        budget = workload["reply_budget"]
        audit.require(len(parsed) == len(prompts) == workload["requests"]
                      and all(row.input_tokens and row.max_output_tokens == budget
                              and row.arrival_s is not None for row in parsed),
                      "Wrong prompt inventory or output budget")
        traffic = sorted((row.arrival_s, row.input_tokens, row.max_output_tokens) for row in parsed)
        traffic_sha256 = audit.digest(json.dumps(traffic, separators=(",", ":"), allow_nan=False).encode())
        short = min(len(row.input_tokens) for row in parsed)
        long = max(len(row.input_tokens) for row in parsed)
        minimum = protocol["min_prompt_tokens"]
        audit.require(short < long and type(minimum) is int and audit.PREFILL_ITEM <= minimum <= long,
                      "The serving corpus or selective split threshold is invalid")
        baseline = solo_tokens(root / "solo-reference.json", workload, prompts)
        results = []
        for leg in (1, 2):
            values = {}
            for policy in POLICIES:
                name = f"{policy}-{leg}"
                metadata = receipt["arms"][name]
                audit.require(metadata["policy"] == policy, f"{name}: policy identity")
                try:
                    values[policy] = score_arm(folder, name, policy, minimum, metadata, workload,
                                               prompts, baseline, short, long)
                except ValueError as error:
                    values[policy] = Measurement(None, str(error))
            audit.require(values["off"].values is not None, "The paired baseline failed its safety checks")
            results.append(values)
        provenance = {"pack_revision": audit.PACK, "corpus_sha256": workload["corpus_sha256"],
                      "offered_traffic_sha256": traffic_sha256,
                      "archive_sha256": audit.digest((bundle / "evidence.tar.xz").read_bytes()),
                      "server_sha256": receipt["binary_sha256"], "requests_per_arm": len(prompts),
                      "short_prompt_tokens": short, "long_prompt_tokens": long,
                      "selective_split_min_prompt_tokens": minimum,
                      "validation_scope": "Reverse process order with the same offered arrivals"}
        return results, provenance


def solo_tokens(path, workload, prompts):
    artifact = audit.read_json(path)
    rows = [row for row in artifact["observations"] if row["phase"] == "measured"]
    tokens = {}
    audit.require(artifact["corpus"]["sha256"] == workload["corpus_sha256"]
                  and artifact["load"]["concurrency"] == 1
                  and artifact["sampling"]["temperature"] == 0
                  and artifact["sampling"]["ignore_eos"] is True,
                  "Wrong solo baseline")
    for row in rows:
        name = row["prompt_id"]
        audit.require(name in prompts and name not in tokens and row["ok"] is True
                      and row["prompt_token_ids"] == prompts[name].input_tokens
                      and len(row["output_token_ids"]) == workload["reply_budget"]
                      and all(type(token) is int and 0 <= token < 248320 for token in row["output_token_ids"]),
                      "Invalid solo reference output")
        tokens[name] = row["output_token_ids"]
    audit.require(tokens.keys() == prompts.keys(), "Incomplete solo reference")
    return tokens


def score_arm(folder, name, policy, minimum, metadata, workload, prompts, baseline, short, long):
    split = policy != "off"
    footprint = metadata["peak_footprint_bytes"]
    audit.require(type(footprint) is int and 0 < footprint <= (128 << 30),
                  f"{name}: footprint exceeds the Flash Next standard")
    archive, tokens, _ = audit.audit_arm(
        folder, name, metadata, workload, prompts, profile=(2, split),
        split_min_prompt_tokens=minimum if policy == "long" else audit.PREFILL_ITEM,
    )
    audit.require(tokens == baseline, f"{name}: greedy tokens differ from the solo reference")
    audit.require((archive["shared_split_requests"] > 0) == split, f"{name}: missing policy contrast")
    artifact = audit.read_json(folder / f"{name}.json")
    rows = [row for row in artifact["observations"] if row["phase"] == "measured"]
    elapsed = max(row["started_at_s"] + row["e2e_latency_ms"] / 1000 for row in rows)
    audit.require(isfinite(elapsed) and elapsed > 0, f"{name}: invalid serving duration")
    rate = sum(len(ids) for ids in tokens.values()) / elapsed
    audit.require(isclose(rate, artifact["summary"]["output_token_throughput_tok_s"], rel_tol=1e-9),
                  f"{name}: serving throughput disagrees with the raw client timeline")
    by_length = archive["by_length_and_peer_prefill"]
    return Measurement((by_length[f"{short}/all"]["per_request_gap_p95_ms"]["p95"],
                        by_length[f"{long}/all"]["arrival_to_first_token_ms"]["p95"], rate))


def repayment_arms():
    model = runpy.run_path(str(REPAY))["replay"](include_grid=True)
    grids = {}
    for label, data in model["decode_sensitivity"].items():
        results = {}
        for row in data["fixed_grid"]:
            share = row["repayment_percent"]
            elapsed = row["long_first_token_added_runner_us_before_short_completion"]
            results[share] = (Measurement(None, "Long first token falls outside the measured schedule")
                              if elapsed is None else Measurement((row["short_runner_gap_p95_us"], elapsed)))
        grids[label] = results
    return [grids["p50"], grids["p10"], grids["p90"]], {
        "input_archive_sha256": model["input_archive_sha256"],
        "fixed_step_widths": model["fixed_step_widths"],
        "validation_scope": "p10 and p90 decoder costs on the same trace, not new traffic",
    }


def frontier(results, directions):
    valid = {setting: tuple(value if direction == "minimize" else -value
                            for value, direction in zip(row.values, directions, strict=True))
             for setting, row in results.items() if row.values is not None}
    return [setting for setting, values in valid.items()
            if not any(other != values and all(a <= b for a, b in zip(other, values, strict=True))
                       for other in valid.values())]


def compare_searches(optuna, datasets, directions, budget, seed):
    training, *validation = datasets
    choices = tuple(training)
    indices = [0] if budget == 1 else [round(i * (len(choices) - 1) / (budget - 1)) for i in range(budget)]
    fixed = {choices[index]: training[choices[index]] for index in indices}
    study = optuna.create_study(directions=directions,
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=2))
    study.enqueue_trial({"setting": choices[0]})
    tried = {}
    duplicates = 0
    integer_range = choices == tuple(range(len(choices)))
    while len(tried) < budget:
        trial = study.ask()
        setting = (trial.suggest_int("setting", 0, len(choices) - 1) if integer_range
                   else trial.suggest_categorical("setting", choices))
        if setting in tried:
            duplicates += 1
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            study.enqueue_trial({"setting": next(value for value in choices if value not in tried)})
            continue
        row = training[setting]
        tried[setting] = row
        if row.values is None:
            trial.set_user_attr("rejection", row.reason)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, row.values)
    tuned = describe(tried, validation, directions, choices[0])
    audit.require(set(tuned["training_frontier"]) == {trial.params["setting"] for trial in study.best_trials},
                  "Optuna and fixed-grid Pareto rules disagree")
    return {"fixed_grid": describe(fixed, validation, directions, choices[0]),
            "optuna_tpe": {**tuned, "duplicate_suggestions_skipped": duplicates}}


def describe(results, validation, directions, baseline):
    finalists = frontier(results, directions)
    return {"evaluated": [{"setting": setting, "values": row.values, "rejection": row.reason}
                          for setting, row in results.items()],
            "training_frontier": finalists,
            "shortlisted_candidates": [setting for setting in finalists if setting != baseline
                                    and improves(results[setting], results[baseline], directions)
                                    and all(improves(data[setting], data[baseline], directions)
                                            for data in validation)],
            "validation": [{"baseline": data[baseline].values,
                            "frontier_trials": [{"setting": setting, "values": data[setting].values,
                                                 "rejection": data[setting].reason} for setting in finalists]}
                           for data in validation]}


def improves(candidate, baseline, directions):
    if candidate.values is None or baseline.values is None:
        return False
    changes = tuple(a - b if direction == "minimize" else b - a
                    for a, b, direction in zip(candidate.values, baseline.values, directions, strict=True))
    return max(changes) <= 0 and min(changes) < 0


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("serving", "repayment"))
    parser.add_argument("--bundle", type=Path, default=SERVING_BUNDLE,
                        help="sealed six-arm mixed-serving benchmark directory")
    parser.add_argument("--holdout-bundle", type=Path, action="append", default=[],
                        help="additional serving campaign with different arrivals and the same executable")
    parser.add_argument("--budget", type=int, help="unique evaluations per search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="write a new JSON report instead of stdout")
    args = parser.parse_args(argv)
    serving = args.study == "serving"
    budget = args.budget if args.budget is not None else (3 if serving else 24)
    if not 1 <= budget <= (3 if serving else 101):
        parser.error("--budget must be between 1 and the number of settings (3 serving or 101 repayment)")
    if not serving and (args.bundle != SERVING_BUNDLE or args.holdout_bundle):
        parser.error("--bundle and --holdout-bundle are supported only for serving evidence")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    try:
        import optuna
    except ImportError:
        parser.error("install the offline study dependency from scripts/requirements-optuna.txt")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    metric_names = SERVING_METRICS if serving else REPAYMENT_METRICS
    directions = ("minimize", "minimize", "maximize") if serving else ("minimize", "minimize")
    datasets, provenance = measured_arms(args.bundle) if serving else repayment_arms()
    if args.holdout_bundle:
        add_holdouts(datasets, provenance, args.holdout_bundle)
    result = {"schema": "ax-engine.flash-next.optuna-study.v1", "study": args.study,
              "metrics": dict(zip(metric_names, directions, strict=True)), "budget": budget,
              "seed": args.seed, "optuna_version": optuna.__version__, "qualification": False,
              "provenance": provenance, **compare_searches(optuna, datasets, directions, budget, args.seed)}
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with args.output.open("x") as handle:
            handle.write(rendered)


def add_holdouts(datasets, provenance, bundles):
    digests = {provenance["offered_traffic_sha256"]}
    verified = []
    for bundle in bundles:
        data, binding = measured_arms(bundle)
        fingerprint = binding["offered_traffic_sha256"]
        audit.require(fingerprint not in digests, "A held-out corpus must offer distinct traffic")
        audit.require(all(binding[key] == provenance[key] for key in (
            "server_sha256", "pack_revision", "selective_split_min_prompt_tokens")),
            "Holdout does not exercise the same executable, checkpoint and split policy")
        digests.add(fingerprint)
        datasets.extend(data)
        verified.append(binding)
    provenance["additional_holdouts"] = verified


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit(f"Flash Next study failed: {error}")
