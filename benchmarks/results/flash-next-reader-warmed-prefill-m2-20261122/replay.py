"""Compare exact resident-reader stage profiles from two measured shapes."""
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median

ORDERS = {"short-first": (2048, 8192), "long-first": (8192, 2048)}
PERIODS = ("before_additional_8k", "after_additional_8k")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def valid_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0.0


def mode_view(data):
    require(data["qualification"] is False and valid_seconds(data["load_seconds"]),
            "Missing diagnostic label or load time")
    require(data["loader"] in ("mmap", "pread") and data["order"] in ORDERS,
            "Invalid test ladder identity")
    widths = ORDERS[data["order"]]
    profiles = data["profiles"]
    require([item["prompt_tokens"] for item in profiles] == list(widths),
            "Missing full geometry profiles")
    tokens = {}
    states = {}
    stages = {}
    for item in profiles:
        width = item["prompt_tokens"]
        cursor = 0
        for offset, count in item["forwards"]:
            require(offset == cursor and 0 < count <= width - cursor, "Discontinuous prefill geometry")
            cursor += count
        require(cursor == width and item["forwards"][-1] == [width - 1, 1],
                "The final singleton or prompt boundary changed")
        token, digest = item["first_token"], item["serialized_state_sha256"]
        require(type(token) is int and 0 <= token < 248320 and len(bytes.fromhex(digest)) == 32,
                "Missing exact token or request state")
        require(valid_seconds(item["traced_prefill_seconds"])
                and item["stages"] and all(valid_seconds(v) for v in item["stages"].values())
                and sum(item["stages"].values()) <= item["traced_prefill_seconds"],
                "Invalid attribution stage time")
        tokens[width], states[width] = token, (digest, item["forwards"])
        stages[width] = item["stages"]
    times = defaultdict(list)
    schedule = [(period, repeat, width) for period in PERIODS for repeat in range(3) for width in widths]
    for row, expected in zip(data["measurements"], schedule, strict=True):
        period, _, width = expected
        require((row["phase"], row["repeat"], row["prompt_tokens"]) == expected
                and row["first_token"] == tokens[width] and valid_seconds(row["prefill_seconds"]),
                "Invalid measurement order, output token, or elapsed time")
        times[period, width].append(row["prefill_seconds"])
    return tokens, states, stages, {key: median(value) for key, value in times.items()}


def compare(data):
    by_mode = {(arm["loader"], arm["order"]): mode_view(arm) for arm in data}
    orders = {order for _, order in by_mode}
    require(len(by_mode) == len(data) == 2 * len(orders), "Need one matched reader pair per order")
    require(all((mode, order) in by_mode for order in orders for mode in ("mmap", "pread")),
            "A warmup order has no reader pair")
    states = [(view[0], view[1]) for view in by_mode.values()]
    require(all(value == states[0] for value in states[1:]),
            "Cross-mode token, state or chunk geometry changed")
    results = {}
    for order in sorted(orders):
        _, _, ref_stages, ref_times = by_mode["mmap", order]
        _, _, pos_stages, pos_times = by_mode["pread", order]
        medians = {period: {str(n): {"mmap_seconds": ref_times[period, n],
                                    "pread_seconds": pos_times[period, n],
                                    "pread_to_mmap": pos_times[period, n] / ref_times[period, n]}
                            for n in ORDERS[order]} for period in PERIODS}
        deltas = {}
        for width in ORDERS[order]:
            left, right = ref_stages[width], pos_stages[width]
            require(left.keys() == right.keys(), "Reader mode changed traced stage roster")
            deltas[str(width)] = {stage: right[stage] - left[stage] for stage in sorted(left)}
        results[order] = {"untraced_medians": medians, "traced_pread_minus_mmap_seconds": deltas}
    return {"scope": "Model-only prefill diagnosis", "exact_state_and_token_identity": True,
            "orders": results}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    receipt = json.loads((root / "m2-verified.json").read_text())
    arms = []
    for name in ("mmap.json", "pread.json"):
        raw = (root / name).read_bytes()
        require(sha256(raw).hexdigest() == receipt["profile_sha256"][name],
                f"Profile digest changed for {name}")
        arms.append(json.loads(raw))
    print(json.dumps(compare(arms), indent=2, sort_keys=True, allow_nan=False))
