"""Check the Flash Next component readings against the recorded reference path."""

import gzip
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
MODES = ("sorted_bonus", "gathered_bonus", "maximum_bonus", "sorted_correction",
         "scatter_correction", "mask_correction")


def read(path):
    return json.loads(path.read_bytes())


def observable(mode, value):
    if mode.endswith("bonus") and value is not None:
        return value[0]
    return value


def summarize(raw):
    if (raw["qualification"] is not False or raw["mlx"] != "0.32.4" or raw["width"] != 248_320
            or raw["dtype"] != "float32" or raw["measurements"] != 60 or raw["warmups"] != 8
            or len(raw["adversarial"]) != 9 or len(raw["wide_adversarial"]) != 36):
        raise ValueError("The recorded component workload changed")
    difficult = {**raw["adversarial"], **raw["wide_adversarial"]}
    results = {}
    for mode in MODES:
        baseline = "sorted_bonus" if mode.endswith("bonus") else "sorted_correction"
        times = sorted(raw["raw_us"][mode])
        values = raw["outputs"][mode]
        if len(times) != len(values) or len(times) != 60 or any(not isfinite(t) or t <= 0 for t in times):
            raise ValueError("A measured timing or output is missing")
        differences = [i for i, (a, b) in enumerate(zip(values, raw["outputs"][baseline], strict=True))
                       if observable(mode, a) != observable(baseline, b)]
        adversarial = [name for name, row in difficult.items()
                       if observable(mode, row[mode]) != observable(baseline, row[baseline])]
        computed = {"median_us": median(times), "p95_us": times[57], "finite_bit_differences": differences,
                    "adversarial_differences": sorted(adversarial)}
        expected = {**raw["summary"][mode], "adversarial_differences": sorted(raw["summary"][mode]["adversarial_differences"])}
        if computed != expected or (mode in ("gathered_bonus", "scatter_correction") and (differences or adversarial)):
            raise ValueError(f"The recorded verifier behavior disagrees in {mode}")
        results[mode] = computed
    return results


def main():
    source = read(ROOT / "source.json")
    if sha256((ROOT / "probe.py").read_bytes()).hexdigest() != source["probe_sha256"]:
        raise ValueError("Component probe source changed")
    captures = {}
    for name in ("m2", "m5-first", "m5-repeat"):
        raw = json.loads(gzip.decompress((ROOT / f"{name}.json.gz").read_bytes()))
        if raw["libmlx_sha256"] != source["libmlx_sha256"]:
            raise ValueError("The component run used a different MLX library")
        host = source["m5_host_samples"].get(name)
        if host is not None:
            eligible = all(90 <= idle <= 100 for idle in host["cpu_idle_percent"])
            if len(host["cpu_idle_percent"]) != 2 or host["timing_eligible"] != eligible:
                raise ValueError("The M5 timing eligibility changed")
        captures[name] = {"hardware": raw["hardware"], "host": host, "routes": summarize(raw)}
    print(json.dumps({"qualification": False, "captures": captures}, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
