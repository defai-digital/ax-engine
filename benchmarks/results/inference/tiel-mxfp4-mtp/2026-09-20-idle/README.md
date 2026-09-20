# Tiel native idle GPU-touch screening

This is an **AX-only native-library diagnostic**, not an implemented server
keepalive, new default, or an updated AX-versus-MTPLX server comparison.
Energy saving is not an acceptance gate; latency, throughput and stability are.

Both exact packs and the Rust 1.97.1 release-pyext extension are unchanged from
[the matched four-host peer campaign](../2026-09-20-peer/README.md).
The measured extension SHA256 is
`4b4d26a19cbdd9e475f777e491c28869a421a897a4928fb2183583230de2bcb6`.
The preceding artifact binds its source, MLX assets and model revisions. Current
doctor/estimator changes are not the measured inference binary.

## Contract

- One model loaded per process, full weights retained, expert streaming Off.
- Greedy seed 0, active throughput MTP cap 3, cold KV/prefix reuse disabled.
- Python-LRU workload: Tiel 194 input tokens, Cyber-Tiel 219, both 256 output.
- Two fresh-process reversed blocks, two warmups and three measurements each.
  Six measured samples per cell; 144 measurements and 96 warmups in total.
- Exactly the same requested 3 s idle in all arms. Actual idle elapsed time is
  retained. All arms preallocate/evaluate the same tiny probe once after load.
- `wired`: explicit wired scale 0.9. `unwired`: explicit scale 0. Both preserve
  model allocations. On M5, wired deliberately overrides existing automatic
  unwiring, so it is not the normal M5 product path.
- `touch`: wired scale 0.9, with a disclosed same-calling-thread GPU operation
  approximately every 0.5 s during idle. This independent screening operation
  is not extracted MTPLX code, a background thread or a server scheduler.
- API entry to first committed callback = TTFT; decode excludes that batch's
  tokens and elapsed time. Completion includes TTFT. No model-load time.
- Exact output IDs match across all three arms, both blocks and warmups, for
  each model on each host. This does not measure answer quality.

## Results

Six-sample medians. TTFT is milliseconds (lower is better); decode and
completion are tokens/second (higher is better). These are not pure prefill
rates or population confidence bounds.

| Host / model | Wired TTFT | Unwired TTFT | Touch TTFT | Wired decode | Unwired decode | Touch decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M5 / tiel | 246.78 | 143.74 | 112.99 | 219.30 | 217.96 | 214.99 |
| M5 / cyber | 253.51 | 143.26 | 120.20 | 250.35 | 248.75 | 244.92 |
| M4 / tiel | 497.00 | 351.22 | 330.85 | 109.18 | 108.29 | 109.26 |
| M4 / cyber | 517.00 | 373.76 | 355.68 | 116.52 | 115.44 | 116.56 |
| M2 / tiel | 493.31 | 277.86 | 248.89 | 134.26 | 132.81 | 132.47 |
| M2 / cyber | 509.90 | 310.10 | 285.34 | 153.37 | 151.92 | 146.33 |
| M3 / tiel | 327.28 | 209.62 | 152.88 | 202.38 | 201.20 | 198.62 |
| M3 / cyber | 336.02 | 209.56 | 159.71 | 218.98 | 217.43 | 215.50 |

| Host / model | Wired completion | Unwired completion | Touch completion | Touch decode vs unwired |
| --- | ---: | ---: | ---: | ---: |
| M5 / tiel | 181.62 | 194.86 | 197.16 | -1.4% |
| M5 / cyber | 201.19 | 219.05 | 220.53 | -1.5% |
| M4 / tiel | 90.33 | 94.60 | 96.03 | +0.9% |
| M4 / cyber | 94.57 | 99.11 | 100.57 | +1.0% |
| M2 / tiel | 107.15 | 116.28 | 117.75 | -0.3% |
| M2 / cyber | 117.78 | 128.79 | 126.36 | -3.7% |
| M3 / tiel | 161.37 | 173.70 | 178.30 | -1.3% |
| M3 / cyber | 170.80 | 185.52 | 190.52 | -0.9% |

Touch lowers median TTFT in all eight coding cells. It is not a universal win:
M2 Cyber-Tiel decode falls about 3.7% versus unwired (4.6% versus wired),
exceeding a 3% regression screen. The current evidence therefore does **not**
justify adding default keepalive or broadening automatic unwiring. It supports
further targeted server experiments, not selecting a production timer.

## Sample spread

TTFT min/median/max and each block median expose drift. Allocator memory is
recorded in raw rows; it is not RSS or a whole-system pressure qualification.

| Host / model / arm | TTFT min / median / max ms | Block 0 / 1 TTFT ms | Decode min / max tok/s |
| --- | ---: | ---: | ---: |
| m2 / cyber / touch | 281.93 / 285.34 / 288.10 | 285.92 / 284.77 | 144.60 / 148.43 |
| m2 / cyber / unwired | 305.29 / 310.10 / 316.19 | 308.93 / 312.99 | 150.96 / 152.92 |
| m2 / cyber / wired | 491.94 / 509.90 / 517.64 | 506.85 / 512.24 | 152.59 / 154.66 |
| m2 / tiel / touch | 248.07 / 248.89 / 251.82 | 249.66 / 248.37 | 132.36 / 133.38 |
| m2 / tiel / unwired | 276.54 / 277.86 / 286.90 | 277.82 / 277.89 | 132.04 / 133.34 |
| m2 / tiel / wired | 474.80 / 493.31 / 498.94 | 491.44 / 495.17 | 133.76 / 134.97 |
| m3 / cyber / touch | 156.81 / 159.71 / 161.76 | 157.55 / 161.29 | 214.82 / 218.54 |
| m3 / cyber / unwired | 200.45 / 209.56 / 212.43 | 208.10 / 211.01 | 215.94 / 219.27 |
| m3 / cyber / wired | 316.21 / 336.02 / 339.58 | 332.36 / 338.17 | 217.62 / 220.78 |
| m3 / tiel / touch | 151.00 / 152.88 / 158.60 | 153.76 / 152.01 | 198.33 / 200.22 |
| m3 / tiel / unwired | 203.99 / 209.62 / 224.18 | 208.70 / 210.55 | 199.63 / 202.72 |
| m3 / tiel / wired | 323.06 / 327.28 / 333.44 | 328.02 / 324.84 | 201.27 / 204.24 |
| m4 / cyber / touch | 348.61 / 355.68 / 371.04 | 355.63 / 355.74 | 116.41 / 116.74 |
| m4 / cyber / unwired | 371.63 / 373.76 / 377.22 | 373.24 / 374.83 | 112.99 / 115.59 |
| m4 / cyber / wired | 514.99 / 517.00 / 526.82 | 517.51 / 516.48 | 116.44 / 116.58 |
| m4 / tiel / touch | 328.98 / 330.85 / 336.47 | 330.54 / 331.16 | 108.22 / 109.41 |
| m4 / tiel / unwired | 350.01 / 351.22 / 352.50 | 350.96 / 351.49 | 105.46 / 108.34 |
| m4 / tiel / wired | 486.52 / 497.00 / 505.57 | 492.40 / 500.53 | 109.08 / 109.36 |
| m5 / cyber / touch | 118.70 / 120.20 / 121.18 | 120.68 / 119.15 | 244.55 / 245.65 |
| m5 / cyber / unwired | 140.76 / 143.26 / 144.80 | 144.20 / 142.53 | 248.56 / 248.96 |
| m5 / cyber / wired | 252.15 / 253.51 / 256.56 | 252.64 / 255.90 | 249.81 / 250.74 |
| m5 / tiel / touch | 112.08 / 112.99 / 113.96 | 112.86 / 113.13 | 214.60 / 215.42 |
| m5 / tiel / unwired | 141.43 / 143.74 / 145.07 | 144.78 / 142.70 | 217.82 / 218.09 |
| m5 / tiel / wired | 240.45 / 246.78 / 247.57 | 244.50 / 246.88 | 218.68 / 219.49 |

## Limits and reproduction

This screen covers one short prompt per pack and 3 s idle only. It does not
qualify long contexts, 30/120 s idle, HTTP/tool-roundtrip p95, concurrent loads,
pressure, thermal endurance, cancellation or server shutdown. Host background
processes were left intact. Fresh whole-system pressure/thermal snapshots were
not collected for these runs; do not reuse the earlier campaign's snapshots
as measurements of this screen. Underlying driver/residency causality remains
unproven. The extra GPU work is disclosed and is outside request timing.

Run `python verify.py` here to check the raw trials and reproduce medians.
The adjacent harness is self-contained apart from the exact AX/MLX environment.
Extract the relevant Python workload into `cases.json`:

```python
import json
from pathlib import Path
raw = json.loads(Path("trials.json").read_text())
Path("cases.json").write_text(json.dumps([
    next(c for c in raw["workloads"]["tiel"] if c["case"] == "python-lru")
]))
```

Run each arm in a fresh process on supported campaign hardware, with the same
model and measured extension on `PYTHONPATH`. Clear unrelated AX overrides.
For the wired control:

```sh
AX_MLX_WIRED_LIMIT_SCALE=0.9 python bench_idle_screen.py \
  --engine ax --model /path/to/model --cases cases.json \
  --stream-experts off --warmups 2 --reps 3 --cooldown 3 \
  --output wired.json
```

Repeat with `--idle-touch` for touch, and with wired scale `0` and no touch flag
for unwired. Run wired/touch/unwired then unwired/touch/wired. Repeat using the
Cyber workload and exact Cyber pack. Never label shorter idle as an engine
speedup or compare these controls to a differently warmed peer server.
