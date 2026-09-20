# Tiel: matched AX Engine and MTPLX native API comparison

This refresh measures the current AX runtime against unmodified MTPLX
**2.11.3** at `7c2205ae4f3b91d7d3852b6ac7510174b4b5d0ae`, independently on:

- MacBook Pro, Apple **M5 Max, 128 GiB**.
- Mac mini, Apple **M4 Pro, 64 GiB**.
- Mac Studio, Apple **M2 Ultra, 192 GiB**.
- Mac Studio, Apple **M3 Ultra, 512 GiB**.

AX base: `574d2d1919d079e7a19c887e5e664888c4ebc666`, plus the Python expert-mode
selection fix committed as `5e07befe` and retained in the source checksums. Numerical inference and default
Auto admission rules are unchanged. All four hosts use the same
Rust **1.97.1**, optimized **release-pyext** extension, SHA-256
`4b4d26a19cbdd9e475f777e491c28869a421a897a4928fb2183583230de2bcb6`.
Both engines use **MLX 0.32.2**, with identical `libmlx.dylib` bytes:
`d24c7a9b9d55a76bfd3bbcd1d042251a185cbadcb6340c3244a6ffa3dcb7c7e8`.
The M3 Ultra was added by explicit user request after the initial campaign.

A subsequent test-only lint cleanup rebuilt the container as
`4e734aac1772697bf7787fa776924f4945721858d67071b130a71adb87e8bf46`.
Every file-backed Mach-O section is byte-identical to the measured extension;
both build records, source snapshots and section hashes are retained. The
timings belong to the measured `4b4d26...` artifact, not a separately timed rebuild.

Exact model snapshots:

| Pack | Revision |
| --- | --- |
| `AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` | `5ab39b24bfd7f65203be9b7823b1840486f58b6d` |
| `AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` | `fe05e871ec69ad9ae8eac01fd285555514ac7daf` |

See [results and per-host limitations](RESULTS.md). The main matrix contains
540 measured requests and 360 warmups; a separate M4 admission diagnostic
adds six measured 16-token requests and six warmups.

## Comparison contract

Every engine receives the exact retained input token arrays. Model loading,
tokenization, HTTP and network latency are excluded. Each engine loads the
model once per process and retains it across every warmup and measured request.
Cold KV means no prompt-state reuse, not reloading model weights.
The three workloads are:

| Workload | Tiel prompt tokens | Cyber prompt tokens | Output tokens |
| --- | ---: | ---: | ---: |
| `random128` (synthetic diagnostic) | 128 | 128 | 128 |
| `python-lru` | 194 | 219 | 256 |
| `rust-jsonl` | 1039 | 1064 | 256 |

- Greedy target and draft sampling, seed zero, ignore EOS, fixed output count.
- MTP requested, maximum draft depth three, cold request KV, no prefix reuse.
  AX uses its explicit throughput-MTP path with conservative depth disabled;
  this is **not** an out-of-box server-default comparison or MTP certification.
- Both engines keep the complete model resident. On M4, AX explicitly sets
  `mlx_stream_experts="off"` (also exercised through the environment);
  the same explicit choice is used on M5, M2 and M3, where Auto already selects resident. AX's normal
  Auto policy pages this pack on 64 GiB because its full-resident estimate
  plus the 48 GiB headroom reserve exceeds capacity. An initial M4 Auto
  warmup was interrupted after stack sampling showed repeated expert-layer
  loads. That diagnostic is excluded from throughput results. The matched
  resident comparison does not represent AX's default M4 paging behavior.
- MTPLX sustained and turbo are both reported. Their profile kernels remain
  unchanged. Context copy, repetition stopping and loop guard are disabled
  for the fixed-output contract. Its profile Q4 draft head is installed, with
  post-norm hidden state and persistent committed MTP history.
- Two fresh-process blocks per engine/model; the second reverses arm order.
  Each block has two warmups and three measured requests per workload.
  Tables pool the **six measured samples** using the arithmetic mean of the
  middle two values. Per-block medians and full min/max ranges expose order
  and background-load effects; accepted/drafted counters expose MTP efficiency. Warmups never enter summaries.
- Synchronize pending work, then idle three seconds before every request.
  One benchmark process at a time per host. Engines never contend with one
  another. Existing unrelated OS/background processes remain untouched.
- Native API entry to first committed callback = **TTFT**. Output count /
  entry-to-last-callback duration = **completion tok/s**, the primary metric.
  **Decode tok/s** excludes the first callback's tokens and elapsed time.
  API-return duration is retained separately. Internal prefill timers have
  different scopes and are not compared across engines.

The engines can generate different token sequences even with greedy sampling,
which changes speculative acceptance. This is a matched-workload comparison,
not a forced-identical-token trajectory, output-quality evaluation or a pure
kernel benchmark. Synthetic random input is not representative coding quality.
These peer results do not replace the project's required `mlx_lm.benchmark`
baseline for broad inference-stack performance claims.

## Wired-memory scope

Within the resident comparison, the normal AX arm preserves the wiring policy:
the audited M5 Max/128 GiB Tiel exports automatically release wired residency;
M4, M2 and M3 retain normal wiring.
M4, M2 and M3 additionally run an **AX unwired diagnostic** using
`AX_MLX_WIRED_LIMIT_SCALE=0`. Its results are separate from the normal-policy
AX-versus-MTPLX comparison. No automatic policy or model default changes here.
The MTPLX native load/generate path is used directly; its separate server
memory-cap setup is not invoked. Its server also implements a conditional GPU
keepalive between foreground requests; that mechanism is not exercised here.
Do not interpret these as server wiring, idle-energy, or server-default user
experience measurements.

The wired-memory API is not M5-specific. AX supports M2-or-newer Macs on
macOS 26+, subject to model compatibility and memory capacity. The policy's
M5/Tiel predicates delimit measured automatic tuning, not API availability.
This campaign does not establish an unwiring speedup across other M2/M3/M4/M5
variants, smaller-memory hosts, other models or long/high-pressure workloads.

## Reproduction

Export each model's exact arrays from `trials.json` into `cases.json`. Point
`PYTHONPATH` at the pinned AX extension or unmodified MTPLX checkout; use the
recorded MLX shared library and clear inherited `AX_`/`MTPLX_` experiment flags.
Run each arm alone, then repeat the arms in reverse order:

```bash
# On the 64 GiB M4 host, match MTPLX's full-resident execution explicitly:
export AX_STREAM_EXPERTS=off
python bench_native_peer.py --engine ax --model "$MODEL_DIR" \
  --cases cases.json --output ax.json --conservative 0 \
  --warmups 2 --reps 3 --cooldown 3 --stream-experts off
python bench_native_peer.py --engine mtplx --model "$MODEL_DIR" \
  --cases cases.json --output sustained.json --profile sustained \
  --warmups 2 --reps 3 --cooldown 3 --stream-experts off
python bench_native_peer.py --engine mtplx --model "$MODEL_DIR" \
  --cases cases.json --output turbo.json --profile turbo \
  --warmups 2 --reps 3 --cooldown 3 --stream-experts off
AX_MLX_WIRED_LIMIT_SCALE=0 python bench_native_peer.py --engine ax \
  --model "$MODEL_DIR" --cases cases.json --output ax-unwired.json \
  --conservative 0 --warmups 2 --reps 3 --cooldown 3 --stream-experts off
python test_verify.py
python verify.py --check-source
```

Use the last arm only for the separately labeled M4/M2/M3 controls. Full manifest
hashes and unchanged MTP norms are checked before and after model execution.
The retained verifier recomputes callback-boundary metrics, sample counts,
input hashes, cold-cache and actual-MTP telemetry, and source identity.
