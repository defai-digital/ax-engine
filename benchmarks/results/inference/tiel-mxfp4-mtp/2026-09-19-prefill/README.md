# Tiel MTP prefill investigation on M5 Max

This campaign evaluates opt-in phase diagnostics and records rejected latency
experiments. It does not establish a new speedup. The predeclared target of
at least 10% lower TTFT on both packs was not met; runtime experiments were
reverted. Only disabled-by-default timing instrumentation remains.

[Measured tables and validation results](RESULTS.md) are recomputed from the
retained trials by the evidence verifier below.

Hardware: MacBook Pro, Apple M5 Max, 128 GiB. Both exact Tiel MXFP4 MTP packs
ran only on this host class. No M3 Mac Studio model execution is included.

## Contract

AX baseline and final diagnostic build use Rust 1.97.1 `release-pyext` and
MLX 0.32.2. MTPLX 2.11.3 at
`7c2205ae4f3b91d7d3852b6ac7510174b4b5d0ae` is unmodified. Both runtimes
resolve the same MLX shared library. Exact revisions and digests are retained
in the evidence. Build and model loading happen outside the timed interval.

The short matrix uses identical input token arrays, fixed 128/256 output
counts, greedy target/draft sampling, depth cap three, cold KV, disabled
prefix reuse, two warmups, five measured repetitions, and three seconds idle
between requests. Each model process runs alone. AX baseline/candidate order
is reversed on the second pack. Both MTPLX sustained and turbo are reported.
AX conservative-depth control is off. MTPLX context copy and loop guard are
off; post-norm hidden state, persistent committed history and Q4 draft head
match the prior peer contract.

- TTFT: native API entry to first committed token callback.
- Completion: fixed output count divided by entry-to-last-callback time.
- Decode: output tokens excluding the first callback batch divided by
  first-to-last-callback time.
- MLX active/cache/peak bytes are sampled outside that interval. They are
  allocator counters, not whole-process RSS or device-wide memory.

These boundaries exclude HTTP and tokenization. Different runtimes can emit
different output sequences, which affects speculative acceptance; this is
matched greedy workload benchmarking, not a forced identical decode trajectory
or a quality ranking. AX baseline versus instrumentation must retain identical
output IDs. The official `mlx_lm.benchmark` raw reference uses its own decode
timer and is reported separately, never as a peer completion ratio.

## Interpretation

The clock-only diagnostic locates most short-prompt latency in the existing
final evaluation/sampling boundary. Graph construction, retained-history
materialization and the cleanup call are comparatively small. Lazy evaluation
means this does not identify a particular kernel, allocation or driver cause.

With the original binary, removing the three-second idle interval greatly
reduces TTFT even though the reusable buffer cache remains small. That rejects
buffer-cache depletion as a sufficient explanation. It is a workload change,
not an engine optimization. Coarse GPU-frequency traces are suggestive of an
idle-to-active transition; they do not prove the specific cause.

Async useful-work submissions gave no material gain. Blocking layer/block
boundaries gave at most roughly 1-2% variation. Larger Metal submission buffer
caps regressed, and a thread-QoS probe gave no material gain. Short 32-output
screening rows are explicitly distinguished from the full peer matrix.

The [operator guide](../../../../../docs/mtp/tiel-prefill-diagnostics.md)
defines each field and its limits. Enabled logging can perturb execution and
is excluded from throughput acceptance. Long-input checks use synthetic
8192-token repeated random input and do not measure coding quality. Bounded
repeat checks do not qualify endurance or prove absence of every memory leak.

No MTP default, certification, or product-performance claim changes here.

## Evidence and reproduction

`trials.json` retains individual timings, callback emissions, output token
IDs, memory readings, relevant route counters, warmups, exact input arrays,
pack revisions, and raw-artifact digests. Unrelated runtime metadata and
private paths are omitted. Rejected screening binaries were not all retained
with digests, so those rows are hypothesis-screening evidence, not release
qualification. Baseline and final builds are digest-bound.

```bash
python3 benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-prefill/verify.py
python3 benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19-prefill/verify.py --check-source
```

Build the recorded AX base and final source in separate checkouts with the
pinned Rust toolchain and `maturin develop --profile release-pyext`. Use
Python 3.12 or later and the recorded MLX version. Export one model's first
three entries from `trials.json`'s `workloads` map to `cases.json`; the fourth
entry is the synthetic long-input holdout. Point `PYTHONPATH` at the intended
native extension directory for AX, or the pinned reference checkout for
MTPLX. Resolve both to the same MLX shared library. On M5 Max:

```bash
python bench_native_peer.py --engine ax --model "$MODEL_DIR" \
  --cases cases.json --output ax.json --conservative 0 \
  --warmups 2 --reps 5 --cooldown 3
python bench_native_peer.py --engine mtplx --model "$MODEL_DIR" \
  --cases cases.json --output mtplx-sustained.json --profile sustained \
  --warmups 2 --reps 5 --cooldown 3
python bench_native_peer.py --engine mtplx --model "$MODEL_DIR" \
  --cases cases.json --output mtplx-turbo.json --profile turbo \
  --warmups 2 --reps 5 --cooldown 3
```

Run from this artifact directory, start each arm in a fresh process with a
clean AX/MTPLX experiment environment, and keep other GPU work idle. The
harness sets the required native MTP, no-prefix-cache and greedy controls.
Do not enable `AX_MLX_PREFILL_TIME_DEBUG` for acceptance measurements. Repeat
with that flag only to collect phase logs. `AX_NO_SPEC=1` selects the direct
smoke control. The repeat control uses only the random128 case with two
warmups and twenty measured requests; the long control uses one warmup and
three measured requests. The 32-output screening probes use `--smoke`.

The official raw reference uses `scripts/bench_mlx_inference_stack.py` with
`--skip-ax-engine --no-build-ax-engine --prompt-tokens 128
--generation-tokens 128 --warmup-repetitions 2 --repetitions 5 --cooldown 3`
and the exact model directory/repository arguments. This invokes
`mlx_lm.benchmark`; its internal decode metric is separate from the callback
boundary above.
