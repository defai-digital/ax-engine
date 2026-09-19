# Tiel idle-prefill residency policy on M5 Max

This campaign validates an automatic, narrowly scoped wired-memory policy against
AX base `f2e52cb7` and unmodified MTPLX 2.11.3 at
`7c2205ae4f3b91d7d3852b6ac7510174b4b5d0ae`. The [measured results](RESULTS.md) record the paired improvement and both peer
profiles. The verifier passes 256 measured requests plus 116 warmups.

Hardware: MacBook Pro, Apple M5 Max, 128 GiB. Both exact Tiel MXFP4 MTP packs
run only on this host class. No M3 Mac Studio model execution is included.

Exact packs:

- `AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` at
  `5ab39b24bfd7f65203be9b7823b1840486f58b6d`.
- `AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` at
  `fe05e871ec69ad9ae8eac01fd285555514ac7daf`.

## Change

For the audited Tiel/Cyber export metadata on Apple M5 Max with at least 128 GiB,
AX releases wired residency after loading weights. Active expert streaming and
numeric `AX_MLX_WIRED_LIMIT_SCALE` overrides prevent automatic application.
Unknown metadata or hardware retain the previous policy. Buffer-cache and
allocation limits, numerical operations, MTP routing and certification are
unchanged. No periodic GPU activity or shortened request idle is introduced.

Metadata hashes identify the tested export configuration, not authenticity of
every weight byte. Re-exported packs fall back to previous wiring until measured
and reviewed. The existing `AX_MLX_WIRED_LIMIT_SCALE=0.9` restores prior wiring;
`0` explicitly disables wiring. Unwired allocations can be evicted under
competing memory pressure. These results cover one resident model in isolation,
not contention or endurance qualification. MLX wired limits have process scope;
mixed-model co-residency in one process is outside this campaign.

## Matched workload contract

- Both AX builds use Rust 1.97.1, `release-pyext`, MLX 0.32.2; both engines
  resolve the same MLX shared library. Model load and tokenization are excluded.
- Exact input arrays, greedy target/draft sampling, fixed 128/256 output counts,
  depth cap three, cold KV, no prefix reuse, two warmups and five measured
  repetitions per short cell; three seconds idle between requests.
- One model process at a time. AX arm order reverses on Cyber. Both MTPLX
  sustained and turbo are retained; context copy and loop guard are disabled.
  Post-norm hidden state, persistent committed history and Q4 draft head match
  the preceding peer campaign. AX conservative-depth control is disabled.
- TTFT: native API entry to first committed callback. Completion tok/s: fixed
  output count divided by entry-to-last-callback time. Decode tok/s excludes
  the first callback batch from both count and elapsed interval.
- AX prefill-forward tok/s uses input token count divided by the existing
  `ax_mlx_prefill_forward_wall_us` counter. It includes evaluation/submission
  waiting and is not GPU-only FLOP/s. Do not compare it with a differently
  bounded MTPLX or MLX-LM timer. Clock-debug logging is off for acceptance.

Across engines output sequences can differ, affecting speculative acceptance.
This is matched greedy workload benchmarking, not forced identical trajectories
or a quality ranking. Within AX, baseline and candidate must emit identical IDs.
The required primary `mlx_lm.benchmark` reference from the preceding campaign
remains [separately reported](../2026-09-19-prefill/RESULTS.md); it is not a peer
completion denominator and is not rerun as a new candidate comparison.

## Controls and attribution

The synthetic 8192-token holdout checks multiple chunks (one warmup, three
samples); it does not evaluate long-context coding quality. Direct smokes use
one warmup and one measured request per case. Zero-idle and explicit 0.9 override
controls use two warmups and three samples; twenty-repeat controls keep cold KV
in one process. Allocator active/cache/peak observations are not process RSS.

Separate Metal System Trace runs use 194 prompt tokens and 32 output tokens.
They locate the waiting removed by an early no-wire override; those instrumented
runs are excluded from the performance matrix. The compiled policy unwires
later, after load, and therefore requires its own acceptance measurements.
The evidence supports residency policy as a causal contributor to pre-GPU
waiting; it does not establish undocumented macOS driver internals.

## Reproduction

Build base and candidate with the pinned Rust toolchain and
`maturin develop --profile release-pyext`. Use the recorded extension digests,
model revisions and shared MLX library. Export the first three entries for
one model from `trials.json`'s `workloads` to `cases.json`. Point `PYTHONPATH`
at the intended AX extension or pinned reference checkout. Start each arm in
a fresh process with experiment environment variables cleared:

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
python verify.py --check-source
```

Use the same harness for controls with the counts above. `AX_NO_SPEC=1` selects
direct controls; `AX_MLX_WIRED_LIMIT_SCALE=0.9` selects the operator override.
Keep those controls separate from the main performance matrix.
