# Experimental MLX GDN prefill — 2026-10-03

AX Engine can opt shared Qwen hybrid prefill into MLX 0.32.3's published
`fast::gated_delta_update` operation. This experiment remains **default-off**.
The dependency upgrade alone did not select this new API.

## Operator selection

Set `AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX=1` before starting the process.
Selection is cached per process. Runtime dispatch requires T=128..2048,
Dk=Dv=128, and one of these (key heads, value heads) pairs:
(24,24), (32,32), (16,32), (16,48), (16,16), (16,64).
Unsupported shapes/dtypes and graph-construction failures retain the AX path.
Later evaluation failures use the existing runtime error handling.

Single-token decode, short MTP verification/checkpoint/replay, and the
separate Flash Next GDN module keep their existing execution. Expert Auto
paging and lazy n-gram/expert loading are unchanged. The option does not
promote MTP-S/P/D, model certificates, or product defaults.

The adapter uses AX's existing gate arithmetic. MLX 0.32.3 casts gate tensors
to the query dtype internally, so AX promotes Q/K/V to FP32 to preserve FP32
decay and recurrent state, then restores the activation output dtype. Beta
retains its original activation rounding. Upstream chunk arithmetic differs
from AX's sequential recurrence, including a small-decay clamp; generated
sequence identity is not promised. No upstream implementation was copied.

## Synthetic kernel measurements

The same Rust 1.97.1 `release-pyext` probe executable and published MLX 0.32.3
runtime assets ran on mini M4 Pro 64 GiB and M2 Ultra 192 GiB. Each host ran
60 cells: B=1/2, T=32/128/512/1024/2048, Hk=16 and Hv=32/48,
Dk=Dv=128, and FP32/FP16/BF16 activations. Seeded inputs and nonzero initial
states are bound by SHA-256. All 120 output/final-state comparisons pass the
recorded absolute error limits (5e-5 for FP32/FP16 outputs and state, 5e-4 for
BF16 outputs). These synthetic tolerances are not model quality gates.

Every latency includes graph construction and synchronous evaluation of
output and final state. Inputs are evaluated before timing. Each backend
gets three warmups and five samples in each of three trials. Backend order
alternates by trial; aggregates are medians of trial medians.

The following complete sequence sweep is B=1, Hk=16, Hv=48, BF16, D=128.
Times are milliseconds; negative changes mean lower latency.

| Tokens | M4 Pro AX | M4 Pro MLX | Change | M2 Ultra AX | M2 Ultra MLX | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.2644 | 0.2645 | +0.0% | 0.2901 | 0.3264 | +12.5% |
| 128 | 0.6983 | 0.4888 | -30.0% | 0.5449 | 0.4808 | -11.8% |
| 512 | 1.9366 | 1.6455 | -15.0% | 1.3790 | 1.2546 | -9.0% |
| 1024 | 3.6525 | 3.3321 | -8.8% | 2.6566 | 2.4220 | -8.8% |
| 2048 | 7.5465 | 6.5270 | -13.5% | 5.3199 | 4.7252 | -11.2% |

T=32 regressed on M2 Ultra, motivating the T>=128 runtime guard. The probe
still admits T=32 to retain the rejected short-call comparison. Benefits vary
by shape and host; no blanket improvement is claimed. Host load averages are
retained; the M2 workstation was not an isolated performance lab.

FP32 promotion can increase memory. For this sweep's T=2048 cell, maximum
trial MLX allocator peaks (including inputs and warmups) are roughly
138/199 MiB for AX/MLX on M4 Pro and 232/355 MiB on M2 Ultra. These are allocator
measurements, not serving RSS or whole-model peak memory.

[Raw cells, all latency samples and provenance](../../benchmarks/results/mlx-gdn-prefill/2026-10-03/)
include embedded source fingerprints, shared executable/runtime hashes,
input hashes, numerical errors, per-trial allocator peaks and optimization
environment values. A portable run outside a checkout reports a null Git
revision; embedded source and executable hashes still bind that run.

Reproduce with the admitted published wheel:

```bash
bash scripts/cargo-pinned.sh build -p ax-engine-microbench \
  --profile release-pyext --bin mlx-gated-delta-prefill-probe
AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX=0 \
  target/release-pyext/mlx-gated-delta-prefill-probe /tmp/gdn-prefill.json 5
```

The probe explicitly calls the candidate adapter while keeping the AX
baseline independent of the opt-in flag. It writes raw results before
returning a nonzero exit code on a numerical failure.

## Validation and remaining model gates

Binding tests check an independent host recurrence and reject malformed
shapes/dtypes. Adapter tests cover six head pairs, three activation dtypes,
nonzero states, distinct batches, strided slices, continuation, partial
chunks, extreme gates, and flag-on/flag-off dispatch. Short calls and decode
match the retained AX route exactly.

Software gate results are retained in the accompanying validation artifact.
The original worktree's script gate encountered pre-existing untracked wiki
references; the unmodified gate passed in a clean worktree containing this
candidate's files. Existing unrelated changes were preserved.

There is no real-weight Qwen 3.8 27B or Flash Next throughput/quality claim
here. Default promotion needs the exact primary 27B pack on mini M4 Pro 64 GB,
matched `scripts/bench_mlx_inference_stack.py` / `mlx_lm.benchmark` evidence,
long-context logits/state and memory checks, default n-gram behavior, and
server/SDK regression coverage. Flash Next needs its own dedicated-path
experiment on Ultra-class 192 GiB+; this flag does not reroute it.

See [runtime admission](mlx-0.32.3-admission-2026-10-03.md) and the
[upstream GDN API change](https://github.com/ml-explore/mlx/pull/4020).
