# Experimental Flash Next MLX GDN prefill — 2026-10-03

The dedicated `qwen4_exp` GDN branch can select MLX 0.32.3's published
`gated_delta_update` operation for long prefill. The experiment is default-off and requires a build of this source change;
it is not a published-release or default-promotion claim:

```bash
AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX=1 ax-engine serve /path/to/flash-next-pack
```

Use the pack admission options required by the installed release. This flag
is separate from `AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX` and never remaps Flash
Next to the shared Qwen 3.5 graph. It does not enable MTP or certify a model.
Restart the process when changing the flag; selection is process-cached.

## Dispatch and arithmetic

Only T=128..2048, Dk=Dv=128, and the MLX 0.32.3 admitted head pairs can
select the experiment. Flash Next's native Hk=16, Hv=48 geometry is covered.
Unsupported shapes and graph-construction errors retain the sequential AX
recurrence. Evaluation errors use existing runtime error handling.

The adapter consumes the dedicated trunk's prepared FP32 Q/K/V, decay and
beta without recomputing gates. It preserves L2 normalization with epsilon
1e-6, query scaling, stable softplus decay, beta rounded through projection
dtype, FP32 recurrent orientation `[B,Hv,Dv,Dk]`, convolution state, output
RMSNorm and sigmoid/silu gating, and projection batch policy.

Singleton decode, short MTP verification and replay, expert Auto paging and
lazy n-gram/expert loading retain their current routes. Long replay is subject
to the same guarded prefill selection as an ordinary long call. The MLX chunk
recurrence changes arithmetic order and clamps small decay; exact generated
sequence identity with the sequential AX graph is not promised. No RAM,
whole-model throughput, MTP-S/P/D, default or certification claim follows.

## Recorded results

The MacBook Pro M5 Max 128 GiB benchmark host ran all 30 synthetic cells:
B=1/2, T=128/131/512/1024/2048 and three projection dtypes, with Hk=16,
Hv=48 and Dk=Dv=128. All comparisons passed the predeclared 5e-5 absolute
core-output/state bounds. All 30 correctness-only cells also passed on the
M2 Ultra 192 GiB functional host, with the same source-bound binary and inputs.
Latencies include graph construction and synchronous
evaluation of output and state; gates and input tensors are evaluated first.
Each backend has three warmups and five samples per trial, three trials with
alternating backend order; entries are medians of trial medians.

This representative sweep uses B=1 and BF16 projection rounding with an FP32
recurrence. It compares the existing dedicated AX sequential graph, rather
than the shared Qwen fused kernel.

| Tokens | AX sequential core (ms) | MLX chunk core (ms) |
| ---: | ---: | ---: |
| 128 | 7.1241 | 0.2493 |
| 131 | 7.3093 | 0.2340 |
| 512 | 28.3516 | 0.4616 |
| 1024 | 56.0558 | 0.8142 |
| 2048 | 120.0046 | 1.2884 |

The same 2048-token cell's maximum trial MLX allocator peaks were
7,197,917,760 bytes for AX and 250,675,584 bytes for the candidate, including
inputs and warmups. This measures the unrolled recurrence graph's working
set; it is neither whole-model RAM demand nor serving RSS. No minimum-RAM
or whole-model speed claim follows from the kernel result.

On the Mac Studio M2 Ultra 192 GiB functional host, both real MXFP4 and 6-bit
layer-zero component comparisons passed. Output maximum absolute error was
0.0001220703125 for each; relative L2 error was 0.00018751 / 0.00015914.
The test-host packs were on an external USB solid-state volume; this is not
internal-NVMe SSD paging performance evidence.
Recurrent-state maximum absolute error was 5.59e-9 / 6.05e-9. Convolution state
and same-state singleton output/state remained exact. Hidden activations were
synthetic, so this evidence does not establish real-prompt model quality.
The native loader used explicit expert paging for this controlled component
check; the product's Auto policy is unchanged.

[Raw measurements and provenance](../../benchmarks/results/flash-next-mlx-gdn/2026-10-03/)
retain all cells, samples, source/input hashes, component weight fingerprints,
runtime/executable identity and software validation results. Benchmark numbers
are signal only and do not qualify the Ultra-class SKU.

## Evidence and reproduction

The probe compares the actual dedicated sequential recurrence against the
candidate operation using prepared Flash Next normalization and gate rounding.
It supports nonzero initial state, distinct batches, partial 131-token chunks,
and FP32/FP16/BF16 projection rounding with FP32 core outputs and state.
Numerical bounds are fixed at 5e-5 absolute for synthetic core output and state.
A failed numerical comparison retains raw JSON and returns nonzero.

```bash
bash scripts/cargo-pinned.sh build -p ax-engine-microbench \
  --profile release-pyext --bin mlx-gated-delta-prefill-probe
AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX=0 \
AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX=0 \
  target/release-pyext/mlx-gated-delta-prefill-probe /tmp/flash-next-gdn.json 5 --flash-next
```

Timing belongs on the designated benchmark host. Add `--correctness-only`
for a functional comparison with no warmups or latency measurements.
The report binds source and seeded input hashes, runtime version, hardware
and raw latency/allocator samples. Kernel measurements are not serving RSS.

The ignored `campaign_mlx_prefill_real_gdn_component` test loads the dedicated
native pack loader and compares layer-zero GDN using real MXFP4 or 6-bit weights
with deterministic synthetic hidden activations. It writes component evidence
before asserting fixed development bounds: output max-absolute <=0.0625,
state max-absolute <=0.002, output/state relative L2 <=0.002, exact convolution
state and exact same-state singleton output/recurrent state. These bounds are
not model quality gates. It binds loaded component values and config/index
hashes; they are not a whole-pack payload identity certificate.

```bash
AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1 \
AX_FLASH_NEXT_PACK_DIR=/path/to/flash-next-pack \
AX_FLASH_NEXT_GDN_REPORT=/tmp/flash-next-component.json \
  bash scripts/cargo-pinned.sh test -p ax-engine-mlx --lib \
  campaign_mlx_prefill_real_gdn_component -- --ignored --nocapture
```

The MXFP4 native development trunk also completed a 131-token prompt followed
by eight greedy decode steps under the candidate flag and unchanged Auto policy.
All logits checked by the smoke remained finite and load read no full n-gram
table payload. This is a functional smoke, not a quality or throughput gate.
The 6-bit native trunk completed the same 131-token prompt and one decode step,
with 36 confirmed MLX GDN prefill dispatches. Its external-USB expert paging
dominated the run: incidental prefill time was about 642 seconds. Sampling
located the wait in expert payload copies, so the synthetic core improvement
must not be interpreted as a whole-model or SSD paging speedup.
The bounded native smoke can select 1..8 decode steps with
`AX_FLASH_NEXT_SMOKE_DECODE_TOKENS` (default 8), and records the number of MLX
prefill dispatches. This diagnostic option does not affect serving.

Full real-prompt logits/state, long-context quality, serving lifecycle and
memory/endurance evidence on Ultra-class 192 GiB+ remain separate open gates.
The primary 27B SKU and canonical product status are unchanged.
