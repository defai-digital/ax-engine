# Qwen 3.8 Flash Next

Status: **Incubating** (dedicated graph implemented; public qualification open)

Best-experience SKU: **Mac Studio M5 Ultra, 256 GB**

Last reviewed: **2026-09-15**

Qwen 3.8 Flash Next is a second product SKU, distinct from Qwen 3.8 27B
and Super-class Qwen 3.8 (2.4T). Its HF identity is `model_type=qwen4_exp`:
125B-A6B hybrid Gated-DeltaNet / sparse-attention MoE plus a 51B n-gram table.
MLX remains the tensor and quantized matrix multiplication engine.

## Implemented development path

- Dedicated GDN, QSA, gated residual streams, MoE, PLE and final mixer.
- Bounded disk n-gram row reads; table payloads are excluded from weight load.
- Transactional request state, cache serialization, prefix restore and verified
  speculative replay across the recurrent, attention and n-gram state.
- Affine expert paging with explicit resident/streaming admission.
- A separately attached, default-off MTP candidate with primary verification.
  A sidecar alone does not establish an attached or qualified draft head.

Tiny F32/BF16 oracle checks and bounded real 4-bit development tests passed.
Development evidence on an M2 Ultra 192 GB includes native HTTP/SSE, request
isolation, cancellation recovery, prefix reuse, long prompts, resident/paged
control tokens, and explicit MTP runner/HTTP checks. This is not M5 Ultra
certification, a trained MTP-head oracle, or a published throughput result.
Quantized prefill comparisons must use the same chunk schedule.

Additional 2-bit and 6-bit controls each match all four generated tokens, full
F32 logits and serialized request state exactly across resident, forced paging
and Auto modes. On the M2, Auto selects resident for the 2-bit pack and paging
for the 6-bit pack. These are bounded execution controls, not a model-quality
comparison between quantization formats. Both packs also pass native completion
and SSE repeat against their own resident-control tokens, with MTP disabled.
Raw numerical and API controls are in the
[affine format evidence](../../benchmarks/results/flash-next-affine-formats-m2-20260915.json).

## Public admission and remaining gates

Conversion preserves the dedicated metadata and validates tensor geometry.
Default auto-generation/load/serve reject `runtime_status.ready=false` and the
legacy blocker `qwen4_exp_native_trunk_not_implemented`. That compatibility
identifier predates the development graph; the remaining gate is public artifact
qualification. An explicit experimental opt-in admits the audited affine
2/4/6-bit expert packs through all ordinary tensor, file and geometry checks.
Expert bit/group pairs must be uniform: 2-bit/group32, 4-bit/group64 or
6-bit/group64. Protected projection layouts are checked against that pack's
expert format. This does not rewrite readiness or imply certification.

There is no download alias or generic Compatible route. Never remap this model
onto `qwen3_5` or Super-class 2.4T. Broader checkpoint/oracle, model quality,
long-context, MTP profitability and target-SKU qualification remain open.
Custom Metal experiments require numerical and measured controls before promotion.

The default local pack remains Qwen 3.8 27B AXQ on Mac mini M5 64 GB:
[Qwen 3.8 27B AXQ certification](qwen3.8-27b-axq.md).

## Operator contract

```bash
python3 scripts/qualify_qwen38_flash_next.py --dry-run
```

The script reports the admission contract without loading weights. A live
`--model-dir` qualification run remains closed pending public qualification.

For development with the audited AXQuant 1.9.0 affine Flash Next exports, the
native server can generate its metadata manifest and load with explicit opt-in:

```bash
AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1 ax-engine-server --mlx \
  --mlx-model-artifacts-dir /path/to/flash-next-4bit \
  --host 127.0.0.1 --port 31418
```

The loader rechecks the audited exporter/source identity and convolution layout.
The 2-bit export additionally requires `AX_ENGINE_2BIT_EXPERIMENTAL=1`; the
family opt-in does not bypass this existing quantization gate. Experimental
auto-conversion reports the underlying validation error when a gate or tensor
check fails, and does not leave an invalid generated manifest behind.

Unsupported bit/group pairs, mixed expert layouts, MXFP4, missing files and
additional blockers remain rejected. Removing a required opt-in rejects the
manifest again. Expert residency uses the existing Auto/On/Off policy; forced whole-layer paging has substantial
transfer cost. MTP remains a separate opt-in candidate and is not certified.
See [Testing](../TESTING.md) and [Supported Models](../SUPPORTED-MODELS.md).

## Selected expert reads (experimental)

With expert streaming active, `AX_MLX_FLASH_NEXT_SELECTED_EXPERTS=1` enables
bounded reads of only the routed experts for singleton forwards, including the
singleton prefill completion. With this flag alone, multi-token and batch>1
forwards keep whole-layer paging. Auto/On/Off residency decisions are unchanged; a resident pack stays
resident even when this flag is set.

The path preserves MLX routing weights and top-k reduction order, and reads the
same compact expert IDs from affine weight/scales/biases triplets. It limits
selected payload to 256 MiB per layer call and evaluates the MoE output before
releasing its owned arrays. No compiled expert closure captures these arrays.

Tiny F32/BF16 controls and bounded M2 2/4/6-bit controls match full logits and
serialized state against their whole-layer controls. This does not qualify
long-context quality, MTP combinations or the M5 Ultra SKU. Cold/warm filesystem
state materially affects paging latency.

The native `/metrics` endpoint exposes
`ax_engine_mlx_flash_next_selected_expert_gathers_total` and
`ax_engine_mlx_flash_next_selected_expert_payload_kib_total`. These count
successful selected gathers observed in engine steps. Payload KiB excludes
headers, whole-layer traffic and failed gathers; it is not physical disk I/O
or total memory. The counters establish actual route use, not just flag state.

Native completion and SSE repeat also pass for all three affine packs with
MTP disabled. Each repeat increases actual selected-gather and payload counters;
output tokens match the same-pack control. The
[selected-expert development evidence](../../benchmarks/results/flash-next-selected-experts-m2-20260915.json)
retains numerical fingerprints, API responses, source/binary hashes and all
repeated process times. Whole-layer prefill remains costly; these controls do
not establish overall serving performance.

### Bounded selected prefill

An additional default-off flag, `AX_MLX_FLASH_NEXT_SELECTED_PREFILL=1`, enables
selected expert unions for batch=1 multi-token Shared forwards. It requires
`AX_MLX_FLASH_NEXT_SELECTED_EXPERTS=1`. The original prefill token shape, routing
slots and MLX gather-QMM arithmetic are retained. Router indices are made
contiguous before host access, including strided multi-token top-k views.

The union must fit the existing 256 MiB affine payload cap across all projections.
A capacity miss falls back to whole-layer paging before selected payload I/O;
invalid metadata, invalid IDs and I/O failures remain errors. This cap excludes
allocator padding and scratch and is not a total-memory bound. Multi-token
RowExact verification and batch>1 keep the existing whole-layer path. Singleton
forwards retain their strict selected payload cap. The prefill flag alone has
no effect, and resident packs retain resident execution.

Tiny F32/BF16 controls for all three affine formats cover exact Shared outputs,
overlapping token selections, capacity fallback, and full hybrid-state recovery
after selected prefill I/O failure. Same-binary M2 controls for all three real
packs match complete prefill and generated logits/state across whole-layer,
singleton-selected and prefill-selected modes. Native completion/SSE also passes
with actual additional prefill gather counts. A 258-token control exercises
capacity fallback with exact logits/state. These bounded controls do not qualify
long-context quality, MTP combinations, sustained throughput or the target SKU.
See the [selected-prefill development evidence](../../benchmarks/results/flash-next-selected-prefill-m2-20260915.json).

### Independent full-checkpoint comparison remains open

An unmodified pinned MLX-VLM reference loaded the same 4-bit checkpoint with
MLX 0.32.2 and the same n-1/singleton schedule. Its four generated tokens matched
AX, but complete logits did not: maximum absolute error was 0.9609375 during
prefill and 2.3427734375 at the first decode step. The highest-scoring token at
the first prefix position also differed. This is retained as an unsuccessful
numerical comparison, not a quality pass.

Pass-through first-layer captures reproduced each implementation's original
logits; AX state fingerprints also stayed exact. Embedding outputs matched.
The first prefill HC difference affected one value, and GDN introduced broader
differences, including during decode with identical HC input. Reference GDN
normalizes Q/K in BF16, while the official Transformers fallback and AX use
FP32 normalization. This identifies a concrete numerical difference, but does
not explain or accept the complete final-logit error. Independent full-checkpoint
correctness remains a gate. See the
[independent comparison evidence](../../benchmarks/results/flash-next-independent-logits-m2-20260915.json).

A subsequent control feeds captured real first-layer Q/K/V, decay, beta and
state to the pinned official Transformers recurrent/chunked functions. All 32
component comparisons pass the previously fixed F32/BF16 oracle tolerances.
Independently carried official recurrent state differs by at most 4.77e-7;
norm/gate output differs by at most 1.53e-5. Capturing these tensors preserves
AX's complete logits and state fingerprints. This supports the tested GDN
arithmetic; it does not qualify the full model or resolve the MLX-VLM mismatch.
See the [official GDN evidence](../../benchmarks/results/flash-next-official-gdn-m2-20260915.json).

### Selected prefill with MTP

Bounded M2 controls now exercise both selected flags with MTP on all three
audited affine packs. Primary and draft state match the direct/full-head
controls immediately after prefill and after each committed step. The tests
separately count selected payload read by MTP steps, retain zero cached whole
expert layers, and cover verifier acceptance, rejection, budget and EOS
boundaries. Synthetic forced-acceptance tests also cover session and runner
terminal behavior.

The same production executable passes completion and SSE with MTP disabled and
required for each pack: 12 requests, matching the corresponding direct output.
Actual draft and selected-read counters confirm execution; the certified
default-on metric remains zero. These short controls do not establish MTP
profitability, a trained-head oracle, long-context quality or full-checkpoint
independent logits agreement. See the
[selected MTP evidence](../../benchmarks/results/flash-next-selected-mtp-m2-20260915.json).

### GDN activation precision correction

A later same-input M2 diagnostic isolated BF16 SiLU and beta sigmoid rounding.
Flash Next now evaluates these activation intermediates in FP32, then rounds
to the original projection dtype before recurrence. Convolution and its cached
tail retain their original precision; other families retain their existing
activation path. A pinned official fixture reproduces both old failures and
passes after the correction, including three cache-boundary splits.

All 20 captured Q/K/V/beta comparisons are now byte-exact with the official
first-layer inputs, and all 32 recurrence/norm component comparisons pass.
The entire first-layer output passes the fixed BF16 tolerance on four of five
forwards; the fifth still fails with maximum error 0.001953125. That residual
is retained as an open numerical discrepancy.

With identical teacher-forced inputs, maximum decode error against the
unmodified full MLX-VLM reference falls from 2.3427734375 to 1.43359375. The
prefix argmax mismatch is resolved, but a decode argmax differs. This is not
full-model acceptance. An unchanged reference control reproduces its original
logits exactly; a hybrid graph using whole official GDN modules remains a
diagnostic, not a complete official model oracle.

The corrected path passes the three affine pack state/runner matrices and 12
native HTTP/SSE requests with MTP disabled/required. Workspace tests report
3,638 passed and 39 ignored; relevant MLX/server Clippy passes. Full workspace
Clippy retains existing core test errors. See the
[GDN precision evidence](../../benchmarks/results/flash-next-gdn-precision-m2-20260915.json).

### MoE activation precision correction

Flash Next expert/shared SiLU and shared-router sigmoid now follow the official
projection-dtype rounding boundary: FP32 activation, cast back, then the
separate low-precision multiplication. Two regressions fail the old arithmetic
and pass the correction. Before changing math, test-only observations preserve
all committed logits and state fingerprints exactly.

On identical real first-layer inputs, 30 corrected activation/product checks
are byte-exact with official Torch, and 30 input-identity checks pass. Router
score checks also pass, with two explicitly recorded equal-score top-k ties
that select different valid expert IDs. An official expert replay isolates
ordered low-precision accumulation differences up to 0.001953125. Routing,
reduction and QMM are unchanged by this correction.

Full numerical agreement remains open: maximum teacher-forced decode difference
against unchanged MLX-VLM increases to 2.453125. That reference's compiled
activation also differs from official Torch on the recorded inputs. This is
operator-alignment evidence, not a full-model accuracy improvement.

The three affine state/runner matrices and 12 completion/SSE requests pass.
Six additional 4-bit chat checks return Paris, 42 and 2, 3, 5 identically with
direct and required MTP. Workspace tests report 3,640 passed and 39 ignored;
relevant strict Clippy passes, with existing full-workspace core test errors
retained. See the
[MoE precision evidence](../../benchmarks/results/flash-next-moe-precision-m2-20260915.json).

### HC and PLE precision correction

Measured real inputs isolate additional rounding differences in HC/PLE
activations, the HC stream mean, and PLE dot accumulation/scalar division.
The correction retains each low-precision product, uses FP32 activation or
accumulation, and preserves the official cast boundaries. Three regressions
fail old arithmetic and pass the correction. All 115 actual HC/PLE boundary
comparisons are exact, including the complete PLE short convolution; 32 GDN
recurrence/norm controls also pass on the changed HC inputs. Test observations
preserve complete logits and state exactly against their plain controls.

Full-model agreement remains open: teacher-forced decode error against
unchanged MLX-VLM reaches 1.84375, with one argmax mismatch. Error is not
uniformly reduced at every step. These operator results do not establish
full-model quality, MTP profitability or M5 Ultra qualification.

All three affine state/runner matrices, 12 completion/SSE requests and six
direct/required-MTP chat checks pass. Workspace tests report 3,643 passed,
39 ignored and zero failed. Relevant strict Clippy passes; the existing core
test failures remain in full-workspace Clippy. See the
[HC/PLE precision evidence](../../benchmarks/results/flash-next-hc-ple-precision-m2-20260915.json).
