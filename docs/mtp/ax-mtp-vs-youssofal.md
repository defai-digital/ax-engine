# AX Engine Native MTP vs Youssofal MTPLX-Optimized Package

When benchmarking Qwen3.6 27B 4-bit with MTP speculative decoding, two MTP
artifact sources are commonly seen:

| Artifact | Publisher | MTP precision | Draft LM head |
| --- | --- | --- | --- |
| `ax-local/Qwen3.6-27B-MTP` | AX Engine (extracted from official Qwen repo) | BF16 (RMSNorm +1.0 delta correction) | BF16 (matching base) |
| `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` | Youssofal (community re-quantization) | INT4 prequantized sidecar | 3-bit affine, group_size=64 |

Both derive from the same official Qwen MTP tensors, but the quantization and
packaging differ. This page explains the trade-offs so you can choose the right
artifact for your use case.

## Why Two Artifacts Exist

MTPLX's `MTPContract` expects a specific fc concat order (`[embed; hidden]`)
and post-norm hidden states. The standard Qwen MTP training convention uses
`[hidden; embed]` and pre-norm states. The Youssofal bundle adapts the weights
to MTPLX's conventions — swapped fc halves and post-norm inputs — which is why
MTPLX only produces meaningful MTP accept rates with that bundle. Using
standard Qwen MTP shards with MTPLX yields ~2% accept rate (speculation
effectively disabled).

AX Engine uses its own loader that reads the standard Qwen layout directly,
so it works with the unmodified BF16 sidecar.

## Pros and Cons

### Youssofal MTPLX-Optimized-Speed

**Pros:**

1. **Out-of-the-box MTPLX compatibility.** Ships a pre-quantized
   `mtp/weights.safetensors` that MTPLX can load and run without manual
   sidecar preparation.
2. **Higher 27B 4-bit decode throughput in the peer comparison.** MTPLX
   with this bundle reaches 64.3 tok/s vs AX Engine's 61.0 tok/s on the
   flappy suite (see [peer comparison](qwen36-peer-comparison.md)).
3. **Only MTP bundle that works with MTPLX.** Standard Qwen shards produce
   near-zero accept rates in MTPLX without the fc/norm adaptation.
4. **MoE variants available.** 35B-A3B Speed and Balance profiles exist.

**Cons:**

1. **Not apples-to-apples with AX Engine.** Different MTP precision (INT4
   vs BF16) and draft LM head precision (3-bit vs BF16) means the
   throughput gap conflates engine differences with artifact differences.
2. **Excluded from fair benchmarks.** The repo's fair benchmark harness
   (`scripts/bench_qwen36_mtp_fair.py`) and sidecar provenance checker
   (`scripts/check_mtp_sidecar_provenance.py`) explicitly reject Youssofal
   bundles from the fair base track.
3. **Precision loss.** INT4 quantization and 3-bit draft LM head are
   lossy transforms. Accept rate impact is negligible (AX 100.0% vs MTPLX
   99.99% on 27B 4-bit), but BF16 remains the cleaner reference for
   quality-sensitive work.
4. **Locks you into MTPLX conventions.** The swapped fc halves and
   post-norm inputs are MTPLX-specific. Running the same weights through
   AX Engine or lightning-mlx requires a different artifact.
5. **Third-party provenance.** Community re-quantization, not an official
   Qwen release. Supply-chain trust is lower than first-party artifacts.
6. **AX significantly faster on MoE.** On 35B-A3B 4-bit, AX leads with
   169.9 tok/s vs MTPLX 138.1 tok/s (~23% gap). The Youssofal bundle
   does not give MTPLX an edge on MoE models.

### AX Engine Native MTP (ax-local/Qwen3.6-27B-MTP)

**Pros:**

1. **BF16 precision.** MTP weights extracted with RMSNorm +1.0 delta
   correction; draft LM head matches the base model at BF16. No
   quantization-induced precision loss.
2. **100% accept rate.** On 27B 4-bit, the BF16 sidecar achieves perfect
   MTP draft acceptance, maximizing speculation efficiency.
3. **Auditable and reproducible.** AX peer rows pass the
   output-degeneracy gate, use fixed seeds (seed 44), and carry full
   artifact provenance tracking.
4. **Fair benchmark baseline.** All fair benchmark scripts use the AX
   sidecar as the reference artifact. Cross-engine results are directly
   comparable.
5. **Strong overall performance.** 61.0 tok/s decode, 812.3 tok/s
   prefill, 396 ms TTFT on 27B 4-bit. On 35B-A3B, AX leads all peer
   engines at both 4-bit and 6-bit.
6. **First-party provenance.** Extracted directly from the official
   Qwen repository with documented extraction methodology.

**Cons:**

1. **27B 4-bit decode slightly below MTPLX.** 61.0 tok/s vs 64.3 tok/s
   (~5% gap). However, this comparison conflates engine and artifact
   differences — the gap cannot be attributed to either factor alone.
2. **Requires AX Engine loader.** The BF16 sidecar uses standard Qwen
   layout conventions. Only AX Engine's native loader reads it correctly;
   MTPLX needs the adapted Youssofal variant.

## Recommendation

| Use case | Recommended artifact |
| --- | --- |
| Running **AX Engine** as inference runtime | `ax-local/Qwen3.6-27B-MTP` (BF16 sidecar) |
| Running **MTPLX** as inference runtime | `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` |
| Cross-engine **fair benchmark** | `ax-local/Qwen3.6-27B-MTP` with AX Engine |
| **Quality-sensitive** work (precision matters) | `ax-local/Qwen3.6-27B-MTP` (BF16) |
| **MoE models** (35B-A3B) | `ax-local` variants — AX leads all peers |

For AX Engine users, the native BF16 sidecar is the recommended choice:
it preserves full MTP weight precision, achieves 100% draft accept rate,
and produces auditable benchmark results that are directly comparable
across engines. The MTPLX 5% throughput advantage on 27B 4-bit comes
from a different artifact, not a controlled engine-only measurement.

The only scenario where the Youssofal bundle is required is when you
specifically use **MTPLX** as your inference runtime — standard Qwen
MTP shards produce near-zero accept rates in MTPLX due to fc/norm
layout incompatibility.

## Related

- [Qwen3.6 MTP peer comparison](qwen36-peer-comparison.md) — full peer
  benchmark results with artifact provenance and fairness limitations
- [MTP draft gate throughput](draft-gate-throughput.md) — gate tuning
  and throughput analysis
- [Supported Models: MTP Downloads](../SUPPORTED-MODELS.md#mtp-downloads) —
  how to download AX Engine MTP packages
