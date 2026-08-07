# AX Native and Youssofal-Optimized MTP Packages

Qwen3.6 peer benchmarks use two MTP packaging families. They serve different
runtime contracts and must not be described as interchangeable weights.

| Package | Normal use | MTP / draft-head precision | Current peer scope |
| --- | --- | --- | --- |
| `ax-local/Qwen3.6-27B-MTP` | AX Engine; verified MTPLX contract and measured lightning-mlx compatibility | BF16 MTP tensors and BF16 matching LM head | Identical 27B 4-bit sidecar across all three engines |
| AX 35B-A3B prepared sidecar | AX Engine | BF16 MTP tensors and BF16 matching LM head | AX side of the 35B-A3B rows |
| `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` | MTPLX / lightning-mlx 35B 4-bit | INT4 MTP sidecar and 3-bit affine LM head | Peer side of the 35B-A3B 4-bit row |
| `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance` | MTPLX / lightning-mlx 35B 6-bit | INT4 MTP sidecar and 3-bit affine LM head | Peer side of the 35B-A3B 6-bit row |

## Current Compatibility

The older statement that MTPLX only works with Youssofal packages is no longer
true. The current `ax-local/Qwen3.6-27B-MTP` snapshot carries a verified
`mtplx_runtime.json` contract. MTPLX 2.1.0 recognizes that contract, passes its
tensor/layout checks, and achieves 97.7% measured draft acceptance in the
2026-08-07 campaign. lightning-mlx 0.6.10 runs the same sidecar at 96.6%
acceptance.

That makes the 27B 4-bit row the closest engine comparison in the matrix:
target weights, draft weights, prompt tokens, seed, sampler, repetitions, and
cooldowns are shared. Runtime implementations and timing scopes still differ.

The 35B-A3B rows are deliberately production-configuration comparisons, not
identical-weight comparisons. AX uses its BF16 sidecar; MTPLX and lightning-mlx
use Youssofal's optimized Speed or Balance package. Decode tok/s is useful
operational evidence there, but the engine effect cannot be separated from the
package/precision effect.

## Fresh Peer Result

Apple M5 Max 128 GB, 2026-08-07, seed 0, sampled decode, 2 warmups, 5 measured
repetitions, clean tracked sources, passing lane-boundary condition gates:

| Target | AX Engine 6.13.3 | MTPLX 2.1.0 | lightning-mlx 0.6.10 | Package boundary |
| --- | ---: | ---: | ---: | --- |
| Qwen3.6 27B 4-bit | 56.1 tok/s | **59.9 tok/s** | 57.3 tok/s | Same verified BF16 sidecar |
| Qwen3.6 35B-A3B 4-bit | 140.9 tok/s | **145.1 tok/s** | 124.2 tok/s | AX BF16 vs Youssofal Speed |
| Qwen3.6 35B-A3B 6-bit | 120.5 tok/s | **125.2 tok/s** | 102.0 tok/s | AX BF16 vs Youssofal Balance |

AX trails MTPLX by 2.9%–6.3% on all three comparable rows. AX trails
lightning-mlx by 2.0% on the identical-sidecar 27B row and leads it by
13.4%–18.2% on the production-configuration 35B rows. Across all three,
AX is 4.3% lower than MTPLX and 9.5% higher than lightning-mlx by geometric
mean.

These results supersede the old claims that AX led every peer row and that the
Youssofal package could not give MTPLX an MoE lead. They do not prove that one
MTP precision is higher quality: draft acceptance is a speculation-efficiency
metric, not an output-quality score.

## Which Package to Use

| Use case | Recommendation |
| --- | --- |
| Serve with AX Engine | Use the matching AX/AutomatosX prepared MTP package |
| Serve 27B 4-bit with MTPLX | Use the verified `ax-local/Qwen3.6-27B-MTP` contract |
| Serve 35B-A3B with MTPLX or lightning-mlx | Use the runtime's documented Youssofal Speed/Balance package |
| Compare engines at 27B 4-bit | Use the same verified BF16 sidecar across engines |
| Compare engines at 35B-A3B | First produce an identical-weight sidecar accepted by every engine; until then label rows production-configuration |
| Quality-sensitive evaluation | Run a separate output-quality suite; do not infer quality from accept rate or quantization labels alone |

## Evidence

- [Qwen3.6 MTP peer comparison](qwen36-peer-comparison.md)
- [Clean 2026-08-07 summary](../../benchmarks/results/mtp-qwen36-matrix/2026-08-07-peer-comparison-apples-to-apples-refresh/summary.json)
- [Output-work diagnostic](../../benchmarks/results/mtp-qwen36-matrix/2026-08-07-peer-comparison-apples-to-apples-refresh/bandwidth_diagnostic.json)
- [Supported Models: MTP Downloads](../SUPPORTED-MODELS.md#mtp-downloads)
