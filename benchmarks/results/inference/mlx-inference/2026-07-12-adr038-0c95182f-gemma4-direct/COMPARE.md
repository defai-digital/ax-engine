# Direct-mode rebench: DiffusionGemma + Gemma4

**Host:** Apple M5 Max 128GB · **method:** median over 5 measure (Gemma 2 warm + 5; Diffusion 1 warm + 5), 15s cooldown, AX direct / no n-gram.

| run | commit | branch |
|---|---|---|
| historical baselines | various Jul 5–10 | published/prior refresh |
| today main | `f2de7866` | `main` |
| today ADR-038 | `0c95182f` | `adr-038-architecture-composition` |

**Note:** first suite accidentally ran on `main`; second suite rebuilt on ADR-038. Both are kept so we can separate *branch delta* from *vs historical*.

## 1. DiffusionGemma 26B-A4B 4-bit (first committed block)

| pt | metric | hist 07-08 | main f2de | **ADR 0c951** | ADR vs hist | ADR vs main |
|---:|---|---:|---:|---:|---:|---:|
| 128 | block decode | 158.9 | 69.6 | **69.6** | -56.2% | -0.1% |
| 128 | prefill | 1151.0 | 905.1 | **902.6** | -21.6% | -0.3% |
| 128 | first-block ms | 1722.8 | 3817.1 | **3821.8** | +121.8% | +0.1% |
| 512 | block decode | 109.6 | 50.8 | **50.9** | -53.6% | +0.2% |
| 512 | prefill | 2794.0 | 1469.1 | **1454.6** | -47.9% | -1.0% |
| 512 | first-block ms | 2520.1 | 5397.5 | **5386.0** | +113.7% | -0.2% |
| 2048 | block decode | 163.5 | 83.8 | **83.7** | -48.8% | -0.2% |
| 2048 | prefill | 3922.3 | 1552.2 | **1571.7** | -59.9% | +1.3% |
| 2048 | first-block ms | 2088.9 | 4373.4 | **4360.7** | +108.8% | -0.3% |

**Verdict:** ~**−50% to −56%** block decode vs Jul-08 historical. **ADR ≈ main** (no extra loss from ADR-038 vs today main). Regression predates or is shared with current main; not introduced solely by latest Gemma draft phases.

## 2. Gemma4 12B ffn4 direct

| pt | metric | hist 07-10 | main f2de | **ADR 0c951** | ADR vs hist | ADR vs main |
|---:|---|---:|---:|---:|---:|---:|
| 128 | decode | 69.5 | 69.6 | **69.6** | +0.1% | -0.0% |
| 128 | prefill | 394.7 | 407.7 | **403.9** | +2.3% | -0.9% |
| 128 | ttft ms | 324.3 | 314.0 | **316.9** | -2.3% | +0.9% |
| 512 | decode | 67.2 | 67.8 | **67.8** | +0.8% | +0.0% |
| 512 | prefill | 524.2 | 531.1 | **528.9** | +0.9% | -0.4% |
| 512 | ttft ms | 976.7 | 964.1 | **968.0** | -0.9% | +0.4% |
| 2048 | decode | 65.4 | 66.1 | **65.9** | +0.9% | -0.2% |
| 2048 | prefill | 556.9 | 559.2 | **558.6** | +0.3% | -0.1% |
| 2048 | ttft ms | 3677.7 | 3662.6 | **3666.0** | -0.3% | +0.1% |

**Verdict:** decode **flat** (±1% vs hist). ADR ≈ main. No material gain/loss on pure direct (expected: frozen KV / rope_dynamic are assistant-MTP only).

## 3. Gemma4 26B-A4B 4-bit direct

| pt | metric | hist 07-07 | main f2de | **ADR 0c951** | ADR vs hist | ADR vs main |
|---:|---|---:|---:|---:|---:|---:|
| 128 | decode | 135.5 | 135.1 | **135.1** | -0.3% | -0.1% |
| 128 | prefill | 1342.4 | 595.6 | **593.6** | -55.8% | -0.3% |
| 128 | ttft ms | 95.4 | 214.9 | **215.6** | +126.1% | +0.3% |
| 512 | decode | 132.3 | 131.9 | **131.8** | -0.4% | -0.0% |
| 512 | prefill | 3031.6 | 1195.9 | **1188.1** | -60.8% | -0.7% |
| 512 | ttft ms | 168.9 | 428.1 | **430.9** | +155.2% | +0.7% |
| 2048 | decode | 127.7 | 126.7 | **127.0** | -0.6% | +0.2% |
| 2048 | prefill | 4613.2 | 1535.4 | **1531.0** | -66.8% | -0.3% |
| 2048 | ttft ms | 443.9 | 1333.9 | **1337.7** | +201.3% | +0.3% |

**Verdict:** decode **flat** vs hist and main. Prefill/TTFT **~−55% to −67% vs Jul-07** but **ADR ≈ main** → shared current-tree issue (or timing-contract/host state), not unique to dual-path fuse on ADR branch alone.

## Bottom line

| model | vs historical | ADR-038 vs today main | promote public claims? |
|---|---|---|---|
| DiffusionGemma | **major loss** (block + prefill) | ~0 | **No** — investigate regression |
| Gemma4 12B direct | flat / tiny + | ~0 | No change |
| Gemma4 26B direct decode | flat | ~0 | No change |
| Gemma4 26B prefill/TTFT | **major loss** | ~0 | **No** — investigate shared regression |

### Artifacts
- ADR-038: `diffusion-gemma-direct/2026-07-12-adr038-0c95182f/`, `mlx-inference/2026-07-12-adr038-0c95182f-gemma4-direct/`
- main control: `.../2026-07-12-adr038-post-phases*` (misnamed; is `f2de7866` main)

### Not measured
- Gemma4 **assistant-MTP** (where Phase D/E should matter): `scripts/bench_gemma4_assistant_mtp.py`
- mlx_lm peer A/B
- docs/README charts **not** updated