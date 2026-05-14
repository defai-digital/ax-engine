# Gemma 4 E2B Direct Pipeline Split

This diagnostic artifact verifies production direct-pipeline timing counters for
Gemma 4 E2B 4-bit. It is intentionally a short 1-repetition run; use the split
shape, not the throughput, as the evidence.

## Artifact

- JSON: `benchmarks/results/mlx-inference/2026-05-14-gemma-e2b-direct-pipeline-split/gemma-4-e2b-it-4bit-direct.json`
- Prompt shape: 128 prompt tokens, 128 generation tokens
- AX policy: `direct_no_ngram_acceleration`
- Repetitions: 1 measured run after 1 AX warmup
- Recorded commit: `1ba410e7a6b70a09d8eddeb208795d342d6e3373`

The artifact was produced from a dirty worktree immediately before committing
the direct-pipeline split counters, so the recorded commit is the pre-commit
HEAD plus local telemetry patches.

## Direct Pipeline Timing

| Metric | Value |
|---|---:|
| Decode tok/s | 166.8 |
| Decode wall us | 758,343 |
| Direct pipeline wall us | 752,922 |
| Direct pipeline steps | 127 |
| Direct pipeline wall us / step | 5,928.5 |

| Direct pipeline bucket | Wall us | Share of direct pipeline |
|---|---:|---:|
| Next forward graph build | 55,308 | 7.3% |
| `argmax` + `async_eval(next)` submit | 696,650 | 92.5% |
| Pending token `eval` barrier | 689 | 0.1% |
| Pending token read | 36 | 0.0% |

## Reading

The direct pipeline is not spending meaningful time waiting for the pending
token to materialize. The measured cost is concentrated in the
`argmax` + `async_eval(next)` bucket, which likely means the async submit call is
back-pressuring on already-queued GPU work rather than returning as a cheap CPU
enqueue.

This gives the next implementation target: inspect whether direct mode is
submitting too much work per step, missing an opportunity to keep logits/token
selection cheaper, or losing overlap before `async_eval(next)`.
