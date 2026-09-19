# Qwen 3.8 27B v7.4.0 release verification

The pinned AXQ 6-bit pack passes the recorded installed-CLI and local HTTP
text-serving product-health protocol on Mac mini M4 Pro 64 GB. AX Engine
v7.4.0 is published on GitHub, PyPI and Homebrew. This record supplements the
[earlier qualification](../2026-09-18-qwen38-27b-product-default/) and preserves
its failed runs and acceptance-policy history.

## Identity

- Source: `a430eb9a82cfb77ffacf45c8a2332b4bb17681a9`, clean checkout.
- Model: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP`, revision
  `3e290738e96972307c6aeb9934ab170ca0eae1c1`; 22 files rehashed on the SKU.
- [Exact-source CI](https://github.com/defai-digital/ax-engine/actions/runs/35402097234)
  and [hosted candidate](https://github.com/defai-digital/ax-engine/actions/runs/35402889454)
  passed. Fresh minimal Rustup needed explicit Clippy/rustfmt installation;
  this release repairs both candidate and PyPI fallback workflow initialization.
- [Public release](https://github.com/defai-digital/ax-engine/releases/tag/v7.4.0).
- Wheel SHA-256: `666d8c41b45fb94366bbda6ba94a5000889ebb5371efbe34a8de217cff678fbc`.
  Public PyPI bytes exactly match the qualified hosted wheel.
- Signed archive SHA-256: `f01467bcd7ac6214263eb02f3bac9f5ddc508c48c0729ca2e7dc4e7c628a229d`.
  Developer ID signatures, Minisign, runtime layout and Apple notarization
  were verified before and after publication. Homebrew references this digest.
- [Release acceptance](release-acceptance.json), [immutable release manifest](released-manifest.json),
  [installed build identity](build-manifest-product-final-installed.json).

## Measured acceptance

| Scope | Result |
| --- | --- |
| Default, explicit direct, explicit MTP sampled QA | 32/32 hard and 7/7 required API probes each |
| Full 79-item QA bank, streaming and non-streaming | 158/158 hard per default/MTP route; zero incomplete; six soft keyword failures per route |
| Fault lifecycle per route | Three rounds, 15 outcomes; cancellation, observed backpressure, quiescence and recovery pass |
| Buffered-event peaks | Default 223; MTP 241, with one bounded-backlog overflow |
| Silent MTP fallback | Zero; default MTP counters remain zero |
| Fresh public wheel installation | Only AX Engine and pip; package-local MLX/JACCL; import and doctor pass |
| Public signed native CLI | Default alias, explicit MTP alias and local directory each reproduce the wheel's 64-token control; servers reaped |

The lifecycle request uses 128 fixed input token IDs, 64 generated tokens and
ignore-EOS for recovery identity. Full QA uses the unchanged 79-item bank,
1024-token cap and `ax.qa.complete_answers.v2`; answers must end naturally.
These controls do not establish broad model accuracy. All raw JSON responses,
soft failures, metric samples, grading and request parameters are retained.

Native inference from a prepared local directory was tested with only the
payload directory on PATH, without Python. Alias controls use an empty Python
3.12 environment without AX Engine or Python MLX. Alias/preparation helpers
require Python 3.12+; online downloads additionally require `huggingface-hub`.
See the [helper setup](../../../../docs/GETTING-STARTED.md#homebrew-model-helpers).

## Reproduce the evidence audit

From the repository root with its QA Python dependencies installed:

```bash
python3 benchmarks/results/qualification/2026-09-18-qwen38-27b-release/verify_sku.py
```

The verifier checks all published file hashes, source and artifact bindings,
route evidence, all 412 QA grades, full-bank coverage, lifecycle recovery and
public standalone output identity. `publication.json` records original and
published hashes. Redaction replaces local paths and private host identifiers;
model answers and measured values are unchanged. The signed release manifest
is retained byte-for-byte. The Homebrew blob receipt records the publication
snapshot; a later helper-documentation-only formula update may change that
blob while retaining the release version and archive digest.

For live reproduction, `collect_lifecycle.py` runs the installed wheel with
the frozen inputs; `verify_standalone_sku.py` exercises the public native
payload. Both take the prepared qualification root as their first argument.
That root must contain `product-final-source` at the release commit, the installed
`product-final-venv`, `model`, and a build manifest rebound to the installed
launcher and binaries. Copy `product-final-lifecycle/soak-prompt.json` to
`product-final-lifecycle-prompt.json` in the root before collecting a new run.
Use a fresh root: the collector refuses existing output/cache directories.
The standalone probe additionally expects the extracted `released-standalone`
payload and an empty Python 3.12 venv at `standalone-python`; it consumes the
preceding lifecycle output and cache. Model weights and generated caches are
intentionally excluded.

## Scope boundaries

This closes the recorded product-health and publication scope. MTP remains
explicit opt-in. MTP-S retains its separate verify/replay safety evidence;
MTP-P performance claims and MTP-D default promotion are not granted here.
Candidate / Tier 2 pending status is unchanged. No 8-hour/72-hour endurance,
broad accuracy, or acceleration multiplier is claimed.
