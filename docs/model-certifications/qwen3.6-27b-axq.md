# Qwen 3.6 27B AXQ Certification

Status: **Candidate**

Default selector candidate: **AXQ 6-bit**

Compact fallback candidate: **AXQ 4-bit**

Last reviewed: **2026-08-06**

This record is the promotion gate for the first AXQ flagship model. A successful
smoke test proves that a checkpoint can load and generate; it does not by itself
make the checkpoint certified or safe to install as an existing bare alias.

## Pinned Checkpoints

| Selector | Repository | Revision | Snapshot bytes |
| --- | --- | --- | ---: |
| `qwen3.6-27b:axq`, `qwen3.6-27b:axq-6bit` | `AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP` | `8c37715c7b5f5ebca00eda6f73be47116a3e4ebc` | 20,857,941,725 |
| `qwen3.6-27b:axq-4bit` | `AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP` | `6182ccbc41c7397ff90670f740c6d9eacfa4b09f` | 19,399,395,845 |

Both artifacts record source model revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. CLI aliases pin the exact
checkpoint revisions above; moving a Hub branch cannot silently change what a
certified selector loads.

## Default Decision

The unqualified `:axq` selector chooses 6-bit because a flagship default should
optimize for quality and stability first. The 4-bit build remains an explicit
compact option for machines or workloads that need lower memory use.

This does not yet change either established bare alias:

- `qwen3.6-27b` remains the mlx-community 4-bit serve target.
- `ax-qwen3.6-27b` remains the AutomatosX OptiQ target.

Changing a bare alias is a separate compatibility decision after certification.
If a future release does so, it must be called out as a user-visible migration
and retain explicit selectors for the former target.

## Current Evidence

The published AXQ plans use AXQuant artifact schema v2 / AXQuant 1.2.0, report
zero quantizer fallbacks, and carry architecture-prior allocation evidence.
That is useful construction evidence, but `evidence_kind=architecture_prior`
is not a measured checkpoint-quality result. Consequently both snapshots stay
`candidate`.

## M5 Smoke Record

On 2026-08-06 the pinned 6-bit snapshot was exercised on an Apple M5 Max with
128 GB unified memory, macOS 26.5.2, and MLX 0.32.0:

- exact offline alias resolution selected revision
  `8c37715c7b5f5ebca00eda6f73be47116a3e4ebc`;
- the downloader rejected a metadata-only partial snapshot, then downloaded
  all six safetensors and generated `model-manifest.json`;
- the native server bound 1,184 tensors and reported 16,920,033,760 model
  buffer bytes; observed server RSS after load was 19,253,184 KiB;
- a non-streaming request returned exactly `AXQ M5 OK`, and an SSE request
  returned exactly `STREAM OK` followed by `[DONE]`.

This proves load and basic generation compatibility on that host, not
certification. Doctor currently reports the model artifact `not_ready` because
the canonical AXQuant plan content digest
`3be9fa6aac78e303369ec0f1cc7d3682c15b82b3ec542af6e4f21b6b0c5dfa02`
does not match manifest `plan_sha256`
`b9f886ac3d0f964ff1409a37303fbe388062839064cd32379d6da413083ad5ce`.
The artifact must be regenerated or republished with internally consistent
lineage before promotion.

## Promotion Gates

All required rows must have reproducible artifacts tied to the checkpoint
revision, AX Engine commit, MLX version, macOS version, and hardware profile.

| Area | Required evidence |
| --- | --- |
| Artifact integrity | `ax-engine doctor` is ready; native manifest and all safetensors validate; AXQuant hashes, source lineage, MTP sidecar, and vision-tower provenance are complete; zero silent fallback |
| Quality | Frozen, versioned A/B corpus against source BF16 plus current OptiQ/uniform baselines; coding, tool calls, strict JSON, multilingual chat, long-context retrieval, and image inputs are covered; acceptance thresholds are declared before the run |
| Runtime correctness | Direct and MTP modes pass deterministic smoke/regression tests; streaming, cancellation, concurrency, prefix/KV reuse, and malformed-input fail-closed behavior are exercised |
| Memory and fit | Peak resident memory and usable context are measured on 32 GB, 64 GB, and 128 GB Apple Silicon; the advertised minimum configuration completes a representative request without swap thrash or OOM |
| Performance | TTFT, prefill, decode, end-to-end latency, MTP acceptance/speedup, and sustained serving behavior are captured with the repository benchmark contracts; regressions versus the selected baseline are explained |
| Release evidence | Raw reports and commands are checked in or durably published, checksums match the pinned snapshots, and the review records why 6-bit or 4-bit should be the default |

The 6-bit checkpoint is promoted only if it clears every quality and runtime
gate and its memory envelope fits the advertised default hardware. If it fails
only the minimum-memory gate, the product must either make 4-bit the default or
state a higher minimum explicitly; it must not silently fall back to a different
checkpoint.

## Verification Commands

Resolution and cache behavior:

```bash
ax-engine serve qwen3.6-27b:axq --dry-run --json
ax-engine serve qwen3.6-27b:axq --offline --dry-run --json
ax-engine download qwen3.6-27b:axq --local-only --json
```

Artifact and real-runtime checks:

```bash
ax-engine doctor --verbose \
  --mlx-model-artifacts-dir /path/to/pinned/axq-6bit/snapshot
ax-engine serve /path/to/pinned/axq-6bit/snapshot --port 31418
```

The real-runtime command must be followed by at least one non-streaming and one
streaming generation request, with the resulting logs and environment captured
as certification evidence.
