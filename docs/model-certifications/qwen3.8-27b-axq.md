# Qwen 3.8 27B AXQ Certification

Status: **Candidate**

Primary optimization target: **AXQ 6-bit MTP** (`qwen3.8-27b:axq`)

Compact sibling: **AXQ 4-bit MTP** (`qwen3.8-27b:axq-4bit`)

Last reviewed: **2026-09-14**

Primary optimization target. Checkpoint Tier 1. MTP Tier 2 pending. AX certification record: Candidate (gates open).

This is the promotion record for the production-size Qwen 3.8 27B pack. It is
the general-purpose default serve target. It is **not** MTP Tier 2 certified
and it is **not** a 72-hour endurance pass. Super-class Qwen 3.8 (2.4T) is a
different, experimental path and is out of this record.

## Pinned Checkpoints

| Selector | Repository | Revision |
| --- | --- | --- |
| `qwen3.8-27b:axq`, `qwen3.8-27b:axq-6bit`, `ax-qwen3.8-27b` | `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` | `3e290738e96972307c6aeb9934ab170ca0eae1c1` |
| `qwen3.8-27b:axq-4bit` | `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP` | `7e865596cb32bd41b29c7a25c5b66b9c3ea25e5e` |
| `qwen3.8-27b:axq-8bit` | `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP` | `4037b7242a4de8deaf71247a685538591cad160a` |
| `qwen3.8-27b:axq-mxfp4` | `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP` | `b2c5354f779e430d0c1733143db848a72b71c16e` |

CLI aliases pin the exact checkpoint revisions above. Moving a Hub branch
cannot silently change what the selector loads.

## Axes (do not collapse)

| Axis | This pack |
| --- | --- |
| Product focus | Primary optimization target (unique general-purpose default) |
| Hub checkpoint | Tier 1 for 6-bit / 8-bit / MXFP4; 4-bit remains a compact candidate |
| AX certification record | Candidate — gates open |
| MTP | Sidecar present; Tier 2 performance certification pending |
| Support tier | Family follows the Qwen 3.x Certified graph path; this *checkpoint* is not `release_ready` |

## Current Evidence

Landed, labeled:

- Default serve alias and revision pin in the CLI.
- 2026-08-30 direct + MTP refresh on Apple M5 Max, 128 GB, from the v7.2.0
  binary at commit `3cea9def`. The host recorded tracked runtime changes;
  treat those numbers as refresh evidence pending a clean-build rerun.
  Artifacts:
  [`mlx-lm` reference](../../benchmarks/results/inference/mlx-lm-reference/2026-08-30-qwen38-27b-axq-6bit-m5-readme-refresh/),
  [AX direct](../../benchmarks/results/inference/ax-direct/2026-08-30-v7.2.0-qwen38-27b-axq-6bit-m5-readme-refresh/),
  [AX MTP](../../benchmarks/results/speculative/mtp-6bit/2026-08-30-v7.2.0-qwen38-27b-axq-6bit-m5-readme-refresh/).
- 2026-08-31 AXQ MTP peer campaign row for this pack (Apple M5 Max, 128 GB):
  [campaign](../../benchmarks/results/mtp-axq-peer/2026-08-31-df-macbookpro-m5/).

Not claimed:

- MTP Tier 2 / `release_ready`
- 8-hour or 72-hour endurance (the published 8.87h soak is
  [Qwen 3.6 27B AXQ](qwen3.6-27b-axq-6bit-8h-endurance-2026-08-08.md))
- Multi-model `load_mode=add`
- P0 multimodal quality for this pack
- Long-context decode-at-depth
- Clean-worktree replacement of the 2026-08-30 refresh

## Qualification

Operator procedure: [Testing](../TESTING.md) and
`python3 scripts/qualify_qwen38_27b.py --dry-run`. Live 27B runs belong on a
Mac mini M5 64 GB host with a clean checkout of the engine under test.

## Related

- [Qwen 3.6 27B AXQ certification](qwen3.6-27b-axq.md) — secondary, evidence-rich candidate
- [Supported Models](../SUPPORTED-MODELS.md)
- [Model Support Policy](../MODEL-SUPPORT-POLICY.md)
