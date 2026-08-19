# Correction: the 2-bit run did not activate DeepSeek V4 MTP

These artifacts are valid raw outputs from an M2 Ultra 192 GB run of
`AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP` (122.3 GB,
loaded with `AX_ENGINE_2BIT_EXPERIMENTAL=1`), but the original interpretation
of them as a DeepSeek V4 nextn validation was incorrect.

The route telemetry is conclusive:

| config | decode tok/s | steps/199 tok | MTP available / drafts | decode route |
| --- | --- | --- | --- | --- |
| direct (`AX_NO_SPEC`) ×2 | 16.82 / 16.80 | 199 | `0 / 0` | 199 direct-pipeline steps |
| originally labelled “MTP” ×2 | 31.99 / 31.90 | 65 | `0 / 0` | 24 n-gram + 40 direct-pipeline steps |

- Both accelerated artifacts report `ax_mtp_available: 0`,
  `ax_mlx_mtp_model_policy: 0`, `ax_mtp_draft_tokens: 0`, and
  `ax_mtp_decode_steps: 0`. `ax_mtp_requested: 1` records request intent; it
  does not prove that a model drafter attached or ran.
- The measured **1.90×** gain is an n-gram-acceleration result relative to the
  all-speculation-off baseline. It is not a DeepSeek V4 nextn MTP claim.
- All four 199-token streams are identical (`7442f1d94f9c5bb3…`), which proves
  determinism for the routes that actually ran, not MTP-on ≡ MTP-off parity.
- The run does validate that this experimental 2-bit pack loads and completes
  direct generation within 192 GB. DeepSeek V4 MTP remains unvalidated until a
  rerun records an attached candidate policy, `ax_mtp_available: 1`, active
  model-draft counters, and an n-gram-isolated comparison.
- The `performance.mtp` block correctly reported the inactive model drafter;
  it was not a counter-mapping defect. The probe should also invoke `env`
  directly because wrapping this runtime with `/usr/bin/time` exited by signal
  before producing usable evidence.
