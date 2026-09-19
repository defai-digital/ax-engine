# Optional Tiel MTP depth control

The Tiel and Cyber-Tiel MXFP4 packs support explicitly requested native MTP.
Use a build containing the [norm metadata correction](tiel-norm-compatibility.md)
for the pinned initial exports. Their already converted sidecar norms must not
receive the HF-to-MLX shift twice.

MTP throughput depends on how many proposals the verifier accepts. The existing
three-token throughput controller returns to depth three after any accepted
proposal and backs off only to two after a complete miss. This works well on
high-acceptance coding requests, but can waste draft and verification work when
even the first proposal is frequently rejected.

`AX_MLX_MTP_CONSERVATIVE_DEPTH=1` enables an experimental request-local policy
for native Qwen linear-attention MTP with a three-token throughput window:

- Observe at least 32 actual proposals at each of the first two positions.
- Use the conservative controller only while first-proposal acceptance is below
  75% and second-proposal acceptance is below 50%. A depth-one window does not
  enter the second-position denominator. These are cumulative request counters,
  not a rolling window.
- After a rejection, set the next depth to the accepted prefix length, with a
  minimum of one. Grow by one after a complete window is accepted, up to three.
- Return to the existing controller when the acceptance condition no longer
  holds. A clipped final window cannot increase depth.

The option is **off by default**. It does not enable MTP automatically or change
the verifier, acceptance rule, rollback, or model files. Explicit
`AX_MLX_MTP_FIXED_DRAFT_DEPTH` takes precedence. Gemma assistant MTP and other
non-Qwen routes retain their existing policies.

```bash
AX_MLX_MTP_CONSERVATIVE_DEPTH=1 ax-engine serve tiel-coder-35b:axq -- \
  --mlx-mtp-policy required --mlx-mtp-disable-ngram-stacking
```

Use `cyber-tiel-coder-35b:axq` for Cyber-Tiel. Unset the environment variable or
set it to `0` to restore the existing controller. Measure the intended workload
before selecting this experimental option; a short random-token diagnostic is
not representative of a coding conversation.

Route telemetry distinguishes configuration from use:

| Key | Meaning |
| --- | --- |
| `ax_mtp_conservative_depth_code` | The option is configured. |
| `ax_mtp_conservative_depth_admission_code` | The request counters meet the statistical condition; this alone does not establish model eligibility or activation. |
| `ax_mtp_conservative_depth_decisions` | Number of next-window decisions actually made by the conservative controller. |

For a controlled benchmark, `scripts/bench_mlx_inference_stack.py` accepts
`--ax-mtp-policy required`. Speculative rows use that server policy, direct rows
use `disabled`, and the harness removes an inherited
`AX_MLX_MTP_FORCE_REQUESTED` override. Keep `--ax-mtp-disable-ngram-stacking` for
head-only comparisons and capture output IDs when checking repeated-run
stability. The existing harness preserves warmups, cold prefix-cache treatment,
and separate prefill/decode timing.

This option does not change checkpoint certification, MTP-S/P/D status, or the
models' automatic MTP policy. Different verification window sizes can change
greedy sequences through floating-point near ties; comparisons against a
separate direct graph remain diagnostic.

The [M5 Max diagnostic artifacts](../../benchmarks/results/inference/tiel-mxfp4-mtp/2026-09-19/README.md)
include both model revisions, measured trials, output IDs, rejected controls,
and a reproducible evidence checker. The acceptance thresholds are empirical;
they are not a guarantee of improvement on other prompts or longer contexts.
