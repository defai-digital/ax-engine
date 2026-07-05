# Sliding-model aligned-prefix snapshot store — verification record (e9a79302)

Corrects and supersedes the verification claim in commit e9a79302's message.
The harness runs cited there executed against a stale `ax_engine` Python
extension (a v5.0.2 wheel from June 2 shadowed the freshly built 6.7.1 —
`maturin develop` had installed into a different environment), so the
telemetry quoted in that message reflected the pre-change retained-cache
mechanism, not the new snapshot store. After force-installing the current
wheel (`maturin build --release` + `pip install --force-reinstall`), the
real results on `gemma-4-e2b-it-4bit` are:

- `warm_repeat`: **PASS 5/5 token-exact**, cold `ax_mlx_prefix_cache_stores` = 4,
  warm `ax_mlx_prefix_cache_hits` = 4, `blocked` = 0
  (`prefix_equiv_e2b_warm_repeat_real.json`). Unaligned prompts now store and
  hit their largest block-aligned prefix; before the change these calls
  recorded `blocked_trim_failure` and stored nothing (confirmed by direct
  Session probe: 61-token prompt → stores=1, trim_failure=0 post-change).
- `warm_extend`: 4/5 exact with one late-token divergence
  (`prefix_equiv_e2b_warm_extend_real.json`). This is NOT a regression of the
  sliding change: the standard-FA control `mlx-community/Qwen3-4B-4bit` —
  whose every-16-token prefix stores predate this work — fails the same mode
  3/5 with the same late-token near-tie signature
  (`prefix_equiv_qwen3_4b_extend.json`). Restore + remainder prefill runs
  different chunk shapes than a single cold pass, so bf16 reduction-order
  drift can flip greedy near-ties in the extended generation on any
  architecture. The enforced merge bar for prefix-cache work is the
  fail-closed `warm_repeat` gate, which is green.

Process note: before using Python-side harness results as evidence, verify
the loaded extension is current (`pathlib.Path(ax_engine.__file__).parent /
"_ax_engine.abi3.so"` mtime, or install the wheel explicitly). `maturin
develop` succeeded while a stale site-packages copy kept winning import
resolution.
