# Correction (2026-07-17, same day): promotion evidence invalid — default reverted to opt-in

The 5-rep Coder-Next A/B in this directory measured a path that never
engaged: every `cn-on-*.txt` log contains 48 one-time
"[Primitive::output_shapes] GatherQMM cannot infer output shapes" errors —
MLX cannot shapeless-compile gather-routed MoE closures, so all 48 layers
fell back permanently and the 1.016 "ratio" was pair noise between
identical imperative runs. The engagement check that the fused-router A/B
used (attempts/hits counters) was skipped here; that omission is the
lesson.

Valid results retained from this directory:
- The 12k/5k soaks and greedy parity: real (they exercise the fallback and
  poison paths, which is what the stability claim needed).
- `gemma-mlc-{off,on}-{0..2}.txt` (added by the correction): the Gemma
  dual-path closure DOES engage; 3 interleaved pairs give off median 95.25
  vs on median 95.56 → ratio 1.003 (neutral), checksums identical.
- `session-caps-{on,off}.txt`: end-to-end python-Session verification of
  the auto-buffer-caps runner-init-order fix (70.19 vs 54.88 tok/s).

Decision per ADR-003 D5: no family shows ≥1.01, so
`AX_MLX_MOE_LAYER_COMPILE` returned to opt-in the same day. The
poison-propagation safety fix stays (it is what made the Coder-Next
failure graceful). Upstream candidate: GatherQMM `output_shapes` support
in MLX compile.
