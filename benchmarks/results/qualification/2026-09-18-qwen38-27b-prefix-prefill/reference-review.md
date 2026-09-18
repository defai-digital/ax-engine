# Reference comparison and independent review

Reference revisions inspected for this continuation are in `references.json`.
They differ from the earlier report's snapshots; the earlier receipts remain
unchanged. References are design input only; no source was copied.

- MTPLX `mtplx/benchmarks/runners/batch_equivalence.py:69-84,132-172` compares
  actual logits, hidden values and materialized argmax for explicit batched
  and sequential suffixes. Such comparisons require the same prefill history.
- oMLX `omlx/patches/mlx_lm_mtp/qwen35_model.py:567-590` describes restoring
  the accepted prefix from retained state. Rollback correctness alone does
  not establish equality with a separately prefilling direct route.
- MLX `mlx/compile.cpp:217-228` enables compilation unless
  `MLX_DISABLE_COMPILE` is set. Merely calling enable_compile does not establish
  a different baseline compilation mode.
- MLX `mlx/array.h:304-305` permits donation only with uniquely owned array
  descriptor and data. A retained shallow clone increases ownership; it does
  not need to deep-copy buffers to inhibit donation. AX's production runner
  and diagnostic deliberately use the same `MlxKVCache::clone` operation.

The decisive implementation comparison is within AX: the runner's aligned
linear-prefix capture invokes cache-only PreserveFinalTokenStep/Blocking on
the head before calling normal prefill on the tail. The previous probe used
only the latter entry. The source comparison preceded the split controls and
CLI repair; the measured server reversal and complete cache witnesses provide
causal evidence for this bounded reproduction defect.

Both reviewers completed initial, comparison-fix and final patch reviews via
AX Code. Receipts retain prompt/output hashes and completion status.

DeepSeek's intermediate missing-state lead was valid: asymmetric missing
states still returned infinity and could exit successfully. The final patch
returns an error and adds a regression. Its final unverified-binding and
unmigrated-call concerns were resolved by the actual FFI signatures, compilation
and full tests. Its claim that a snapshot must deep-copy buffers is incorrect
under the reference ownership rule and the production runner's own lazy clone.

MiniMax found no final blocker. Its description of the unsplit token as "wrong"
is too strong: that token is the clean cache-off server's own result. This
work repairs an invalid comparison between execution histories; it does not
judge answer correctness. A small margin does not by itself identify one
faulty kernel or establish harmlessness across workloads.

Both-absent state handling checks comparison validity, not the model's expected
layer topology. The separate cache observer validates all48 linear and16
full-attention layers, and the CLI continues to state that matching history
IDs alone does not establish live-server cache identity. These are bounded
experiments, not reviewer-issued release approval.
