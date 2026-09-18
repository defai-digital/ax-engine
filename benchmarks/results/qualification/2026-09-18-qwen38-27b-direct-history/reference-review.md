# Reference and independent review

MTPLX `mtplx/benchmarks/runners/batch_equivalence.py:69-84,132-172` measures logit and actual argmax agreement for explicitly replayed routes. oMLX `omlx/patches/mlx_lm_mtp/qwen35_model.py:567-614` restores an accepted prefix; rollback alone does not establish direct-route numerical equivalence. Pinned revisions are in `references.json`. No reference implementation is copied.

DeepSeek completed a self-contained patch review and found no indexing blocker. Compiling and running the probe resolve its API-signature concerns. Single-token validation is deliberately retained as a prefill-only boundary check; it does not establish decode coverage. MiniMax M3 timed out; MiniMax M2.7 completed a bounded evidence review. Its cache-root-cause and sampling-noise assertions were rejected as unsupported: these are greedy runs, and equal IDs do not establish state identity. The common mismatch only excludes replacing synchronous calls with pipeline APIs as a sufficient repair.

These reviews guide experiments and do not certify quality or release readiness.
