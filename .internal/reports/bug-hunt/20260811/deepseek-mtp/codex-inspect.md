## Residual risks

Greedy remaining at `T=1.0` and removal of the hybrid `0.7` hardcode are confirmed.

- **High — think-boundary mismatch remains.** The next draft uses stale `state.ngram_in_think` in both [hybrid](/Users/akiralam/code/ax-engine/crates/ax-engine-mlx/src/runner/mod.rs:9964) and [pure MTP](/Users/akiralam/code/ax-engine/crates/ax-engine-mlx/src/runner/mod.rs:10144), despite post-result state being available at [line 9723](/Users/akiralam/code/ax-engine/crates/ax-engine-mlx/src/runner/mod.rs:9723). After `</think>`, stochastic drafting can use `T=1.0`, while the next acceptance recomputes `T=0.7`; entering think reverses the mismatch. Therefore claim (1) is not always true.

- **Low — coverage/API regression risk.** The passing unit test checks only resolver values, not a proposal→accept cycle across think boundaries or the hybrid path. Legacy mode-only temperature helpers also remain publicly callable.