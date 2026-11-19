# Qwen 3.8 Flash Next MXFP4 runner state

The [publisher inventory check](inventory-check.json) verified all 49
files of the MXFP4 pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
The controls ran on MacBook Pro M5 Max 128 GB using macOS 27.2.1 and
MLX 0.32.4. The [provenance](provenance.json) binds the pack, test
executables, source patches, and reports.

| Control | Result |
| --- | --- |
| [Trained private cursor and verifier transactions](trained-reader.json) | 15 cases passed, including failure after an accepted draft and distinct terminal cases. |
| [Permuted draft-output negative control](permuted-reader.json) | 13 cases passed, including the forced verifier controls. |
| [Default cost calibration](default-cost.json.gz) | The routes returned the same eight IDs. All four aligned states and the continuation logits matched byte for byte. The strict MTP diagnostic failed because the direct calibration used the entire token budget. |
| [Aligned runner and continuation with MTP](aligned-reader.json) | The routes returned the same eight IDs. MTP performed four verifications and accepted three drafts, with zero step errors. Trunk states at consumed positions 48, 50, and 52 and the singleton continuation state and logits matched byte for byte. |

The state observer checks that repeated observations at the same consumed
position serialize identically. Draining a pending direct token can emit
without advancing the trunk. The observer compares one snapshot per
consumed position across the two routes.

The raw [trained](trained.json.gz), [permuted](permuted.json.gz), and
[aligned](aligned.json.gz) reports bind each retained tensor or state
file by SHA-256. The independent readers checked the state files and
continuation logits. These controls cover one fixed-budget, greedy
request on the canonical singleton target schedule. MTP-S, MTP-P, and
MTP-D remain open.

## Reproduce the aligned runner control

Set `MODEL_DIR` to the pinned local model directory, `RECEIPT_DIR` to
this evidence directory, and `RESULT_DIR` to a fresh output path.

```bash
AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1 AX_FLASH_NEXT_MTP_COST=0 \
AX_FLASH_NEXT_REAL_PACK="$MODEL_DIR" \
AX_FLASH_NEXT_PROMPT_IDS="$(cat "$RECEIPT_DIR/prompt.json")" \
AX_FLASH_NEXT_ALIGNED_RESULT_DIR="$RESULT_DIR" \
rustup run 1.99.1 cargo test -p ax-engine-mlx --lib \
  runner::flash_next_tests::aligned_state_tests::flash_next_real_runner_aligned_states_and_continuation \
  -- --exact --ignored --nocapture --test-threads=1

python3 scripts/verify_flash_next_runner_states.py "$RESULT_DIR"
```
