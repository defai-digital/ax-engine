# Flash Next physical width control

The ignored native diagnostic compares two physical prefill widths while
holding each logical end and final singleton fixed. Each width starts
from an empty request cache with the same resident weights. The reference
width runs twice. Both controls have to reproduce the serialized trunk
state at each logical cut, all greedy IDs, and the full F32 logits
after two identical teacher-forced continuation tokens. The diagnostic
then compares the smaller width against those controls.

The [synthetic results](synthetic.json) use two existing tiny MXFP4
fixtures with F32 and BF16 activations on M2 Ultra 192 GB, macOS 27.2.1,
Rust 1.99.1 and MLX 0.32.4. Both use tokens 1 through 22 and logical
cuts at 11 and 22. Widths four and two match both state hashes, the
two teacher-forced logit vectors and every one of the 24 greedy IDs
in both fixtures. The tiny fixtures establish behavior only for these
shapes and weights.

The [real-pack M2 diagnostic](real-pack.json) uses the second prompt
from the [earlier M5 partition replay](../flash-next-mxfp4-prefill-partition-m5-20261121/README.md).
The prompt has 2,048 tokens. The probe verifies all 49 publisher members
and uses widths 512 and 256 at the same logical cut with the same last
singleton. The two width-512 controls reproduce all 96 greedy IDs,
serialized state and the two same-input logit vectors. Width 256 changes
the serialized state and both logit vectors, with maximum absolute
differences of 1.0625 and 0.5. It changes 92 of 96 greedy output tokens
and first differs at token four. This disqualifies width 256 from the
exact path for the tested input. The probe runs outside HTTP serving.

```bash
cargo +1.99.1 test -p ax-engine-mlx --lib \
  flash_next_compare_prefill_widths_at_equal_logical_cuts -- --ignored --nocapture
```

Set `AX_FLASH_NEXT_ORACLE_DIR` to the Tiny MXFP4 fixture for a synthetic
run. Use the pinned resident `AX_FLASH_NEXT_REAL_PACK` and a JSON array
in `AX_FLASH_NEXT_PROMPT_IDS` for the real diagnostic. The prompt must
contain one to four complete 2,048-token items.
