# Path A residual: force packed Gemma4 prefill (mbp-m5, 2026-07-26)

**Decision: `reject_keep_split_prefill` / not_yet.** Gates unchanged.

## Residual

`AX_MLX_GEMMA4_SPLIT_PREFILL_FFN` defaults **ON**: long Gemma4 multi-token
prefill prefers split dual steel qmm even when `gate_up_packed` is loaded.

Kill-switch `=0` forces packed steel dual-output + prefill compile (Path A
attempt to beat split keep_base under thr-b8 pure env).

Pure helpers unit-tested:

- `parse_gemma4_split_prefill_ffn` — fail-closed to default ON
- `prefer_split_dense_ffn_gate_up_for` — routing predicate

## Pure A/B under thr-b8 keep_base (3 cold reps)

Env: cache_eval + async_chunk + pipe layer + eval block:8 + pack weights ON.

| variant | median ms | ratio vs split |
|---------|----------:|---------------:|
| split_prefill (default) | **7823** | 1.000 |
| packed_prefill (`SPLIT_PREFILL=0`) | 8149 | **1.042** |

Text parity OK. Keep bar ≤0.96: **not met** (packed regresses pure).

## Concurrent dual-target S1 smokes (1-rep)

| config | thr | gap | ax thr |
|--------|----:|----:|-------:|
| thr-b8-util split-prefill (default) | **1.145** | 1.201 | 20.96 |
| thr-b8-util packed-prefill | 1.119 | 1.168 | 20.49 |

Default split-prefill remains best thr (near gate); gap still ≫0.90.
Packed-prefill taxes thr without clearing gap.

## Conclusion

Keep default **split prefill**. Packed dual-output is not a pure ≤0.96 unlock
under thr-b8 keep_base and does not transfer concurrent thr headroom.
No cool formal (smoke thr near 1.15 but gap 1.20; packed thr regresses).
Flip remains **not_yet**.
