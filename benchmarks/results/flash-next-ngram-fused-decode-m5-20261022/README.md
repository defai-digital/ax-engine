# Flash Next n-gram reads and fused decode stretches (2026-10-22)

Apple M5 Max, 128 GB, macOS 27.0.1, pack on the internal SSD. Campaign host, signal
only: not a certification run. Analysis:
[docs/performance/flash-next-ngram-reads-and-fused-decode-2026-10-22.md](../../../docs/performance/flash-next-ngram-reads-and-fused-decode-2026-10-22.md).

Pack: `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` @
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`, MTP off, direct decode, one request at a
time, `--prefill-chunk 2048 --max-batch-tokens 2048`, `--total-blocks 2048`.

## Protocol

Three server builds, each run in a fresh process after `purge` (page cache emptied) so
every n-gram row is cold: `before` is the build from `main` db2a8b92 plus the DFlash2
branch (which does not touch Flash Next); `n-gram reads + read-ahead` is the new build
with `AX_FLASH_NEXT_FUSE=0`; `plus fused stretches` is the new build with defaults at the time
(the first seven stretches; the later GDN segment stretches add 0.8% decode, same text).
Prompts are first-seen Rust source excerpts (`Explain what this Rust code does and list
its main risks.` followed by 8,000 characters of a repository file, chosen at random
with seed 5; `long1` concatenates four of them), greedy, 128 generated tokens. Decode is
the rate after the first token over the streamed response. The client records the
SHA-256 of the generated text; it is identical in all three arms for every prompt.

## Files

- `summary.json`: the five prompts x three arms (TTFT, prefill, decode, text SHA-256).
- `raw/`: the per-arm client records (`out-g_base-*` = before, `out-g_io-*` = reads +
  read-ahead, `out-g_full-*` = plus fused stretches; `-s` is the four short prompts in
  one session, `-l` the 8,976-token prompt alone).
- `harness/`: `fn_ab.py` (server launch and streaming client), `run_final2.sh` (the
  three arms), `final_report.py` (the table).

A 39,225-token prompt (not in `raw/`): TTFT 123.0 s before, 76.4 s after, decode 26.9 to
31.6 tok/s, text SHA-256 identical (48 generated tokens).
