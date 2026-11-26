# Qwen 3.8 27B DFlash2: 8 against 16 tree rows by output length

**Date:** 2026-11-26 · **Host:** MacBook Pro M5 Max 128 GB, macOS 27.2,
pinned MLX 0.32.4 · **Pack:** Qwen 3.8 27B AXQ MXFP4 with the DFlash2
drafter · **Build:** AX Engine 8.1.0+8108ac93 (commit `8108ac93c`)

Note: [27B DFlash2 tree rows](../../../docs/performance/qwen38-27b-dflash2-tree-rows-2026-11-26.md).

## Files

- `runs.jsonl`: one record per request. `tree_rows` is
  `AX_MLX_DFLASH2_TREE_ROWS`, `run_order` the arm (1 to 4),
  `max_tokens` the output cap, `first_to_last_text_seconds` the time
  from the first to the last streamed text, `text_sha256_prefix` the
  first 12 hex digits of the reply's SHA-256.
- `run.py`: the client. Usage: `run.py <port> <model> <corpus.jsonl>
  <caps> <tokenizer.json>`.

## Method

- Four arms in the order 8, 16, 8, 16 rows. Each arm is a fresh server
  started with 150 seconds of rest before it:
  `AX_MLX_DFLASH2=1 AX_MLX_DFLASH2_DIR=<drafter> AX_MLX_DFLASH2_TREE_ROWS=<rows>
  ax-engine-server --mlx --mlx-model-artifacts-dir <pack>
  --max-concurrent-requests 1 --mlx-mtp-policy auto --stream-experts off`.
- The corpus is the eight prompts of the campaign's DFlash2 paired corpus
  (code, math and prose; `qwen38_dflash2_paired.jsonl`). Each arm sends one
  warmup request, then every prompt at caps 64, 128, 192 and 256 with
  `temperature` 0, streamed `/v1/completions`.
- Decode rate is completion tokens after the first chunk over
  `first_to_last_text_seconds`, summed over the prompts of one cap. The
  short-answer prompt stops after one token and is left out.
- Prompts repeat across caps, so later caps reuse the stored prompt
  prefix. That changes first text, not the decode rate.
