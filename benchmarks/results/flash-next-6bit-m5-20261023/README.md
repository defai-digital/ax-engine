# Qwen 3.8 Flash Next affine 6-bit MTP on MacBook Pro M5 Max 128 GB — 2026-10-23

Pack: `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP` at
`d514dcebf3086068ed7968caf395083c95ebcfca` (affine 6-bit/group64, 167 GB including the
53.6 GiB n-gram table), a local copy on the internal SSD; its config, AXQuant manifest,
expert-stream manifest and tensor index match the pinned revision's recorded hashes. The
Hub repository is not public. AX Engine at main `0f9e44a3` plus the ADR-050 admission
change (release build, Homebrew Rust 1.99.1). One request at a time, MTP off, greedy,
`--total-blocks 4096`. Signal-level evidence on one machine, not a qualification record.

| Expert placement | Prompt tokens | Output tokens | TTFT (s) | Decode tok/s | Server RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `--stream-experts auto` (default) | 26 | 64 | 30.5 | **0.17** | 14.1 |
| `--stream-experts off` (resident) | 26 | 64 | 0.27-0.30 | 46.2-46.5 | 102.2-102.3 |
| `--stream-experts off` (resident) | 1,905 | 128 | 3.31 | 44.33 | 102.3 |
| `--stream-experts off` (resident) | 2,207 | 128 | 3.78 | 41.57 | 102.5 |
| `--stream-experts off` (resident) | 8,976 | 128 | 30.67 | 41.07 | 103.2 |

- Auto streams the experts from the SSD on this host (its admission rule keeps a large
  headroom over the full-resident estimate), which makes decode about 270x slower.
  Resident, the pack fits: swap did not grow (1.79 GB before and after) and
  `memory_pressure` reported 15-17% free during the resident runs.
- Generated text is coherent in every row (`Paris`; an iterative Fibonacci function;
  summaries of the prompt code). No QA cohort, long-context (above 9k), concurrency,
  MTP or endurance run was made, and the 96 GiB peak-RSS bound of the MXFP4 release
  gates is exceeded (about 103 GiB), so the MXFP4 gates do not transfer.
- For comparison, MXFP4 on the same host decodes 49.9 / 47.3 / 45.7 tok/s at 1.9k /
  2.2k / 9k with 78 GiB peak RSS (`../qualification/2026-10-23-flash-next-mxfp4-m5max-release-gates/`).
- Earlier 6-bit receipts are from the 192 GiB M2 Ultra; there the same build decodes
  30.7-31.4 tok/s resident at 102.5 GiB RSS.

Files: `auto-streaming.jsonl`, `resident-short.jsonl`, `resident-long.jsonl` (one JSON row
per request: TTFT, decode, prompt and output tokens, RSS, swap, text head),
`harness_resident.py` (the resident-run script; the auto run used the same script
without `--stream-experts off`).
