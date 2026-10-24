# Flash Next affine 6-bit MTP retest at the public revision — 2026-10-24

Pack: `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP` at
`3c1dd547d70cb7bf01dd0a89c41c400dde32ed39`, now public. The earlier pin `d514dceb` no
longer exists on the Hub; the new revision keeps the same config, tensor index, expert
manifest and weight shards and changes only AXQuant metadata. Every one of its 57 files
was verified on the host against the Hub (LFS sha256 for the shards, git blob ids for the
rest). Host: MacBook Pro M5 Max 128 GB. Runtime: main `318d315d` plus the repin (the
released v7.9.0 predates 6-bit admission and rejects this pack with "standalone affine
packs are retired").

| Run | Result |
| --- | --- |
| Auto expert placement, 26-token prompt | 0.17 tok/s decode (experts streamed from SSD), RSS 14 GiB |
| `--stream-experts off`, 26 / 1,905 / 2,207 / 8,976 prompt tokens | 46.2 / 44.5 / 41.5 / 41.0 tok/s decode, TTFT 0.29 / 1.73 / 3.27 / 29.9 s, RSS 85-103 GiB, no swap growth |
| Frozen 105-item QA cohort, resident, direct and default modes | **103/105** in both modes (failures: `format_csv_pair`, `knowledge_water_formula`), identical text across modes, normal stops, peak RSS 109.8 GiB, ready 69-73 s, 29,774-token item 122-124 s |

For comparison the MXFP4 pack scores 101/105 on the same cohort (failures
`format_status_colon`, `instruction_alphabet_first`, `knowledge_water_formula`,
`science_gravity_earth`) at 78 GiB. The QA driver (`qa_6bit_resident.py`) reuses the
frozen cohort runner and grader unchanged and only adds `--stream-experts off`; it does
not bind a pack inventory, so this is not a release-gate record, and the 109.8 GiB peak
is above the 96 GiB bound those gates use.

Files: `six-auto.jsonl`, `six-resident.jsonl`, `qa/` (per-item rows, server logs,
`result.json` with grades), `qa_6bit_resident.py`, `retest_m5.sh`.
