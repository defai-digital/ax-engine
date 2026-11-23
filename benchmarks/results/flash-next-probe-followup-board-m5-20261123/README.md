# Flash Next storage-probe follow-up release board

This snapshot records six of nine release gates on signed native build
`26c0ccefe`, MacBook Pro M5 Max 128 GB, and macOS 27.2.1. Delivery,
lifecycle, memory, readiness, long lookup, and MTP policy pass. Integrity,
independent quality, and reference-relative timing remain open.

The unchanged command-line evaluator reads the
[fresh delivery](../flash-next-probe-followup-fresh-m5-20261123/README.md),
[QA](../flash-next-probe-followup-qa-m5-20261122/README.md),
[lifecycle](../flash-next-probe-followup-lifecycle-m5-20261122/README.md),
and [retained matrix](../flash-next-probe-followup-retained-m5-20261122/README.md)
receipts. The replay reruns each archived verification and binds the raw
receipts and evaluator inputs by SHA-256.

The first native matrix matches all 124 signed output arrays. Its final
78.15% CPU idle sample misses the 90% timing floor. The pinned
[ordinary MLX-VLM reference](../flash-next-ordinary-reference-m5-20261121/README.md)
supplies three QA rows and no timing cells. A qualifying same-build
native repeat and a complete ordinary reference are separate results.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
