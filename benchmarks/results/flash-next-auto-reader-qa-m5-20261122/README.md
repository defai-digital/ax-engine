# Flash Next automatic reader native QA on M5 Max

The signed `88c480c03` release-server binary from the
[independent delivery](../flash-next-auto-reader-fresh-m5-20261122/README.md)
ran the frozen QA cohort against a retained APFS view of the same pinned
MXFP4 publisher files. The MacBook Pro used M5 Max, 128 GB of unified
memory, an internal NVMe SSD, macOS 27.2.1, and MLX 0.32.4.
CPU idle was 98.95% before the cohort and 98.77% afterward.

| Mode | Checks passed | Ready | 29,774-token lookup | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| MTP disabled | 101/105 | 108.046 s | 42.124 s | 78.746 GiB |
| Ordinary default | 101/105 | 99.708 s | 42.799 s | 78.773 GiB |

All 210 responses match the signed native baseline in text, finish reason,
and prompt, completion and total token counts. Both modes reproduce the same
four frozen checker failures and report zero verifier steps and route
errors. Each advertises 32,768 context tokens. The memory and long lookup
gates pass on this retained snapshot.

The replay binds the raw responses to the signed executable, the verified
delivery, and the original answer checkers. Lifecycle, repeated serving
matrix, and independent MLX-VLM evidence are still required for release.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
