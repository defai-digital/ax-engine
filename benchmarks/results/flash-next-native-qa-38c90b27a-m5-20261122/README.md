# Flash Next native QA on the `38c90b27a` M5 production build

This replay uses the release-server binary from the [fresh delivery](../flash-next-fresh-delivery-38c90b27a-m5-20261122/README.md)
and an APFS view of its verified MXFP4 checkpoint on a MacBook Pro
M5 Max 128 GB with internal NVMe storage and macOS 27.2.1. The disabled
and ordinary product launches each advertise 32,768 context tokens.
Host CPU idle is 99.56% before the cohort and 98.35% after it.

| Mode | Responses | Frozen checks passed | 29,774-token lookup | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 105 | 101 | 41.682 s | 78.704 GiB |
| Product default | 105 | 101 | 41.663 s | 78.793 GiB |

All 210 responses match the signed native evidence in text, finish reason,
and prompt, completion and total usage counts. The modes reproduce the
same four frozen checker failures, and every response stops normally.
Both report zero MTP verification steps and route errors. The native
96 GiB RSS and 400-second lookup gates pass.

The startup times within this staged QA are 106.098 and 116.165 seconds.
The separate fresh-delivery warm readiness, lifecycle, serving matrix
and ordinary independent comparison use their own receipts.

The [archive](evidence.tar.xz) retains the raw responses, contract and
receipt. The replay binds them to the fresh-delivery executable, publisher
inventory, signed response oracle and original closed-answer checkers.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
