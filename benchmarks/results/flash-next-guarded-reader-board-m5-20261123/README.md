# Flash Next guarded-reader release board on M5 Max

The current direct-default board passes six of nine gates for the
signed guarded-reader build `d2220c26f`. The measured profile is the
pinned MXFP4 pack on MacBook Pro M5 Max 128 GB, macOS 27.2.1, with
one resident request, up to 32,768 context tokens, and MTP off by default.

| Passing gate | Native result |
| --- | --- |
| Delivery | All 49 publisher members verified from an empty cache |
| Lifecycle | All 14 actions across both direct modes with counters drained |
| Memory | At most 70.6 GiB RSS during QA |
| Readiness | 30.212 s after delivery and 31.147 s for the slowest retained launch |
| Long lookup | 29,774 tokens in 42.532 s and 43.137 s |
| MTP policy | Zero default verifier steps and explicit Required mode verified |

Integrity, independent quality, and reference-relative timing remain
open. The archived ordinary reference gives three QA rows and no timing
cells. A historical complete ordinary reference passes the quality and
timing checks in a replay control on macOS 27.0.1. The unchanged
integrity gate rejects that cross-version combination.

The replay reruns the signed build's
[fresh delivery](../flash-next-guarded-reader-fresh-m5-20261123/README.md),
[lifecycle](../flash-next-guarded-reader-lifecycle-m5-20261123/README.md),
[QA](../flash-next-guarded-reader-qa-m5-20261123/README.md), and
[native timing matrix](../flash-next-guarded-reader-matrix-m5-20261123/README.md)
verifiers. It assembles their original raw rows for the existing
release evaluator and checks the frozen evidence and verifier hashes.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -E -s -B replay.py | cmp comparison.json -
```

With `--ordinary-capture`, the replay accepts the collector's verified
reference directory. It binds the captured result, admission, monitor,
and reference provenance to the collection receipt. The evaluator
checks that completed ordinary reference against the same native
artifacts and thresholds.
