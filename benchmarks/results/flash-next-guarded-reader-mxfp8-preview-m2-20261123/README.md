# Flash Next guarded-reader MXFP8 preview on M2 Ultra

The signed server from the [MXFP4 delivery](../flash-next-guarded-reader-fresh-m5-20261123/README.md)
also runs the separate opt-in MXFP8 preview on Mac Studio M2 Ultra 192 GB.
The test uses the exact revision `48b9aa5f002f29a9f8ab7cfe5c7185e342595b1b`,
resident experts, explicit Required MTP and an explicit buffered-reader
request. The native log confirms mapped reads for all 27 selected shards
and 1,842 tensors. The launch retains the 32,768-token context.

The runner validates all 61 publisher files (189,240,515,155 bytes)
before and after serving. The 16- and 512-token prompts reproduce all 64
output IDs from the [canonical MXFP8 preview](../flash-next-mxfp8-resident-loader-m2-20261121/README.md).
MTP verifies 34 steps with zero route errors. Health is good, the
owned process exits cleanly, and all 326 pressure samples are normal.

The successful run uses a 1,800-second startup limit. The first limit of
900 seconds expired during shard loading. A subsequent run stopped at an
obsolete aggregate log assertion after serving. The final replay checks
the native per-shard log and retains both raw HTTP results. This is functional
coverage on the M2 host. Catalog admission and M5 qualification remain open.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
