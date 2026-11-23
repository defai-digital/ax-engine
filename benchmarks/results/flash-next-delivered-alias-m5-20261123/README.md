# Flash Next ordinary offline alias after fresh delivery

The signed CLI from the [fresh delivery](../flash-next-guarded-reader-fresh-m5-20261123/README.md)
resolves `qwen3.8-flash-next:axq` to its pinned MXFP4 snapshot with
`--offline` on MacBook Pro M5 Max 128 GB. The plain model alias starts
the same signed native server without an engine preset or reader override.
It selects the aligned buffered reader and reaches HTTP readiness in
30.242 seconds including local resolution. The logs confirm a 32,768-token
context, one live request, and resident execution from internal SSD.

The run checks all 32 output IDs from the signed 512-token [crossover case](../flash-next-automatic-buffered-reader-m5-20261123/README.md),
with zero MTP steps and route errors. Health remains good and all five
work counters drain. The replay binds that input witness to the launch
receipt. The receipt records the matched count and launch assertions.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
