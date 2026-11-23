# Flash Next restored-prefix suffix matrix with resident loading on M5 Max

The signed diagnostic runs on MacBook Pro M5 Max 128 GB with macOS
27.2.1, internal SSD, Rust 1.99.1, and MLX and MLX Metal 0.32.4. It
verifies every publisher member before and after the pinned MXFP4 run.

The initial raw-weight-loader attempt stops at the first pressure
warning, after 211.2 seconds and 105 normal samples. Its guarded exit
and original pressure trace are retained. The new test uses the real
`MlxRunner` resident loading policy with one active request and a
32,768-token budget. The fixture and published model stay pinned.

Cold and rehydrated state agree after all ten suffixes, from one to
2,048 tokens, and after four further decode operations. All 50 output
IDs match. The native diagnostic records 8,640 singleton GDN dispatches
and 100 normal pressure samples. Peak sampled process RSS is
74.25 GiB. The [M2 matrix](../flash-next-prefix-suffix-matrix-m2-20261123/README.md)
uses the same input fixture. Both hosts have independent cold runs.

The evidence verifies state and outputs for the tested lengths. The
HTTP and product qualification campaigns have separate gates.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
