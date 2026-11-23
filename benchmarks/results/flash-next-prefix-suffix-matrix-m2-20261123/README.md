# Flash Next restored-prefix suffix matrix on M2 Ultra

The signed diagnostic uses the pinned Flash Next MXFP4 pack on a Mac
Studio M2 Ultra 192 GB with external SSD, macOS 27.2.1, Rust 1.99.1,
and MLX and MLX Metal 0.32.4. It verifies all 49 publisher members
before and after the run. The reproducible test is `cdbbb572a`.

The probe compares cold and rehydrated native state after a 2,048-token
prefix and ten suffix lengths, from one token through 2,048. It rebuilds
the prefix independently for every cold row. The original prefill
width scales to 512 for the 2,048-token suffix. The shorter suffixes
retain the 1,024-token base width. All 50 output IDs match. Every
serialized state matches after suffix prefill and after four more decode
operations. The retained seed digest stays the same, and the ten suffix
states differ. The probe records 7,992 native singleton GDN dispatches
and 73 normal pressure samples.

The receipt covers state and output equivalence at the sampled lengths.
Serving admission, concurrency latency, and product qualification use
separate tests. The original pack, signed test, fixture, and raw
observations are bound in the archive.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
