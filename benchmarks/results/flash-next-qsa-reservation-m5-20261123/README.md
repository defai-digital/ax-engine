# Flash Next QSA reservation through the standard M5 server

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX 0.32.4,
and Rust 1.99.1. Two signed release-server executables load the
same pinned 49-member MXFP4 pack with the ordinary one-slot profile,
a 32,768-token context, and MTP off. The matched native SSE workload
disables prefix reuse and paired QSA gathering. It warms both arms
and measures the same 512-, 2,048- and two 8,192-token prompts with
a 64-token output budget.

The correction in `3d94e0d40` prices lazy selected-attention scratch
across the physical prefill chunk. The live startup plan on the
standard host records:

| Plan field | Before | After |
| --- | ---: | ---: |
| Prefill scratch reserve | 1.69 GiB | 12.02 GiB |
| Projected working set | 81.52 GiB | 91.85 GiB |
| Auto ceiling | 115.20 GiB | 115.20 GiB |
| Expert placement | Resident | Resident |

Both arms use the AlignedBuffered reader with no wired shortfall.
The measured requests match **256 output token IDs** and drain all
five resource counters. The corrected server reaches readiness in
34.27 seconds and its peak sampled physical footprint is **75.52 GiB**.
All **163** corrected-build pressure samples stay normal. The prior
control reaches readiness in 32.31 seconds and records 96 normal
pressure samples. These are functional and memory-accounting controls.
Readiness differences retain their original host conditions.

The bound evidence extends the
[M2 and planner tests](../flash-next-qsa-prefill-scratch-20261123/README.md)
to the ordinary primary profile. The 6-bit projection and the
independent MXFP4 qualification board retain separate scopes.
The before executable is also bound in the
[paired HTTP crossover](../flash-next-paired-qsa-http-m5-20261123/README.md).

The [capture](evidence.tar.xz) preserves original output, resource
gauges, pressure samples, startup plan excerpts, and signed input
and build receipts. The [comparison](comparison.json) replays the
accounting delta and exact measured output.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
