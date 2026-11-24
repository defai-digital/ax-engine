# Flash Next ordinary reference quiet-entry retry on M5 Max

The frozen MLX-VLM reference heartbeat ran with MLX and MLX Metal 0.32.2
on a MacBook Pro M5 Max with 128 GB unified memory, internal SSD and
macOS 27.2.1. The pinned MXFP4 pack, reference source and signed reference
anchor are bound by the [retained preflight](raw/reference-preflight.json).
The CPU preflight reported 90.71% idle, normal pressure and 6.19 GiB RSS
for the largest background process against the 6.44 GiB signed margin.
The runner's own idle sample was 91.26%.

The reference materialized all 2,326 arrays in 24.113 seconds and
entered the first QA case. It emitted zero tokens before the monitor
observed critical system memory pressure and stopped the process.
The ten resource samples contain eight normal, one warning and one
critical observation. The final sample records 79.10 GiB of compressed
pages, 25,412.19 MiB swap in use and 4.70 GiB for the largest other
process. These host-wide counters describe coincident conditions.

In the [earlier attempt](../flash-next-reference-heartbeat-pressure-m5-20261123/README.md),
the same pinned reference completed three cases before the pressure
stop at a larger sampled background RSS. Both attempts leave independent
quality, integrity and reference-relative timing open on the
[guarded-reader release board](../flash-next-guarded-reader-board-m5-20261123/README.md).
This attempt did not start the full 105-item, 15-cell ordinary reference.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -E -s -B replay.py | cmp comparison.json -
```
