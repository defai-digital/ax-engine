# Flash Next paired mapped copies on M2 Ultra

The Mac Studio M2 Ultra 192 GB used external APFS storage, macOS 27.2.1,
and MLX and MLX Metal 0.32.4. The test selects the four largest resident
weight tensors from each of two startup-sensitive MXFP4 shards, 17 and
18. Each selection is 1.5625 GiB. The harness checks the complete shard
hash against the pinned publisher inventory before making its timed
clones.

MLX's parallel file reader splits data into disjoint destination ranges
and joins the workers before returning the buffer. This probe applies the
same ownership rule to the existing mapped copy. Both modes map the file
and allocate one owned MLX buffer per tensor in the same order. For tensors of at least
64 MiB, the candidate runs the two byte copies concurrently
and joins before creating the array. Elapsed time includes allocation,
worker creation, copying and joining. Byte and Metal checks run after the
file mapping is released.

| Shard | Initial state | Serial median | Paired median | Change |
| --- | --- | ---: | ---: | ---: |
| 17 | No selected pages resident | 21.480 s | 13.046 s | −39.26% |
| 18 | No selected pages resident | 8.718 s | 11.078 s | +27.08% |
| 17 | All selected pages resident | 0.271 s | 0.163 s | −39.74% |
| 18 | All selected pages resident | 0.269 s | 0.167 s | −38.04% |

All 16 processes in the counterbalanced orders match the independently
hashed selected weights, allocation size and Metal output. Memory
pressure is normal at each measurement boundary. A separate 64 MiB
plus 257 byte control exercises a nonaligned split remainder, matches
the exact bytes and Metal output, and rejects a source range past EOF.
Codex GPT-6.2 Sol high reviewed the ownership change and prompted that
remainder control.

Read the clone results as initially nonresident file pages. APFS clones
can still share underlying storage caches. The cold-file directions differ
by shard, so the measurement does not support changing the startup loader.
The fully resident rows measure the bounded copy component on M2. M5
whole-model startup and warmed serving require primary-tier evidence.

The two patch files apply to the [previous probe's source and driver](../flash-next-paired-resident-reads-m2-20261122/README.md).
`replay.py` reconstructs both inputs and checks their hashes before reading
the observations. For a device rerun, compile the reconstructed C++20
source against the pinned MLX headers and library, stage the pinned inventory
and selection, and call `run_m2.sh` with `PYTHON`, the pack path and isolated
APFS scratch space. `run_tail.py` takes the same executable.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
