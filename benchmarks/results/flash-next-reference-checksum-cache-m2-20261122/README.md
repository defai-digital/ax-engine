# Flash Next independent reference checksum cache experiment on M2 Ultra

Mac Studio M2 Ultra 192 GB, internal APFS SSD, macOS 27.2.1. The
ordinary reference verifies the pinned pack using 4 MiB buffered reads
before constructing the language graph. This diagnostic reads the
pinned 5,283,244,832-byte `model-00001-of-00025.safetensors` shard with
the same chunk size. In each process sequence the modes are buffered,
`F_NOCACHE`, `F_NOCACHE`, buffered. Every arm uses a separate fresh APFS
clone with zero resident pages at its first `mincore` sample. All eight
checksums match the published SHA-256 and all pressure readings are normal.

| Reader buffer | Descriptor cache policy | Post-checksum resident pages, both repeats | Resident fraction |
| --- | --- | ---: | ---: |
| Python buffered reader | Default | 322,464 | 100% |
| Python buffered reader | `F_NOCACHE` | 241,824 | 75% |
| Page-aligned read buffer | Default | 322,464 | 100% |
| Page-aligned read buffer | `F_NOCACHE` | 0 | 0% |

The page-aligned arm uses unbuffered `readinto` and an anonymous 4 MiB
mapping. The measured cache-bypass behavior depends on that reader
configuration. Checksum time is diagnostic because system activity
and the device cache are uncontrolled. The probe covers one file and
does not measure inference, model materialization or whole-pack memory.

The [previous M5 materialization diagnostic](../flash-next-reference-materialization-batches-m5-20261122/README.md)
stopped at a system pressure warning. These shard results justify a
separate paired preflight and materialization trace on M5. Keep the
published hashes, pinned MLX-VLM graph, and ordinary qualification
evidence bound under their existing contracts.

The [runner](hash_cache_probe.py), [buffered reader receipt](buffered-reader.json),
and [aligned reader receipt](page-aligned.json) preserve the method and
observations. Replay them against the pinned publisher inventory.

```bash
python3 benchmarks/results/flash-next-reference-checksum-cache-m2-20261122/replay.py
```
