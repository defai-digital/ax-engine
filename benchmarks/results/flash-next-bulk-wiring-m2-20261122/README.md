# Flash Next bulk GPU residency diagnostic on M2 Ultra

The probe uses the pinned MXFP4 checkpoint and MLX and MLX Metal 0.32.4
on a Mac Studio M2 Ultra 192 GB with external APFS storage. It compares
all 158 selected resident tensors in each of two startup-sensitive
shards, 17 and 18. The selected data is 5,144,009,280 bytes per shard.

At MLX tag `v0.32.4` (`36969c094`), the Metal allocator tracks allocations
even at a zero wired limit. Within the current limit, insertion commits
the affected set immediately. Raising the limit can place the tracked
allocations together and commit once per affected set. AX's normal runner
sets its wired budget before loading weights. The probe compares that
order with setting the same 8 GiB budget immediately after loading the
owned arrays and before their Metal verification. Both arms use the same
mapped reader and array creation order. Total time includes the budget
change, mapping, allocation, copy and unmap.

| Shard | Initially selected pages | Bulk change, first pair | Bulk change, reverse pair |
| --- | --- | ---: | ---: |
| 17 | None resident | +2.17% | −64.68% |
| 18 | None resident | +12.25% | −71.23% |
| 17 | All resident | +3.74% | +1.44% |
| 18 | All resident | +2.25% | +2.35% |

All 16 arms preserve all 2,528 selected payload observations, allocator
sizes and Metal checksums. Memory pressure is normal at each boundary.
The reused-input medians are 2.30–2.59% slower with bulk wiring. The
two large apparent savings come from reverse-pair early arms that take
77.643 and 93.408 seconds. Both initially nonresident first pairs favor
early wiring. APFS clones attest initial page residency for their own
mappings and can share underlying storage caches. The measured component
does not support a startup policy change.

The probe's 8 GiB budget applies to one fresh process at a time. The
serving process uses a global wired budget with shared-runner lifetimes.
Any future serving experiment needs separate validation of that ownership
and matched M5 startup and warmed serving observations.

The patches apply to the [existing mapped-reader probe](../flash-next-paired-resident-reads-m2-20261122/README.md).
The workload and complete shard hashes come from the previous startup
holdout and the frozen publisher inventory. The replay reconstructs the
source hashes and checks the raw observations and both measured intervals.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
