# Flash Next bounded PLE quiet-host repeat on M5 Max

MacBook Pro M5 Max 128 GB, internal NVMe SSD, macOS 27.2.1. The repeat
uses the same pinned publisher pack, MLX-VLM graph, affine8 PLE reader,
and native comparison matrix as the
[previous bounded reference](../flash-next-affine8-reference-refresh-m5-20261121/README.md).
The launcher verified each frozen source and input before startup.

All 105 QA outputs and grades and all 15 fixed 128-token trajectories
match the signed control through the earlier bounded replay. All 31
pressure samples are normal. Peak MLX allocation is 67.91 GiB and peak
process RSS is 53.51 GiB. The six reference-relative timing cells meet
their numeric ratios. Before-run and after-run CPU idle measured
90.83% and 88.60%, respectively. The after-run sample misses the 90%
floor, so these remain diagnostic observations. The ordinary MLX-VLM
qualification and its timing gate remain open.

The [bounded result](bounded-result.json.gz), [memory samples](memory-samples.jsonl.gz),
and [source bindings](source.json) retain the input and outcome.
Reproduce [comparison.json](comparison.json) from the repository root.

```bash
python3 benchmarks/results/flash-next-affine8-reference-quiet-m5-20261122/replay.py
```
