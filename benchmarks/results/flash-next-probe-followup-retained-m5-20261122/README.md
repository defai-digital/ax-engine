# Flash Next storage-probe follow-up retained matrix on M5 Max

The signed `26c0ccefe` release-server binary completed the retained
native workload on MacBook Pro M5 Max 128 GB with internal NVMe SSD,
macOS 27.2.1, and MLX 0.32.4. The artifact binds the same executable
and publisher inventory as the [native HTTP](../flash-next-probe-followup-http-m5-20261122/README.md),
[QA](../flash-next-probe-followup-qa-m5-20261122/README.md), and
[lifecycle](../flash-next-probe-followup-lifecycle-m5-20261122/README.md)
receipts.

All four measured processes reproduce the 124 signed output arrays.
The selected Disabled and Default processes reach HTTP readiness in
103.421 and 103.377 seconds, respectively. CPU idle is 93.50% before
and 78.15% after the matrix. The final sample misses the 90% timing
floor, so the full matrix is queued for a new run with the same signed
executable, pinned pack, workloads, and thresholds.

The explicit reader override `0` reaches readiness in 105.900 seconds
and reproduces all seven native default responses and activation
counters. Its log records no sidecar pre-read. CPU idle is 98.59%
before the control and 92.61% after it, with normal memory pressure
at both samples. This starts on the retained pack.

Replay binds the raw native receipts, matrix repeats, and mapped
responses to the signed candidate. The empty-cache delivery and
ordinary MLX-VLM comparison retain their own receipts.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
