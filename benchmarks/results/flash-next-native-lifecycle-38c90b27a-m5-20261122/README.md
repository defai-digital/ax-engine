# Flash Next lifecycle on the `38c90b27a` M5 production build

This run uses the binary and verified MXFP4 checkpoint from the
[fresh delivery](../flash-next-fresh-delivery-38c90b27a-m5-20261122/README.md)
and the same internal-SSD APFS view as the [native QA](../flash-next-native-qa-38c90b27a-m5-20261122/README.md).
The host is a MacBook Pro M5 Max 128 GB with macOS 27.2.1.

Each mode passes all seven actions in the frozen order. The baseline,
SSE, one-token budget, two-token budget, stop, disconnect and recovery
results match the [prior production controls](../flash-next-current-lifecycle-m5-20261122/README.md)
in stable response fields and activation counters. TensorFold
`f0cc0a134` checks cancellation and returns the lane's reservation at
completion. The AX HTTP replay checks that all five generation counters
drain after each action and that both owned servers exit cleanly.

CPU idle is 98.75% before and 98.70% after the run. The disabled and
default servers reach readiness in 105.088 and 124.249 seconds on the
staged view. The lifecycle gate passes. The default startup observation
exceeds 120 seconds.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
