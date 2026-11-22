# Flash Next current native release board on M5 Max

The complete native sequence uses the production server built from
`4bd15681fc7e6eec0c88f62a1a12ec9261cfcce4` with Rust 1.99.1 and MLX
0.32.4, on a MacBook Pro M5 Max with 128 GB unified memory, internal NVMe
SSD and macOS 27.2.1. The four native results bind the same server
binary and publisher inventory for the pinned MXFP4 pack. The independent
MLX-VLM receipt is an incomplete run with the same pinned inventory
and hardware profile.

Five of the nine frozen gates pass. Delivery, lifecycle, memory, long
lookup and the default MTP policy pass. The delivered-copy default
started in 122.002 seconds and the Required control in 125.258 seconds.
Both exceed the 120-second warm limit. The independent reference
record leaves integrity, quality comparison and relative timing open.
The current native QA completes 105/105 cases for each direct arm,
with identical text and four retained checker failures in each arm.

The evaluator rereads and regrades all 210 original AX QA rows. It checks
the frozen request order, generation budget, normal stop and exact text
agreement independently of the summary and records a digest for each
mode's raw rows. The unchanged signed v7.9.0 evidence passes 9/9 on
the same evaluator. A negative control that blanks one archived default
answer with the summary untouched now fails the quality gate.

The [delivery](../flash-next-mxfp4-current-delivery-m5-20261122/README.md),
[QA](../flash-next-current-qa-m5-20261122/README.md),
[lifecycle](../flash-next-current-lifecycle-m5-20261122/README.md),
[native matrix](../flash-next-current-matrix-m5-20261122/README.md)
and [partial reference](../flash-next-ordinary-reference-m5-20261121/README.md)
remain in their original packages. The [comparison](comparison.json)
records the input hashes and the evaluator hash. The separate staged
run will measure the current runtime source.

From the repository root, replay the board with:

```bash
python3 -B benchmarks/results/flash-next-current-native-board-m5-20261122/replay.py
```
