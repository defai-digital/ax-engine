# Consolidated native lifecycle repeat

The independently relaunched lifecycle used the same signed native
server, verified snapshot, MacBook Pro M5 Max 128 GB, macOS 27.2.1 and
single-stream 32,768-token resident profile as the [first run](../README.md).
All 14 stable actions match the first run exactly, with zero final
verifier steps and drained counters. Default readiness was 30.170
seconds, and explicit disabled readiness was 31.256 seconds.

The CPU samples were 90.63% before and 88.34% after. The first run
ended at 82.31%. The frozen integrity check still requires at least
90% at both endpoints and a complete independent reference. With this
repeat in place, the same six of nine release gates pass.

`python3.12 -E -s -B replay.py` regenerates `comparison.json` using
the consolidated native archive and the frozen release evaluator.
