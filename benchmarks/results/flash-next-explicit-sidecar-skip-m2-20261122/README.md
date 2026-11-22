# Flash Next explicit MTP sidecar bypass on M2 Ultra

Mac Studio M2 Ultra 192 GB, external SSD, macOS 27.2.1. The pinned
MXFP4 pack and all 49 publisher files were verified before the native
controls. The release-server binary used Rust 1.99.1 and MLX 0.32.4.

Both default arms have identical response choices, prompt totals and
completion totals in all seven direct cases. Both have zero verifier
steps and zero attachment failures. The explicit
`AX_MLX_SKIP_MTP_SIDECAR=1` arm takes 6 microseconds in the isolated
sidecar stage. The ordinary arm takes 893,170 microseconds.

The required arm with the sidecar available runs 107 verifier steps.
With the bypass enabled, required mode exits at startup with
`MlxMtpRequiredButUnavailable`. Its timed sidecar stage takes 2
microseconds. The ordinary default still loads a present sidecar and
executes direct generation.

The initial comparator compared every usage field and stopped before
running the negative control. The usage objects carry per-request
prefill and decode timing. Three requests in the bypass process also
reported 16 cached tokens where the paired process reported zero. The
follow-up replays the original responses, compares their stable usage
fields and runs the separate negative control with the same executable
and manifest. The three-arm startup order does not attribute the
cache difference to the switch or measure a delivery readiness gain.

Replay the [control receipt](receipt.json.gz) and compare against
[comparison.json](comparison.json).

```bash
python3 benchmarks/results/flash-next-explicit-sidecar-skip-m2-20261122/replay.py
```
