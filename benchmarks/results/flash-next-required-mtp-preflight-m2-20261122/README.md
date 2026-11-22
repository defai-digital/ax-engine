# Flash Next required-MTP startup preflight

The native preview server returns `MlxMtpRequiredButUnavailable` when
Flash Next is launched with both `--mlx-mtp-policy required` and
`AX_NO_SPEC=1`. The session checks the same cached process switch used
by runner activation after validating the model metadata.

The [previous M5 negative control](../flash-next-delivery-bound-startup-m5-20261122/README.md)
copied 15 resident shards in 74.776 seconds and loaded the MTP sidecar
in 10.510 seconds before returning this error. The amended release
server on Mac Studio M2 Ultra 192 GB with MLX 0.32.4 and the pinned
publisher MXFP4 pack returned the same error in 0.684 seconds from
process launch. The startup log recorded zero runner initialization,
wiring, resident shard and sidecar markers. Times describe their
respective hardware and measurement scopes.

The live [check](check.py) verifies all 49 member sizes, three
runtime metadata hashes and the exact admitted manifest. It uses the
existing native required policy command. The
[build receipt](build.json) binds the executable and touched source
files. Replay joins the [observed error](server.txt) to the previous
M5 startup trace and reproduces [comparison.json](comparison.json).

```sh
python3 benchmarks/results/flash-next-required-mtp-preflight-m2-20261122/replay.py
```
