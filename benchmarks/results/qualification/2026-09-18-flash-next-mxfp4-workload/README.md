# Flash Next MXFP4 failed M5 fixed workload

The frozen installed runtime `85a2bab0dd72c9e2076f455237dca052ff8d12b7`
completed with failure on Apple M5 Max 128 GiB using the pinned MXFP4 pack
`0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35` on NAS storage. It predates
the later QSA correction. Qualification, release readiness and performance
claims remain false.

One of six AX cells completed: 512-token direct, with two warmups and three
measurements, each producing 128 tokens. All five raw trial records and
their repeated-output, route and timing validation are retained. The first
512-token required-MTP warmup exceeded the frozen 3600-second request
deadline; zero required-MTP trials completed. The 2048/8192-token cells
never started. Three primary mlx_lm.benchmark cells remain unsupported,
so there is no complete nine-cell comparison or MTP speedup result.

The exact terminal error is `TimeoutError: Frozen request deadline exceeded`.
The required server's exit -9 records supervisor cleanup after SIGTERM and
a 30-second grace. Its log contains only startup; the client log is empty
and no completed/partial-stream JSON exists. This does not establish zero
tokens, a spontaneous native crash, OOM, or a NAS/prefill/decode root cause.
The 1800-second socket timeout bounds individual operations; the
129600-second collection deadline was not reached.

All 68 installed identities passed recorded checks before and after the
attempt, including 48 pack-directory files (47 snapshot files plus the
installed model manifest). The collection observer verified stable remote
terminal/log hashes and matching local copies. Supervisor, server and
client processes were absent at both recorded observations. Large remote
model payloads were not rehashed by the curator.

`result.json.gz` preserves the complete sanitized terminal,
`trials.json.gz` all five original trial records, `logs.json.gz` all small
logs, and `integrity.json.gz` the frozen 68-file identity map, source
hashes, observations and independent saved-artifact validation. The full
matrix validator still fails on missing cells. `summary.json` binds each
deterministic gzip artifact to compressed/uncompressed SHA-256 values.
Numeric records are preserved; private paths, NAS authority and user
identities are redacted. No weights or activation payloads are included.
