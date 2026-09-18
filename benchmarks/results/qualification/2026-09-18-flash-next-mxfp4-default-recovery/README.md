# Flash Next MXFP4 installed default recovery

The installed AX Engine source `85a2bab0dd72c9e2076f455237dca052ff8d12b7`
completed one default-transport recovery of an existing partial NAS cache on
Apple M5 Max with 128 GiB unified memory. All 47 published files matched the
original expected sizes and SHA-256 values, totaling 132,261,853,669 bytes.
The generated native manifest was verified separately. Installed runtime and
dependency identities were unchanged before and after; the CLI exited 0,
its child was reaped without forced termination, and retained process
observations confirm the supervisor, CLI and helper had ended.

This is **recovery success**, with fresh-cache delivery still unqualified.
The original fresh default attempt and first default recovery both failed and
remain in the evidence. The successful invocation used the same installed
CLI, pinned revision, original partial cache and default transport settings;
no timeout, worker-count or Xet override was introduced.

| Identity | Value |
| --- | --- |
| Repository | `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` |
| Revision | `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35` |
| Wheel SHA-256 | `b05b1ef032761bd4d05f3c9401621618f70bf2f2ba151f489ab7e364359275f3` |
| Generated manifest SHA-256 | `4252bb87db718f88e7e2eae3a08abe7e118999ffe8b664e301574348b5195512` |
| Original terminal SHA-256 | `285f680e336e3ccd60ac5bd696b159c26e7098bfe6ef1eb6db0528247f2d5fba` |

Before the successful recovery, 31 published snapshot entries already existed;
their recorded metadata remained unchanged afterward. The final snapshot had
47 verified published members plus one separately generated native manifest.
These counts and verified payload sizes are not measurements of newly
transferred network bytes. No fresh-empty-cache or download-throughput claim
follows from them.

The observed CLI interval was 372.420701 seconds; the overall interval,
including complete payload verification, was 673.804133 seconds. These are
invocation/verification durations, not pure network timings or a performance
benchmark. All model, MTP, lifecycle, paging, current-candidate and release
qualification remains open; this evidence belongs only to the installed
source identified above.

- [summary.json](summary.json): scope, verdict and artifact integrity.
- [recovery.json.gz](recovery.json.gz): whitelisted terminal fields, both
  complete expected/verified 47-file inventories, all installed runtime
  hashes before/after, snapshot metadata, cleanup and retained failures.
- [integrity.json.gz](integrity.json.gz): original source, transfer and
  validation identities. Runtime paths use unique venv-relative labels;
  private machine paths and unselected raw text/logs are omitted.

The JSON gzip artifacts are deterministic with mtime zero. No model weights
or array payloads are included. `qualification=false` and
`release_ready=false` are preserved.
