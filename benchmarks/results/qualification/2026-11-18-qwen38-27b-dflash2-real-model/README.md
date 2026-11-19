# Qwen 3.8 27B DFlash2 real-model controls

The three previously skipped real-weight controls ran on MacBook Pro
M5 Max 128 GB with macOS 27.2.1 and MLX 0.32.4. The run verified
all 21 pinned target files, both drafter files, and the six original
MLX-VLM 0.7.11 reference files.

- Target hidden-state capture passed across 15 prompt-layer pairs. The
  lowest mean cosine similarity was 0.99438 against the test's
  greater-than-0.98 requirement. The lowest individual token was 0.77410.
- The MXFP4 drafter matched 119 of 126 reference draft tokens across 18
  rounds. The test requires at least 90% for MXFP4.
- The target LM-head projection probe completed for one through eight
  rows. Its timings are diagnostic.

The [receipt](m5-real-model-tests.json) binds both pack revisions, the
test executable and the original reference checksums. Join each
`log_lines` array with newline characters to recover the exact logged
bytes. This run provides bounded functional and diagnostic evidence.
