# Flash Next MXFP4: M5 original-full numerical diagnostic

**Collection valid; numerical identity failed.** This is the completed native
collection of the original two prompts, each capped at 256 new tokens. The
collector stops at the first greedy mismatch: it establishes 15 and 6 equal
generated positions, not two 256-token matching completions.

The target was Apple M5 Max, 128 GiB, using the recovered pinned MXFP4 pack on
NAS. Native binary `c7cceff3` corresponds to AX source
`cd207324af669f37d5ebfe29430c0e4ea41f9a81`. It used the installed MLX 0.32.2
libraries, experimental Flash Next admission, forced expert streaming and a
one-layer expert cache. This is not the installed default HTTP route or an
MTP release qualification. The source is distinct from the old installed
`85a2bab0` delivery/runtime evidence; using its admitted MLX libraries does not
make this an installed current-source wheel test.

| Original request | Input IDs | Equal generated positions | First token difference (zero-based) | Earliest stored-state difference |
| --- | ---: | ---: | --- | --- |
| `reasoning_cause_effect` | 46 | 15 | Position 15, bonus: direct 6745 / MTP 5073; direct top-two margin 0.125 | Consumed prefix 2; layer 1 `ple.conv`, BF16 `[1,9,10240]`, max absolute difference `2.384185791015625e-7`; 105/109 state arrays differ |
| `reasoning_syllogism_roses` | 41 | 6 | Position 6, correction: direct 198 / MTP 271; direct top-two margin 0 | Consumed prefix 3; layer 2 `gdn.conv`, BF16 `[1,3,10240]`, max absolute difference `0.03125`; 104/109 state arrays differ |

Both prompt prefills have 109 exact state arrays and exact repeat-prefill
controls. The cause/effect collector compares common states at consumed
prefixes 2, 4, 6, 8, 10, 11 and 13; the roses collector compares prefixes 1, 3
and 5. Their mismatch windows are retained in full, including tokens that
were proposed/consumed by the candidate beyond the equal ordinary prefix.
Those candidate tokens are not counted as additional matching output.

The retained checkpoint replays reproduce their respective sessions, ordinary
trajectory and checkpoint singleton controls. Their starting MTP states already
differ from the ordinary trajectory. Matching an argmax within that checkpoint
therefore does not repair the earlier divergence. The roses zero-margin tie
remains a failure. Earliest stored-state differences do not establish exact
incoming hidden inputs or identify a PLE, GDN, projection, paging or storage
cause. The evidence does not select a numerical implementation change.

`native-diagnostic.json.gz` retains the entire native JSON, including token IDs,
logits summaries, dtypes, shapes, all state rows, windows, checkpoint replays and
tensor metadata. Its decoded JSON equals the original record exactly.
`integrity.json.gz` retains sanitized worker/supervisor/freeze/transfer receipts,
root and independent terminal reviews, source/build/pack metadata, preflight,
launch, commands, environment, loaded dependencies and complete native/launch/
supervisor log text. Numeric types/values and hashes are preserved; private
paths, host/user identifiers and sensitive text are sanitized with collision
checks. The unrelated whole-host process-listing string is retained only as a
SHA-256 marker; structured owned-process commands, PID and cleanup records remain.
No weights, captured arrays, executable, source patch or crash payload
is published. The raw stopped M2 full attempt remains separate
[context from the short evidence](../2026-09-18-flash-next-mxfp4-m5-short/separate-m2-context.json).

The retained before/after receipts match all 47 original model payloads plus
the generated manifest, and preserve the exact 48-entry canonical snapshot
layout. Model payload size is 132,261,853,669 bytes; this is content size, not
network transfer volume. The local curation rechecks retained receipts and
source identities; it does not repeat target payload hashing. Target runtime
pins were checked before/after by the frozen worker and independently recorded
in the final transfer receipt. Native, worker and supervisor exited 0 without
signals; owned processes were reaped and their groups observed absent. Exit 0
means the diagnostic collection completed, not that numerical identity passed.

The frozen anchored dyld parser records 549 paths. Independent parsing of every
occurrence finds 550; the extra is the allowed system AGX Metal driver. This
limitation remains explicit instead of claiming every loaded path was admitted
by the original parser.

| Identity | Value |
| --- | --- |
| Pack | `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` |
| Revision | `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35` |
| Native binary SHA-256 | `c7cceff3aba9163ce5444c63481dd6cf208210e777743bf8526b1ac70295b457` |
| Generated manifest SHA-256 | `4252bb87db718f88e7e2eae3a08abe7e118999ffe8b664e301574348b5195512` |
| Collector freeze SHA-256 | `310bdb2505309f95a2f7c647c600fb04ea2c8a85bf73d21dab5356a68b36b50f` |
| Original native JSON SHA-256 | `6e22b0ca63f824c9b88deec05daca90e7c7e6f27e55febd7f70ea45ee2abe6b4` |

`summary.json` binds both deterministic gzip artifacts with compressed and
decoded hashes/sizes. Gzip filename is empty and mtime is zero. Retained timings
are collection observations, not throughput comparisons. Default admission,
quality, lifecycle, memory, performance and release gates remain open; this
failed diagnostic closes none of them.
