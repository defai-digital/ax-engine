# Flash Next MXFP4 QSA query/output attribution

Supplementary Apple M2 Ultra 192 GiB evidence for source `358bac91d5340e2b45b55a14e6cfb13b5fb5c63f`,
before the Q/output correction, with pack `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35`.
Qualification and release readiness remain false. A separately bound short
candidate control is complete and still fails state parity.

The unchanged five-token roses capture contains 470 arrays, all 47 pre/post
model hashes, identical control/capture results, both accepted pairs and the
original first-state failure at generated prefix 4, layer 8 (`gdn.conv`,
90 differing arrays). Five equal tokens do not establish state parity.

At inherited prefix 41, exact gated inputs produce one different QSA output
projection value. At prefix 43, exact HC inputs produce one different packed
Q/gate value (query half). Both maxima are 0.0001220703125. Original-weight
MLX 0.32.2 replays reproduce every native batch and singleton projection value
exactly; repeat schedules and rowwise/singleton controls are exact. K/V caches
are exact within and across the two groups. Prefix-41 residual writes remain
exact; prefix-43 residual writes differ. Prefix-43 O inputs already differ, so
its six output differences are not independently attributed to O scheduling.

The first probe attempt stopped before any GPU operator because canonical HF
snapshot blob links were rejected. Its failure metadata and source/log hashes
are retained in integrity.json.gz. A bounded path-admission fix preserved
link/target/content identities and passed 28 CPU controls locally and on M2;
both subsequent serial projection replays exited 0. No operator or environment
change was made to obtain those replay results.

The five deterministic gzip artifacts retain sanitized component/replay/control
records, numeric summaries, selected changed scalars and integrity metadata.
They contain no model weights or captured activation payloads. summary.json
records compressed/uncompressed hashes and numeric-preservation counts.

candidate-control.json.gz separately binds the Q/output correction to base
`7d0d2727d680e1e131e6c3d27c228c7757234358` plus its three-file source patch,
frozen release binary, runtime, and all 47 matching pre/post model hashes. Its
whole prefill result, five generated IDs, and both accepted pairs match the
pre-correction control. State prefix 2 remains exact; at prefix 4, the first
stored difference moves from layer 8 (90 arrays) to layer 34 (`gdn.conv`,
maximum 0.03125, 32 arrays). This is partial progress, not state parity. Five
tokens were emitted but only four were consumed; prefix 5 was not compared.
This candidate control does not replace the pre-correction attribution.

The
[full pre-correction diagnostic failure](../2026-09-18-flash-next-mxfp4-qsa-full/README.md)
remains retained. The candidate has no full-diagnostic result in this bundle.
This artifact does not establish M5/NAS installed behavior, quality, lifecycle,
throughput or release readiness.
