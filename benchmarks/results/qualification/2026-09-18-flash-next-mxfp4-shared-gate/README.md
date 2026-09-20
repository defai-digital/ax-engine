# Flash Next MXFP4 shared-gate attribution

Supplementary Apple M2 Ultra 192 GiB evidence for source `6d2fdbf5fe1730c4b3b929ab89ae63685ba4afca` and pack `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35`. Qualification and release readiness remain false.

The original layer-33 collection retains 614 tensor summaries, complete control/capture identity, both accepted pairs and 47 pre/post model hashes. Its five-token control still has a stored-state failure at generated prefix 4, layer 34, with 32 differing arrays.

Prefix 41 is an exact-output control. At prefix 43, captured exact BF16 MLP inputs produce one shared-gate difference at output index 524: -0.482421875 versus -0.48046875, maximum 0.001953125. Original packed weight/scales replay reproduces every native batch and singleton output. Repetitions are exact; contiguous rowwise projection equals independent singletons for both groups. Shared-down inputs already differ, so its later differences are not an independent attribution.

The replay rehashes the complete selected shard and index before/after. Its other 47-file identity metadata is inherited from the original capture; the local curator does not reread remote weights. Runtime, source, header, payload hashes and canonical snapshot-link identities are retained.

The separately bound candidate control reports 5 equal generated tokens; stored states are still different. Only consumed prefixes [2, 4] were compared. Its complete native result and any continued failure remain in candidate-control.json.gz; no state-equality condition was imposed to admit this evidence.

Four deterministic gzip artifacts preserve sanitized analysis, replay, collection and candidate records, all numeric values and dtypes, integrity metadata and bounded changed-value examples. They contain no captured tensor payloads or model weights. summary.json provides compressed/uncompressed hashes and numeric counts.

This is a short diagnostic, not a full-model correction, M5/NAS installed-default, quality, throughput, lifecycle or release qualification.
