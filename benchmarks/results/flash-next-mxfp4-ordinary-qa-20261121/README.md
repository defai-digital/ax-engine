# Flash Next 32K ordinary-launch QA on M5 Max, 2026-11-21

MacBook Pro M5 Max 128 GB, macOS 27.2.1, internal SSD, Rust 1.99.1 and
MLX 0.32.4. The `release-server` executable from `b36b7b287` and the
pinned MXFP4 pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3` were
hash-bound before and after the replay. The
[cohort runner](../../../scripts/run_flash_next_qa_cohort.py) ran with
`--ordinary-launch` and SHA-256
`a5abdec748e75d545372e52ce6dba7c2746d88d4e468f0ed16b5152b63f8a02a`.
The [startup controls](../flash-next-mxfp4-startup-envelope-20261121/README.md)
exercise the same runtime build's admission and disconnect behavior.

| Serving arm | Advertised context | Graded passes | Normal stops | First readiness | 29,774-token lookup | Sampled RSS peak after readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Explicit direct control | 32,768 | 101/105 | 105/105 | 46.4 s | 43.2 s | 78.71 GiB |
| Ordinary startup | 32,768 | 101/105 | 105/105 | 46.2 s | 41.9 s | 78.77 GiB |

The two arms match all 105 response texts. Both recorded zero MTP verifier
steps and zero route errors. The four checker failures match both AX arms
of the [signed v7.9.0 run](../qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/README.md).
The independent MLX-VLM run in that archive also fails the same four
items. The [result](result.json) retains the case IDs, grades, runtime
hashes, and per-arm advertised model metadata. The compressed
[responses](responses.json.gz) retain the 210 HTTP output rows.

CPU idle measured 90.40% before the replay and 88.76% afterward. Timing
and RSS values are observations. The signed v7.9.0 record remains the
release qualification. Flash Next remains Candidate.
