# Historical AX-only MTP rerun

This evidence set is retained for diagnosis and reproducibility, but it is
superseded and is not eligible for README, release, or current-runtime
performance claims.

All completed aggregates were produced from commit `44a2c6c7`, before the
native generation service and engine failure/batched-state hardening changes.
They used one warmup plus five measured repetitions. Current publication
evidence requires a clean build, at least two warmups, five measured runs, and
the current native service path.

`evidence-set.json` is the machine-readable authority for this directory. The
Gemma 4 31B aggregate remains intentionally marked incomplete and must not be
filled or interpreted as a completed row without a fresh rerun.
