# Qwen 3.8 27B MXFP4 product-health assessment

The live product-health gate passed on MacBook Pro M5 Max 128 GB with
a NAS-backed materialized snapshot. It used the isolated
AX Engine 8.0.7 wheel built from
`0e97f0056a09b259b8a878d618970e31839fb940` and the
[verified MXFP4 inventory](../2026-11-18-qwen38-27b-mxfp4-inventory/).

The default launch, explicit direct and required MTP arms passed 32/32
hard QA checks each and their required HTTP surface checks. The paired
64-token direct/MTP greedy probe matched at every position. Required MTP
drafted 63 tokens, accepted 41 and reported zero direct fallbacks.
[The qualification summary](qualification-summary.json) includes the
synthetic route inputs, outputs and provenance hashes.

Run `scripts/qualify_qwen38_27b.py --run` with the clean source checkout,
the installed wheel and executables, and a build manifest that supplies
the pinned file inventory and hashes.

The checkpoint remains Candidate. MTP-S, MTP-P and MTP-D require
separate evidence.
