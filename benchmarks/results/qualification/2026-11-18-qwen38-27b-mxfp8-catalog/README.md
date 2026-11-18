# Qwen 3.8 27B MXFP8 catalog Candidate

`qwen3.8-27b:mxfp8` and `qwen3.8-27b:axq-mxfp8` select
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP8-MTP` at
`11d81dd63869cc80be5c541520071aa9192192b6`.
The 29 published files total 29,541,733,616 bytes. Each file was
checked against the publisher's revision metadata using its LFS SHA-256
or Git blob ID. The [inventory](pack-inventory.json) records each payload's
SHA-256. The [functional receipt](functional.json) also binds the
AX Engine generated manifest and the installed server binary.

On MacBook Pro M5 Max 128 GB with MLX 0.32.4, the AX Engine 8.0.7
release wheel built from `0e97f0056a09b259b8a878d618970e31839fb940`
served the materialized snapshot. The default, direct and required MTP
arms each passed 32/32 hard QA cases and the required HTTP surface checks.
The matrix used the same 16 prompts and streamed and non-streamed requests
as the MXFP4 product-health campaign, with seed 20260917.

The paired raw greedy diagnostic used input token IDs 1 through 16 and a
fixed budget of 64 output tokens with EOS ignored. Direct and required MTP
first differed at output token 30. Required MTP drafted 73 tokens, verified
101, accepted 38 and recorded zero direct fallback steps over 27 decode
steps. The probe does not capture logit margins. Normal launch kept MTP
inactive.

These short functional controls support catalog Candidate admission.
MXFP8 SKU qualification, measured performance and MTP-S/P/D remain open.
