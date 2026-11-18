# Qwen 3.8 Flash Next MXFP8 functional preview

The [publisher inventory](pack-inventory.json) binds all 61 files of
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP8-MTP` at
`48b9aa5f002f29a9f8ab7cfe5c7185e342595b1b`. Files were verified against
their LFS SHA-256 or Git blob ID and the materialized file tree was
used to generate and validate an AX Engine manifest.

On MacBook Pro M5 Max 128 GB with MLX 0.32.4, Auto residency selected
expert streaming. A single-request control enabled selected expert
reads and supplied one raw input token with a four-token greedy budget.
Direct and required MTP returned the same four tokens. The required
route ran two verify steps with zero direct fallbacks. Both proposed
drafts were rejected. Each arm reported 192 selected expert gathers
and 9,504,000 KiB of selected payload reads over the request.
The [paged receipt](m5-auto-paged.json) records the exact inputs,
preview flags and development server binary hash.

The [follow-up](m5-auto-paged-acceptance.json) used four raw input
tokens and an eight-token budget. The two routes matched again.
Required MTP accepted two of four drafts and reported zero direct
fallbacks.

The [resident direct control](m2-resident-direct.json) completed
twelve tokens on Mac Studio M2 Ultra 192 GB with expert streaming off.
The resident and paged controls retain separate development binary hashes.

The preview uses the canonical singleton MTP verifier. The format's
catalog admission, published performance, SKU qualification and
MTP-S/P/D remain open.
