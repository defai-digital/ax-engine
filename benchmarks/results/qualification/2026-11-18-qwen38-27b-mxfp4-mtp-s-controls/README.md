# Qwen 3.8 27B MXFP4 same-state greedy verifier controls

The ignored Rust test
`runner::tests::qwen38_27b_greedy_mtp_commits_same_state_verifier_choices`
ran on MacBook Pro M5 Max 128 GB with Rust 1.99.1, MLX 0.32.4 and the
[pinned MXFP4 pack](../2026-11-18-qwen38-27b-mxfp4-inventory/).
It used native requested MTP with DFlash2 disabled and input IDs 1
through 16. The [receipt](m5-greedy-audit.json) binds the test executable
and model manifest.

The test computes the first maximum of each target-head verifier row
on the host after production acceptance. All 69 accepted drafts
matched their same-state target decisions across 38 verified cycles.
The fixed-budget request emitted 64 tokens and continued after rejection.
The EOS case stopped after 39 tokens. The short-budget request emitted
exactly seven tokens.

This is bounded MTP-S evidence for one unprocessed, greedy raw-token
input on the pinned pack. Broader MTP-S assessment, MTP-P performance
qualification and MTP-D default promotion remain open.
