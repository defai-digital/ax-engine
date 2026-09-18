# Flash Next installed lifecycle after the active-stream fix

Result: **PASS for the fourteen-request installed lifecycle only. Product qualification remains open.**

Source `5d028881` and wheel `96509d63` ran on the M5 Max 128 GiB MXFP4/NAS target with explicit experimental Flash Next family admission. Here default means unchanged MTP/paging policy within that opt-in, not default family certification. Both default and MTP-required modes complete baseline, full SSE, one/two-token budgets, stop, an unfinished disconnect and identical recovery. Raw text, finish reasons and three token totals match; request-specific cached-token metadata is validated separately. Active-stream count is positive before disconnect and work drains afterward. All 47 model payloads and 52 installed runtime files pass the original pre/post checks; both servers and the worker are gone.

The bundled record retains raw responses/events, metrics, contracts, installation receipts, logs and independent root recomputation. The new wheel passes 81 isolated local packaging tests. Exact-source CI succeeds in eight jobs, but all eight real-weight Run steps are skipped. Strict Clippy still has 1,798 unchanged baseline diagnostics. None of these facts closes the full QA, broader MTP, memory, performance, primary-reference or fresh-delivery gates.

Prior native numerical evidence remains source `bf06cbb2`. A pinned comparison of complete unchanged numerical/runtime Git trees permits testing this new server; it does not relabel an old numerical run. The earlier cache-comparison and active-stream failures remain failed in [their original evidence bundle](../2026-09-18-flash-next-mxfp4-lifecycle/README.md). No prompt, token budget, timeout or active-producer requirement was relaxed.

Artifact identity is in [summary.json](summary.json). Private locations are stable hash labels; numeric/boolean/null values are preserved. Gzip timestamp is zero. No release or default-MTP promotion is implied.
