# Forced greedy replay regression evidence

This patch repairs a bounded control-flow contract in AX's explicitly forced
singleton replay. It does not change the default relaxed verifier and does not
claim to resolve the observed default Qwen 27B output split.

The failing-before test extracts the production-used acceptance/replay seam
without correcting it, injects deliberately false batched predictions, and
executes real singleton forwarding on a tiny dense-attention model with
token-dependent nonzero KV. The assertion fails because the legacy path
retains three accepted drafts instead of zero at the first mismatch.
`skip-before.log` separately records the failing stale skip-state test.
Fixture compilation/setup failures are excluded from this curated evidence;
they are not defect reproductions.

| Retained files | Meaning |
| --- | --- |
| `before.json`, `before.log`, `before.patch` | Meaningful injected false-acceptance failure; exit 101 |
| `skip-before.json`, `skip-before.log`, `skip-before.patch` | Stale skip-state seam failure; exit 101 |
| `final.json`, `final.patch` | Frozen complete correction, source hashes, formatting and test receipts |
| `final-focused.json`, `final-focused.log` | Three focused tests pass |
| `final-linear-mtp.json`, `final-linear-mtp.log` | Thirteen related tests pass |
| `final-ngram-tests.json`, `final-ngram-tests.log` | Fifty ngram tests pass |
| `final-fmt.log`, `final-diff-check.log` | Successful empty formatting/diff logs, hashes in `final.json` |
| `reference-contract.md`, `reference-identity.json` | Design-only source comparison and five pinned file hashes |

The final three test-run patch files are byte-identical to `final.patch` and
are deduplicated in the publication index. Earlier failing runs retain their
own source/patch identities; they must not be described as final-source runs.
Local filesystem prefixes and unrelated inherited PATH/provider entries in
receipts are replaced with consistent labels. Original log/source hashes
remain in receipts and the publication index; sanitized log bytes have their
own published hashes.

The final tests cover rejection before consumption, later mismatch, complete
and empty drafts, a provisional batched rejection cap, KV contents, returned
predictions, hidden-row selection, accepted-token telemetry, scope restoration,
stale skip-state handling and exclusion of default/sampled/processor paths.
Hidden-row and telemetry assertions use production helpers, while the test
does not execute the full runner or an actual MTP-head refold. Input primary
and initial cache remain preconditions; whole-request direct identity,
hardware-model numerical equivalence, performance and release quality are
not established.
