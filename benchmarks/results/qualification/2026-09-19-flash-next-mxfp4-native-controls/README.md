# Flash Next current-source native MTP controls

Result: **PASS for the original bounded native controls only. Product qualification remains open.**

Source `5f583018032c406cd3016c6c1c016612dd8b2fbd` ran on the MacBook Pro M5 Max 128 GiB with the original MXFP4 MTP pack over NAS/SMB. This uses experimental family admission, forced expert paging, a one-layer expert cache, selected expert/prefill routes and the canonical singleton MTP candidate. The runtime environment, source, wheel and native binary identities are retained in the bundle. This is not installed-default or performance evidence.

Both trained and permuted heads complete the original 104 inputs with a 16-token cap and the original EOS rules. Each excludes 55 short requests and scores 49. The trained head accepts **95/117 (81.20%)** proposals; the permuted head accepts **0/209**. The original >=70% and <10% thresholds pass, with exact paired generated-token identity. Independent reconstruction of all recorded verifier events finds zero invalid same-state acceptances. The permuted control observes no accepted path; rejected draft logits are not independently recomputed.

The state control compares four emitted tokens, two state steps and 109 arrays, and retains forced acceptance/rejection, budget, EOS and zero-budget checks. The runner control matches three tokens but does **not** compare prefill or decode state. Original checks cover all 47 payload digests before and after, 52 installed runtime files and termination of all six owned processes. The live native library observation binds `libmlx` and `libjaccl` to the current installation; it does not prove GPU metallib mapping.

MTP-S, MTP-P and MTP-D remain **not_assessed**. Installed/default routing, broader safety and rollback coverage, full QA, reference agreement, performance, memory and fresh delivery remain separate requirements. Earlier failed native and QA records retain their original verdicts. No release, speed multiplier or default promotion follows from these controls.

`native-controls.json.gz` retains sanitized raw results, complete logs, prompt IDs, execution/installation receipts and the independent review. Private locations use stable hash labels; numeric, boolean and null values are preserved. Original input hashes identify the private records before sanitization. Gzip timestamp is zero. The offline reader recomputes the public trace, state and paired-head assertions; it does not rerun the model or independently rehash remote weights.

From this directory:

```sh
python3 -B verify_evidence.py
python3 -B -m unittest test_verify_evidence.py
```

Artifact identities and scoped counts are in [summary.json](summary.json).
