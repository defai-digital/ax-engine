# Flash Next current-source installed QA

This is a complete installed QA evidence collection, not product qualification.
Read [summary.json](summary.json) for the original QA verdict, exact quality
failure IDs, direct/MTP text-pair count and immutable artifact identities.
Every original failed checker remains failed.

Source `5f583018032c406cd3016c6c1c016612dd8b2fbd` ran on a MacBook Pro M5 Max
128 GiB using the original MXFP4 MTP pack over NAS/SMB. Both installed modes
use explicit experimental family admission. The environment, commands, wheel,
source, pack-layout checks and process-cleanup records are in the artifact.
The hardware execution is not relabelled with the later publication commit.

The frozen cohort contains 104 short closed-answer cases and one deterministic
29,774-input-token lookup per mode: 210 responses total, with MTP disabled and
required. Original output caps are 256 tokens for short cases and 64 for the
long case, using natural termination. These QA request timings are not the
128-output-token controlled performance matrix or an MTP-P speed claim.

The original QA protocol jointly checked text identity, checker identity,
normal stops and quality. Its verdict is preserved as `original_qa_passed`.
Under ADR-033, independent direct/MTP text differences are disclosures; they
neither establish nor fail same-state MTP-S safety by themselves. No near-tie
explanation is inferred without measured logits. MTP-S, MTP-P and MTP-D all
remain `not_assessed`; release qualification and default promotion remain off.

`qa.json.gz` contains sanitized original responses, checker results, complete
logs, contract, launch, transfer and independent-review records. The portable
reader reruns every original checker from the exact owned frozen inputs and
checks original budgets, source, commands, MTP activity and cleanup. It does
not rerun the model or rehash remote model weights. Input and checker hashes
are in `reader-inputs.json`; original private-record hashes and sanitized
artifact hashes identify distinct byte streams. Gzip timestamp is zero.

From this directory, without model weights or network access:

```sh
python3 -B verify_evidence.py
python3 -B -m unittest test_verify_evidence.py
```

The public mutation controls reject changed answers, checker verdicts, budgets,
cohort length, source, cleanup, diagnostic environment and unsupported release
or certification claims. Earlier quality, numerical, timeout and delivery
failures retain their original verdicts. Full safety, reference/performance,
memory, fresh-delivery and product requirements remain separate work.
