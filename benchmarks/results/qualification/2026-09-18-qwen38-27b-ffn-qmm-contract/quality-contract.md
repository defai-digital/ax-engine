# LINE_SET score interpretation

The retained **0/12 strict exact-set matches per route** is accurate for the
selected 512-token diagnostic. It must not be described as original source-task
accuracy or as twelve semantic vulnerability-localization failures.

The imported source questions ask for a representative primary-bug location
or a small necessary group. Their native matcher accepts a nonempty reported
set contained within the audited accepted-location key. AX preserves the
question wording but deliberately uses a stricter status: equality with the
entire key. AX separately records accepted-location subset detection. The
source-native question and its matcher are coherent; the two evaluation
contracts measure different things.

This interpretation is historically grounded. The full `ds4_eval.c` hash in
both retained imported campaign banks and manifests matches the clean local
source containing the subset matcher. The matcher dispatches by declared
LINE_SET answer kind, including an explicitly declared non-COMPSEC case.
This is source evidence, not a claim that AX ran the external grader.

The audit verified all 24 retained request dictionaries and gold fields against
the unchanged selected-case bank and documented diagnostic request builder.
Each route contains the same 12 unique cases: eleven COMPSEC and one NIST Juliet.
The requests retain the original questions and use the existing answer-only
system instruction, 512-token cap, greedy sampling, and disabled thinking.
This workload differs from the original long-thinking campaign.

| Recorded metric | Direct | MTP |
| --- | --- | --- |
| Strict exact-set matches | 0/12 | 0/12 |
| Strict wrong | 7 | 7 |
| Truncated | 5 | 5 |
| Existing accepted-location subset detections | 7/12 | 7/12 |

The subset count is an existing AX diagnostic field, not a retroactive grade
change or an end-to-end native-scoring result. Completion and formatting still
matter. Truncated responses remain bounded-completion failures. This selected
failure subset cannot estimate overall model accuracy.

No harness, gold, historical status, runtime default, or release gate changed
as part of this audit. Reporting should name strict exact-set equality and
accepted-location detection separately. Exact source identities, historical
hash matches, retained-artifact identities and counts are in
[quality-contract-audit.json](quality-contract-audit.json).

A separate two-case `complete_set_v1` variant replaces exactly one
location-reporting instruction with a stricter complete relevant-line-set task.
It permits noncontiguous root-cause and affected-operation locations without
requiring unrelated intervening lines. Code, safety instructions, output
formatting, sampling settings and gold remain unchanged. This is a stricter
new diagnostic task, not a repair of the source-native location contract or a
replacement qualification. Reusing an accepted-location key does not prove it
is an exhaustive causal-line annotation. Execution results must be reported
separately from this preparation audit.

The completed d121f107 cold experiment is recorded separately in
[quality-contract-controls.json](quality-contract-controls.json). All eight
requests completed: each original route has one truncation and one strict-wrong
accepted-location answer; each stricter variant route has two strict-wrong
answers. This selected experiment does not replace the retained twelve-case
scores or establish a release-quality accuracy bar.

The two stricter prompts were also run on the pinned independent direct
reference after full prompt/token parity with installed AX was verified.
Both answers, finish reasons, output counts and unchanged grades match both
AX routes. This places those selected observations in the reference as well;
it does not establish correctness or identify the longer route split's cause.
See [independent-reference.json](independent-reference.json) for artifact
identities, input hashes, results and measurement limits.
