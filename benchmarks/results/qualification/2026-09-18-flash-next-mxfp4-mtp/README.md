# Flash Next MXFP4 target MTP controls

Result: **failed exact identity; not release-ready**.

MacBook Pro M5 Max 128 GiB, NAS over SMB, native test source
`2642a6286211ae540c2dd4cdb308f8149987685a`. The frozen supervisor completed
all four jobs and its post-run model/runtime integrity verification. The
separate pair validator rejected the result; successful process exits do not
mean MTP qualification passed.

| Control | Result |
| --- | --- |
| Real head | 94/116 accepted (81.03%); direct/MTP greedy identity failed |
| Permuted head | 0/209 accepted; direct/MTP greedy identity passed |
| Cohort | 104 requests per head, at most 16 generated tokens; 55 short requests excluded from each acceptance aggregate |
| Across heads | 102/104 generated-token arrays identical |
| State | Four generated tokens and two state steps within the frozen tolerance; not byte-exact state |
| Runner | Bounded control passed; decode state was not compared |

The real-head cause/effect request first differs at position 15 (zero-based),
direct token 6745 versus MTP 5073, direct top-two margin 0.125. The roses
request first differs at position 1, tokens 13 versus 271, margin 0.0. These
selected-route native results do not establish the cause of the longer
default-route HTTP QA text differences. No near-tie exempts exact identity.

The frozen oracle has a reporting defect: after the first mismatch it stops
comparison but reports the total generated length. Roses therefore reports
16 compared positions although only positions 0 and 1 were observed. Raw
values remain unchanged; [summary.json](summary.json) records the bounded
coverage separately. Future test code counts unique observed positions,
including the first mismatch, without counting repeated pending-token checks.

[native-controls.json.gz](native-controls.json.gz) retains both heads and
state/runner results. [integrity.json.gz](integrity.json.gz) retains the
supervisor result, frozen identities, original prompts and source lines;
private locations are replaced with stable hash labels. Gzip timestamps are
zero. Artifact hashes are in the summary. No thresholds, IDs, scores or
failed verdicts were changed. This is correctness evidence, not throughput.
