# Flash Next MXFP4 historical four-graph numerical evidence

The frozen numerical bounds passed for eight inputs and 3,316 positions per
graph. AX data was collected on Apple M5 Max 128 GiB with NAS storage from
source `2642a6286211ae540c2dd4cdb308f8149987685a`, native binary `9c1d319a...`,
using the pinned installed libraries from source `85a2bab0...`.
The AX route forced streaming, one retained expert layer, selected expert reads,
selected prefill with capacity fallback, and a 128-token prefill chunk.
It does not establish acceptance of source `358bac91...`, the default installed
route, MTP, full QA, throughput, lifecycle, delivery or a release.

Three independently collected reference graphs and the retained AX graph use
pack revision `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35`. CPU scoring ran on
Apple M2 Ultra after collection. The reference is official-chunked; the numerical
floor is official-recurrent. MLX-VLM is an additional comparison, not a replacement
for the required primary benchmark.

| Frozen measure | AX | Limit |
| --- | ---: | ---: |
| Mean KL | 0.018450284676548776 | 0.0253247336844727 |
| Top-1 disagreement rate | 0.08172496984318456 | 0.10705669481302775 |
| AX-only high-margin rate | 3 / 3316 = 0.0009047044632086852 | 0.01 |

The first two limits are 1.25 times the official-recurrent floor. High margin
means reference margin strictly greater than 1.0. All 101 AX-only top-1
disagreements remain in the statistics, including all three high-margin cases.
AX-only means AX differs while official-recurrent agrees with official-chunked;
MLX-VLM also differs in two of those three high-margin cases and agrees in one.
Passing these bounds is not exact token or state parity.

`statistics.json.gz` retains every numeric field and all 9,948 comparison rows.
`integrity.json.gz` retains collection metadata, the frozen contract, source and
payload identities, terminal records, transfer history and both analysis logs.
All 64 array/sidecar hashes matched before and after scoring; the 17-file AX
transfer comprised 3,294,517,142 bytes. These are saved observation records,
not a new remote integrity or process check performed during curation.
Historical pending wording in frozen collection records remains intact;
the completed analysis terminal provides the later numerical disposition.

`summary.json` records full source identities, exact bounds, scope and artifact
hashes. Gzip headers have `mtime=0`. Private locations and user strings are
replaced with stable hash labels; numeric JSON fields and verdicts are unchanged.
No model weights or logit arrays are published. Qualification, performance and
release claims remain false.
