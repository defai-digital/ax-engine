# Qwen 3.8 Flash Next

Status: **Candidate; release qualification open**

Second SKU. MXFP4 MTP target; native support and checkpoint qualification pending. MTP Tier 2 pending. AX certification record: Candidate (gates open).

Target SKU: **Mac Studio, Ultra-class Apple Silicon (M2 Ultra or newer), 192 GB+**
(corrected from the original MacBook Pro M5 Max 128 GB target by
[ADR-037](../../.internal/adr/ADR-037-FLASH-NEXT-ULTRA-192GB-TARGET-CORRECTION.md),
2026-09-22; campaign host `um-macstudio-m2`). Target pack: **MXFP4 MTP**.
Historical affine real-pack evidence is from **Apple M2 Ultra, 192 GB**. The
M5 Max 128 GB evidence recorded below remains historical evidence for the
Auto expert-paging code path; it does not by itself qualify the corrected
target. Last reviewed: **2026-09-23**.

On 2026-09-23 the MXFP4 MTP direct-fallback behaviour was root-caused and
instrumented. The draft cursor is created only on a genuine cold-start prefill,
so a request whose prefill reuses a cached prefix decodes direct for its whole
duration — deterministically, not with the per-step probability an earlier
throughput note implied; that note is corrected in place. A cursor-restore path
(Spec A/B) and per-reason MTP `/metrics` telemetry landed as engine
observability. **No gate closes from any of this.** MTP-S/P-D remain
`not_assessed`, default admission stays fail-closed, no default-MTP promotion
happens, and no throughput or acceleration claim is made. Local (non-public)
detail: `.internal/reports/flash-next-mtp-fallback-review-20260923/`.

Later the same day the MTP attach-failure signal became observable:
`ax_engine_flash_next_mtp_attach_failed_total` now publishes the engine's
`ax_mlx_flash_next_mtp_attach_failed` step counter, so an operator who
requests MTP and sees no verification can tell "the draft head never
attached" apart from "attached but every step blocked". A checked-in
omlx/MTPLX/ds4 peer-benchmark contract
(`scripts/flash_next_peer_bench_plan.json`, checked by
`scripts/check_flash_next_peer_bench_plan.py`) fixes the comparison host, pack
revision and required `/metrics` series, and fails closed when its
preconditions are absent. **No gate closes from either change; the peer
numbers remain unverified because no Flash Next weights and no Ultra-class
host exist on the authoring machine.**

"MTP Tier 2 pending" here uses the same three-gate vocabulary as the
[Qwen 3.8 27B record](qwen3.8-27b-axq.md#what-mtp-tier-2-pending-means):
MTP-S (in-path safety), MTP-P (performance claim license) and MTP-D (default
promotion) are evidenced separately. For this SKU all three are open. Default
admission remains fail-closed; the opt-in experimental evidence below does not
by itself close any gate.

## Current target and open gates

The target is
[`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP/tree/0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35)
at revision `0b0bf6c1603054df4a8eef0d4bc96bd4672d2c35`. Published file
metadata totals 132,261,853,669 bytes. Default MXFP4/group32 coexists with
per-tensor affine8/group32 overrides and an affine8/group64 output head;
the protected MTP sidecar is BF16.
The current campaign uses a NAS-backed Hugging Face cache over SMB; storage
conditions are part of qualification. MXFP4 paging requires the explicit
`AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1` opt-in. Small generated-tensor controls
cover bounded U8 scale reads, full-layer/selected-row equivalence, malformed
layouts and I/O recovery; target whole-model qualification is still open.

[Pinned pack audit](../../benchmarks/results/qualification/2026-09-17-flash-next-mxfp4-paging/pack-audit.json)
records full file hashes and quantization geometry. Successful Hub staging does
not establish installed AX delivery or model execution.

[Installed default recovery](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-default-recovery/README.md)
now completes on the M5/NAS target using source `85a2bab0`: all 47 published
file hashes and the generated native manifest match, runtime identities remain
unchanged, and the owned processes have exited. The invocation reused the
original partial cache with unchanged default transport settings. Both earlier
failed attempts remain recorded; this recovery does not qualify fresh-cache
delivery, later runtime candidates, model execution or lifecycle behavior.

Installed M5 fixed-input controls now pass both direct and required MTP on the
[selected route](../../benchmarks/results/qualification/2026-09-17-flash-next-mxfp4-paging/native-selected-r2.json)
and [default Auto route](../../benchmarks/results/qualification/2026-09-17-flash-next-mxfp4-paging/native-default-r2.json),
including post-run file hashes and identical text/usage (46 input, 32 output
tokens). These four requests do not establish full QA, numerical certification,
or throughput. The required Auto request includes intrusive stack sampling;
its elapsed time is not benchmark evidence.

[Current-source installed QA](../../benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-qa/README.md)
on source `5f583018` completes all 105 original inputs in each mode, with 210
normal stops and 105/105 identical direct/MTP text pairs. Each mode still has
102 quality passes and three original failures: alphabet comma spacing, the
gravity answer `9`, and the water formula's subscript representation. Both modes
return `1734` for the 29,774-token lookup. Original payload/runtime integrity
and owned-process cleanup pass; the original QA verdict remains **failed**.
The portable evidence reader independently replays the unchanged checkers.
These request timings do not establish MTP acceleration, and text identity
does not by itself establish same-state MTP-S safety.

The [earlier installed target QA](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-target-qa/README.md)
retains its separate historical result: all 105 original inputs in each mode, with 210 normal stops
and clean shutdowns. Each mode has 102 quality passes and three retained
failures. Direct/MTP text matches on 103/105 pairs; two reasoning answers
differ despite identical checker results. Both modes pass the 29,774-token
lookup. The frozen supervisor completed all pre/post integrity checks, but
the overall QA verdict is **failed**. Generated token IDs and logit margins
were not recorded by this endpoint; a near-tie explanation is not established.

[Current-source native controls](../../benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-native-controls/README.md)
pass their original bounded contract on source `5f583018`: real-head acceptance
is 95/117 and the permuted control is 0/209, with all 104 paired outputs equal.
Independent same-state trace reconstruction finds zero invalid acceptances.
The state control covers two steps and 109 arrays; the three-token runner
control does not compare stored state. All original payload/runtime integrity
checks and owned-process cleanup pass. This selected-route diagnostic retains
MTP-S/P/D as `not_assessed`; installed/default and full product qualification
remain open. Earlier results below retain their original source and failures.

[Current-source installed lifecycle](../../benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-lifecycle/README.md)
also passes its fourteen-action contract on source `5f583018`: default and
required MTP complete SSE, budgets, stop, active-producer disconnect and
identical recovery. Original payload/runtime integrity and clean owned-process
exit pass. This uses the installed paging defaults within experimental family
admission. The current QA collection above is complete with retained quality
failures; broader numerical coverage, memory, performance and fresh delivery
remain open. MTP-S/P/D are still `not_assessed`.

[Target native MTP controls](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-mtp/README.md)
are also complete and **failed exact identity**. Real-head acceptance is
94/116 (81.03%), while the permuted head accepts 0/209. Both use 104 inputs
with 55 short exclusions; only 102/104 generated-token arrays agree across
heads. Bounded state/runner controls and post-run integrity verification
passed. The recorded coverage overstatement after a first mismatch is
preserved and explained in the evidence; it does not waive the failure.

[Supplementary MXFP4 GDN attribution](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-gdn/README.md)
reproduces batch/singleton rounding differences with actual BF16 inputs and
original QKV/output weights. Verifier-only per-row projections move the first
changed stored state from layer 0 to layer 43 in the original three-token
roses control. That GDN-only candidate retains eleven differing state arrays;
this is a component correction, not whole-model or M5 identity qualification.
Direct and affine routing retain
their existing policies, and default admission remains closed.

[Supplementary MXFP4 QSA attribution](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-qsa/README.md)
isolates main-key and indexer projection scheduling differences on identical
captured inputs. The main-key replay exactly reproduces the native outputs;
the indexer replay has no captured native projection. Applying the verifier
policy to these two MXFP4 projections leaves no recorded state difference in
the same three-token control, with all three output tokens equal and all 47
pack hashes unchanged. This bounded M2 result does not establish longer
trajectory identity or qualify the M5/NAS target; those gates remain open.

[The longer QSA diagnostic](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-qsa-full/README.md)
retains a failed identity result on that candidate. Cause/effect matches 89
token positions including EOS, but stored state differs after prefix 2.
Roses first differs in stored state after prefix 4 and in output at position
13. Both prefill comparisons are exact and all 47 pack pre/post hashes match.
The passing three-token control therefore does not close whole-model identity.

[Supplementary QSA query/output attribution](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-qsa-query-output/README.md)
reproduces two further projection differences with identical captured inputs
and original weights. Applying the existing verifier policy to MXFP4 Q/gate
and output projections preserves all five generated tokens, both accepted
pairs and the repeated prefill in a separate M2 control. At consumed prefix 4,
the first stored-state difference moves from layer 8 to layer 34; 32 arrays
still differ. This candidate has not completed the full diagnostic or M5/NAS
qualification. The earlier failures remain retained, and default admission
remains closed.

[Supplementary shared-gate attribution](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-shared-gate/README.md)
reproduces one further MXFP4 projection difference on identical captured inputs
and original weights. The verifier correction preserves all five output tokens,
both accepted pairs and prefill in the final-source M2 control; all 47 pre/post
pack hashes match. At consumed prefix 4, the first stored-state difference
moves from layer 34 to layer 41, with 16 arrays still different. Only prefixes
2 and 4 were compared. Complete state identity therefore still fails, and the
original M2 full diagnostic stopped incomplete with native SIGBUS and no
post-run payload hashes; it supplies no full numerical verdict. M5/NAS
installed qualification, performance, lifecycle and default-MTP promotion
remain open.

[The same-source M5 short diagnostic](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-m5-short/README.md)
now completes with valid collection and failed state identity. The 41-input,
five-output-token control preserves all five greedy IDs and exact prefill,
but state first differs at consumed prefix 3, zero-based layer 2, across
104 of 109 arrays. Prefixes 1, 3 and 4 were compared; the fifth emitted token
was not consumed by a further forward pass. All 47 model pre/post hashes,
the generated manifest and runtime identities match, and owned processes
exited cleanly. This forced-streaming native diagnostic uses a one-layer
expert cache; it does not qualify installed default behavior or identify a
particular faulty operator. Full target identity remains open.

[The same-source M5 original-full diagnostic](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-m5-full/README.md)
also completes with valid collection and failed numerical identity. The two
original requests have 256-token caps and stop at their first greedy mismatch:
cause/effect matches 15 positions, then differs at zero-based position 15
(bonus, direct 6745 / MTP 5073); roses matches six positions, then differs at
position 6 (correction, direct 198 / MTP 271). Their earliest stored-state
differences are consumed prefix 2, layer 1 `ple.conv` (105/109 arrays), and
prefix 3, layer 2 `gdn.conv` (104/109 arrays). Both prefills are exact.
All 47 pre/post payload hashes, manifest, runtime/source and process cleanup
checks pass. These forced-streaming native results retain the zero-margin
roses mismatch as a failure; they do not identify an operator cause or qualify
installed default behavior, MTP, performance or release readiness.

The current implementation candidate uses ordinary single-token target forwards
for both MTP decisions and retained state within the audited mixed MXFP4 pack
format. Load-time classification checks the resolved quantization modes and
validated expert-paging metadata; unclassified formats cannot attach an MTP
head. Existing affine verifier behavior is preserved. Local synthetic controls
cover accepted/rejected drafts, output budgets, terminal tokens, state ownership
and failed-step recovery. The [same-source M5 canonical diagnostic](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-m5-canonical-full/README.md)
now passes independently reviewed collection and numerical identity on source
`bf06cbb2`. The original two requests finish at EOS after 90 and 133 equal
output positions; 256 remains the per-request cap. Repeated prefill and every
compared complete serialized state, stream hidden and canonical full-logit
checkpoint are exact. All 47 pre/post payload hashes, generated manifest,
source/runtime bindings and owned-process cleanup pass. This uses forced
streaming with one cached expert layer and the pinned diagnostic runtime.
It does not qualify the candidate wheel's installed/default behavior, independent
holdout, full QA, lifecycle, memory or throughput. The `cd207324` failures above
remain the comparison baseline. Experimental opt-in, qualification and release
gates remain unchanged, and no MTP speedup is claimed.

[The same-source four-prompt holdout and candidate installation](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-m5-holdout-installed/README.md)
now pass on M5. The four frozen inputs contain 66/72/76/70 IDs; each reaches
64 equal output positions, with exact compared primary/final state, hidden and
full logits. Each compares 63 consumed positions and 109 prefill state arrays;
the final emitted token is not consumed by another forward pass. All 47
pre/post payload hashes and clean owned-process checks pass. This retains the
forced-streaming native diagnostic scope and one-layer expert cache.

The proper `bf06cbb2` wheel is also installed in a separate target environment.
All 21 AX package members match; 28 tokenizer members and 52 runtime files are
bound. Isolated import, bundled-runtime doctor, default MXFP4 rejection and
experimental metadata validation pass without replacing the existing manifest.
Installed runtime libraries match the diagnostic. Dependency versions and
observed tokenizer bytes are recorded; prior immutable third-party wheel
reproduction is not claimed. This installation check executes no model request.
Installed API lifecycle, full QA/long context, broader numerical controls,
memory/latency, performance, the primary baseline and fresh delivery remain
open. Neither result promotes the product default or qualifies a release.

[The installed `bf06cbb2` lifecycle attempts](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-lifecycle/README.md)
retain two failures. The first comparator incorrectly required cold/warm cache
metadata equality despite matching text, finish reason and token totals. After
that harness correction, baseline, SSE, budgets and stop pass, but the disconnect
probe exposes a stale active-stream gauge: zero active streams while one job is
pending and only the preceding five requests have completed. Drain/recovery and
required MTP remain untested in those attempts. Both runs lack their original
postchecks; separate later integrity followups pass without changing the failed
lifecycle verdicts.
The server now publishes stream ownership before its first decode burst, with
a local before/after regression. This does not change model arithmetic or
promote the product default.

[The rebuilt `5d028881` installation and lifecycle](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-lifecycle-gauge-fix/README.md)
now pass on the M5/NAS target. Both default and required-MTP modes complete
all seven lifecycle actions: baseline, full SSE, one/two-token budgets, stop,
unfinished disconnect and identical recovery. The active-stream gauge is one
before each disconnect, then work drains. Both servers exit cleanly, all owned
processes are gone, and the original pre/post checks verify all 47 model payloads
and 52 installed runtime files. Independent reconstruction validates all 14
actions against raw responses, events and metrics.

Family admission still requires explicit experimental opt-in; "default" here
means the unchanged MTP/paging policy within that opt-in. The new wheel passes
81 isolated packaging tests. Exact-source CI passes eight jobs but skips all
eight real-weight execution steps; strict Clippy retains 1,798 unchanged
baseline diagnostics. Prior numerical evidence keeps source `bf06cbb2`, with
complete unchanged numerical/runtime trees bound by a source comparison.
The new 105-input-per-mode QA campaign is launched separately; no result is
claimed yet. Broader MTP, checkpoint, performance, memory, primary-reference
and fresh-delivery gates remain open.

[MXFP4 reference evidence](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-references/README.md)
now includes all three numerical reference graphs, eight inputs and 3,316
positions per graph, with completed artifact and post-run hash verification.
The [four-graph comparison](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-four-graph/README.md)
passes the frozen aggregate numerical bounds across those 3,316 positions.
Its AX data comes from source `2642a628` on the M5 with selected-expert paging;
it does not qualify the later QSA correction, MTP or default paging. Three
AX-only high-margin disagreements remain, below the frozen 1% limit. This is
statistical acceptance, not exact token or logit identity. The separate 105-item
functional reference replay has 101 quality passes and four retained failures.
Neither result closes AX QA or the missing `mlx_lm.benchmark` primary baseline.
No failed answer was normalized into a pass.

[The installed M5 fixed-workload attempt](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-workload/README.md)
completed one of six AX cells: 512-token direct, with two warmups and three
measurements of 128 output tokens each. The first required-MTP warmup exceeded
the frozen 3,600-second request deadline; no required trial completed, and
the four larger-context AX cells did not start. All 68 recorded installed
identities match before/after. This failed attempt used source `85a2bab0`,
not the later QSA correction. Forced server cleanup after the timeout does
not establish a native crash or its cause. The three primary reference cells
remain unsupported; no speedup or completed performance matrix is claimed.

[An isolated required-512 diagnostic](../../benchmarks/results/qualification/2026-09-18-flash-next-mxfp4-required512-diagnostic/README.md)
subsequently completed all 128 outputs on the same installed `85a2bab0` source
within the original deadlines. It omitted the five preceding direct requests
and added progress/resource observation, so it does not replace the failed
matrix or identify the original timeout's cause. Its first 37 token IDs match
all five direct trials, then the trajectories diverge; greedy identity fails.
SSE, client and terminal records agree, and the frozen helper completes both
68-file integrity checks. This is bounded diagnostic evidence, not a throughput,
quality or later-candidate qualification result.

Six-bit is excluded from this target campaign. The former Studio M5 Ultra
256 GB target, the intermediate MacBook Pro M5 Max 128 GB target, and all
affine results below are historical; they do not qualify MXFP4 on the
current Ultra-class 192 GB+ target. Existing download aliases retain their
original pack identity.

| Gate | Current target state |
| --- | --- |
| MXFP4 execution | Diagnostic mode binding and U8 scales implemented; small-tensor controls and four installed M5 fixed-input direct/MTP controls pass. Full target validation remains open |
| Immutable delivery | Installed85 default recovery verifies all 47 published files and the generated manifest on target NAS. Original failed attempts remain retained; fresh-cache and final-candidate delivery qualification remain open |
| Numerical and MTP | Canonical source `bf06cbb2` passes the original two-prompt M5 diagnostic (90/133 positions to EOS) and four-prompt holdout (64 outputs each), including compared state/hidden/logits and complete integrity checks. Broader checkpoint/default/MTP qualification remains open. All `cd207324` failures and incomplete M2 SIGBUS remain retained |
| Installed QA and lifecycle | Source `5d028881` passes installation and all 14 lifecycle actions with original pre/post integrity checks. New 105-input-per-mode QA is launched, without a result yet. Historical installed85 QA failures and both bf06 lifecycle failures remain retained |
| Throughput and memory | Failed fixed workload on installed `85a2bab0`: one of six AX cells complete; first 512-token MTP warmup times out; four AX cells unstarted. Three primary-reference cells unsupported. Target peak memory and cold latency pending |
| Release | Candidate; no release-ready or default-MTP promotion |

## Existing affine implementation and historical evidence

HF family `qwen4_exp` identifies the 125B-A6B hybrid Gated-DeltaNet / sparse
attention MoE with its 51B n-gram table. It is distinct from Qwen 3.8 27B
and Super-class 2.4T. The dedicated AX graph implements GDN, QSA, gated
residual streams, PLE disk-row gathers, MoE, and the final mixer. MLX owns
quantized matrix multiplication. This is not a `qwen3_5` remap or an adapter.

Request-owned state, prefix serialization/restore, expert paging and native
HTTP/SSE are implemented. N-gram table payloads are excluded from weight-load
evaluation. Selected-expert and selected-prefill paths remain opt-in.

The CLI maps `qwen3.8-flash-next:axq` and `qwen3.8-flash-next:axq-6bit` to
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP` and
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP`, respectively. An alias is
not proof of published-pack availability or successful download qualification.
Audited legacy manifests identify source `Qwen/Qwen3.8-Flash-Next` revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`.

## Historical affine gates

| Gate | Current result | Remaining requirement |
| --- | --- | --- |
| Support tier | Experimental graph; checkpoint Candidate | Complete reproducible checkpoint qualification |
| Pack delivery | Fresh default-transport and two-worker downloads each verify both public revisions and all 63 LFS files. The a7 installed follow-up exposed a Hub snapshot link rejection; c101 snapshot controls now complete eight requests; the original six-bit 600-second timeout remains open. | Complete full installed QA, cold-latency acceptance and target-SKU admission |
| Numerical | Frozen 4-bit and 6-bit holdouts pass unchanged aggregate bounds across 3,316 aligned positions each | Target and installed qualification remain open; retain historical 22 high-margin disagreements and the earlier threshold revision |
| Functional QA | Installed c101 four-bit default-residency QA: 105 direct/required pairs match text and checker results; each mode has 102 hard passes, 105 normal stops and clean shutdown | Six-bit default QA failed on its fourth request at 1,800 seconds; resolve latency and complete remaining installed/target qualification; reference/NLL were not rerun |
| Long context / NLL | Long-context lookup completed across all three routes; 3,999 scored tokens, AX mean NLL 1.96226 versus reference 1.96609 | Broader contexts; recover matching historical harness or rerun with frozen provenance |
| Trained head | Recorded real acceptance 95/114 (83.3%), permuted 0/207 | Only 50 of 104 requests contribute to acceptance; 54 short cases are excluded. This is a bounded falsification control, not Tier 2 |
| MTP integration | All 12 HC state/runner controls and 105-item native QA parity pass; four installed direct/required pairs have text and usage identity | Full installed/default-route QA and target qualification before promotion |
| HTTP / SSE | HC candidate: six modes and 12 requests pass. Installed wheel: eight HTTP completions with 32-token budgets, correct ready health and clean shutdown pass | Extend installed streaming/lifecycle coverage and target qualification |
| Throughput | Historical fixed-output matrix: 11/18 complete; two failed cells and five without results; 128 tokens in every measured sample | Collect a complete current-candidate matrix; retain historical budget/reference GPU failures and qualify target-SKU memory |
| Former target hardware | No Studio M5 Ultra 256 GB result; intermediate MacBook Pro M5 Max 128 GB target evidence retained for the Auto expert-paging code path | Superseded by the Ultra-class 192 GB+ target above (ADR-037) |
| Release | Not release-ready | Close the numerical, QA, MTP, throughput, delivery and target-hardware gates above; merged validation alone is insufficient |

The statistical threshold was adjusted on the collected sample: the earlier
zero-high-margin-disagreement rule failed, while the later at-most-1% rule
passes at 22/3,260 (0.67%). The aggregate mean KL is 0.0860 against a 0.1011
limit, and top-1 disagreement is 2.85% against 3.22%. Do not describe this as
an independent confirmation or exact full-model parity.

Historical MTP artifacts contain `primary_state_exact_each_step` and
`draft_state_exact_each_step` flags derived from greedy token identity, while
the state arrays were checked with numerical tolerances. Those flags do not
prove byte-exact state equality. New diagnostics report tolerance checks and
whether state comparison stopped after a near-tie separately. The initial
primary/draft prefill state comparison remains byte-exact.

Draft agreement uses its actual comparison-sample count as the denominator.
Session proposal/acceptance counters remain a separate metric; historical
aggregate agreement rates divided by proposals and can differ when those
counts diverge. New reports include agreement matches and samples explicitly.

## Latest campaign failures

The original frozen-holdout attempt collected eight 4-bit prompts on each of the official
chunked, official recurrent and pinned MLX-VLM graphs. AX then failed before
inference: the diagnostic test reads an existing historical manifest whose
`runtime_status.ready` is false. No holdout acceptance score was produced.
The original manifests, frozen dependencies and failure remain preserved.

Fresh CLI delivery separately failed before transfer because its private
installation omitted the documented `download` dependency group. The dependent
throughput campaign consequently collected no measurements. The
[campaign failure record](../../benchmarks/results/flash-next-campaign-failures-m2-20260917.json)
retains these outcomes. Recovery uses separate admission fixtures and installation
outputs; an attempted or queued recovery does not close any release gate.

## Frozen 4-bit holdout result

The [frozen 4-bit holdout](../../benchmarks/results/flash-next-holdout-4bit-m2-20260917.json)
passes its predeclared aggregate rule across eight prompts and all 3,316 aligned
positions, including 1,024-token and 2,052-token inputs. Mean KL is 0.02823 against
0.03757; top-1 disagreement is 8.655% against 10.442%. Four AX-only high-margin
disagreements remain (0.121%, below the frozen 1% cap). High margin means strictly
`reference_margin > 1.0`; a fifth disagreement exactly at 1.0 remains in the raw
record but is outside that category. This is statistical acceptance, not exact
token or logit parity. The threshold was frozen before these holdout outputs;
the historical threshold revision above remains disclosed.

The original numerical binary and inputs are unchanged. The recovery uses an
independent derived manifest differing only in runtime status, and reuses the
three frozen reference collections with artifact hashes. Earlier pre-inference
failures remain recorded. [Raw comparison statistics](../../benchmarks/results/flash-next-holdout-4bit-raw-m2-20260917.json.gz)
retain all rows with only the internal host alias/IP removed. The final
[post-campaign integrity check](../../benchmarks/results/flash-next-holdout-integrity-m2-20260917.json)
passed for both pack fixtures and all 51 frozen dependencies. This M2 result does
not qualify installed/default QA, delivery, throughput, or the target Studio SKU.

## Frozen 6-bit holdout result

The [frozen 6-bit holdout](../../benchmarks/results/flash-next-holdout-6bit-m2-20260917.json)
passes the same predeclared aggregate rule across eight prompts and all 3,316
aligned positions. Mean KL is 0.01939 against 0.02803; top-1 disagreement is
8.625% against 11.196%. Four AX-only high-margin disagreements remain
(0.121%, below the frozen 1% cap), using the same strict margin greater than 1.0.
All four graphs were freshly collected, including the two official paths and
MLX-VLM. [Raw comparison statistics](../../benchmarks/results/flash-next-holdout-6bit-raw-m2-20260917.json.gz)
preserve the numerical results with only the internal host alias/IP removed.

The original numerical binary, prompt IDs, teacher streams and frozen thresholds
were retained. Final post-campaign payload and dependency integrity verification passed.
This M2 result does not qualify installed/default QA, fresh delivery, throughput,
or the target Studio SKU, and does not enable default MTP or Tier 2 status.

## Immutable public pack metadata

The 4-bit alias is pinned to `680573112360bfd3f71556082f875c907c21a6e7`;
the 6-bit alias to `d514dcebf3086068ed7968caf395083c95ebcfca`. Both repositories
are publicly accessible. Their config, AXQuant manifest, tensor index and expert
stream manifest match the native test packs byte-for-byte. See the
[metadata identity record](../../benchmarks/results/flash-next-public-pack-metadata-20260917.json).
Metadata alone does not verify weight bytes. The full payload check below
adds that evidence; later fresh-download results are recorded below, while full installed-runtime qualification remains open.

## MTP state and runner coverage

The [HC verifier matrix](../../benchmarks/results/flash-next-hc-mtp-matrix-m2-20260917.json)
passes all twelve 2/4/6-bit controls with the original bounds unchanged. Both the
five-token tie prompt and 69-token primary prompt are covered. Six state controls
record zero relative logit/state divergence; six runner controls produce identical
direct/MTP tokens. The correction keeps normal projections and expert paging
Shared while using per-row MLX projections for verifier HC. Ordinary prefill and
direct execution keep their prior policy. [Raw native records](../../benchmarks/results/flash-next-hc-mtp-matrix-raw-m2-20260917.json.gz)
are retained separately. This closes the recorded short state/runner failures;
it does not establish arbitrary-context parity or Tier 2 certification.

The historical cells below use test binary `79f30efe` (SHA-256 prefix), built from commit
`1819e4bb`. A pass applies to the recorded prompt and tolerance contract.

| Pack | Primary state | Primary runner | Tie state | Tie runner |
| --- | --- | --- | --- | --- |
| 2-bit | Pass | Pass | Fail: bonus margin 0.9375 > 0.5 | Pass |
| 4-bit | Pass | Pass | Fail: logit relative error 0.100864 > 0.1 | Fail: margin 1.3125 > 0.5 |
| 6-bit | Pass | Pass | Pass | Pass |

A [corrected 4-bit tie runner replay](../../benchmarks/results/flash-next-runner-margin-replay-m2-20260917.json)
measures a direct-to-MTP token gap of 0.125 at position 2, below the unchanged
0.5 limit. The historical 1.3125 margin above duplicated the pipeline bootstrap
token. The output sequences still differ; this is a bounded tie pass, not text
identity. The later HC matrix above separately fixes the state failure and
regenerates all twelve controls with the corrected runner diagnostic.

The pre-merge native API matrix uses server binary `f61f46a0` (SHA-256 prefix), also
from `1819e4bb`. Each mode runs completion and SSE with the same five input
tokens and four output tokens. All six modes return usage and a terminal SSE
marker, repeat their own text, preserve pack metadata, and exit cleanly.
Default-on MTP remains zero. Required MTP produces different text from direct
for 4-bit and 6-bit; 2-bit agrees for this prompt. These strict identity
failures remain visible even though the bounded 6-bit runner controls pass.
This short API control does not establish long-request quality or stability.

The merged source `ea4eb15b` was rebuilt on M2 Ultra with Rust 1.97.1
(`release-server`, binary SHA-256 prefix `6c5c0188`). Its separate six-mode,
12-request rerun again passes four modes and fails 4/6-bit required MTP
text identity. All transport, repeat identity, usage, metadata and clean-exit
checks pass. The 4/6-bit runs remove both experimental family/2-bit opt-ins;
2-bit retains both. Selected-expert/prefill opt-ins remain enabled.
Local merged validation passes 3,681 Rust tests (46 ignored), pinned Clippy,
formatting, script gates and 50 Python CLI tests with 92 subtests.
These bounded checks do not close the release gates above.

The later HC candidate passes all six modes and twelve requests on the same
frozen prompt and four-token budget. Completion/SSE text and direct/MTP text
are identical for all three packs; usage, terminal events, selected-expert
activity, unchanged metadata and clean shutdown pass. MTP default-on remains
zero. Source and binary hashes are preserved in the
[HC HTTP matrix](../../benchmarks/results/flash-next-hc-http-m2-20260917.json).
The earlier failures above remain historical evidence. This short control
still does not qualify longer generation. Health also exposes the legacy
manifest's `qwen4_exp_native_trunk_not_implemented` blocker despite successful
generation; reconciling manifest and active runtime status remains open.

The HC candidate also replays both original QA mismatch cases with identical
direct/required text: `reasoning_cause_effect` emits 160 tokens and
`reasoning_syllogism_roses` emits 211 tokens in each mode. Both answer checks,
normal stop and server shutdown pass. The
[focused QA evidence](../../benchmarks/results/flash-next-hc-focused-qa-m2-20260917.json)
records the exact inputs and responses. A process sample overlapped direct
decode, so recorded durations are diagnostic only.

The later [full HC QA rerun](../../benchmarks/results/flash-next-hc-full-qa-m2-20260917.json)
completes all 105 items in both modes with identical text and checker results.
Every response stops normally and both servers exit cleanly. Each mode passes
102 closed-answer checks; the three failures retain the same text and verdict
as the prior reference run. Both modes answer the 29,774-token lookup with
`1734`. [Raw inputs and responses](../../benchmarks/results/flash-next-hc-full-qa-raw-m2-20260917.json.gz)
retain the exact binary and harness identities. This run uses selected-expert
opt-ins on M2; final installed/default-route and target-SKU qualification remain
separate. Reference and NLL were not rerun.

Full local payload hashing now verifies every LFS file at both immutable public
revisions: 28 files for 4-bit and 35 for 6-bit, including the model shards and
MTP sidecars. File sets, byte counts and SHA-256 values match public metadata;
[the payload identity record](../../benchmarks/results/flash-next-public-payload-identity-20260917.json)
preserves each result. This was a complete read of existing local files, not a
fresh network download or installed-runtime qualification.

A separate uncontaminated resident 4-bit paired control completes two warmup
pairs and twelve measured pairs, alternating direct/MTP order with 32 output
tokens and cleared prefix stores. All 28 requests have identical tokens.
The paired MTP/direct decode-time ratio has median **0.790404**, ranging from
0.785718 to 0.799137. Median direct/MTP decode times are 1.541547/1.220216 seconds.
See the [paired cost record](../../benchmarks/results/flash-next-hc-paired-cost-m2-20260917.json)
and its linked raw artifact. This closes the bounded verifier cost decision,
not general profitability, paged throughput, reference comparison or target-SKU
qualification. The earlier process-sampled run is excluded from these timings.

## Evidence and provenance

These artifacts preserve development outcomes, including failures and missing
cells. `qualification=false` and `release_ready=false` are intentional.
Recorded binary/harness hashes are never replaced with hashes of newer files.
A mismatching or unavailable harness is an open reproducibility gate.

Selected-expert `/metrics` totals from the installed `c10162e6` runtime
undercount reads during intermediate single-token decode: the SDK omitted
those steps' route reports, including their per-step read deltas. These
counters establish selected-read activity but cannot quantify complete logical
payload or physical disk I/O. The SDK now preserves the two read deltas while
omitting the full route map. Corrected native counter validation remains open;
the recorded artifacts and their binary identities are unchanged.

- [Statistical acceptance](../../benchmarks/results/flash-next-statistical-acceptance-m2-20260916.json)
- [Native build identity and checked source hashes](../../benchmarks/results/flash-next-native-build-m2-20260916.json)
- [Completed QA, long context and NLL](../../benchmarks/results/flash-next-extended-qa-v3-m2-20260916.json)
- [Trained-head falsification control](../../benchmarks/results/flash-next-mtp-head-oracle-m2-20260916.json)
- [Merged native build identity](../../benchmarks/results/flash-next-native-build-merged-m2-20260917.json)
- [Merged native HTTP / SSE matrix](../../benchmarks/results/flash-next-http-merged-m2-20260917.json)
- [Pre-merge native HTTP / SSE matrix](../../benchmarks/results/flash-next-http-m2-20260916.json)
- [Batched MTP matrix](../../benchmarks/results/flash-next-mtp-batched-verify-m2-20260916.json)
- [Observed six-bit Auto paging](../../benchmarks/results/flash-next-sixbit-paging-m2-20260917.json)
- [Throughput audit with explicit failure and missing counts](../../benchmarks/results/flash-next-throughput-audit-m2-20260917.json)
- [Throughput matrix, including incomplete cells](../../benchmarks/results/flash-next-throughput-ab-m2-20260916.json)
- [Earlier affine 2/4/6-bit execution controls](../../benchmarks/results/flash-next-affine-formats-m2-20260915.json)
- [Earlier native selected-prefill controls](../../benchmarks/results/flash-next-selected-prefill-m2-20260915.json)

Earlier operator-level, cache, QSA-boundary and numerical comparisons remain in
`benchmarks/results/flash-next-*.json`. They describe their recorded snapshots;
they do not supersede the current gate table or validate a later merged binary.

The fresh comparison fixes the earlier EOS mismatch: AX uses the native
fixed-output endpoint with `ignore_eos=true`, and both routes emit 128 tokens.
The post-first-token decode interval covers 127 tokens. AX client-wall timing
and reference in-process timing remain distinct; native runner timings are
retained separately. The baseline is pinned MLX-VLM because `mlx_lm` has no
`qwen4_exp` graph; this is not an `mlx_lm.benchmark` result.

The host had background indexing/sync activity. AX RSS is an end-of-request
snapshot, and its MLX peak covers the server lifetime; the reference resets
MLX peak per request. No controlled performance or equivalent memory-peak
claim follows from these fields. Although the historical configuration label
says `auto_resident_on_192gib`, observed 6-bit Auto execution uses expert
paging. A stack sample in the first prefill reaches
`ExpertStackPager::ensure_layer` and `load_safetensors_mmap_filtered`.
The first 512-token prefill takes about 448 seconds; subsequent samples take
about 14 seconds. A one-second diagnostic sample was taken during this
unmeasured first warmup. Read actual behavior rather than that stale
residency label.

## Admission and operator contract

The [installed native controls](../../benchmarks/results/flash-next-installed-native-m2-20260917.json)
complete all eight HTTP requests on M2 Ultra 192 GB: 4-bit and 6-bit, selected
and default routes, each with direct and required-MTP generation. All four
pairs have identical text and usage, ready health, unchanged manifests and
normal server exits. No experimental admission override is enabled. The
32-token output budgets cover one 46-token input, not the full QA bank.
The [raw records](../../benchmarks/results/flash-next-installed-native-raw-m2-20260917.json.gz)
retain metrics and outputs with local path prefixes redacted.

The 6-bit default direct request takes about 1,020 seconds and required MTP
about 186 seconds. Auto selects layer paging on this 192 GB host. These are
single diagnostic observations with different cache histories; no speedup or
target-SKU performance claim follows. The former 256 GB target remained untested.

The [installed offline alias check](../../benchmarks/results/flash-next-offline-alias-m2-20260917.json)
passes for both pinned 4-bit and 6-bit aliases after correcting the download
helper's HC role requirements and I64 metadata binding. Both manifests are
regenerated and validated by the native tool. The fixture reuses previously
verified payloads, so that fixture alone does not establish fresh network download; the later delivery campaigns below provide separate evidence.
The rebuilt wheel passes 79 installed/packaging tests and has byte-identical
native binaries, MLX runtime and Metal assets to the earlier packaging wheel.

The [local wheel packaging check](../../benchmarks/results/flash-next-wheel-packaging-m3-20260917.json)
passes isolated native import and 79 installed/packaging tests on M3 Max.
Its negative control rejects an earlier wheel with a Mach-O loading error.
This validates the wheel packaging; it does not establish installed Flash Next
generation, a fresh pack download, or target-SKU qualification. The wheel is
not published.

Audited affine 4-bit/group64 and 6-bit/group64 manifests can be admitted
without an environment variable. `runtime_status.ready` expresses loader
admission, not checkpoint certification. Unknown exporter layouts, invalid
geometry and mixed expert layouts remain rejected. MXFP4/group32 now has an
explicit family opt-in for diagnostic validation; it is not release-qualified.
2-bit/group32 requires both `AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1` and
`AX_ENGINE_2BIT_EXPERIMENTAL=1`. Auto/On/Off expert residency is unchanged.

```bash
python3 scripts/qualify_qwen38_flash_next.py --dry-run --json
python3 scripts/qualify_qwen38_flash_next.py --model-dir /path/to/converted-pack
```

The second command is a metadata preflight only. It does not load tensors,
validate exporter identity/geometry, exercise generation, or certify the pack.
Native `ax-engine doctor` and actual server loading remain required. A legacy
manifest may retain its old trunk blocker; this conservative preflight rejects
it even when the native loader can re-audit and admit the same artifact.

```bash
ax-engine-server --mlx \
  --mlx-model-artifacts-dir /path/to/converted-pack \
  --host 127.0.0.1 --port 31418
```

An available MTP sidecar attaches automatically, but default-on MTP remains
disabled. Explicit `--mlx-mtp-policy required` exercises the experimental
verified path. The test contract records identity until an observed tie and
bounded numerical divergence; it does not promise universal text identity.
The historical QA mismatches remain recorded failures under the original
all-items identity rule. The HC two-item replay fixes both observed mismatches,
and the full HC rerun above now passes the 105-item text-identity contract.
The historical 4-bit tie runner margin of 1.3125 was measured after replaying
the pipeline bootstrap token twice. It is not a valid margin for the divergent
position. The corrected diagnostic skips tokens already represented by the
snapshot, verifies the direct prediction, and measures the gap to the actual
MTP token. The historical state error (0.100864 versus 0.1) was a separate
HC projection issue, now passing in the twelve-cell HC matrix above. Wider
QA, API, cost and target-hardware requirements remain independent.

The primary default remains [Qwen 3.8 27B AXQ](qwen3.8-27b-axq.md) on
Mac mini M4 Pro 64 GB. See [Supported Models](../SUPPORTED-MODELS.md) and
[Testing](../TESTING.md) for the wider operator contract.

The throughput campaign stopped after entering 6-bit reference / 2,048 tokens.
The SSH command exited 255; the process was absent on inspection, and its
termination cause is unconfirmed. The five missing rows are not successful
measurements. The 6-bit direct / 8,192-token server was deliberately terminated
by the 900-second progress supervisor; the reference / 512-token Metal GPU
timeout was a separate recorded failure. See the
[supervisor record](../../benchmarks/results/flash-next-throughput-budget-m2-20260917.json).

## Verifier diagnostic controls

Test binaries support `AX_FLASH_NEXT_VERIFY_DIAGNOSTICS=1`. Each verify call
reports `batched_row0` separately from a `singleton` decision whose source is
`rejection_replay` or `one_slot_budget`. The existing correction margin retains
its meaning. Missing comparisons are null; an accepted window does not trigger
an extra singleton forward for diagnostics. These synchronized records cannot
be used for throughput claims and are absent from production builds.

The [tiny-model control](../../benchmarks/results/flash-next-verify-diagnostics-m3-20260917.json)
checks accepted, rejected, terminal and one-slot cases on M3 Max. Enabling the
diagnostic preserves committed tokens and state hashes. It does not resolve the
real-pack numerical failures above. The newer throughput audit re-curates the
same immutable M2 input: 11 complete, two failed, five missing; it is not a new
hardware run.


The [fresh two-worker delivery record](../../benchmarks/results/flash-next-fresh-delivery-workers2-m2-20260917.json)
verifies both pinned public aliases from an empty cache: 28 four-bit LFS files
(136,112,205,574 bytes) and 35 six-bit files (167,045,770,831 bytes). The six-bit
transfer recovered from one shard read timeout. This uses the explicit existing
`AX_ENGINE_HF_MAX_WORKERS=2` override; it does not qualify default transport.
The [raw record](../../benchmarks/results/flash-next-fresh-delivery-workers2-m2-20260917-raw.json.gz)
retains the retry and the subsequent a7 wheel's four-bit load failure: its path
resolver rejected a legitimate Hub snapshot link into its own blob store.
No inference, throughput or full QA completed in that follow-up. The later c101 controls below verify the snapshot path correction; full QA and
cold-latency acceptance remain open. Earlier installed controls are separate evidence, not a pass for this
fresh-snapshot attempt.


## Default-transport fresh delivery retry

The [default-transport delivery record](../../benchmarks/results/flash-next-fresh-delivery-default-m2-20260917.json)
verifies both pinned public aliases from a new empty cache on Apple M2 Ultra
192 GiB, with no worker-count or timeout override. Both CLI commands exit zero
and report ready. All 28 four-bit and 35 six-bit LFS payloads match their
expected sizes and SHA-256 hashes: 136,112,205,574 and 167,045,770,831 bytes.

Each pack encountered one shard read timeout and resumed successfully. The
[raw bundle](../../benchmarks/results/flash-next-fresh-delivery-default-m2-20260917-raw.json.gz)
retains those messages and the earlier failed default attempt. This successful
retry is not a measured reliability rate. It exercises the a7 downloader;
the c101 native runtime has separate installed acceptance. No numerical,
throughput, MTP-default or target-SKU gate is promoted by this download result.


## c101 installed snapshot controls

The [c101 snapshot controls](../../benchmarks/results/flash-next-installed-snapshots-c101-m2-20260917.json)
complete eight HTTP requests on the verified public four-/six-bit snapshots,
without rewriting their links or manifests. Selected and default route pairs
produce identical text and usage for one 46-token input and 32-token outputs,
with ready native health and clean server exits.

The [raw bundle](../../benchmarks/results/flash-next-installed-snapshots-c101-m2-20260917-raw.json.gz)
also retains the original six-bit selected request's 600-second timeout and
the subsequent default recovery's port-preflight failure before model load.
Later six-bit retries completed; the default request took 1,045.268 seconds and
required MTP took 186.209 seconds with different cache histories. These are not
a speedup comparison or cold-latency acceptance. Full QA has separate records
below; streaming lifecycle, the complete current matrix and target-SKU
qualification remain open.


## c101 installed four-bit full QA

The [installed four-bit QA record](../../benchmarks/results/flash-next-installed-full-qa-4bit-c101-m2-20260917.json)
completes all 105 fixed inputs through the default residency route with direct
and required-MTP policies. Every pair matches text and checker results; all
210 requests stop normally, both servers exit zero, and the manifest remains
unchanged. Direct records zero drafted tokens; required MTP records 487.

Each mode has 102 hard passes and retains the gravity, water-formula and CSV
failures. The 29,774-token lookup returns `1734` in both modes. Its request
durations are 428.043 and 439.078 seconds with different cache histories, not
a speedup comparison. The [raw record](../../benchmarks/results/flash-next-installed-full-qa-4bit-c101-m2-20260917-raw.json.gz)
preserves every response, check and metric with local paths redacted. This
closes the four-bit installed QA collection and pairing requirement only;
six-bit default QA, throughput, lifecycle, cold latency and target qualification
remain open. Selected six-bit QA is recorded separately below. MTP default and
release status are unchanged.


## c101 six-bit default full-QA timeout

The [six-bit default QA failure](../../benchmarks/results/flash-next-installed-full-qa-6bit-c101-failure-m2-20260917.json)
records three completed direct requests, followed by a 1,800-second timeout
on `reasoning_cause_effect` (46 input tokens, 256-token output budget). The
first three hard checks pass; the first request takes 1,401.716 seconds. The
server exits zero after the timeout. Required-MTP QA never starts, and the
dependent throughput and lifecycle runs collect no new measurements.

The [raw failure bundle](../../benchmarks/results/flash-next-installed-full-qa-6bit-c101-failure-m2-20260917-raw.json.gz)
retains outputs, checks, the exception and supervisor result. Earlier short
controls and successful warmed retries do not close this full-QA failure.
This result is from the supplementary M2 host; release and target qualification
remain open.


## c101 installed selected six-bit full QA

The [selected six-bit QA record](../../benchmarks/results/flash-next-installed-selected-full-qa-6bit-c101-m2-20260917.json)
completes all 105 fixed inputs with direct and required-MTP policies on the
supplementary M2 Ultra 192 GiB host. Every pair matches text and checker
results; all 210 requests stop normally, both servers exit zero without a
forced kill, and the manifest remains unchanged. Prompt-token accounting and
output budgets were independently checked. Direct records zero draft tokens;
required MTP records 518. Each mode has 103 hard passes and retains the
water-formula representation and incorrect CSV-literal failures.

The 29,774-token lookup returns `1734` in both modes, taking 1,174.861 and
1,098.861 seconds respectively. These are single observations with different
cache histories, not a speedup comparison. The [raw record](../../benchmarks/results/flash-next-installed-selected-full-qa-6bit-c101-m2-20260917-raw.json.gz)
preserves all responses, checks and metrics with local paths redacted.

This run uses existing selected-expert and selected-prefill opt-in flags. It
closes collection and pairing for this route only; the default six-bit
1,800-second full-QA timeout and earlier selected 600-second timeout remain
open. Full throughput, installed lifecycle, cold latency and the former M5 Ultra
256 GiB qualification were separate open requirements in this historical campaign. Product defaults, MTP
certification and release status are unchanged.
