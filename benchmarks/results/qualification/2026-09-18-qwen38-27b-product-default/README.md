# Qwen 3.8 27B product-default qualification

Source: `a142a5ed0652d98510700c71a1cc769bebb9bc42`, AX Engine 7.4.0 candidate.
Target: Mac mini M4 Pro 64 GB (Mac16,11), macOS 26.6.2. Pack:
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` at
`3e290738e96972307c6aeb9934ab170ca0eae1c1`.

Assessment: **product-health candidate qualification passed** for the pinned
6-bit pack's installed CLI and local HTTP text-serving scope on this SKU.
Default/direct/explicit-MTP qualification, full-bank default/MTP quality and
bounded lifecycle tests passed. This can enter the formal release workflow;
it is **not a published release**, MTP default promotion or broad model-accuracy
certification.

## Reproduced defect and repair

The previous clean `22affab5` wheel requested and activated MTP on an ordinary
launch: the fixed 64-token response recorded 60 drafts, 41 accepted tokens and 21
MTP decode steps. Explicit direct recorded zero. Publisher speed/enable
metadata had been treated as permission to promote linear-Qwen Auto MTP,
although AX MTP-D had never authorized that promotion.

The runtime now requires explicit opt-in for the linear-Qwen candidate routes.
It retains sidecar attachment, route safety and Required requests. Dense Qwen
and other families retain their existing defaults. No quantization, kernel,
acceptance threshold, gold answer or numerical tolerance changed. Correcting
the accidental MTP default can change outputs and latency versus that default.

Schema 4 adds an actual launch without acceleration overrides, alongside the
existing explicit direct and MTP cells. MTP QA now uses Required explicitly;
the benchmark preflight's force-requested override is confined to its child
process. Qualification requires default MTP requested/active false and explicit
MTP active with zero direct-fallback steps. Missing, skipped, failed or partial
evidence fails. The paired 64-token identity probe remains diagnostic only.

## Installed artifact

Wheel SHA256:
`2afba9b7530cb16f268d63a34c3bf4753fbebd17efadbfe46f4fba3e5652d7d0`.
The canonical wheel builder used release-server and release-pyext profiles,
bundled MLX 0.32.2 libraries/metallib, checked product binary macOS minimums and
passed isolated native import. A fresh Python environment on the target SKU
contains the installed wheel without a Python MLX package. The loader resolves
MLX/JACCL from `ax_engine/.dylibs`, not the build machine. All installed wheel
members, native executables, console launcher and 22 model files were verified.

The first build completed but its clean-check rejected an untracked build-venv
symlink. Replacing that owned build dependency with an ignored environment
directory and repeating the canonical build passed the clean-check. The first
log is retained by hash; no source-cleanliness check was skipped.

## Qualification and broader checks

| Route | Contractual QA | Required API probes | MTP requested/active | Direct fallback |
| --- | --- | --- | --- | --- |
| Normal default |32/32|7/7|false/false|0|
| Explicit direct |32/32|7/7|false/false|0|
| Explicit MTP |32/32|7/7|true/true|0|

All 96 sampled results have `finish_reason=stop`; no soft checks failed.
MTP recorded 60 drafted/82 verified tokens on the route probe. Its 64 output IDs
match the paired direct output. Independently, the new default and direct match
the old explicit-direct 64 IDs, and new explicit MTP matches old MTP 64 IDs.
The inputs, sampling and output budget match; the qualification MTP arm uses a
different registered model label, disclosed in the control record.

## Preserved failures and revised quality contract

The first broader run on `030ba288` used all 79 bank items in both streaming
forms, with a uniform 256-token budget. Both routes originally scored 154/158.
Valid numeric answers failed the prose-coherence heuristic; ordered alphabet
lists with separator-adjacent spaces failed literal matching. Independent
review, the declared list-only prompt and existing numeric-list aliases support
an explicit whole ordered-list checker. Extra items, changed duplicate counts,
wrong order and surrounding prose fail. Prompts, golds, aliases and regular
expressions are unchanged, as checked against the old bank.

The old grader also passed incomplete answers when a gold substring appeared.
The revised `ax.qa.complete_answers.v2` contract requires natural `stop`.
Regrading all original responses yields **152/158 default and 154/158 MTP**:
six/four truncated responses now fail. These original failures remain under
`original/`; `quality-contract-audit.json` records every score change. The new
full-bank run uses a declared uniform 1024-token cap, with no selected reruns.
This is a different workload, not a retrospective pass for the 256-token one.

The first lifecycle runs restored quiescence and fixed outputs, but failed the
mandatory pressure gate: no buffered events or overflow were observed. The old
512-token workload and eight-second hold could finish inside the 512-event
channel plus 256-event worker backlog and TCP buffers. Stalling after headers
could also cancel a queued job before output. The repaired harness requests
a small receive buffer before connecting and waits for actual output before
holding. The final protocol uses three rounds at concurrency four per route,
8192-token stalled requests held for 180 seconds, 512-token slow/disconnect
requests and 32-token normal requests. Actual backpressure remains mandatory.

A separate bounded loopback probe on the SKU found that macOS enlarged the
requested 1024-byte buffer to 326640 bytes at connect. Setting it again after
connect did not impose a durable bound. The executed `a142a5ed` artifact's
`receive_buffer_bytes` is the **before-connect** value, not an effective cap.
`d5d6ba59` adds a separate connected-buffer reading with a failing-before
regression and 12 passing fault-harness tests; it changes no workload or verdict.
`transport-buffer-probe.json` and its standalone probe preserve this diagnostic.
Backpressure evidence comes from the actual worker metrics, not buffer requests.

## Final full-bank and lifecycle results

| Check | Product default | Explicit MTP |
| --- | --- | --- |
| Full bank, 79 items x two streaming forms |158/158 hard passes|158/158 hard passes|
| Natural stop / incomplete / transport error |158 / 0 / 0|158 / 0 / 0|
| Non-blocking keyword checks failed |6|6|
| Fault workload, three rounds x five requests |15/15 expected outcomes|15/15 expected outcomes|
| Sampled buffered-event peak |222|242|
| Backlog-overflow counter delta |0|1|
| Direct-fallback counter after QA and soak |0|0|
| Lifecycle gauges after cleanup |all zero|all zero|
| Cold / warm / post-cancellation 64 IDs |identical|identical|
| Owned server cleanup |reaped|reaped|
| Fault phase elapsed |544.67 s|545.31 s|
| RSS before / after fault phase |16.597 / 15.809 GiB|16.409 / 15.807 GiB|

All normal and slow requests completed; disconnect and stalled clients reached
their declared cancellation conditions. MTP's bounded-backlog overflow was
observed in the final counter delta, followed by quiescence and an unchanged
recovery answer. Periodic samples did not catch the counter increment, so their
sampled peak must not replace that final delta. RSS is a short-run observation,
not a leak-free or endurance claim. Default MTP draft/accept counters stayed
zero throughout. Explicit MTP remained active on the cold/warm/recovery probes,
with zero fallback across the entire process.

The 316 full-bank results were independently regraded with the current source;
every hard and soft check detail matches. The six keyword misses per route are
reported, not erased. The final evidence verifier also reproduces the corrected
historical 152/158 and 154/158 scores.

The candidate runner implements v2 completion checks, but its reporter omitted
the supplied version field. The follow-up `b80ee67e` fixes serialization only,
with a failing-before regression and 89 passing offline QA tests. Raw candidate
reports are unchanged: their `commit` and individual hard completion checks
bind the grading contract. The wheel and hardware harness remain exactly
`a142a5ed`; no claim is made that later documentation/reporting commits were
compiled into that artifact.

## Validation and review

On runtime source `030ba288`, Rust 1.97.1: 3725 tests passed, 46 ignored. Python: 209 passed, 26 skipped and 140
subtests. Formatting, script gates, both qualification dry-runs, primary claims,
CI-policy Clippy, and strict production-library/binary Clippy pass. Strict
all-target Clippy still exits 101 on existing test restriction lints; it is not
reported as green. Focused policy and qualification regressions pass. The new
Auto-policy test failed against the old implementation before the repair.
The candidate has identical `crates/`, `metal/`, `python/` and Cargo inputs;
subsequent changes are QA/fault harness and reporting only. Full script gates
were rerun after those changes. Receipts distinguish these source identities.

DeepSeek and MiniMax provided bounded read-only AX Code reviews. DeepSeek
identified both invalid publisher-metadata activation branches. MiniMax's
suggestion to remove only the throughput branch was rejected: a publisher's
optimized stamp is also not AX default-promotion authority. Its assertion of no
customer impact was rejected, as was making MTP-P depend on MTP-D. Local
reference designs were inspected; no reference implementation was extracted.

## Scope and limits

This record separates product-health qualification from MTP certification and
release publication. The qualifier marks MTP-S/P/D `not_assessed`. The accepted
MTP-S same-verifier greedy verify/replay contract is unchanged; its acceptance
and replay regressions remain in the passing Rust suite. No speed multiplier,
MTP-D promotion, sampled-distribution equivalence or optimistic-mode certification
is claimed. Known independent 192-token route differences at 116/155 remain
separate disclosed diagnostics.

Historical long-reasoning budget exhaustion and wrong answers are retained.
LINE_SET full enumeration is an additional metric beyond its original
accepted-location subset contract. Independent same-pack reference controls also
reproduce the saved wrong recovery answer. Neither selected failure subsets nor
this basic question bank establish general model accuracy. No gold was changed.

There is no 8h/72h endurance, long-context-at-depth, multimodal or multi-model
promotion here. Signing/notarization, exact-source hosted CI and public
GitHub/PyPI/Homebrew publication remain release-workflow steps. The checkpoint
retains Candidate status; MTP-P and MTP-D remain open.

## Reproduction and evidence

Run `scripts/qualify_qwen38_27b.py --run` from the clean source checkout with the
installed candidate's manifest, wheel, CLI, server, bench and pinned model,
as documented in `docs/TESTING.md`. The installed CLI form for explicit MTP is:

```sh
ax-engine serve qwen3.8-27b:axq -- --mlx-mtp-policy required
```

The additional lifecycle collector is included as `collect_lifecycle.py`.
Prepare an empty workspace containing `product-final-source` at the recorded
commit, the pinned `model/`, `product-final-venv` with the recorded wheel,
`build-manifest-product-final-installed.json` with installed launcher hashes,
and `product-final-lifecycle-prompt.json` copied from `lifecycle/soak-prompt.json`.
Run from a clean environment with no AX/MLX/DYLD/Metal overrides:

```sh
"$WORKSPACE/product-final-venv/bin/python" collect_lifecycle.py "$WORKSPACE"
```

The collector runs installed `ax-engine serve` outside the checkout, creates a
private offline cache using hardlinks to the already verified snapshot, and
owns/reaps its servers. Port 31503 must be unused. Output/cache directories must
not already exist. It uses the frozen seeded 128-token native prompt so the
isolated target does not need a Python MLX package. QA prompts use the standard
chat path. The published collector differs from the executed one only by the
workspace-root expression; `collector-receipt.json` records both hashes.

Raw artifact hashes and path-redacted published hashes are distinct. The offline
verifier checks recorded provenance and outcomes; it does not execute model
arithmetic or replace hardware qualification. From this evidence directory:

```sh
python3 verify_evidence.py --source-root /path/to/ax-engine
```

Omit `--source-root` for a standalone receipt/hash check. The optional source
check additionally reproduces every final grade and both historical regrades.
