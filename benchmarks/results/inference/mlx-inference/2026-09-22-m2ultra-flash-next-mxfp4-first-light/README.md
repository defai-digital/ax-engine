# Qwen 3.8 Flash Next MXFP4: first working generation on the ADR-037 target (2026-09-22)

Host: um-macstudio-m2 (Apple M2 Ultra, 192 GiB), the ADR-037-corrected Flash
Next campaign/target host. Binary: current main (a2016f51) built with
`cargo build --release-server --bin ax-engine-server` (unwind profile).
Pack: `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP`, local copy at
`/Volumes/Ext16TR0/models/`, 123 GB on disk. Launch:
`AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1 AX_MLX_NATIVE_CONFIRM=1
ax-engine-server --mlx --mlx-model-artifacts-dir <pack> --model-id <pack>
--port <p> --mlx-mtp-disable-ngram-stacking`.

This is a smoke/diagnostic record, not a qualification result. No throughput
claim, no quality claim, no default/route change implied.

## Load time

Cold load (external RAID, page cache not warmed) took approximately 22
minutes end to end (process start to `/health` returning 200) on this run.
Root-caused via `sample <pid> 3` (macOS stack sampler) during load: the
dedicated `ax-native-generation` thread's entire stack was
`load_resident_tensors -> load_safetensors_mmap_filtered ->
MlxArray::from_raw_data -> mlx_array_new_data -> mlx::core::array::init ->
_platform_memmove`, i.e. legitimate single-threaded sequential tensor
loading (mmap + memcpy per tensor, no parallelism across the ~277 large
tensors), not a hang. This is broadly consistent with the historical
`.internal/reports/flash-next-2026-09-15` measurement of ~512s (8.5 min) load
on the smaller 4-bit pack. Loading every tensor on one thread is a real,
separate optimization opportunity (the tensors are independent) not
attempted here.

An earlier attempt on the 6-bit pack (156 GB, out of scope for this
qualification campaign per the PRD, and larger than the resident budget
even on this 192 GiB host once the engine's ~48 GiB reserve is added) hit
Auto expert-paging and did not complete a single generation request within
20-25 minutes of waiting; killed rather than diagnosed further, since 6-bit
is not the target format and MXFP4 (the actual target) does not have this
problem (below).

## First generation, quality probe

Once loaded, MXFP4 generation is fast and normal: a trivial 8-token
completion ("Say hi in one word." -> "Hi") returned in 2.6s.

Re-ran the three quality-cohort items named as failing in
`.internal/prd/PRD-FLASH-NEXT-PRODUCT-CLOSURE.md` (2026-09-19 review) --
`instruction_alphabet_first`, `science_gravity_earth`,
`knowledge_water_formula`, from
`.internal/reports/flash-next-product-20260917/candidate-5f583018/qa-preparation/full-qa-items.json`
-- against this build, greedy (temperature 0), max_tokens 32:

| item | expected (exact_answer) | got | diagnosis |
| --- | --- | --- | --- |
| instruction_alphabet_first | `A,B,C,D,E` | `A, B, C, D, E` | checker strictness: model added conventional comma-spacing; content is correct |
| science_gravity_earth | `10` | `9` | genuine, minor content disagreement: model answers a truncated/rounded-down convention rather than the "g ~ 10 m/s^2" pedagogical rounding the checker expects; not a formatting bug |
| knowledge_water_formula | `H2O` | `H₂O` (actual Unicode subscript-2 character) | checker strictness: model rendered the formula with a typographic Unicode subscript instead of the plain-ASCII digit the checker expects; content is correct |

Two of three failures are exact-match checker strictness on substantively
correct model output (the item schema already supports
`exact_answer_aliases`; adding `"A, B, C, D, E"` and `"H₂O"` as aliases
would close them without loosening the checker's match semantics, per the
PRD's instruction not to loosen checkers -- this is recognizing an
already-valid alternate rendering, not relaxing the rule). The third
(gravity) is a real, low-severity content difference, not attributable to
this build or to AX-specific behavior without comparing against the
official reference's answer to the same prompt (not done here).

## Rough throughput (not a qualification measurement)

Three repeats, greedy, 128 max_tokens, a short prompt, same warm process
(no separate warmup excluded): 10.70s / 6.92s / 6.91s wall time including
prefill, settling at approximately 18.5 tok/s end-to-end by the third
repeat. No comparison baseline collected in this run; the nine-cell
512/2048/8192-token matrix from the completion plan's WI-7 was not run.
