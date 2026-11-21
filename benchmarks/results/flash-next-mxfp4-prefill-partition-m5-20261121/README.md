# Flash Next prefill partition replay on M5 Max

MacBook Pro with M5 Max, 128 GB unified memory, internal SSD, macOS 27.2.1,
MLX 0.32.4, and Rust 1.99.1. The resident MXFP4 pack was
`AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP` at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`.
The `b36b7b287` release-server binary has SHA-256
`0a7c6049a5af88ed1ebdfce02c648416349b2b1eed301381cb6eff5b310e6af5`.
The pack inventory was verified before and after the replay.

| Request slots | Batch tokens | Arrivals tested | Changed outputs | Peak sampled footprint |
| ---: | ---: | --- | ---: | ---: |
| 1 | 2,048 | Serial baseline | 0/4 | 72.10 GiB |
| 1 | 2,047 | Serial, forced split | 2/4 | 73.09 GiB |
| 2 | 2,048 | Serial, simultaneous, during decode | 2/12 | 73.91 GiB |
| 2 | 4,096 | Serial, simultaneous, during decode | 0/12 | 72.21 GiB |

The 2,047-budget solo arm split each prompt at cursor 2,047 before
finishing at 2,048. The second prompt's first token changed from 760 to
27775 on both repeats. All 96 output IDs matched its two changed
concurrent outputs under the 2,048 budget, with the same prefill cursors.
In the other two overlapping trials at that budget, the first prompt was
split and retained its solo output. The reported inner chunk switched
from 512 to 1,024 when the scheduler split a prompt.

The 4,096 budget preserved the full 2,048-token scheduled prefill and all
output IDs for both prompts, including every request in the four
overlapping trials. Both requests recorded engine steps that scheduled
two requests in each of those trials. All memory samples reported normal
system pressure. This is a two-prompt replay with MTP and coalescing off.
Longer concurrent inputs, different prompts, and sustained traffic
remain unqualified. The supported profile remains one active request.

## Reproduction

Start a fresh server per table row with the pinned pack and set
`--max-concurrent-requests` and `--max-batch-tokens` to the row values.
Use `--mlx --model-id qwen3.8-flash-next:axq` and
`--prefill-chunk 1024 --total-blocks 2048 --stream-experts auto
--mlx-mtp-policy disabled --disable-ngram-acceleration` and disable
native prefix reuse, the disk prefix cache, and direct coalescing with
`AX_ENGINE_PREFIX_REUSE_DISABLED=1`,
`AX_MLX_PREFIX_CACHE_DISK_DISABLED=1`,
`AX_MLX_PREFIX_CACHE_MAX_BYTES=0`,
`AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0`, and
`AX_MLX_FLASH_NEXT_DIRECT_COALESCING=0` with `AX_MLX_NATIVE_CONFIRM=1`.

Send the [two token arrays](prompts.jsonl.gz) to
`/v1/generate/stream` with 96 output tokens, temperature zero, and
`sampling.ignore_eos` enabled. The decode-arrival control starts the second
client when the first receives a token and swaps the leading prompt on the
second repeat. Compare the terminal output IDs against the serial baseline
and use each request's recorded prefill cursors and scheduler step counts
to identify actual partitioning and overlap. The [raw observation](observation.json.gz)
retains all 32 requests, their forwarded scheduler events, and one-second
memory samples.

## Fixed-chunk and aligned-cut follow-up

The follow-up used the same hardware, binary, model inventory, and two
prompts. Each single-request arm ran both prompts twice. Within an arm,
both repeats matched across all 96 output tokens for each prompt.
System pressure stayed normal and sampled process footprint stayed under
74 GiB in every arm.

| Scheduler budget | Configured chunk | Observed prompt cursors | Prompt 0 changed positions | Prompt 1 changed positions |
| ---: | ---: | --- | ---: | ---: |
| 2,048 | 1,024 | 2,048 | 0 | 0 |
| 2,047 | 512 | 2,047, 2,048 | 11 | 57 |
| 1,536 | 512 | 1,536, 2,048 | 0 | 85 |
| 1,536 | 1,024 | 1,536, 2,048 | 1 | 93 |

Every arm emitted first token 760 for both prompts. With the 2,047 cut
and 512 chunk, the first changes were at zero-based output positions 41
and 38. The 1,536 cut at the same chunk size changed prompt 1 starting
at position 11. The 1,536 cut at chunk size 1,024 changed prompts 0 and 1
starting at positions 41 and 3. Use the
[raw follow-up](alignment-observation.json.gz) for the full trajectories.

An intermediate greedy prefill item reserves a singleton forward for its
own last token. An aligned scheduler cut therefore still changes the
physical forward boundaries. The replay establishes the effect on these
output trajectories. The serialized trunk state and logit margins need
their own evidence. The existing two-request 4,096-budget control preserved
the whole scheduled prompt for both requests.

## One decode token and one 2,048-token prefill

The same binary, pack, token arrays and controls ran with two active
slots and `--max-batch-tokens 2049 --multi-prefill-fair
--max-prefill-tokens-per-request-per-step 2048
--max-inflight-prefill-requests 1`. Each request kept a single
2,048-token scheduled prefill. Both requests in each of the four
overlapping trials reported the same shared 2,049-token step.
Across all six trials, all 12 requests reproduced all 96 tokens from
their single-request baselines.

| Two-slot arrival | Median time for two replies | Trials |
| --- | ---: | ---: |
| Serial | 9.072 s | 2 |
| Simultaneous | 8.921 s | 2 |
| Second request enters during decode | 8.962 s | 2 |

The largest interval between successive streamed tokens for the
already-decoding request was 2.204–2.234 seconds in the four overlapping
trials. Peak sampled process footprint was 72.277 GiB and peak RSS
was 73.728 GiB. All pressure samples were normal. These observations
cover two 2,048-token prompts with 96 generated tokens each and no
native direct coalescing. Longer context, steady traffic, broader
prompt coverage and any throughput claim need separate evidence.

The [raw reservation receipt](reservation-observation.json.gz)
records every streamed token, per-request step time and memory sample.
