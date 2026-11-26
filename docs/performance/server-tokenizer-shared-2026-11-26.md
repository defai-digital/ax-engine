# Text requests: sharing the tokenizer instead of copying it

**Date:** 2026-11-26 · **Hosts:** MacBook Pro M5 Max 128 GB and Mac mini
M4 Pro 64 GB, macOS 27.2 · **Packs:** Qwen 3.8 Flash Next AXQ MXFP4, Qwen
3.8 27B AXQ MXFP4

## Summary

Every request that carries text (completions, chat, tokenize) spent about
10 ms copying the tokenizer before it tokenized anything. The server now
shares one tokenizer between requests. First text of a streamed completion
is 10 to 12 ms earlier at every prompt length; token ids are unchanged.

## What happened

The server keeps one parsed tokenizer per model in a process cache and
hands each request a clone. The wrapper held a `tokenizers::Tokenizer` by
value, and that type clones by value: the vocabulary and the merge table
were copied and freed again for every request. For the 248,320-entry Qwen
tokenizer that is about 10 ms, whatever the prompt length. Tokenizing a
36-token prompt itself takes 0.02 ms.

It showed up as a constant gap between `/v1/generate` with token input and
`/v1/completions` with the same prompt as text. A sample of the server
under `/v1/tokenize` load put the time in `BPE::clone`, the copy of its
vocabulary map and the drop of the model.

## What changed

`EngineTokenizer` holds the tokenizer behind an `Arc`, so a clone shares
it. Nothing mutates a cached tokenizer.

The upstream BPE model keeps a word cache that every copy started empty.
It now persists across requests. It is bounded, only fills, and takes its
lock without waiting, so concurrent requests do not queue on it.

## Measurements

MacBook Pro M5 Max 128 GB, Flash Next, medians:

| | Before | After |
| --- | ---: | ---: |
| `/v1/tokenize`, short prompt | 10.5 ms | 0.18 ms |
| Streamed completion, first text, 39 input tokens | 165.4 ms | 153.6 ms |
| Streamed completion, first text, 235 input tokens | 232.6 ms | 222.6 ms |
| `/v1/generate` with token input, 39 tokens (no tokenizer) | 155.5 ms | 155.3 ms |

Mac mini M4 Pro 64 GB, 27B:

| | Before | After |
| --- | ---: | ---: |
| `/v1/tokenize`, short prompt | 10.9 ms | 0.19 ms |
| Streamed completion, first text, 36 input tokens | 448 ms | 437 ms |

Rested paired reruns on the M5 Max against oMLX 0.8.2, first text in
seconds
([records](../../benchmarks/results/qwen38-m5-modes-20261125/README.md#every--row-after-the-shared-tokenizer)):

| | Before | After | oMLX |
| --- | ---: | ---: | ---: |
| Flash Next, 36 input tokens | 0.174 | 0.165 | 0.164 |
| Flash Next, 231 | 0.237 | 0.226 | 0.225 |
| Flash Next, 413 | 0.308 | 0.294 | 0.292 |
| Flash Next, 803 | 0.451 | 0.438 | 0.429 |
| Flash Next, 1,492 | 0.702 | 0.689 | 0.689 |
| 27B, 1,492 | 1.501 | 1.490 | 1.510 |

The "before" Flash Next column is a scan of commit `df8c86ecf` in the same
protocol earlier the same day; the 27B row compares reruns 8 and 9. From 2,961
tokens up Flash Next stays 3 to 12% behind oMLX; that gap is in the
sparse-attention layers
([note](flash-next-qsa-masked-dense-2026-11-25.md#what-is-left)).

## Reproduce

Send `/v1/tokenize` a short text in a loop and read the latency, or
compare first text of `/v1/completions` with the wall time of
`/v1/generate` given the same prompt as token ids.
