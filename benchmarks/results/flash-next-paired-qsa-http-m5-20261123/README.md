# Flash Next paired QSA gather through native HTTP on M5 Max

MacBook Pro M5 Max 128 GB, internal SSD, macOS 27.2.1, MLX 0.32.4 and
Rust 1.99.1. The same signed release-server executable runs in
stock, paired, paired, stock order. The native SSE route uses the
verified 49-member pinned MXFP4 pack and the ordinary resident
one-request, 32,768-token profile with MTP off. Prefix reuse is
disabled throughout. Each process warms up and replays the same 512,
2,048, and two independent 8,192-token prompts. The request order
reverses in the second pair. Every measured request emits 64 tokens.

All 16 measured responses match **512 output token IDs** between their
paired comparisons. The two measured long treatment prompts each
report 432 paired tiles in both orders. The two shorter prompts and
every stock request report zero. All lifecycle counters drain, no
MTP verifier or route errors are observed, and all **750** pressure
samples are normal. The maximum sampled physical footprint is
**75.66 GiB**.

| Prompt | First order stock / paired TTFT | Reverse stock / paired TTFT | First / reverse endpoint eligibility |
| --- | ---: | ---: | --- |
| 512 tokens | 1.155 / 1.542 s | 1.332 / 1.667 s | Yes / No |
| 2,048 tokens | 2.821 / 2.737 s | 2.747 / 2.658 s | No / No |
| 8,192 tokens, replay | 9.620 / 9.630 s | 9.875 / 9.503 s | No / No |
| 8,192 tokens, holdout | 9.813 / 9.677 s | 10.093 / 9.819 s | Yes / No |

The first holdout is the only engaged long comparison that meets
90% endpoint idle in both arms. It measures 1.38% lower TTFT, with
full-request idle of 93.77% stock and 82.28% paired. The other three
long comparisons miss the endpoint threshold. The short control
shows that end-to-end TTFT also moves when the gather never engages.
The archived [native forward probe](../flash-next-paired-qsa-quiet-m5-20261123/README.md)
establishes a bounded 2.50% model-forward reduction on its fixed
prompt. This serving comparison supports functional engagement and
exactness. The serving timing remains diagnostic, and the gather
stays opt-in.

The [capture](evidence.tar.xz) includes the signed build and source
receipts, both orders, prompts and inventory, original streamed
output and telemetry, and all sampled memory observations.
The [comparison](comparison.json) recomputes prompt identity,
kernel engagement, request and endpoint CPU idle, and TTFT by arm.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
