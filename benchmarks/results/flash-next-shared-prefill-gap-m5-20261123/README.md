# Flash Next shared prefill and short-stream output gaps

This replay uses the sealed [M5 Max natural-arrival crossover](../flash-next-natural-short-drain-m5-20261123/README.md).
It checks the source replay first and joins the 29 matched short
requests to their 8,192-token peers by prompt IDs and logical
prefill cursor. Each 512-token request emits 96 tokens. Consecutive
short output steps link each interval to the engine step ending it.
Any interval spanning multiple engine steps receives its own label.

| Ending step | Measured gaps | Median gaps, by order | Observed maximum |
| --- | ---: | ---: | ---: |
| Shared prefill, 512 tokens | 261 | 817.46 / 787.31 ms | 1,074.82 ms |
| Shared prefill, 511 tokens | 87 | 825.81 / 798.64 ms | 958.84 ms |
| Shared prefill, one token | 87 | 63.62 / 60.93 ms | 103.51 ms |
| Two decoders | 2,320 | 43.33 / 43.08 ms | 60.75 ms |
| Short decode, drain arms | 2,755 | 20.14 / 20.65 ms | 31.19 ms |

Every mixed-arm request reaches its largest output gap on a shared
511- or 512-token prefill. All 348 gaps ending on those steps exceed
739 ms. The single-token closure of the same logical prefill reaches
100 ms once across the two orders.

Client intervals include transport and host scheduling. The separate
[runner trace](../flash-next-mxfp4-prefill-step-cost-m5-20261121/README.md)
measures actual model execution at the same physical widths.
The [width control](../flash-next-prefill-width-control-m2-20261122/README.md)
records state and token divergence when the previous 512-token
physical cut was reduced to 256 on its real-pack input.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
