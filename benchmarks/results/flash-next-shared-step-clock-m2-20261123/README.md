# Flash Next shared-step timing on M2 Ultra

Four Flash Next short/long prompt pairs are selected from the
[sealed M5 natural-arrival crossover](../flash-next-natural-short-drain-m5-20261123/README.md).
Each is the median-ranked or largest per-request gap in its width group.
The fixture extractor verifies that archive's replay before selecting
the inputs and matched engine steps. The native fingerprint is the
same signed executable as the M5 capture. The Mac Studio M2 Ultra
192 GB used MLX 0.32.4, internal SSD, resident experts, two slots,
and MTP disabled.

The stepwise probe submits the same 8,192-token long prompt, advances it
to the recorded 2,048-token cursor, and admits its 512-token peer.
The target step must have the recorded prefill history, physical width,
logical boundary, and short output depth. All 41 recorded short
token IDs match the same-host solo control. One of four short
prefixes also matches the original M5 SSE capture. That comparison
changes both host and transport and leaves the cause of the token
differences unverified.

| Long prefill width | Mixed runner | Prefill forward | Engine time outside runner |
| ---: | ---: | ---: | ---: |
| 511 | 1,349.50 ms | 1,309.76 ms | 45 µs |
| 511 | 1,346.91 ms | 1,308.00 ms | 44 µs |
| 512 | 1,337.44 ms | 1,296.67 ms | 41 µs |
| 512 | 1,312.49 ms | — | 45 µs |

The prefill fields in the route are cumulative across the selected
requests. The first three rows subtract the prior step with the same
participants to recover the prefill forward within the target runner
interval. That forward accounts for 97.1% of runner time at the median.
The fourth row's prior step contains only the short request, so the
capture lacks the long request's cumulative baseline. Its runner and
engine clocks remain valid. The separate solo decode control takes
27.86–28.90 ms inside the runner.

This diagnostic uses the native step endpoint, explicit request
snapshots, and post-cancellation drain checks. It measures the
existing mixed route. M5 attribution and an ordinary SSE repeat
are separate controls. The standard product profile uses one
active request.

## Replay

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```

The [fixture extractor](fixtures.py) verifies and reads the sealed
natural-arrival archive. To rerun the hardware probe, extract the
following inputs from that archive's `bundle/` alongside `probe.py`
and `fixtures.json`: `build.json`, `source.patch`,
`mxfp4-inventory.json`, `check_flash_next_native_support.py`,
`qualify_qwen38_flash_next.py`, and `run_qwen38_reference_serving.py`.
Supply the matching signed binary and verified pack with the
`model-manifest.json` used by the capture. The probe verifies the
publisher inventory before and after the run.
