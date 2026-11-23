# Flash Next medium request handoff on M2 Ultra

The same signed experimental server from `71f891b25` runs on Mac
Studio M2 Ultra 192 GiB, internal SSD, macOS 27.2.1, and MLX and MLX
Metal 0.32.4. It uses the MXFP4 pack
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`, mapped resident weights,
two slots, and MTP off. Both arms enable short-first and differ only
in the short decode drain switch. The three prompts come from one
matching handoff in the M5 natural-arrival workload.

Start an 8,192-token prompt and advance its prefill to 2,048 tokens.
Give the 512-token request the next whole prefill turn and let it
finish. Submit the 2,048-token successor as soon as that slot frees.
The native stepwise HTTP snapshots verify the wait through the long
request's final decode.

| Policy after the short prefill turn | Long prefill cursor at short completion | Long output IDs emitted at short completion | Successor wait in engine steps |
| --- | ---: | ---: | ---: |
| Mixed prefill and decode | 8,192 | 81 | 15 |
| Short decode drain | 2,048 | 0 | 98 |

The successor stays `waiting` with zero prefill and output in every
snapshot through long completion. Both arms start the successor
with one whole 2,048-token prefill on the next engine step.
All 576 completed output IDs match the M2 solo oracle, all five
resource counters drain, and all 184 pressure samples are normal.
Peak process footprint is 85.90 GiB. All 49 publisher members and
the derived manifest match before and after the run.

The snapshots establish the admission boundary for the measured
pair. The step counts describe this controlled workload. M5
natural-arrival timing and supported concurrency require separate
evidence.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
