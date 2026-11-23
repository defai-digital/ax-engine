# Flash Next Required MTP sampling fallback on M2 Ultra

The same signed guarded-reader server `d2220c26f` runs both MTP
policies on Mac Studio M2 Ultra 192 GiB, external SSD, macOS 27.2.1,
and MLX and MLX Metal 0.32.4. The live process maps the pinned MLX
library. Both arms verify all 49 publisher files and the derived
manifest for MXFP4 revision `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`
before and after serving.

Each arm sends six native SSE requests on one resident slot, using
the same frozen 512-token input and a 64-token output budget.
Prefix reuse, generic n-gram acceleration, and MTP cost parking
are disabled for these controls.

| Request | Required verified steps | Required non-strict direct steps |
| --- | ---: | ---: |
| Greedy before the fallback controls | 41 | 0 |
| Seed A, temperature 0.9 | 0 | 63 |
| Seed B, temperature 0.9 | 0 | 63 |
| Repeat seed A after seed B | 0 | 63 |
| Greedy with repetition penalty 1.2 | 0 | 63 |
| Greedy after the fallback controls | 41 | 0 |

Required and Disabled emit identical arrays at all 384 corresponding
positions. Seed A repeats and seed B changes the sequence in both
policies. The greedy output also repeats before and after the fallback
controls in each process. Every request drains all five resource
counters with zero MTP errors. All 91 pressure samples are normal.
Peak process footprint is 80.34 GiB.

The first capture expected an OpenAI completion sentinel on the
native SSE route. The corrected parser uses the typed terminal
response. A subsequent attempt recorded all six Disabled responses
before an auto-placement log assertion rejected the explicitly
resident launch. The completed replay checks the mapped resident
reader event and both post-run inventories.

This bounds request-level sampling fallback and greedy recovery on
the tested native route and hardware. MTP acceleration, default
promotion, and the standard M5 release retain their own evidence.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
