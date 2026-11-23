# Flash Next priority short decode drain on M2 Ultra

Mac Studio M2 Ultra 192 GiB, macOS 27.2.1, internal NVMe SSD, and
MLX and MLX Metal 0.32.4. Both controls use the same signed release
server from experimental commit `71f891b25` and the pinned MXFP4
pack `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. The runner
verifies all 49 publisher members and uses mapped resident weights,
two slots, and MTP off.

The long request prefills to 2,048 of 8,192 tokens. In both controls
the 512-token short request uses one whole priority prefill and emits
its first of 96 greedy IDs.

| Control after the priority turn | Observed resume |
| --- | --- |
| Mixed decode and prefill | The next step advances the long cursor to 2,560 alongside one short decode token. |
| Short decode drain | The long cursor stays at 2,048 for 95 short-only decode steps. After the short request completes, the next step advances the long cursor to 4,096. |

Cancelling the short request after its second token gives the original
long request the next 2,048-token turn. A replacement stays at zero
prefill during that turn. Both completed pairs and the completed
cancellation replacement pair match all 576 solo output IDs. The
cancelled prefix also matches. All five native counters drain and all
191 memory pressure samples are normal. Peak footprint is 85.89 GiB.

This is a same-binary functional control. Primary-tier timing and
sustained admission evidence remain separate.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
