# Flash Next independent reference heartbeat stopped under memory pressure

The MacBook Pro M5 Max 128 GB ran the frozen four-case MLX-VLM
heartbeat for the pinned MXFP4 pack on macOS 27.2.1. The preflight
reported normal memory pressure and 4.33 GiB for the largest
background process, within the 6.44 GiB signed memory margin.

The monitor completed three cases. On the fourth it observed 114
generated tokens before stopping the owned process at critical
memory pressure. The 288 host observations contain 98 normal,
189 warning and one critical reading. The final observation
reports 59.02 GiB of compressed pages and 32,819.50 MiB of
swap in use. The largest other process had 18.30 GiB RSS at that
sample. These are host-wide observations.

The heartbeat did not admit the ordinary reference workload.
The reference still owes the full 105-item QA result and the
15 fixed timing cells for the current release comparison.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
