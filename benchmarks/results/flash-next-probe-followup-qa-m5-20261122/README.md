# Flash Next storage-probe follow-up QA on M5 Max

The signed `26c0ccefe` release-server binary from the
[retained native controls](../flash-next-probe-followup-http-m5-20261122/README.md)
ran the frozen QA cohort using the pinned MXFP4 pack. The MacBook Pro
used M5 Max, 128 GB unified memory, internal NVMe SSD, macOS 27.2.1,
and MLX 0.32.4. CPU idle was 96.38% before the cohort and 98.91% after.

| Mode | Frozen checks passed | HTTP ready | 29,774-token lookup | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| MTP disabled | 101/105 | 98.990 s | 42.118 s | 78.747 GiB |
| Ordinary default | 101/105 | 103.028 s | 40.811 s | 78.879 GiB |

All 210 responses match the signed baseline in text, stop reason, and
prompt, completion, and total token counts. Both modes reproduce the
same four retained checker failures. They advertise 32,768 context
tokens and report zero verifier steps and route errors. Memory and long
lookup pass their frozen bounds on the retained snapshot.

Replay binds the build, inventory, original QA checkers, response
oracle, and raw responses. The independent delivery, lifecycle,
repeat matrix, and ordinary MLX-VLM reference have their own receipts.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
