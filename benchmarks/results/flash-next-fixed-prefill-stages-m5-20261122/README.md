# Flash Next fixed split prefill stage costs on M5 Max

The M5 Max 128 GB repeats the [M2 fixed-geometry experiment](../flash-next-fixed-prefill-stages-m2-20261122/README.md)
with the same release-server test executable and pinned 8,192-token
prompt. The launcher verifies the current production matrix inventory
against the internal-SSD pack. The 20 physical forwards retain the four
512, 512, 512, 511, 1 sequences. The traced and untraced final logits
and serialized state match bit for bit. CPU idle is 91.50% before and
99.20% after the test.

| Synchronized attribution | Time | Share of traced wall |
| --- | ---: | ---: |
| Expert compute and routing | 3.403 s | 33.1% |
| QSA stages combined | 3.174 s | 30.9% |
| Gated DeltaNet | 1.644 s | 16.0% |
| Cache-only terminal stateless suffix | 0.102 s | 0.99% |

The terminal suffix overlaps the compute stage families in the table.
It is a measurement of the proposed savings scope, so do not sum that
row with the family rows. The largest single stage is expert compute
at 3.090 s. QSA selected-key SDPA and gather use 1.522 s and 0.845 s.

The untraced run takes 9.159 s with a median of 0.562 s across 511/512
forwards. The synchronized run takes 10.267 s. The top two stage
families justify prioritizing expert compute and exact QSA on this
device. The model-only profile has no concurrent request or HTTP
streaming, so client gaps require a separate serving comparison.

The original [stage intervals](attribution.json) replay against the
confirmed [probe and input hashes](inputs.sha256).

```bash
python3 benchmarks/results/flash-next-fixed-prefill-stages-m5-20261122/replay.py
```
