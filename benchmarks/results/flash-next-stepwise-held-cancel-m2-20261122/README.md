# Flash Next cancellation of a held prefill over stepwise HTTP

Mac Studio M2 Ultra 192 GB, external APFS storage, Rust 1.99.1
and MLX and MLX Metal 0.32.4. The native test uses the pinned
MXFP4 pack and the experimental resident two-slot profile with
MTP disabled. TensorFold's lane-ledger cancellation tests motivate
checking both exact request identity and the return of a slot.

The prior `2b905c012` trace admits an 8K request while its short
peer has 84 output tokens. Seven consecutive singleton decode
steps leave the long cursor at zero. The stepwise response omits
the hold marker. The next mixed step advances the long cursor
to 512. The archived scheduler counters, HTTP snapshots and old
binary identity reproduce the missing-observation condition.

The SDK change in `14235e84b` preserves the scheduler's hold marker
in its compact step-local route. The fixed server rejects a third
request with HTTP 429 while both slots are occupied. Its next
step reports one held singleton decode. Cancelling the long
request returns a terminal snapshot with zero prompt progress
and zero generated tokens. The short request is still at token
85 while the replacement is admitted to the released slot.
Both surviving requests finish and match the 192 total
[same-host solo output IDs](../flash-next-server-prefill-early-m2-20261122/README.md).
All five work counters drain.

All 58 memory-pressure samples are normal. The fixed run checks
all 49 publisher files before and after the controls. The software
suite passes 4,791 Rust and 345 Python tests. The old trace and
fixed receipt have separate executable digests. This is functional
cancellation and reporting evidence for the experimental profile.
The corrected M5 serving timing has a separate test.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
