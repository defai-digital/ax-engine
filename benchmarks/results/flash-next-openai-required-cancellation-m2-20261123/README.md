# Flash Next Required MTP cancellation through OpenAI SSE

The signed server previously tested through the
[native endpoints](../flash-next-required-cancellation-m2-20261123/README.md)
runs the pinned MXFP4 pack on Mac Studio M2 Ultra 192 GB, internal SSD,
macOS 27.2.1, and MLX and MLX Metal 0.32.4. Both arms use one resident
slot, mapped weights, a frozen 512-token input and a 64-token greedy
output budget. The fixture disables prefix reuse and generic n-gram
acceleration and holds the experimental cost controller off.

Each process completes an OpenAI-compatible `/v1/completions` SSE
request, disconnects a second request after receiving text, and
repeats the original request. The completed streams each emit
one length terminal, the usage record, and `[DONE]`. In each
policy, the recovery text matches the baseline character for
character and the text visible before disconnect is a prefix.

The Required arm reports four verifier steps and one acceptance
while its interrupted stream is active. Metrics advance to six
verifications and two acceptances after the interruption drains.
Each completed Required request records 41 verified steps and 22
acceptances. Disabled completes the same text with both counters
at zero. MTP error counters stay at zero and all five
admission and stream counters drain after every action. All 71 pressure
samples are normal. Peak process footprint is 80.52 GiB.

The launcher checks all 49 publisher members, the derived manifest
and the exact MLX library. This evidence covers the measured
OpenAI streaming completion lifecycle and exported counters on M2.
The native evidence retains its own corresponding output IDs.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
