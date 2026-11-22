# Flash Next early arrival and held cancellation on M5 Max

This functional check uses the same signed `14235e84b` release-server
binary, pinned MXFP4 pack and resident two-slot policy as the
[counterbalanced late-arrival comparison](../flash-next-server-prefill-tail-fixed-m5-20261122/README.md).
The host is a MacBook Pro M5 Max 128 GB with macOS 27.2.1 and internal
NVMe storage. Rust is 1.99.1 and MLX/MLX Metal is 0.32.4.

- With the long request sent at short-output token 48, all four
  logical prefill items finish before the short stream reaches token
  84. With arrival at token 65, the fourth logical item retains its
  physical cuts across token 84. Both trials produce zero hold markers
  and match all 384 frozen solo output IDs.
- The stepwise API reports a held singleton decode at step 556. The
  original long request still has zero prompt and output progress.
  While both slots are occupied, a third request receives HTTP
  `429 concurrency_limit`. Cancelling the held request frees the slot
  for a replacement while the short request remains active at token 85.
  Both survivors match all 192 frozen solo output IDs.

Every completed case drains all five work counters. All 187 sampled
pressure readings are normal, with a peak process footprint of
75.28 GiB. CPU idle is 92.00% before and 92.57% after. Revalidation
after the controls matches the 49-member publisher inventory.

The archive binds the exact binary and inputs to all three native
traces. The replay checks request order, token trajectories, prefill
geometry, cancellation, slot reuse and drainage. Sustained mixed
traffic has its own acceptance gate. The product qualification still
uses the one-slot resident profile.

```sh
shasum -a 256 -c SHA256SUMS
python3 -B replay.py | cmp comparison.json -
```
