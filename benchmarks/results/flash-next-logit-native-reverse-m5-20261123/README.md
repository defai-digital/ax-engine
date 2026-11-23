# Flash Next canonical logit selection in reverse order on M5 Max

The [original serving pair](../flash-next-logit-native-m5-20261123/README.md)
runs the baseline first. This repeat runs the same signed candidate first and
reverses the prompt order in both processes. Both use the pinned resident
MXFP4 pack, canonical Required MTP, one active request, a 32,768-token context,
normal EOS, and the same ten frozen prompts. Hardware is MacBook Pro M5 Max
128 GB with internal SSD, macOS 27.2.1 and MLX 0.32.4. Full publisher inventory
checks and signed executable digests match before and after the run.

| Process order | Workload | Original decode | Candidate decode | Change |
| --- | --- | ---: | ---: | ---: |
| Baseline then candidate | Python modules | 44.05 tok/s | 44.96 tok/s | +2.07% |
| Candidate then baseline | Python modules | 43.55 tok/s | 43.78 tok/s | +0.52% |
| Baseline then candidate | General long form | 43.06 tok/s | 42.92 tok/s | -0.34% |
| Candidate then baseline | General long form | 45.34 tok/s | 45.16 tok/s | -0.39% |

In the reverse order the candidate CPU idle endpoints were 98.73% and 98.35%,
and the baseline endpoints were 99.17% and 98.88%. Every measured request
began and ended at normal memory pressure. All four processes match all 8,002
output IDs, including the two-token short-answer control. Prompt hashes,
finish reasons, verifier schedules, and counts for verified, accepted and
emitted tokens agree across process orders.

On the reverse coding panel, correction and bonus together fall from 64.106
to 63.738 seconds. The same long-form total rises from 101.887 to 102.193
seconds. The coding gain varies with process order, and long-form throughput
is slightly lower in both pairs. Keep the change on the experimental branch.
The public release, product profile, and MTP defaults keep their existing gates.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
