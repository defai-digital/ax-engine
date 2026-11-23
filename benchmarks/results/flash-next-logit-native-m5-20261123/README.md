# Flash Next canonical logit selection on M5 Max

The signed baseline at `26c0ccefe` and candidate at `ec51bbbf2` ran
with the same MXFP4 pack, MLX 0.32.4, macOS 27.2.1, and 32,768-token
resident single-sequence profile on MacBook Pro M5 Max 128 GB with
internal SSD. Both processes use Required MTP and the canonical
singleton verifier. Each warms once and runs the full set of ten
natural prompts with its original budget and EOS handling. Prefix reuse
and generic n-gram acceleration are disabled. The embedded patch and
two build receipts identify the exact executable inputs.

| Workload | Exact output IDs | Original decode | Candidate decode | Change |
| --- | ---: | ---: | ---: | ---: |
| Python modules | 3,000 | 44.05 tok/s | 44.96 tok/s | +2.07% |
| General long form | 5,000 | 43.06 tok/s | 42.92 tok/s | -0.34% |
| Short arithmetic | 2 | 19.89 tok/s | 19.07 tok/s | One measured interval |

The baseline ran first. Its endpoint CPU idle readings were 98.93%
and 91.74%. The candidate readings were 95.48% and 93.37%. Each
request began and ended at normal pressure. Prompt hashes, all 8,002
output IDs, stop reasons, and verification counts agree between builds.
The outputs also match the archived direct serving control.

In Python modules, target correction and bonus wall time totals
63.217 seconds before the change and 61.924 seconds with it. For long
form, the same total is 107.169 and 107.337 seconds. The [independent
logit component measurement](../flash-next-logit-reductions-20261123/README.md)
measures the row selection separately. This serving pair records a
small coding gain and flat long-form throughput at one process order.
The candidate stays isolated pending the reverse order. MTP product
qualification and the standard serving default retain their own gates.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
