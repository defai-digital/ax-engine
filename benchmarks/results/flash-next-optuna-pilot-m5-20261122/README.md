# Flash Next offline Optuna scheduling pilot

The [study driver](../../../scripts/tune_flash_next_serving.py) compares
seeded Optuna TPE and a fixed grid under the same unique evaluation
budget. Duplicate TPE suggestions are recorded and replaced with the
first unseen setting. The measured study uses the
[sealed M5 Max serving campaign](../flash-next-mxfp4-selective-prefill-m5-20261121/README.md).
Each policy has 220 requests per process. The audited raw streams must
match the independently recorded solo greedy tokens, retain the fixed
prefill boundaries, demonstrate the selected split route, drain serving
counters and keep sampled pressure normal. Throughput is recomputed
from the original client timeline.

| Policy | First-order short p95 stream gap | First-order long p95 arrival TTFT | First-order throughput |
| --- | ---: | ---: | ---: |
| Off | 52.83 ms | 40.05 s | 23.11 tok/s |
| All eligible prompts | 63.78 ms | 48.67 s | 22.99 tok/s |
| Long prompts only | 404.20 ms | 46.43 s | 22.85 tok/s |

Both searches evaluate all three measured policies and retain only
off on the training frontier. The reverse process order replays the
same traffic. An extra `--holdout-bundle` supplies new offered traffic
on the same executable, checkpoint and split threshold. The driver
fingerprints token IDs, arrival offsets and output budgets to reject a
renamed copy of the same traffic. A candidate must beat the baseline
without regression in training and every validation dataset.

The other study uses the [fixed-width repayment model](../flash-next-prefill-repayment-sweep-m5-20261122/README.md),
with median measured decoder costs for search and p10/p90 costs for
validation. It records no long first-token objective outside the
captured schedule. With 24 unique settings per search, both choose
0% repayment. Both also choose 0% when they exhaust all 101 settings.
These results replay archived measurements and fixed-cost projections.
They supply no new hardware timings or release qualification.

Run from the repository root with the developer-only dependency.

```bash
python3 -m venv /tmp/ax-engine-optuna
/tmp/ax-engine-optuna/bin/python -m pip install -r scripts/requirements-optuna.txt
/tmp/ax-engine-optuna/bin/python scripts/tune_flash_next_serving.py serving
/tmp/ax-engine-optuna/bin/python scripts/tune_flash_next_serving.py repayment --budget 24
```

The [measured](serving.json) and [projected](repayment.json) reports use
Optuna 5.1.0 and seed 42.
