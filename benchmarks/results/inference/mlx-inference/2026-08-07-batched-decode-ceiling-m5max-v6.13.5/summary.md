# Batched decode ceiling: Shared vs RowExact

- Host: Apple M5 Max
- Engine: 6.13.5
- Commit: `93b897731b8863365465df5b08ee174cb9fc56c1`
- Repetitions: 5 per policy
- Publication candidate: true

| Policy | Batch | Median agg tok/s | Median per-policy scaling | Median step µs |
|---|---:|---:|---:|---:|
| shared | 1 | 82.1 | 1.00× | 12182 |
| shared | 2 | 146.8 | 1.79× | 13625 |
| shared | 4 | 271.0 | 3.30× | 14761 |
| shared | 8 | 328.9 | 4.01× | 24325 |
| row_exact | 1 | 82.2 | 1.00× | 12166 |
| row_exact | 2 | 91.1 | 1.11× | 21962 |
| row_exact | 4 | 99.2 | 1.21× | 40326 |
| row_exact | 8 | 102.6 | 1.25× | 77970 |

At batch 8, Shared / RowExact has a paired median ratio of **3.20×** (5 wins, 0 ties, 0 losses).
