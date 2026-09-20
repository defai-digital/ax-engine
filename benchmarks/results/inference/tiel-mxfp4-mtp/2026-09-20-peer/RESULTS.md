# Tiel peer results: M5 Max, M4 Pro, M2 Ultra and M3 Ultra

UTC campaign date: 2026-09-20. See the [contract and exact provenance](README.md).
All primary cells use six measured samples from two reversed-order blocks.
Primary metric is completion tok/s including TTFT. These are resident,
explicit throughput-MTP native-library measurements, not default server performance.

**M4 requires explicit expert-stream Off. Auto retains the 48 GiB reserve and pages these packs.**

Both engines retain the full model throughout measured requests. Unwired
means the OS residency lock is disabled; it does not unload the model.

## Coding-workload overview

Observed AX normal-wiring completion medians versus MTPLX sustained across
the four Python/Rust coding cells per host. Ranges are per-cell differences,
not a pooled speedup or confidence interval. Synthetic input is reported below.

| Host | AX difference versus sustained |
| --- | ---: |
| M5 | +3.1% to +18.0% |
| M4 | -2.4% to -0.5% |
| M2 | -2.1% to +8.6% |
| M3 | +2.2% to +4.7% |

## MacBook Pro M5 Max / 128 GiB

| Model / workload | AX normal wiring | MTPLX sustained | MTPLX turbo | AX vs sustained |
| --- | ---: | ---: | ---: | ---: |
| tiel / random128 | 118.35 | 109.98 | 112.43 | +7.6% |
| tiel / python-lru | 194.88 | 177.44 | 178.47 | +9.8% |
| tiel / rust-jsonl | 172.08 | 145.82 | 139.93 | +18.0% |
| cyber / random128 | 170.88 | 172.83 | 113.93 | -1.1% |
| cyber / python-lru | 219.43 | 194.15 | 201.52 | +13.0% |
| cyber / rust-jsonl | 169.14 | 163.99 | 159.01 | +3.1% |

AX normal wiring means the ordinary wiring policy **within this explicitly resident comparison**:
the M5 audited-export policy releases wiring; M4, M2 and M3 keep existing wiring.
The M4 resident override is not the shipped Auto paging path.

### TTFT and decode

| Model / workload | AX TTFT ms | Sustained TTFT ms | Turbo TTFT ms | AX decode tok/s | Sustained decode | Turbo decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiel / random128 | 138.09 | 162.44 | 162.73 | 134.57 | 126.87 | 130.27 |
| tiel / python-lru | 141.77 | 156.80 | 157.64 | 217.85 | 198.40 | 199.72 |
| tiel / rust-jsonl | 281.58 | 293.99 | 293.13 | 211.48 | 174.49 | 165.86 |
| cyber / random128 | 139.92 | 163.57 | 164.56 | 208.45 | 220.86 | 132.44 |
| cyber / python-lru | 142.08 | 158.41 | 156.99 | 249.01 | 219.84 | 229.06 |
| cyber / rust-jsonl | 279.60 | 293.04 | 296.14 | 206.59 | 201.13 | 194.22 |

### Sample spread, order and draft efficiency

| Model / workload / arm | Completion min / median / max tok/s | Block 0 / 1 median | Accepted / drafted | Output variants |
| --- | ---: | ---: | ---: | ---: |
| cyber / python-lru / ax | 218.91 / 219.43 / 220.58 | 220.43 / 219.08 | 1104/1248 (88.5%) | 1 |
| cyber / python-lru / sustained | 192.93 / 194.15 / 194.66 | 194.10 / 194.49 | 1110/1278 (86.9%) | 1 |
| cyber / python-lru / turbo | 200.43 / 201.52 / 202.55 | 201.26 / 201.75 | 1110/1260 (88.1%) | 1 |
| cyber / random128 / ax | 170.66 / 170.88 / 171.11 | 170.89 / 170.87 | 516/702 (73.5%) | 1 |
| cyber / random128 / sustained | 170.58 / 172.83 / 174.68 | 172.88 / 172.78 | 552/636 (86.8%) | 1 |
| cyber / random128 / turbo | 113.03 / 113.93 / 114.32 | 113.90 / 113.96 | 420/1014 (41.4%) | 1 |
| cyber / rust-jsonl / ax | 168.32 / 169.14 / 169.42 | 169.19 / 169.07 | 1044/1434 (72.8%) | 1 |
| cyber / rust-jsonl / sustained | 163.59 / 163.99 / 164.59 | 164.12 / 163.87 | 1074/1368 (78.5%) | 1 |
| cyber / rust-jsonl / turbo | 158.77 / 159.01 / 159.39 | 159.05 / 158.94 | 1062/1410 (75.3%) | 1 |
| tiel / python-lru / ax | 194.48 / 194.88 / 195.67 | 194.83 / 194.93 | 1056/1404 (75.2%) | 1 |
| tiel / python-lru / sustained | 177.21 / 177.44 / 177.90 | 177.60 / 177.28 | 1062/1416 (75.0%) | 1 |
| tiel / python-lru / turbo | 177.91 / 178.47 / 178.81 | 178.57 / 178.38 | 1062/1410 (75.3%) | 1 |
| tiel / random128 / ax | 118.06 / 118.35 / 118.70 | 118.15 / 118.40 | 342/1044 (32.8%) | 1 |
| tiel / random128 / sustained | 108.93 / 109.98 / 110.09 | 109.99 / 109.97 | 390/1098 (35.5%) | 1 |
| tiel / random128 / turbo | 111.32 / 112.43 / 112.89 | 112.38 / 112.47 | 420/1044 (40.2%) | 1 |
| tiel / rust-jsonl / ax | 171.64 / 172.08 / 172.79 | 172.66 / 171.76 | 1056/1380 (76.5%) | 1 |
| tiel / rust-jsonl / sustained | 145.42 / 145.82 / 146.71 | 145.85 / 145.80 | 1008/1578 (63.9%) | 1 |
| tiel / rust-jsonl / turbo | 139.37 / 139.93 / 140.16 | 139.98 / 139.50 | 990/1626 (60.9%) | 1 |

Draft efficiency is each engine's accepted/drafted counter ratio across measured
requests, not output quality or a forced-trajectory comparison. Block medians expose
order/background drift; six samples do not establish population confidence bounds.

### MLX allocator memory

Measured request-end active ranges and highest recorded allocator peak/cache.
These are not process RSS or whole-system memory totals.

| Arm | Active min / max GiB | Highest peak GiB | Highest cache GiB |
| --- | ---: | ---: | ---: |
| ax | 21.25 / 21.25 | 22.41 | 0.10 |
| sustained | 19.95 / 19.95 | 21.09 | 1.72 |
| turbo | 19.95 / 19.95 | 21.09 | 1.73 |

OS: 27.0 (26A428); Python: 3.14.7; MLX 0.32.2.
Observed boundary pressure levels: ['1'].
Largest recorded MLX allocator peak: 22.41 GiB (not process RSS).
Swap snapshots: total = 1024.00M  used = 519.75M  free = 504.25M  (encrypted).
Selected background process CPU snapshots (% of one core per process, not an aggregate): `{"audiomxd": {"max": 0.0, "min": 0.0}, "configd": {"max": 0.0, "min": 0.0}, "fseventsd": {"max": 0.0, "min": 0.0}, "mds": {"max": 0.1, "min": 0.0}, "mds_stores": {"max": 0.1, "min": 0.0}, "mdworker_shared": {"max": 10.9, "min": 0.0}, "mediaanalysisd": {"max": 0.0, "min": 0.0}}`.

## Mac mini M4 Pro / 64 GiB

| Model / workload | AX normal wiring | MTPLX sustained | MTPLX turbo | AX vs sustained |
| --- | ---: | ---: | ---: | ---: |
| tiel / random128 | 63.09 | 64.53 | 62.50 | -2.2% |
| tiel / python-lru | 90.46 | 92.23 | 93.62 | -1.9% |
| tiel / rust-jsonl | 64.62 | 64.94 | 60.97 | -0.5% |
| cyber / random128 | 86.94 | 58.66 | 57.88 | +48.2% |
| cyber / python-lru | 94.37 | 96.65 | 94.30 | -2.4% |
| cyber / rust-jsonl | 67.71 | 68.92 | 70.17 | -1.8% |

AX normal wiring means the ordinary wiring policy **within this explicitly resident comparison**:
the M5 audited-export policy releases wiring; M4, M2 and M3 keep existing wiring.
The M4 resident override is not the shipped Auto paging path.

### TTFT and decode

| Model / workload | AX TTFT ms | Sustained TTFT ms | Turbo TTFT ms | AX decode tok/s | Sustained decode | Turbo decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiel / random128 | 405.18 | 322.51 | 321.04 | 78.21 | 76.47 | 73.62 |
| tiel / python-lru | 493.77 | 393.47 | 393.62 | 109.12 | 107.04 | 108.94 |
| tiel / rust-jsonl | 1344.37 | 1244.43 | 1230.86 | 97.39 | 94.49 | 85.92 |
| cyber / random128 | 396.17 | 325.69 | 320.49 | 117.97 | 68.48 | 67.22 |
| cyber / python-lru | 519.00 | 414.17 | 415.76 | 116.29 | 114.25 | 110.93 |
| cyber / rust-jsonl | 1364.65 | 1271.92 | 1267.96 | 105.51 | 104.34 | 107.19 |

### M4 unwired diagnostic (expert streaming still Off)

This is an explicit `AX_MLX_WIRED_LIMIT_SCALE=0` control, not automatic tuning or a default promotion.

| Model / workload | AX wired completion | AX unwired completion | Completion change | Wired TTFT ms | Unwired TTFT ms | Decode change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiel / random128 | 63.09 | 67.31 | +6.7% | 405.18 | 258.42 | -1.2% |
| tiel / python-lru | 90.46 | 94.41 | +4.4% | 493.77 | 350.96 | -1.0% |
| tiel / rust-jsonl | 64.62 | 65.47 | +1.3% | 1344.37 | 1200.46 | -3.4% |
| cyber / random128 | 86.94 | 95.08 | +9.4% | 396.17 | 253.93 | -1.3% |
| cyber / python-lru | 94.37 | 98.97 | +4.9% | 519.00 | 372.73 | -1.0% |
| cyber / rust-jsonl | 67.71 | 68.81 | +1.6% | 1364.65 | 1229.64 | -3.2% |

### Sample spread, order and draft efficiency

| Model / workload / arm | Completion min / median / max tok/s | Block 0 / 1 median | Accepted / drafted | Output variants |
| --- | ---: | ---: | ---: | ---: |
| cyber / python-lru / ax | 93.93 / 94.37 / 94.55 | 94.33 / 94.40 | 1080/1302 (82.9%) | 1 |
| cyber / python-lru / ax-unwired | 97.10 / 98.97 / 99.13 | 98.94 / 99.10 | 1080/1302 (82.9%) | 1 |
| cyber / python-lru / sustained | 96.25 / 96.65 / 97.02 | 96.53 / 96.95 | 1104/1296 (85.2%) | 1 |
| cyber / python-lru / turbo | 93.74 / 94.30 / 94.51 | 94.37 / 94.12 | 1074/1374 (78.2%) | 1 |
| cyber / random128 / ax | 86.20 / 86.94 / 87.17 | 86.94 / 86.95 | 540/630 (85.7%) | 1 |
| cyber / random128 / ax-unwired | 94.94 / 95.08 / 95.38 | 95.06 / 95.35 | 540/630 (85.7%) | 1 |
| cyber / random128 / sustained | 58.48 / 58.66 / 58.76 | 58.68 / 58.64 | 408/1068 (38.2%) | 1 |
| cyber / random128 / turbo | 57.50 / 57.88 / 58.06 | 57.82 / 57.95 | 390/1110 (35.1%) | 1 |
| cyber / rust-jsonl / ax | 67.50 / 67.71 / 67.77 | 67.69 / 67.75 | 1056/1398 (75.5%) | 1 |
| cyber / rust-jsonl / ax-unwired | 68.47 / 68.81 / 68.96 | 68.74 / 68.92 | 1056/1398 (75.5%) | 1 |
| cyber / rust-jsonl / sustained | 68.55 / 68.92 / 69.29 | 68.58 / 68.94 | 1068/1392 (76.7%) | 1 |
| cyber / rust-jsonl / turbo | 69.95 / 70.17 / 70.51 | 70.13 / 70.21 | 1068/1398 (76.4%) | 1 |
| tiel / python-lru / ax | 89.94 / 90.46 / 90.65 | 90.54 / 90.17 | 1050/1392 (75.4%) | 1 |
| tiel / python-lru / ax-unwired | 93.57 / 94.41 / 94.59 | 94.40 / 94.59 | 1050/1392 (75.4%) | 1 |
| tiel / python-lru / sustained | 91.90 / 92.23 / 92.66 | 92.49 / 92.01 | 1068/1386 (77.1%) | 1 |
| tiel / python-lru / turbo | 92.94 / 93.62 / 93.85 | 93.63 / 93.55 | 1068/1404 (76.1%) | 1 |
| tiel / random128 / ax | 62.54 / 63.09 / 63.28 | 63.27 / 62.73 | 432/918 (47.1%) | 1 |
| tiel / random128 / ax-unwired | 66.82 / 67.31 / 67.41 | 67.28 / 67.33 | 432/918 (47.1%) | 1 |
| tiel / random128 / sustained | 63.96 / 64.53 / 64.88 | 64.66 / 64.31 | 444/966 (46.0%) | 1 |
| tiel / random128 / turbo | 61.66 / 62.50 / 62.65 | 62.52 / 62.48 | 420/1020 (41.2%) | 1 |
| tiel / rust-jsonl / ax | 64.43 / 64.62 / 64.66 | 64.47 / 64.65 | 1014/1488 (68.1%) | 1 |
| tiel / rust-jsonl / ax-unwired | 65.23 / 65.47 / 66.21 | 65.48 / 65.45 | 1014/1488 (68.1%) | 1 |
| tiel / rust-jsonl / sustained | 64.79 / 64.94 / 65.26 | 64.95 / 64.87 | 1014/1536 (66.0%) | 1 |
| tiel / rust-jsonl / turbo | 60.70 / 60.97 / 61.09 | 60.95 / 61.03 | 954/1728 (55.2%) | 1 |

Draft efficiency is each engine's accepted/drafted counter ratio across measured
requests, not output quality or a forced-trajectory comparison. Block medians expose
order/background drift; six samples do not establish population confidence bounds.

### MLX allocator memory

Measured request-end active ranges and highest recorded allocator peak/cache.
These are not process RSS or whole-system memory totals.

| Arm | Active min / max GiB | Highest peak GiB | Highest cache GiB |
| --- | ---: | ---: | ---: |
| ax | 21.25 / 21.25 | 22.46 | 0.10 |
| sustained | 19.95 / 19.95 | 21.06 | 1.72 |
| turbo | 19.95 / 19.95 | 21.06 | 1.73 |
| ax-unwired | 21.25 / 21.25 | 22.46 | 0.10 |

OS: 26.6.2 (25G83); Python: 3.14.6; MLX 0.32.2.
Observed boundary pressure levels: ['1'].
Largest recorded MLX allocator peak: 22.46 GiB (not process RSS).
Swap snapshots: total = 1024.00M  used = 17.06M  free = 1006.94M  (encrypted).
Selected background process CPU snapshots (% of one core per process, not an aggregate): `{"audiomxd": {"max": 79.4, "min": 56.8}, "configd": {"max": 37.8, "min": 15.3}, "fseventsd": {"max": 0.0, "min": 0.0}, "mds": {"max": 0.0, "min": 0.0}, "mds_stores": {"max": 0.0, "min": 0.0}, "mdworker_shared": {"max": 0.0, "min": 0.0}}`.

## Mac Studio M2 Ultra / 192 GiB

| Model / workload | AX normal wiring | MTPLX sustained | MTPLX turbo | AX vs sustained |
| --- | ---: | ---: | ---: | ---: |
| tiel / random128 | 73.91 | 72.47 | 74.98 | +2.0% |
| tiel / python-lru | 107.28 | 104.96 | 103.12 | +2.2% |
| tiel / rust-jsonl | 81.24 | 82.98 | 84.90 | -2.1% |
| cyber / random128 | 71.89 | 63.18 | 60.69 | +13.8% |
| cyber / python-lru | 117.59 | 108.27 | 113.78 | +8.6% |
| cyber / rust-jsonl | 84.80 | 84.51 | 84.46 | +0.3% |

AX normal wiring means the ordinary wiring policy **within this explicitly resident comparison**:
the M5 audited-export policy releases wiring; M4, M2 and M3 keep existing wiring.
The M4 resident override is not the shipped Auto paging path.

### TTFT and decode

| Model / workload | AX TTFT ms | Sustained TTFT ms | Turbo TTFT ms | AX decode tok/s | Sustained decode | Turbo decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiel / random128 | 437.86 | 313.50 | 307.13 | 98.56 | 87.28 | 90.98 |
| tiel / python-lru | 494.41 | 346.18 | 352.92 | 134.60 | 122.13 | 119.69 |
| tiel / rust-jsonl | 1019.26 | 872.80 | 872.93 | 119.42 | 115.25 | 118.89 |
| cyber / random128 | 414.09 | 296.78 | 306.26 | 92.76 | 73.80 | 70.59 |
| cyber / python-lru | 505.93 | 341.83 | 358.27 | 152.53 | 125.98 | 134.92 |
| cyber / rust-jsonl | 1010.16 | 860.06 | 891.54 | 126.67 | 117.52 | 119.15 |

### M2 unwired diagnostic (expert streaming still Off)

This is an explicit `AX_MLX_WIRED_LIMIT_SCALE=0` control, not automatic tuning or a default promotion.

| Model / workload | AX wired completion | AX unwired completion | Completion change | Wired TTFT ms | Unwired TTFT ms | Decode change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiel / random128 | 73.91 | 82.95 | +12.2% | 437.86 | 257.64 | +0.6% |
| tiel / python-lru | 107.28 | 115.76 | +7.9% | 494.41 | 298.34 | -1.1% |
| tiel / rust-jsonl | 81.24 | 84.33 | +3.8% | 1019.26 | 824.60 | -3.4% |
| cyber / random128 | 71.89 | 77.35 | +7.6% | 414.09 | 262.95 | -1.7% |
| cyber / python-lru | 117.59 | 128.28 | +9.1% | 505.93 | 312.62 | -0.6% |
| cyber / rust-jsonl | 84.80 | 87.30 | +3.0% | 1010.16 | 837.64 | -3.9% |

### Sample spread, order and draft efficiency

| Model / workload / arm | Completion min / median / max tok/s | Block 0 / 1 median | Accepted / drafted | Output variants |
| --- | ---: | ---: | ---: | ---: |
| cyber / python-lru / ax | 117.20 / 117.59 / 118.42 | 117.51 / 117.60 | 1116/1236 (90.3%) | 1 |
| cyber / python-lru / ax-unwired | 127.86 / 128.28 / 128.90 | 128.17 / 128.87 | 1116/1236 (90.3%) | 1 |
| cyber / python-lru / sustained | 106.86 / 108.27 / 108.41 | 108.35 / 107.85 | 1092/1320 (82.7%) | 1 |
| cyber / python-lru / turbo | 112.63 / 113.78 / 114.47 | 113.27 / 114.42 | 1104/1278 (86.4%) | 1 |
| cyber / random128 / ax | 71.56 / 71.89 / 72.13 | 71.97 / 71.84 | 402/930 (43.2%) | 1 |
| cyber / random128 / ax-unwired | 75.47 / 77.35 / 78.90 | 77.19 / 77.43 | 402/930 (43.2%) | 1 |
| cyber / random128 / sustained | 62.89 / 63.18 / 63.52 | 63.12 / 63.24 | 390/1122 (34.8%) | 1 |
| cyber / random128 / turbo | 59.96 / 60.69 / 61.72 | 60.54 / 61.01 | 378/1146 (33.0%) | 1 |
| cyber / rust-jsonl / ax | 83.44 / 84.80 / 85.01 | 84.93 / 84.16 | 1044/1446 (72.2%) | 1 |
| cyber / rust-jsonl / ax-unwired | 87.07 / 87.30 / 87.74 | 87.24 / 87.33 | 1044/1446 (72.2%) | 1 |
| cyber / rust-jsonl / sustained | 84.30 / 84.51 / 84.83 | 84.72 / 84.50 | 1068/1392 (76.7%) | 1 |
| cyber / rust-jsonl / turbo | 83.68 / 84.46 / 84.89 | 83.80 / 84.80 | 1068/1392 (76.7%) | 1 |
| tiel / python-lru / ax | 106.85 / 107.28 / 107.45 | 107.42 / 107.23 | 1068/1398 (76.4%) | 1 |
| tiel / python-lru / ax-unwired | 115.25 / 115.76 / 116.44 | 115.64 / 115.87 | 1068/1398 (76.4%) | 1 |
| tiel / python-lru / sustained | 104.02 / 104.96 / 105.64 | 104.57 / 105.56 | 1074/1374 (78.2%) | 1 |
| tiel / python-lru / turbo | 102.43 / 103.12 / 103.46 | 103.38 / 102.85 | 1056/1410 (74.9%) | 1 |
| tiel / random128 / ax | 72.91 / 73.91 / 75.50 | 73.55 / 74.26 | 432/864 (50.0%) | 1 |
| tiel / random128 / ax-unwired | 81.49 / 82.95 / 84.27 | 82.75 / 83.15 | 432/864 (50.0%) | 1 |
| tiel / random128 / sustained | 71.96 / 72.47 / 72.96 | 72.55 / 72.39 | 444/948 (46.8%) | 1 |
| tiel / random128 / turbo | 74.23 / 74.98 / 75.77 | 74.67 / 75.29 | 456/912 (50.0%) | 1 |
| tiel / rust-jsonl / ax | 80.90 / 81.24 / 81.40 | 81.34 / 81.03 | 996/1506 (66.1%) | 1 |
| tiel / rust-jsonl / ax-unwired | 83.99 / 84.33 / 84.82 | 84.56 / 84.00 | 996/1506 (66.1%) | 1 |
| tiel / rust-jsonl / sustained | 82.76 / 82.98 / 83.19 | 83.05 / 82.92 | 1050/1440 (72.9%) | 1 |
| tiel / rust-jsonl / turbo | 84.68 / 84.90 / 85.17 | 84.89 / 84.94 | 1062/1404 (75.6%) | 1 |

Draft efficiency is each engine's accepted/drafted counter ratio across measured
requests, not output quality or a forced-trajectory comparison. Block medians expose
order/background drift; six samples do not establish population confidence bounds.

### MLX allocator memory

Measured request-end active ranges and highest recorded allocator peak/cache.
These are not process RSS or whole-system memory totals.

| Arm | Active min / max GiB | Highest peak GiB | Highest cache GiB |
| --- | ---: | ---: | ---: |
| ax | 21.25 / 21.25 | 22.46 | 0.10 |
| sustained | 19.95 / 19.95 | 21.06 | 1.72 |
| turbo | 19.95 / 19.95 | 21.06 | 1.73 |
| ax-unwired | 21.25 / 21.25 | 22.46 | 0.10 |

OS: 27.0 (26A428); Python: 3.14.7; MLX 0.32.2.
Observed boundary pressure levels: ['1'].
Largest recorded MLX allocator peak: 22.46 GiB (not process RSS).
Swap snapshots: total = 2048.00M  used = 421.81M  free = 1626.19M  (encrypted).
Selected background process CPU snapshots (% of one core per process, not an aggregate): `{"audiomxd": {"max": 0.0, "min": 0.0}, "configd": {"max": 0.2, "min": 0.0}, "fseventsd": {"max": 105.0, "min": 0.0}, "mds": {"max": 175.5, "min": 0.3}, "mds_stores": {"max": 452.1, "min": 0.0}, "mdworker_shared": {"max": 55.9, "min": 0.0}, "mediaanalysisd": {"max": 0.0, "min": 0.0}, "rsync": {"max": 82.0, "min": 0.0}}`.

## Mac Studio M3 Ultra / 512 GiB

| Model / workload | AX normal wiring | MTPLX sustained | MTPLX turbo | AX vs sustained |
| --- | ---: | ---: | ---: | ---: |
| tiel / random128 | 107.54 | 102.86 | 95.73 | +4.6% |
| tiel / python-lru | 160.35 | 153.48 | 153.48 | +4.5% |
| tiel / rust-jsonl | 127.19 | 121.80 | 110.72 | +4.4% |
| cyber / random128 | 143.62 | 93.88 | 88.86 | +53.0% |
| cyber / python-lru | 170.65 | 162.99 | 155.93 | +4.7% |
| cyber / rust-jsonl | 134.49 | 131.54 | 132.20 | +2.2% |

AX normal wiring means the ordinary wiring policy **within this explicitly resident comparison**:
the M5 audited-export policy releases wiring; M4, M2 and M3 keep existing wiring.
The M4 resident override is not the shipped Auto paging path.

### TTFT and decode

| Model / workload | AX TTFT ms | Sustained TTFT ms | Turbo TTFT ms | AX decode tok/s | Sustained decode | Turbo decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiel / random128 | 303.17 | 225.86 | 229.75 | 142.12 | 124.96 | 115.00 |
| tiel / python-lru | 329.74 | 225.93 | 227.63 | 201.30 | 176.46 | 176.94 |
| tiel / rust-jsonl | 593.25 | 457.63 | 453.80 | 179.58 | 154.96 | 137.22 |
| cyber / random128 | 303.70 | 226.92 | 228.15 | 216.99 | 111.90 | 105.01 |
| cyber / python-lru | 336.72 | 227.31 | 228.74 | 218.80 | 189.95 | 180.37 |
| cyber / rust-jsonl | 603.73 | 465.75 | 467.73 | 195.94 | 172.25 | 173.67 |

### M3 unwired diagnostic (expert streaming still Off)

This is an explicit `AX_MLX_WIRED_LIMIT_SCALE=0` control, not automatic tuning or a default promotion.

| Model / workload | AX wired completion | AX unwired completion | Completion change | Wired TTFT ms | Unwired TTFT ms | Decode change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiel / random128 | 107.54 | 115.64 | +7.5% | 303.17 | 209.00 | -0.5% |
| tiel / python-lru | 160.35 | 172.32 | +7.5% | 329.74 | 209.83 | -0.6% |
| tiel / rust-jsonl | 127.19 | 137.06 | +7.8% | 593.25 | 432.47 | -1.1% |
| cyber / random128 | 143.62 | 159.91 | +11.3% | 303.70 | 205.88 | -2.0% |
| cyber / python-lru | 170.65 | 184.92 | +8.4% | 336.72 | 211.19 | -0.9% |
| cyber / rust-jsonl | 134.49 | 145.33 | +8.1% | 603.73 | 442.55 | -1.3% |

### Sample spread, order and draft efficiency

| Model / workload / arm | Completion min / median / max tok/s | Block 0 / 1 median | Accepted / drafted | Output variants |
| --- | ---: | ---: | ---: | ---: |
| cyber / python-lru / ax | 169.28 / 170.65 / 171.60 | 170.84 / 170.46 | 1086/1290 (84.2%) | 1 |
| cyber / python-lru / ax-unwired | 183.85 / 184.92 / 185.92 | 184.72 / 185.13 | 1086/1290 (84.2%) | 1 |
| cyber / python-lru / sustained | 161.89 / 162.99 / 164.61 | 162.94 / 163.77 | 1104/1296 (85.2%) | 1 |
| cyber / python-lru / turbo | 155.34 / 155.93 / 156.76 | 155.93 / 155.93 | 1074/1374 (78.2%) | 1 |
| cyber / random128 / ax | 141.87 / 143.62 / 145.55 | 143.10 / 144.53 | 540/630 (85.7%) | 1 |
| cyber / random128 / ax-unwired | 157.30 / 159.91 / 160.45 | 160.02 / 159.80 | 540/630 (85.7%) | 1 |
| cyber / random128 / sustained | 92.63 / 93.88 / 94.83 | 94.06 / 93.78 | 408/1068 (38.2%) | 1 |
| cyber / random128 / turbo | 87.98 / 88.86 / 89.27 | 88.90 / 88.59 | 390/1110 (35.1%) | 1 |
| cyber / rust-jsonl / ax | 133.95 / 134.49 / 135.62 | 134.50 / 134.47 | 1056/1398 (75.5%) | 1 |
| cyber / rust-jsonl / ax-unwired | 145.11 / 145.33 / 145.71 | 145.24 / 145.42 | 1056/1398 (75.5%) | 1 |
| cyber / rust-jsonl / sustained | 131.09 / 131.54 / 132.74 | 132.17 / 131.51 | 1068/1392 (76.7%) | 1 |
| cyber / rust-jsonl / turbo | 131.77 / 132.20 / 133.38 | 132.65 / 131.87 | 1068/1398 (76.4%) | 1 |
| tiel / python-lru / ax | 158.30 / 160.35 / 162.05 | 160.88 / 159.03 | 1050/1392 (75.4%) | 1 |
| tiel / python-lru / ax-unwired | 171.64 / 172.32 / 174.07 | 171.95 / 172.68 | 1050/1392 (75.4%) | 1 |
| tiel / python-lru / sustained | 152.28 / 153.48 / 154.36 | 153.76 / 153.20 | 1068/1386 (77.1%) | 1 |
| tiel / python-lru / turbo | 152.92 / 153.48 / 154.21 | 153.97 / 153.47 | 1068/1404 (76.1%) | 1 |
| tiel / random128 / ax | 106.03 / 107.54 / 108.71 | 108.16 / 106.06 | 432/918 (47.1%) | 1 |
| tiel / random128 / ax-unwired | 114.59 / 115.64 / 115.97 | 115.72 / 114.97 | 432/918 (47.1%) | 1 |
| tiel / random128 / sustained | 102.21 / 102.86 / 104.50 | 103.89 / 102.53 | 444/966 (46.0%) | 1 |
| tiel / random128 / turbo | 94.51 / 95.73 / 97.07 | 95.07 / 96.40 | 420/1020 (41.2%) | 1 |
| tiel / rust-jsonl / ax | 125.75 / 127.19 / 128.89 | 128.48 / 126.04 | 1014/1488 (68.1%) | 1 |
| tiel / rust-jsonl / ax-unwired | 136.11 / 137.06 / 138.21 | 136.89 / 137.12 | 1014/1488 (68.1%) | 1 |
| tiel / rust-jsonl / sustained | 120.77 / 121.80 / 122.84 | 122.25 / 121.64 | 1014/1536 (66.0%) | 1 |
| tiel / rust-jsonl / turbo | 110.00 / 110.72 / 111.67 | 110.65 / 110.80 | 954/1728 (55.2%) | 1 |

Draft efficiency is each engine's accepted/drafted counter ratio across measured
requests, not output quality or a forced-trajectory comparison. Block medians expose
order/background drift; six samples do not establish population confidence bounds.

### MLX allocator memory

Measured request-end active ranges and highest recorded allocator peak/cache.
These are not process RSS or whole-system memory totals.

| Arm | Active min / max GiB | Highest peak GiB | Highest cache GiB |
| --- | ---: | ---: | ---: |
| ax | 21.25 / 21.25 | 22.46 | 0.10 |
| sustained | 19.95 / 19.95 | 21.06 | 1.72 |
| turbo | 19.95 / 19.95 | 21.06 | 1.73 |
| ax-unwired | 21.25 / 21.25 | 22.46 | 0.10 |

OS: 26.6.2 (25G83); Python: 3.14.6; MLX 0.32.2.
Observed boundary pressure levels: ['1'].
Largest recorded MLX allocator peak: 22.46 GiB (not process RSS).
Swap snapshots: total = 1024.00M  used = 6.00M  free = 1018.00M  (encrypted).
Selected background process CPU snapshots (% of one core per process, not an aggregate): `{"OrbStack Helper": {"max": 94.3, "min": 18.4}, "audiomxd": {"max": 0.0, "min": 0.0}, "configd": {"max": 0.1, "min": 0.0}, "fseventsd": {"max": 0.0, "min": 0.0}, "mds": {"max": 0.5, "min": 0.0}, "mds_stores": {"max": 99.5, "min": 0.0}, "mdworker_shared": {"max": 0.0, "min": 0.0}, "mediaanalysisd": {"max": 0.0, "min": 0.0}}`.

## M4 Auto admission diagnostic (16 output tokens)

One warmup and one measured request per cell, python-lru prompt. This low-sample
diagnostic is separate from every primary 128/256-token result above.
AX uses the same explicit throughput-MTP settings; Auto changes expert residency only.

| Model / arm | Completion tok/s | TTFT ms | Active MLX GiB |
| --- | ---: | ---: | ---: |
| tiel / ax-auto | 1.54 | 2224.14 | 5.71 |
| tiel / ax-off | 24.71 | 499.10 | 21.25 |
| tiel / sustained | 30.36 | 395.47 | 19.95 |
| cyber / ax-auto | 1.37 | 2235.72 | 5.71 |
| cyber / ax-off | 23.28 | 514.26 | 21.25 |
| cyber / sustained | 28.78 | 404.74 | 19.95 |

Auto preserves the mandated 48 GiB reserve. Its lower allocation and slower
generation expose the cost of expert paging on this host; no policy change or
general speedup claim follows from this diagnostic.

## Interpretation and limits

- Normal wiring and unwired results remain separate. Small differences with overlapping sample ranges are close results, not robust universal wins.
- Engines can emit different greedy trajectories, changing MTP acceptance. Raw tokens and counters are retained; no output-quality or forced-trajectory claim.
- This short cold-KV suite does not qualify long context, co-residency, pressure or endurance on M4, M2 or M3.
- OS/power/pressure observations are process-boundary snapshots, not continuous thermal or RSS monitoring. Existing OS background load was not stopped.
- M2-or-newer API availability is independent of the measured M2/M3/M4/M5 speed outcome. No automatic no-wire policy expansion was made.

## Verification

Run `python verify.py --check-source` and `python test_verify.py` from this directory.
The verifier checks 540 measured requests and 360 warmups, exact inputs/output counts,
callback phase accounting, active MTP, cold KV, source and model integrity.
The Python mode probe confirms environment precedence and required-pack rejection.
Rust: 3,749 passed. Python: 211 passed, 26 skipped. Format, Python-library Clippy,
script checks, qualification dry-runs and canonical claims passed. Full workspace
Clippy still fails on existing core/SDK test lint errors; this is not an all-green gate claim.
