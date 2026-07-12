# 6-bit MTP AX approximate diagnostic

This non-publishable artifact records an explicit optimistic speed diagnostic. It does not establish exact-distribution MTP acceleration.

The diagnostic ratio is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.

| Target | Suite | AX direct decode | Approx. MTP decode | Diagnostic ratio | Draft quality | MTP step coverage | Fallback prompts |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen3.6-27b-6bit` | `flappy` | 24.3 tok/s | 45.8 tok/s | 1.89x | 61.4% match | 21.1% | 2/4 |
| `qwen3.6-27b-6bit` | `long_code` | 24.6 tok/s | 66.0 tok/s | 2.69x | 98.8% match | 43.9% | 1/4 |
| `qwen3.6-27b-6bit` | `python_modules_long` | 24.7 tok/s | 66.6 tok/s | 2.70x | 95.4% match | 100.0% | 0/3 |
| `qwen3.6-35b-a3b` | `flappy` | 121.4 tok/s | 151.7 tok/s | 1.25x | 99.9% match | 100.0% | 0/4 |
| `qwen3.6-35b-a3b` | `long_code` | 121.0 tok/s | 121.6 tok/s | 1.00x | 21.1% match | 15.1% | 3/4 |
| `qwen3.6-35b-a3b` | `python_modules_long` | 121.5 tok/s | 152.5 tok/s | 1.26x | 88.9% match | 100.0% | 0/3 |
| `gemma-4-12b` | `flappy` | 41.3 tok/s | 99.5 tok/s | 2.41x | 99.2% verified | 82.6% | 1/4 |
| `gemma-4-12b` | `long_code` | 41.1 tok/s | 98.3 tok/s | 2.39x | 99.8% verified | 100.0% | 0/4 |
| `gemma-4-12b` | `python_modules_long` | 41.4 tok/s | 86.1 tok/s | 2.08x | 95.2% verified | 100.0% | 0/3 |
| `gemma-4-26b` | `flappy` | 105.0 tok/s | 151.2 tok/s | 1.44x | 99.7% verified | 100.0% | 0/4 |
| `gemma-4-26b` | `long_code` | 103.9 tok/s | 145.1 tok/s | 1.40x | 100.0% verified | 100.0% | 0/4 |
| `gemma-4-26b` | `python_modules_long` | 105.2 tok/s | 141.5 tok/s | 1.35x | 96.9% verified | 100.0% | 0/3 |
| `gemma-4-31b` | `flappy` | 18.7 tok/s | 46.2 tok/s | 2.47x | 99.8% verified | 100.0% | 0/4 |
| `gemma-4-31b` | `long_code` | 18.8 tok/s | 45.6 tok/s | 2.42x | 99.9% verified | 100.0% | 0/4 |
| `gemma-4-31b` | `python_modules_long` | 19.4 tok/s | 43.1 tok/s | 2.22x | 96.5% verified | 100.0% | 0/3 |

This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.

Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.
