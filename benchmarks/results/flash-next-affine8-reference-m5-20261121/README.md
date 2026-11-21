# Flash Next bounded PLE diagnostic on M5 Max

The complete replay ran on a MacBook Pro with Apple M5 Max, 128 GB of
unified memory, an internal SSD, and macOS 27.2.1. It used the MXFP4 MTP
pack at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`, MLX and MLX Metal
0.32.2, and MLX-VLM `6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb`. The
diagnostic ran greedy text generation with the independent model graph,
an inspected row reader for the affine8 PLE table, and 1,024-token
prefill chunks. The runner bound all 1,750 runtime files and verified
the published pack inventory. CPU idle was 96.26% before and 97.39%
after the full run. All 25 memory-pressure samples were normal.

The smoke run matched all 96 generated IDs. The full run matched the
signed M5 reference on 105 QA rows, including text, generated IDs, stop
state, and grades. The checkers passed 100 cases in both runs. The
29,774-token lookup completed in 40.114 seconds and returned the same
IDs. The full run also matched the prompt hashes and all 128 generated
IDs in each of the 15 matrix runs, with two warmups and three measured
runs per length.

The bounded run used 67.91 GiB of peak MLX tracked memory and 56.61 GiB
of peak process RSS. The signed ordinary run recorded 121.56 GiB of peak
MLX tracked memory on macOS 27.0.1. These observations span different
host software, loader paths, and cache conditions.

| Prompt tokens | Bounded median TTFT | Bounded median decode | Signed ordinary median TTFT | Signed ordinary median decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2.197 s | 38.31 tok/s | 4.554 s | 15.56 tok/s |
| 2,048 | 5.646 s | 37.14 tok/s | 13.874 s | 14.23 tok/s |
| 8,192 | 15.149 s | 37.15 tok/s | 90.080 s | 12.66 tok/s |

The earlier native MXFP4 matrix on macOS 27.2.1 used the same prompt
hashes. Across its two n-gram arms and three lengths, native decode was
1.195–1.298 times the bounded reference and TTFT was 0.248–0.629 times
the bounded reference. Those observations are bound to the archived
native executable digest `0a7c6049a5af88ed1ebdfce02c648416349b2b1eed301381cb6eff5b310e6af5`.
The release contract continues to use the ordinary MLX-VLM profile and
its same-host evidence requirements. Both bounded receipts carry the
diagnostic profile and leave qualification and release readiness false.
[Earlier native matrix](../flash-next-mxfp4-native-matrix-m5-20261121/quiet-result.json.gz),
[signed ordinary reference](../qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json),
[ordinary rerun](../flash-next-ordinary-reference-m5-20261121/README.md).

## Evidence replay

From the repository root, run the following command and compare stdout
with [comparison.json](comparison.json). The replay checks the source
bindings, workload identity, and output equality, then uses only the
release evaluator's numeric reference-relative check for the diagnostic
comparison.

```sh
python3 benchmarks/results/flash-next-affine8-reference-m5-20261121/compare_outputs.py
```

- Full run [raw receipt](result.json.gz), [memory samples](memory-samples.jsonl.gz),
  [source manifest](source.json), [exact runner](run_flash_next_affine8_reference.py.gz).
- Smoke run [raw receipt](smoke-result.json.gz), [memory samples](smoke-memory-samples.jsonl.gz),
  [source manifest](smoke-source.json), [exact runner](smoke-run_flash_next_affine8_reference.py.gz).
