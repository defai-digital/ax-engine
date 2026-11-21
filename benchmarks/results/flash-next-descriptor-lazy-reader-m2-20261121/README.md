# Descriptor-bound lazy materialization of the Flash Next MTP sidecar

This comparison ran on a Mac Studio M2 Ultra 192 GB with macOS 27.2.1,
MLX 0.32.4 and an external APFS SSD. Each process loaded and evaluated
all 31 BF16 tensors from the exact 5.21 GB Flash Next MXFP4 sidecar.
MLX used its usual path overload or a positional reader holding a
duplicate of the original file descriptor. The timer ends before
digesting the selected arrays. All 16 observations reproduced the
digest of names, dtypes, shapes and complete payload bytes.

| Selected OS pages before loading | Stock MLX path | Positional descriptor |
| --- | ---: | ---: |
| All 318,256 pages resident | 0.4025 s | 0.4111 s |
| Fresh APFS files with one page resident | 2.1905 s | 2.1639 s |

The results report medians of four observations per reader and
condition. The two process orders alternate, and each fresh clone
has its own file identity on the same volume. The SSD controller
cache was uncontrolled. Median RSS immediately after evaluation
was about 4.872 GiB for both readers in both conditions. The
positional prototype uses synchronous reads within each tensor
load. Stock MLX parallelizes large reads internally. The internal
SSD on the M5 has a separate test for this bandwidth difference.

## Reproduction

Use the [recorded source receipt](source.json) with the exact
[publisher sidecar selection](../flash-next-mtp-sidecar-reader-m2-20261121/selection.json).
Restore the selection entry's `path` to the verified local
`mtp.safetensors`. Build [full_sidecar.cpp](full_sidecar.cpp) as
`full_sidecar` beside [bench_adapter.py](bench_adapter.py) with
`xcrun clang++ -std=c++17 -O2`, the installed MLX include and library
directories, its library rpath and `-lmlx`.

Put the [shared selected-weight trial runner](../flash-next-weight-loader-lab-20261121/run_load_order.py)
on `PYTHONPATH` and invoke [run_pair.py](run_pair.py) with the adapter,
workload, output and `--storage 'external SSD' --passes 4`. Run the
original series after hashing the complete original file. Add
`--fresh-clones --scratch-dir` on the pack volume for the second series.

- [Original-file observations](original.jsonl)
- [Fresh-clone observations](fresh-clones.jsonl)
