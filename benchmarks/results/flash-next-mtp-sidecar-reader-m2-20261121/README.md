# Flash Next MXFP4 MTP sidecar materialization on M2 Ultra

This isolates the pinned Flash Next MXFP4 MTP sidecar on a Mac Studio
M2 Ultra 192 GB with macOS 27.2.1, MLX 0.32.4 and an external APFS SSD.
The 5,214,305,544-byte file contains 31 BF16 tensors and 5,214,301,696
selected payload bytes. The test stops its timer after evaluation
and hashes every selected name, dtype, shape and payload. All 22
observations have the same digest.

| Selected page condition | Name-order mmap | MLX lazy CPU | RSS mmap | RSS lazy CPU |
| --- | ---: | ---: | ---: | ---: |
| Fully resident original | 1.377 s | 0.398 s | 9.728 GiB | 4.873 GiB |
| Fresh APFS clones | 58.304 s | 7.846 s | 9.728 GiB | 4.873 GiB |

The table reports four observations per reader and condition. Each
condition alternates which reader goes first. The original file is
hashed against the publisher inventory before the runs. The runner
records page residency immediately before each read. Each cloned
sample uses a separate APFS file on the same physical SSD. Device
cache and underlying storage are shared and were uncontrolled.

The initial three-reader series also compared file offset order.
Name order already follows file offset order for all 31 tensors.
The equivalent mmap traversals still varied between 28.42 and
93.79 seconds as the reader position rotated. Lazy CPU took 2.17
to 2.19 seconds in the initial six-observation series. The repeated
two-reader study retains all results, including higher readings.

Process RSS is sampled after evaluation and includes file-backed
pages. These measurements cover the sidecar phase. Whole-server
startup, required-MTP execution and M5 promotion have separate
controls. The native attachment contract applies under both MTP
policies.

## Reproduction

Build the [existing probe](../flash-next-weight-loader-lab-20261121/README.md)
at the pinned source in the [receipt](source.json). Restore the
`path` field in each [selection](selection.json) entry using the
published `mtp.safetensors` file and verify its SHA-256 first. Run
the shared trial runner with `--passes 2 --fresh-clones` for the
three-reader series. Use [two_mode_sidecar.py](two_mode_sidecar.py)
with `--passes 4` for each repeated series and
`--fresh-clones --scratch-dir` for the cloned condition. Put the
shared trial runner on `PYTHONPATH` for the two-mode wrapper.

- [Initial three-reader fresh clones](three-reader-fresh-clones.jsonl)
- [Counterbalanced original file](two-mode-original.jsonl)
- [Counterbalanced fresh clones](two-mode-fresh-clones.jsonl)
