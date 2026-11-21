# Selected safetensors weight loader comparison

This experiment compared three materialization paths on a Mac mini
M4 Pro 64 GB with macOS 27.0, internal SSD and MLX 0.32.4. Each process
loaded a named 1.5 GiB subset from one of three real Qwen 3.8 27B MXFP4
shards. It materialized the selected arrays before stopping the timer,
then hashed their names, dtypes, shapes and full byte contents.

| Loader | Warm median | Fresh clone median | Median process RSS after eval |
| --- | ---: | ---: | ---: |
| AX mmap in name order | 181.8 ms | 1,926.4 ms | 3.019 GiB |
| AX mmap in file offset order | 182.7 ms | 1,939.9 ms | 3.019 GiB |
| MLX lazy CPU load and selected eval | 61.1 ms | 259.7 ms | 1.518 GiB |

Three loaders ran in rotating order for two passes per cache condition.
All 36 materializations produced the same per-shard digests. `mincore`
reported all selected pages resident before every warm observation and
zero or one resident before each fresh-clone observation. Fresh APFS
clones have separate file pages and share the SSD's underlying storage.
The RSS samples include file-backed pages visible to each process.

The lazy loader parsed the whole safetensors header, retained the selected
arrays, then evaluated them. In a fixture with an invalid excluded dtype,
the lazy loader returned an error and the filtered mmap paths loaded the
selected tensors exactly. The timings above cover selected weight bytes.
Full Flash Next startup, token parity and eligibility for a default still
need pack-specific evidence. On the tested data, pursue the lazy CPU
loader as the next candidate.

The [actual Flash Next header audit](flash-next-headers.json) checked
every named tensor's dtype and shape through the pinned MLX CPU reader.
It parsed all 26 MXFP4 and 37 MXFP8 safetensors files, including their MTP
sidecars, with zero change in active MLX allocation. The audit read
array metadata and left materialization to the subsequent weight trial.

- [Warm observations](m4-warm.jsonl), [fresh-clone observations](m4-unresident.jsonl),
  [exact selection](selection.json), and [source and build binding](source.json)
- [Probe patch](loader-lab.patch), [selection script](select_shards.py),
  and [trial runner](run_load_order.py)

Apply the probe patch with `git apply --unidiff-zero` to the recorded
source commit in a scratch checkout. Build the `flash_load_order_ab` example from package `mlx-sys`
with the pinned release-profile tools. Refresh the workload with the
selection script and the pinned local checkpoint, then give the trial
runner the example path, workload file, output file and storage type.
Use `--fresh-clones` for the cloned condition on the same APFS volume
as the weights. The runner records its per-observation page state.
