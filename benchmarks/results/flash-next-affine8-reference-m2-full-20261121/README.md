# Flash Next full M2 bounded PLE diagnostic

The `88cd09972` diagnostic ran on Apple M2 Ultra 192 GB, macOS 27.2.1,
and an external SSD. It used MLX and MLX Metal 0.32.2, MLX-VLM
`6102cb4ad`, and the pinned MXFP4 pack at
`ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`. The source checked all
publisher members and 1,750 runtime files before and after the run.
The two CPU-idle samples were 91.12% and 92.4%. All resource samples
reported normal system memory pressure.

The 105 QA rows and all 15 matrix outputs match the archived reference
on the same M2 hardware class. QA text and every frozen checker grade
match too, including the same five failures. The three matrix lengths
each completed two warmups and three measured 128-token runs. The
input hashes match the two current AX native arms. The 29,774-token
lookup took 77.673 seconds and returned the same output IDs.

- Peak MLX tracked memory was 67.909 GiB. Peak process RSS was
  64.337 GiB. The PLE reader returned 197,080,740 selected payload bytes
  across the full workload.
- [Raw outputs](result.json.gz), [memory samples](memory-samples.jsonl.gz),
  [archived-reference adjudication](comparison.json) and
  [source identity](source.json) retain the execution evidence.
- The [archived reference](../qualification/2026-10-05-flash-next-mxfp4-release-gates/reference/result.json)
  supplies the M2 parity oracle. The frozen M5 release comparison uses
  its own runtime and measured host envelope.

## Packed-row timing

The [standalone probe](bench-row-io.py) ran after the model process
exited. It read the same scattered affine8 row payloads through serial
pread, four-worker Python pread, and read-only mapped batches. A first
prototype exhausted its per-process file descriptor limit. The measured
retry reused one mapping per physical file across all its planes.
All 54 observations returned identical byte digests within their paired
windows. The first-window and repeated-window observations share the OS
page cache. Timings include the actual byte copy.

| 1,024-token-shaped window | Serial pread | Four-worker pread | Mapped |
| --- | ---: | ---: | ---: |
| First-window median | 36.251 ms | 138.599 ms | 37.168 ms |
| Repeated-window median | 35.025 ms | 131.691 ms | 3.470 ms |

The [raw observations](row-io.json) retain the underlying rotations and
include one-token windows. The repeated-window gain measures packed-row
access. Prefill also includes index synchronization and the model graph.

## Row timing during inference

The `3de6f14da` host profiler ran the three 32-token bounded smoke
cases on the same M2 with the same pack and pinned runtime. Its pre- and
postflight hashes passed, and every sampled pressure reading was normal.
All three prompt hashes and 96 output IDs match the previous [bounded
smoke](../flash-next-affine8-reference-m2-20261121/bounded.json).

| Prompt tokens | Wall time | First token | PLE row selection and reads |
| ---: | ---: | ---: | ---: |
| 512 | 5.990 s | 4.567 s | 3.652 s |
| 2,048 | 20.753 s | 18.588 s | 14.971 s |
| 8,192 | 122.467 s | 121.014 s | 104.299 s |

The selection and read stage includes file operations and payload
assembly across prefill and generation. Its 104.299 seconds account for
85.2% of the 8,192-token case wall time. The separate index conversion
bucket can wait for Metal. The graph-construction bucket includes host
payload copies. Other model work runs outside the measured callbacks.
The diagnostic uses Python serial reads. The AX native runner uses its
own parallel positional reads and bounded read-ahead.

The [raw smoke receipt](profiled-smoke.json), [memory samples](profiled-smoke-samples.jsonl.gz)
and [source binding](profiled-smoke-source.json) preserve the measurement.
