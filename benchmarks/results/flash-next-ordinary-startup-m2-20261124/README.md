# Flash Next ordinary MLX-VLM startup memory on M2 Ultra

A Mac Studio M2 Ultra 192 GiB on macOS 27.2.1 ran four frozen QA prompts
with the pinned MXFP4 pack on an external USB SSD. The runner checked all
49 publisher members before and after execution and rechecked its installed
MLX-VLM 0.7.0rc0 and MLX 0.32.2 files. The audited Qwen4 language module
matches the ordinary M5 reference. The probe uses the same load batches,
1,024-row prefill geometry, greedy decode and QA budgets. The four prompts
are 29–46 tokens long. This is a startup diagnostic.

MLX-VLM's ordinary loader retains the sharded PLE embeddings. Following
oMLX's distinction between allocator accounting and host memory pressure,
the probe records active, cached and peak MLX bytes, process peak RSS and
system pressure separately. It samples at the existing materialization
and prefill boundaries and around the first token. The trace adds no MLX
evaluation or synchronization calls.

| Boundary | MLX active | MLX peak so far | Process peak RSS |
| --- | ---: | ---: | ---: |
| Lazy model load returned | <0.01 GiB | <0.01 GiB | 0.49 GiB |
| All 2,326 arrays materialized | 117.46 GiB | 117.46 GiB | 104.70 GiB |
| First token ready | 117.58 GiB | 117.69 GiB | 104.70 GiB |

All 97 checkpoints recorded normal memory pressure. The allocator peak
across four cases was 117.74 GiB. Most observed allocator demand occurs
during weight materialization. The [M5 pressure stop](../flash-next-reference-quiet-entry-m5-20261124/README.md)
still requires attribution on its own host. The M2 probe uses a different
chip, memory tier and storage medium. Its timings include sampling overhead.

All four QA cases pass grading. Three generated sequences match the signed
v7.9.0 M5 reference on macOS 27.0.1 exactly. The reasoning case differs
from the fifth generated token and still passes its grader. The
[original runner control](raw/original-control.json) used the same M2, pack
and package installation in a fresh process. Its
[adapter](raw/original-control-adapter.py) only enables the diagnostic
hardware and storage and labels the host. Across all four M2 cases,
the traced and original runs match all 119 generated IDs, EOS, text and grades.

The [raw result](raw/result.json) includes all generated token IDs. The
[raw trace](raw/startup-memory.jsonl) retains each observed checkpoint.
[Input digests](raw/run-inputs.json) identify the measured runner, original
runner, pack inventory and original-runner adapter. The checked-in tracer
adds an explicit Mac Studio hardware-identifier check to the measured
runner. It uses the same observation and generation code.

```sh
shasum -a 256 -c SHA256SUMS
python3.12 -B replay.py | cmp comparison.json -
```
