# Qwen 3.8 MXFP4 greedy decode and effective prefill on M5 Max

The MacBook Pro M5 Max 128 GB runs the original
[pinned MXFP4 packs and engine builds](../qwen38-m5-peer-20261125/provenance.json).
Each arm opens one loopback HTTP stream at a time. All requests use
temperature zero. The 27B DFlash2 arms use the same local draft
checkpoint. The separate Flash Next routes use its pinned MXFP4 pack.

## Measurement

- Decode uses the same [fixed corpus](../../corpora/serving/qwen38_dflash2_paired.jsonl)
  as the original comparison. After one warmup, two passes use code, math
  and prose prompts with a 64-token output limit. The throughput numerator
  is the retokenized visible output minus the first text chunk's token
  count, and the denominator is elapsed time between the first and last
  nonempty text chunks. The result pools the six streams.
- Effective prefill is the median over two independently paired input
  sizes. Each probe asks for one output token. The rate divides the input
  token count difference by the first-text latency difference. The second
  pair reverses the prompt-length order. Each prompt uses a distinct marker
  at the beginning to limit prefix reuse. The estimate includes any
  input-length-dependent scheduling, tokenization and HTTP effects.
- The prefill probes run after each decode pass. They are outside the decode
  timer. Results use the same tokenizer and accounting across engines.
- A zero indicates that the tested configuration could not complete the
  workload or prove its selected route. Omitted engine/mode combinations
  have no score. The [original matrix](../qwen38-m5-peer-20261125/README.md)
  records its original loader and first-text results.
- AX Engine's explicit direct mode disables MTP and n-gram acceleration.
  Its DFlash2 arm attaches the local drafter and requires live round traces.
  oMLX's DFlash2 arm enables its native batched drafter per model and
  requires an observed DFlash2 load. Splash 1.7.0 uses its own DFlash2
  implementation. MTPLX uses target-only autoregressive generation and
  mlx-lm serves without a speculative model.

[README tables and charts](../../../README.md#performance) ·
[Per-request measurements and failures](results.jsonl) ·
[Build and pack provenance](provenance.json)

## Replay

Stage the binaries and packs using the
[original matrix instructions](../qwen38-m5-peer-20261125/README.md#replay).
Use fresh scratch directories and the admitted 27B
[local DFlash2 checkpoint](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4).
Run the two models one after the other and use the fixed corpus at the
stage's `qwen38_dflash2_paired.jsonl` path.

```bash
export AX_BENCH_MLX_LIB=/path/to/site-packages/mlx/lib
campaign=benchmarks/results/qwen38-m5-modes-20261125
for model in dense flash; do
  python "$campaign/run.py" --model "$model" \
    --stage /path/to/stage --draft /path/to/27b-dflash2 \
    --baseline-runner benchmarks/results/qwen38-m5-peer-20261125/run.py
done
python "$campaign/render_charts.py" --check
```
