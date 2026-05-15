### Embedding throughput

ax-engine matches `mlx-lm` on Qwen3-Embedding 4B and 8B and is
within ~10% on 0.6B (small-model Python/PyO3 overhead). Use the
batched API — `embed_batch_array` in Python, `embed_batch_flat`
in Rust, or `{"input": [[ids], ...]}` over HTTP. Single-sentence
loops are 2–3× slower in both backends.

Source: `benchmarks/results/embedding/2026-05-14-readme/`.

#### In-process batched (sustained, hot-loop)

| Model | mlx-lm | ax-engine-py | ax-engine Rust |
|---|---:|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 9,813 | 8,877 | 8,816 |
| Qwen3-Embedding 4B 4-bit | 2,368 | 2,384 | 2,397 |
| Qwen3-Embedding 8B 4-bit DWQ | 1,474 | 1,449 | 1,446 |

Median tok/s, 5 warmup + 15 timed trials, no cooldown. 10-sentence
corpus with lengths [10,15,13,8,3,8,10,8,10,10], `last` pooling,
l2-normalized.

#### HTTP serving (`/v1/embeddings`)

Three serving contracts on the same 10-sentence corpus:

| Model | Sequential | Concurrent (microbatcher) | Batched POST |
|---|---:|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 315 | 1,783 | 3,890 |
| Qwen3-Embedding 4B 4-bit | 234 | 1,161 | 1,729 |
| Qwen3-Embedding 8B 4-bit DWQ | 180 | 877 | 1,158 |

- **Batched POST** `{"input": [[ids], ...]}` is the fastest path.
- **Concurrent** single-input POSTs are coalesced server-side by
  `EmbeddingMicroBatcher` (20 ms window in this measurement). Use
  this when the caller can't pre-batch.
- **Sequential** is the worst case — every POST round-trips through
  the GPU on its own.

#### Cold start (session construction)

| Model | Default | `AX_MMAP_WEIGHTS=1` |
|---|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 807 ms | 834 ms (+3%) |
| Qwen3-Embedding 4B 4-bit | 894 ms | 1016 ms (+14%) |
| Qwen3-Embedding 8B 4-bit DWQ | 1011 ms | 1238 ms (+22%) |

Default loader (`mlx_load_safetensors`) is the recommended choice
on warm OS page cache. `AX_MMAP_WEIGHTS=1` selects a Rust mmap +
`mlx_array_new_data` path that may be faster on true-cold disk
(where the C loader's userspace `read()` pipeline costs more than
mmap-on-demand) but is slower on warm cache because it adds JSON-
header parsing and per-tensor registration overhead on top of the
required copy. See
[`docs/EMBEDDING_COLDSTART.md`](docs/EMBEDDING_COLDSTART.md)
for the true-cold (post-`sudo purge`) measurement procedure and the
default-on criteria.

#### Reproducing

```bash
bash scripts/bench_embedding_readme.sh
```

Detailed methodology: [`docs/EMBEDDINGS.md`](docs/EMBEDDINGS.md).
