### Embedding throughput

ax-engine matches `mlx-lm` on Qwen3-Embedding 4B and 8B and is
within ~10% on 0.6B (small-model Python/PyO3 overhead). Use the
batched API — `embed_batch_array` in Python, `embed_batch_flat`
in Rust, or `{"input": [[ids], ...]}` over HTTP. Single-sentence
loops are 2–3× slower in both backends.

Source: `benchmarks/results/embedding/2026-05-15-postfix/`.

#### In-process batched (sustained, hot-loop)

| Model | mlx-lm | ax-engine-py | ax-engine Rust |
|---|---:|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 9,629 | 8,780 | 8,785 |
| Qwen3-Embedding 4B 4-bit | 2,370 | 2,374 | 2,374 |
| Qwen3-Embedding 8B 4-bit DWQ | 1,471 | 1,443 | 1,449 |

Median tok/s, 5 warmup + 15 timed trials, no cooldown. 10-sentence
corpus with lengths [10,15,13,8,3,8,10,8,10,10], `last` pooling,
l2-normalized.

#### HTTP serving (`/v1/embeddings`)

Three serving contracts on the same 10-sentence corpus:

| Model | Sequential | Concurrent (microbatcher) | Batched POST |
|---|---:|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 320 | 1,837 | 3,950 |
| Qwen3-Embedding 4B 4-bit | 240 | 1,203 | 1,809 |
| Qwen3-Embedding 8B 4-bit DWQ | 184 | 902 | 1,179 |

- **Batched POST** `{"input": [[ids], ...]}` is the fastest path.
- **Concurrent** single-input POSTs are coalesced server-side by
  `EmbeddingMicroBatcher` (20 ms window in this measurement). Use
  this when the caller can't pre-batch.
- **Sequential** is the worst case — every POST round-trips through
  the GPU on its own.

#### Cold start (session construction)

| Model | Default | `AX_MMAP_WEIGHTS=1` |
|---|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 161 ms | 205 ms (+27%) |
| Qwen3-Embedding 4B 4-bit | 219 ms | 384 ms (+75%) |
| Qwen3-Embedding 8B 4-bit DWQ | 279 ms | 603 ms (+116%) |

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
