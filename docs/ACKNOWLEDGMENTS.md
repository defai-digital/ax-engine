# Acknowledgments

AX Engine is grateful to the open-source projects and people that make its
runtime, validation, and benchmarking work possible. This page records each
project's role so an acknowledgment does not imply a code dependency, a shared
benchmark contract, or a published performance result where none exists.

## Runtime foundation

- [MLX](https://github.com/ml-explore/mlx) provides the array runtime and MLX
  C++ APIs used by AX Engine's native MLX execution path on Apple Silicon.

## Benchmark peers and reference implementations

- [mlx-lm](https://github.com/ml-explore/mlx-lm) is used for MLX text and
  embedding reference runs, numerical validation, and the explicit delegated
  compatibility route.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) provides Metal/GGUF
  reference rows and an explicit delegated compatibility route. Its benchmark
  rows are shape-compatible external references, not prompt-hash-parity MLX
  comparisons.
- [MTPLX](https://github.com/youssofal/MTPLX) is an MTP benchmark peer for the
  published Qwen MTP comparisons.
- [lightning-mlx](https://github.com/samuelfaj/lightning-mlx) is an MTP
  benchmark peer for the published Qwen MTP comparisons.
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) is the
  EmbeddingGemma reference implementation and correctness oracle, and provides
  paired embedding benchmark reference runs.

## Engineering and multi-model serving references

- [MLXcel](https://github.com/lablup/mlxcel) is a reference for the AX
  multi-model serving comparison work and for KV-cache engineering exploration.
  AX does not vendor or depend on MLXcel source code. The repository contains
  comparison tooling for an MLXcel baseline; that tooling alone does not claim
  a published AX-versus-MLXcel result.

## People and community

- Thanks to [Samuel Faj](https://www.samuelfaj.com/en/), author of
  [lightning-mlx](https://github.com/samuelfaj/lightning-mlx), for generous
  support during AX Engine's MTP peer-comparison work.
- [Remote Code](https://www.remotecode.io/) is a recommended resource for
  people building LLM tooling on Apple Silicon. It is not an AX Engine runtime
  dependency or benchmark source.

## Scope

This page acknowledges projects with a direct technical, validation,
benchmarking, or community-support relationship to AX Engine. Projects named
only in unmeasured or unsupported benchmark plans are not listed as benchmark
peers.
