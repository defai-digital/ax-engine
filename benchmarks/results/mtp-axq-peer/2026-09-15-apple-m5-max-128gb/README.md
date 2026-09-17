# Qwen 3.8 27B AXQ 6-bit: latest-runtime peer campaign

Apple M5 Max, 128 GB (campaign host, not the Mac mini M4 Pro 64 GB SKU). Pack:
`AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` @ `3e290738e96972307c6aeb9934ab170ca0eae1c1`.

Contract: repository `flappy` suite, four cases, 256 generated tokens, greedy,
two warmups, five measured repetitions, three-second cooldown, prefix-cache
off, n-gram stacking off. Decode is the **median of 20 measured runs**.

Same checkpoint directory was offered to every runtime. No GGUF or community
4-bit substitute. A lane that cannot load this AXQ 6-bit snapshot is
**unsupported**, not filled with another pack.

## Measured

| Runtime | Version (latest checked 2026-09-15) | Decode | Prefill |
| --- | --- | ---: | ---: |
| AX Engine | 7.4.0 product-path MTP (recurrent depth 3) | **76.90 tok/s** | **795.3 tok/s** |
| MTPLX | **2.11.2** (PyPI / mtplx.com Latest) | 70.62 tok/s | 686.6 tok/s |
| mlx-lm | **0.31.3** (PyPI Latest), direct AR on the same `flappy` prompts | 27.90 tok/s | — (decode-only harness) |
| OMLX | **0.6.4**, `import_mtplx_sidecar` then Lightning MTP depth 1 | 38.47 tok/s | — (generate-wall tok/s) |

Prefill for AX is `prefill_tok_s` (20-run median). Prefill for MTPLX is
`prompt_tokens / prompt_eval_time_s` (20-run median). Prompt lengths are
264–432 tokens.

Raw: `ax_engine.json`, `mtplx.json`, `mlx_lm.json`, `omlx.json`. mlx-lm is greedy
`stream_generate` (temp 0), 256 tokens, 2 warmups + 5 reps per case; decode
excludes the first generated token. It does **not** use the MTP sidecar.

## Latest runtimes that could not load this pack

| Runtime | Latest checked | Result |
| --- | --- | --- |
| mlxcel | **v0.7.0** (GitHub Latest, 2026-09-09) | Load error: AXQ affine 6-bit embed `group_size=32` inferred bits=16 |
| OMLX | **0.6.4** (GitHub Latest release, 2026-08-29) | Raw Hub tree still fails Lightning load; **imported snapshot measured 38.47 tok/s** (see Measured) |
| llama.cpp | Homebrew **0.4.1** formula; host binary **0.4.0** (b10809) | Not GGUF (`gguf_init_from_reader: failed to read magic`) |
| mistral.rs | **v0.9.3** (GitHub Latest, 2026-09-07) | Not installed on the campaign host; GGUF/HF path, not this MLX AXQ snapshot |
| exo | **v1.0.71** | Cluster runtime; not a single-host load of this snapshot |
| rMLX | Pushkinist/rMLX latest release **v0.4.1** (2026-09-02) | No campaign binary on the host; not measured |
| uzu | **0.5.26** (PyPI, 2026-09-05) | Ships Mirai checkpoints, not this AXQ snapshot |
| vLLM | **0.29.0** (PyPI) | CUDA/Linux; not an Apple Silicon load of this pack |

mlxcel and OMLX were invoked at those latest releases against the **same directory** AX and MTPLX used. llama.cpp was invoked as `llama-cli -m <snapshot>`.
