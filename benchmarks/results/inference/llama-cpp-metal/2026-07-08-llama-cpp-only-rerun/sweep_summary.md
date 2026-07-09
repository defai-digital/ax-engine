# llama.cpp Metal sweep summary

- elapsed: 2950s
- downloaded: 194.1 GB
- freed: 0.0 GB

| slug | status | repo | quant | notes |
|---|---|---|---|---|
| gemma-4-e2b-it-4bit | ok | unsloth/gemma-4-E2B-it-GGUF | Q4_K_M |  |
| gemma-4-e2b-it-6bit | ok | unsloth/gemma-4-E2B-it-GGUF | Q6_K |  |
| gemma-4-e2b-it-8bit | mlx_model_dir_missing | unsloth/gemma-4-E2B-it-GGUF | Q8_0 | No Hugging Face cache snapshot found for MLX repo mlx-community/gemma-4-e2b-it-8bit. Cannot generate prompt artifact. |
| gemma-4-e4b-it-4bit | ok | unsloth/gemma-4-E4B-it-GGUF | Q4_K_M |  |
| gemma-4-e4b-it-6bit | ok | unsloth/gemma-4-E4B-it-GGUF | Q6_K |  |
| gemma-4-26b-a4b-it-4bit | ok | unsloth/gemma-4-26B-A4B-it-GGUF | UD-Q4_K_M |  |
| gemma-4-26b-a4b-it-6bit | ok | unsloth/gemma-4-26B-A4B-it-GGUF | UD-Q6_K |  |
| gemma-4-31b-it-4bit | ok | unsloth/gemma-4-31B-it-GGUF | Q4_K_M |  |
| gemma-4-31b-it-6bit | ok | unsloth/gemma-4-31B-it-GGUF | Q6_K |  |
| qwen3_6-27b-4bit | ok | unsloth/Qwen3.6-27B-GGUF | Q4_K_M |  |
| qwen3_6-27b-6bit | ok | unsloth/Qwen3.6-27B-GGUF | Q6_K |  |
| qwen3_6-27b-8bit | mlx_model_dir_missing | unsloth/Qwen3.6-27B-GGUF | Q8_0 | MLX cache snapshot for mlx-community/Qwen3.6-27B-8bit is not AX-ready: /Volumes/Ext4T/models/hub/models--mlx-community--Qwen3.6-27B-8bit/snapshots/c5a593c1475a746e43a543b0a02bd2b357e5745f; missing model-manifest.json. Cannot generate prompt artifact. |
| qwen3_6-35b-a3b-4bit | ok | unsloth/Qwen3.6-35B-A3B-GGUF | UD-Q4_K_M |  |
| qwen3_6-35b-a3b-6bit | ok | unsloth/Qwen3.6-35B-A3B-GGUF | UD-Q6_K |  |
