# QA matrix catalog (direct + MTP)

This documents the model matrix used for local verification. Paths are resolved
from the active Hugging Face hub cache at runtime by `scripts/run_qa_matrix.py`
(via inventory written to `$QA_SCRATCH/qa-matrix.txt`).

## Direct catalog aliases attempted

See `MODEL_PROFILES` / server presets: Gemma 4, Qwen 3.5/3.6, GLM 4.7 Flash,
Llama 3.1/3.3/4 Scout, Mistral Small/Ministral/Devstral, GPT-OSS 20B/120B.

## MTP packages attempted

ax-local assistant-MTP and Qwen fused MTP sidecars under
`models--ax-local--*` when present in the HF hub cache.

## Runner

```bash
cargo build -p ax-engine-server
# inventory (example): python helpers or re-run goal inventory into $QA_SCRATCH/qa-matrix.txt
export QA_SCRATCH=/path/to/scratch
export QA_SAMPLE=8 QA_SEED=20260716
python3 scripts/run_qa_matrix.py
```

Evidence from the 2026-07-16 full matrix: **28/28 cells completed with 0 engine
failures** (load/chat/null). Residual partial fails were model-quality on hard
prompts (e.g. `unit_test_clamp` incomplete code), documented in run logs.
