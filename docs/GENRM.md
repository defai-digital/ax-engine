# Generative Reward Models (GenRM)

AX Engine supports **NVIDIA GenRM-style judges** as a **chat workload** on top
of existing causal families (primarily **Qwen3**). GenRM is **not** a new
layer-forward architecture and is **not** Nemotron Embed or Nano Omni.

## What GenRM is

A Generative Reward Model is an LLM fine-tuned to evaluate assistant answers
(often against a **principle** such as correctness or helpfulness). Inference is
ordinary autoregressive generation. The published Qwen3-Nemotron GenRM-Principle
recipe then derives a **scalar reward** from **Yes vs No token logprobs**.

| Model example | Architecture | AX graph | Product note |
| --- | --- | --- | --- |
| `nvidia/Qwen3-Nemotron-32B-GenRM-Principle` | Qwen3 32B | `qwen3` (direct) | **MVP target** |
| `nvidia/Llama-3_3-Nemotron-Super-49B-GenRM` | Llama 3.x | `llama3` preview path | Secondary |
| `nvidia/NVIDIA-Nemotron-3-Ultra-*-GenRM` | Nemotron-H LatentMoE ~550B | not productized | **Out of scope** on Apple Silicon |

## Requirements

1. Convert or obtain **MLX safetensors** + `model-manifest.json` for the base
   family (`qwen3` for Qwen3 GenRM).
2. Host memory for **32B 4-bit** is typically a **64 GB+** class Mac; 16–32 GB
   hosts should not expect comfortable residency.
3. Use OpenAI-compatible chat with role **`principle`**.

## Request shape (Principle GenRM)

```json
{
  "model": "qwen3-nemotron-32b-genrm-principle",
  "messages": [
    {"role": "user", "content": "What is 1+1?"},
    {"role": "assistant", "content": "1+1=2"},
    {"role": "principle", "content": "correctness"}
  ],
  "temperature": 0,
  "max_completion_tokens": 4096
}
```

AX accepts `principle` and renders it as a Qwen ChatML turn:

```text
<|im_start|>principle
correctness<|im_end|>
```

Multi-turn history may prepend earlier `user` / `assistant` pairs as on the
Hugging Face model card.

## Scoring status

| Capability | Status |
| --- | --- |
| Serve GenRM as chat (`principle` role) | **Phase A — landed** |
| Numeric reward = logprob(Yes) − logprob(No) | **Phase B — planned** (needs logprobs surface + tokenizer token ids) |
| Managed download alias / multi-model allowlist | Not yet |
| Ultra LatentMoE GenRM | Not supported |

Until Phase B ships, operators can still run the judge as a **text generator**
and inspect free-form critique, but that is **not** the official scalar reward
contract and should not be used for RLHF training claims.

## Relation to other Nemotron work

| Workstream | Family / path | Use |
| --- | --- | --- |
| GenRM (this page) | Base chat graph (`qwen3` …) | Preference / principle scoring |
| Nemotron 3 Embed | `nemotron_embed` | RAG vectors |
| Nemotron 3 Nano Omni | `nemotron_h` + media | Multimodal chat |

Do not mix GenRM with embedding indexes or Omni media tensors.

## OpenClaw / agents

Keep the **default chat model** as Qwen 3.5/3.6 for OpenClaw. Use GenRM only as
an optional **evaluation** model (second process or multi-model later), not as
the interactive assistant.
