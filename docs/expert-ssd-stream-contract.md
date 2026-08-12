# AX Expert SSD Stream — shared contract (v1)

**Repos:** `axquant` (`feat/expert-ssd-stream`) and `ax-engine` (`feat/expert-ssd-stream`)
**Goal:** Run Super-class MoE (Qwen3.8-2.4T-A95B) on Macs whose unified memory is smaller than the full 2-bit expert table (~0.8 TB). A 512 GB Mac cannot resident-load that pack.
**Non-goal:** Copy, translate, or vendor mlx-optiq. Clean-room implementation only.
**v1 scope:** *layer-stack paging* — page one MoE layer's fused expert stack from SSD, run the existing `gather_qmm`, then evict. Not per-expert unfused GEMM (v2). Enabled families: Qwen 3.8 and DeepSeek V4 Flash (inferred from native expert roles when `ax_expert_stream.json` is absent).

## Why this shape

AX Engine already executes fused MoE via `gather_qmm` on packed tensors
`[num_experts, out, in]`. Per-expert streaming would require a new kernel path
and an unfused layout (ADR-0005 deferred that). Layer-stack paging reuses the
current kernel: peak RAM ≈ resident trunk + one layer's experts + KV.

For a 2.4T / ~80-layer model at 2-bit, one layer is on the order of several GB,
not 0.8 TB. That is what makes 192–512 GB Macs viable.

## On-disk contract

AXQuant writes `ax_expert_stream.json` next to `config.json` / weight shards.

```json
{
  "schema_version": "axquant.expert-stream.v1",
  "generated_by": "axquant",
  "required": true,
  "mode": "layer-stack",
  "num_experts": 256,
  "experts_per_tok": 8,
  "estimated_resident_bytes": 40000000000,
  "estimated_full_resident_bytes": 800000000000,
  "estimated_max_layer_expert_bytes": 10000000000,
  "resident_roles": ["embedding", "attention", "router", "shared_expert", "norm", "lm_head", "mtp"],
  "streamed_roles": ["expert"],
  "tensors": [
    {
      "name": "model.layers.0.mlp.switch_mlp.gate_proj.weight",
      "file": "model-00001-of-00080.safetensors",
      "layer": 0,
      "proj": "gate_up",
      "expert_axis": 0,
      "num_experts": 256,
      "bits": 2,
      "group_size": 64
    }
  ]
}
```

Rules:

- `required=true` means AX Engine **must** stream. Loading the pack without
  `--stream-experts` / `AX_STREAM_EXPERTS=1` is a hard error. Do not silently
  attempt a full resident load (that OOMs or swap-thrashes).
- `mode` v1 is only `"layer-stack"`. Unknown modes fail closed.
- `tensors[].name` is the **runtime / sanitized MLX module path** AX Engine
  already uses (`switch_mlp.*` / packed expert roles), not the raw HF name.
- `tensors[].file` is repo-relative. One safetensors file may hold many layers;
  the engine must load **only the named tensors** from that file, never
  `eval` the whole file.
- `proj` is one of: `gate_up`, `gate`, `up`, `down`.
- `expert_axis` is 0 for packed `[E, out, in]` (and packed quantized views).
- Resident roles stay in unified memory for the process lifetime.
- Streamed roles are **absent** from the initial `load_weights` map.

Also record a pointer on the existing artifact manifest:

- `RuntimeMetadata.memory_policy["expert_stream"] = "required" | "optional" | "off"`
- `RuntimeMetadata.memory_policy["expert_stream_manifest"] = "ax_expert_stream.json"`

Do **not** bump `axquant.artifact.v2` in a breaking way. Additive memory_policy
keys are enough for v1.

## AXQuant work (this repo)

Implement in `/Users/akiralam/code/_ssd-stream/axquant`.

1. **Schema**
   - Pydantic model `ExpertStreamManifest` in `src/axquant/schema/artifacts.py`
     (or a new `src/axquant/schema/expert_stream.py` imported from schema `__init__`).
   - JSON Schema `schemas/axquant.expert-stream.v1.schema.json`.
   - Register in `src/axquant/schema/registry.py` as compatibility_class
     `operational`, freeze_policy `additive-ok`.
   - Update `docs/schema-catalog.md` via the existing catalog renderer if that
     is how other schemas are listed; otherwise add the row consistently.
   - Round-trip tests.

2. **Qwen 3.8 adapter** (`src/axquant/architectures/qwen38.py`)
   - `adapter_id = "qwen38-moe-v1"`
   - `product_family = "qwen3.8"`
   - Match `model_type in {"qwen3_5_moe_text", "qwen3_5_moe"}` **and** a Qwen 3.8
     2.4T-A95B identity (name regex `qwen[._-]?3[._-]?8` plus a 2.4T / A95B
     size hint, or a config signature once known). Do **not** steal Qwen 3.6
     35B-A3B (`qwen3_5_moe` + 40 layers / 256 experts / hidden 2048) from
     `qwen36-v1`.
   - `declared_tier = CONVERTIBLE`, support_level SUPPORTED for the catalog
     2.4T-A95B text MoE only; other qwen3.8 sizes stay inspect-only.
   - Notes: development evidence; AX Engine stream required; no cert track yet.

3. **Policy** (`src/axquant/support_policy.py`)
   - Add a `qwen3.8` family policy: investment `THIN`, `cert_track=False`.
   - **Replace** the old “Do not invent SSD expert streaming” rule. New rule:
     SSD expert streaming is an explicit Super-class product path for packs
     whose full-resident size exceeds target unified memory. Do not copy
     mlx-optiq. Do not claim OptiQ quality or AX Engine cert by association.
   - Keep Nemotron Super/Ultra inspect-only until a stream convert exists for
     those families (out of v1 scope).

4. **Emit the manifest on convert**
   - When converting a stream-capable adapter (qwen38-moe-v1) **or** when
     `--expert-stream required|auto` is set:
     - Classify packed expert tensors (reuse `module_paths` packed-expert
       aliases / fused switch names).
     - Write `ax_expert_stream.json`.
     - Set `required=true` when estimated full-resident bytes exceed 256 GiB
       (auto) or when the flag is `required`.
   - CLI: `--expert-stream {off,auto,required}` default `auto`.
   - `off` on a Super-class pack must fail closed with a message that a 512 GB
     Mac still cannot run the pack resident.
   - Convert must still fail closed if the plan does not cover modules.

5. **Docs**
   - New `docs/expert-ssd-stream.md` (operator + contract).
   - Update `docs/qwen38-optiq-experimental.md`: OptiQ remains the *current*
     published experimental serve path; AXQ+Engine stream is the in-progress
     native path. Do not delete the OptiQ warning until Engine can load a
     real Qwen 3.8 pack.
   - Update `docs/known-issues.md` and the Qwen 3.8 README paragraph
     (dedupe the triple-copied paragraph if you touch README).

6. **Tests**
   - Adapter match / non-match (must not claim Qwen 3.6 35B-A3B).
   - Schema validation + reject unknown `mode`.
   - Manifest emission from a tiny synthetic packed-expert inventory (do not
     require the 2.4T weights).
   - Policy text no longer forbids the stream path for qwen3.8.

7. **Do not**
   - Import or copy mlx-optiq.
   - Claim quality vs BF16.
   - Change GPT-OSS / Qwen 3.6 certified paths.
   - Reshard weights into one-file-per-expert in v1.

## AX Engine work (ax-engine repo)

Implement in `/Users/akiralam/code/_ssd-stream/ax-engine`.

1. **Parse** `ax_expert_stream.json` from the model directory. Unknown
   `schema_version` / `mode` → hard error.

2. **Admission**
   - Flag: `--stream-experts off|auto|on` (default **auto**; bare `--stream-experts` is `on`).
     Env `AX_STREAM_EXPERTS=off|auto|on` (also `0`/`1`). Auto streams required packs
     and packs that cannot fit in unified memory plus 48 GiB headroom.
   - If manifest `required=true` and streaming is off → fail closed
     (`ExpertStreamRequired`) with estimated_full_resident_bytes in the
     message. Never fall through to full `load_weights`.
   - If streaming is on and no manifest → fail closed (do not guess).

3. **Initial load** (`crates/ax-engine-mlx/src/weights.rs`)
   - Today both the C loader and `AX_MMAP_WEIGHTS` load **every tensor in a
     file** and `eval` them all. That is the bug for Super-class MoE.
   - When streaming: build the name map from **resident tensors only**.
     Skip names listed in the stream manifest. Do **not** `eval` skipped
     tensors. If a safetensors file is expert-only, do not open it at init.
   - `LayerWeights.gate_up_exps_packed` / `gate_exps` / `up_exps` /
     `down_exps` stay `None` until paged.

4. **Pager** (new module, e.g. `crates/ax-engine-mlx/src/expert_stream.rs`)
   - Cache key: `(layer_idx, proj)`.
   - On MoE forward for layer L: if L's expert stack is not resident, load
     only that layer's streamed tensors from their files (single-tensor
     safetensors slice or mmap+copy of that tensor only), construct the same
     `QuantizedWeight` the resident path would have built, insert into cache.
   - Then call the existing `gather_qmm` path unchanged.
   - After the layer (or when cache exceeds budget): drop the LRU layer
     stack and free MLX arrays.
   - Default budget: `max(1 layer, AX_STREAM_EXPERT_LAYERS)` env, default 1.
   - Optional prefetch of layer L+1 after routing L (can be a follow-up if
     L+1 load is easy; do not block v1 on prefetch).

5. **Serve / doctor**
   - Surface stream mode in doctor/runtime-check JSON:
     `expert_stream: {enabled, required, resident_bytes, max_layer_bytes, cached_layers}`.
   - Existing disk prefix cache stays KV-only. Do not overload it.

6. **Family**
   - Accept `qwen3_5_moe_text` as the Qwen 3.8 text MoE family if the model
     config uses that `model_type`. Map it onto the existing Qwen 3.5/3.6 MoE
     forward if the layer math matches; otherwise add a thin alias. Do not
     silently treat it as Qwen 3.6 35B-A3B.

7. **Tests**
   - Unit: manifest parse; required+flag-off fails; skip-list excludes
     streamed names from initial load.
   - Synthetic: tiny packed expert stack (2 layers × 4 experts), stream
     load layer 0, run a dummy gather or at least prove the tensor appears
     in cache and layer 1 is still absent; then evict.
   - Do not require the 2.4T checkpoint.

8. **Do not**
   - Copy mlx-optiq.
   - Change dense / certified Qwen 3.6 27B/35B-A3B default load (stream
     stays off unless a stream manifest is present).
   - Use `AX_MMAP_WEIGHTS` as a substitute (it still materializes every
     tensor).

## Sequencing

Independent after this contract:

- AXQuant can land schema + adapter + emission + tests without Engine.
- Engine can land parser + skip-load + pager against a hand-written
  `ax_expert_stream.json` fixture without a real AXQ convert.

Integration later: convert a toy packed-MoE with `--expert-stream required`
and `ax-engine` load it with `--stream-experts`.

## Acceptance (v1)

- A pack with `required=true` cannot be fully resident-loaded by AX Engine.
- With `--stream-experts`, initial RSS does not include streamed expert
  tensors (tested on the synthetic fixture).
- Existing Qwen 3.6 / GPT-OSS / Gemma tests stay green.
- No mlx-optiq imports, comments, or copied control flow.
- Policy docs no longer say AX will never do SSD expert streaming.
