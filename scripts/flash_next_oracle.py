#!/usr/bin/env python3
"""Generate small, checkpoint-free Flash Next numerical oracle artifacts.

Requires PyTorch and Transformers installed at the pinned commit. This is a
validation tool, not an AX inference adapter. All model operations execute in
the upstream implementation; no model implementation is copied here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

TRANSFORMERS_COMMIT = "bd15bc95a89e728bbc1224084eb3b5829428c353"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--mtp-candidate", action="store_true",
                        help="Add synthetic draft weights; these are not an official MTP oracle")
    args = parser.parse_args()

    distribution = importlib.metadata.distribution("transformers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_COMMIT:
        raise SystemExit(f"Oracle requires Transformers commit {TRANSFORMERS_COMMIT}")

    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as upstream

    torch.set_num_threads(1)
    torch.manual_seed(7901)
    config = Qwen4ExpTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        hc_count=4,
        hc_lowrank=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        indexer_head_dim=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        output_gate_type="sigmoid",
        ple_layer_ids=[2],
        ple_embed_dim=16,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=2,
        eos_token_id=31,
        bos_token_id=31,
        seed=1234,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    )
    config._attn_implementation = "eager"
    model = upstream.Qwen4ExpForCausalLM(config).float().eval()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, payload: dict) -> None:
        (output_dir / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    with torch.no_grad():
        # Use nontrivial norm deltas and PLE convolution weights; initialization
        # deliberately zeros these, which would hide important graph mistakes.
        for name, parameter in model.named_parameters():
            if "hc_norm" in name or "ple.norm_" in name or "ple.conv1d" in name:
                parameter.copy_(torch.randn_like(parameter) * 0.15)
        residual = model.model.layers[0].attn_hyper_connection
        packed = torch.randn(1, 3, 64)
        branch = torch.randn(1, 3, 16)
        mixed, original, injection = residual(packed)
        combined = original + (branch.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)
        dump("residual.json", {
            "hidden_size": 16,
            "streams": 4,
            "epsilon": config.rms_norm_eps,
            "input": packed.tolist(),
            "branch": branch.tolist(),
            "weights": {name: value.tolist() for name, value in residual.state_dict().items()},
            "mixed": mixed.tolist(),
            "injection": injection.tolist(),
            "combined": combined.tolist(),
        })
        tokens = torch.tensor([[1, 2, 3, 4, 5, 31, 6, 7, 8]], dtype=torch.long)
        lookup = model.model.layers[1].ple.ple_embedding
        # Each embedding row stores its row index, making the upstream lookup
        # observable without duplicating its hash implementation.
        saved_embedding = lookup.ngram_embedding.weight.clone()
        lookup.ngram_embedding.weight.copy_(
            torch.arange(lookup.ngram_embedding.num_embeddings).unsqueeze(-1).expand_as(saved_embedding)
        )
        rows = lookup(tokens, None).reshape(1, tokens.shape[1], 4, 4)[..., 0].to(torch.int64)
        dump("ngram.json", {
            "vocabulary": config.vocab_size,
            "eos": config.eos_token_id,
            "multipliers": lookup.layer_multipliers.tolist(),
            "head_sizes": lookup.ngram_heads_vocab_sizes.tolist(),
            "head_offsets": lookup.ngram_heads_offsets.tolist(),
            "table_rows": lookup.ngram_embedding.num_embeddings,
            "heads_per_order": config.heads_per_ngram,
            "tokens": tokens[0].tolist(),
            "rows": rows[0].tolist(),
        })
        lookup.ngram_embedding.weight.copy_(saved_embedding)

        from transformers.cache_utils import DynamicCache

        gdn = model.model.layers[0].linear_attn
        gdn_input = torch.randn(1, 7, config.hidden_size)
        gdn_cache = DynamicCache(config=config)
        gdn_output = gdn(gdn_input, cache_params=gdn_cache)
        tiny_qk = torch.tensor([[[[0.0, 1e-8, -2e-6, 3e-5], [0.1, -0.2, 0.3, -0.4]]]])
        dump("gdn.json", {
            "input": gdn_input.tolist(),
            "output": gdn_output.tolist(),
            "weights": {name: value.tolist() for name, value in gdn.state_dict().items()},
            "conv_state": gdn_cache.layers[0].conv_states[0].tolist(),
            "recurrent_state": gdn_cache.layers[0].recurrent_states[0].tolist(),
            "near_zero_qk": tiny_qk.tolist(),
            "normalized_qk": upstream.l2norm(tiny_qk, dim=-1, eps=1e-6).tolist(),
        })
        ple = model.model.layers[1].ple
        ple_hidden = torch.randn(1, tokens.shape[1], config.hidden_size * config.hc_count)
        ple_embedding = ple.ple_embedding(tokens, None)
        ple_output = ple(ple_hidden, tokens, None)
        dump("ple.json", {
            "input": ple_hidden.tolist(),
            "embedding": ple_embedding.tolist(),
            "output": ple_output.tolist(),
            "weights": {name: value.tolist() for name, value in ple.state_dict().items()
                        if not name.startswith("ple_embedding.")},
            "kernel": config.ple_conv_kernel_size,
            "dilation": config.ngram_size,
            "hidden_size": config.hidden_size,
            "streams": config.hc_count,
            "epsilon": config.rms_norm_eps,
        })

        attention = model.model.layers[3].self_attn
        qsa_input = torch.randn(1, 19, config.hidden_size)
        positions = torch.arange(19).reshape(1, 1, 19).expand(3, 1, 19)
        rotary = model.model.rotary_emb(qsa_input, positions)
        visible = torch.ones(1, 1, 19, 19, dtype=torch.bool).tril()
        selected = attention.indexer(qsa_input, rotary, visible, None)
        additive_mask = torch.where(visible, 0.0, torch.finfo(torch.float32).min)
        attention_output, _ = attention(qsa_input, rotary, additive_mask, None)
        dump("qsa.json", {
            "input": qsa_input.tolist(),
            "selected": selected[:, 0].tolist(),
            "output": attention_output.tolist(),
            "weights": {name: value.tolist() for name, value in attention.state_dict().items()},
            "heads": config.indexer_n_heads,
            "head_dim": config.indexer_head_dim,
            "rotary_dim": 4,
            "ratio": config.indexer_compress_ratio,
            "budget": config.indexer_budget,
        })
        dump_qsa_bf16_rotary(dump, upstream, model, config)
        moe = model.model.layers[0].mlp
        moe_input = torch.randn(2, 5, config.hidden_size)
        dump("moe.json", {
            "input": moe_input.tolist(),
            "output": moe(moe_input).tolist(),
            "weights": {name: value.tolist() for name, value in moe.state_dict().items()},
            "hidden_size": config.hidden_size,
            "expert_count": config.num_experts,
            "top_k": config.num_experts_per_tok,
            "intermediate_size": config.moe_intermediate_size,
        })

        model = model.to(getattr(torch, args.dtype))
        whole = model(input_ids=tokens, use_cache=False).logits
        cache = None
        steps = []
        for token in tokens[0]:
            result = model(input_ids=token.reshape(1, 1), past_key_values=cache, use_cache=True)
            steps.append(result.logits)
            cache = result.past_key_values
        decode = torch.cat(steps, dim=1)
        chunk = model(input_ids=tokens[:, :3], use_cache=True)
        tail = model(input_ids=tokens[:, 3:], past_key_values=chunk.past_key_values, use_cache=True)
        chunked = torch.cat([chunk.logits, tail.logits], dim=1)
        tolerance = {"float32": 2e-5, "bfloat16": 1e-3, "float16": 1e-4}[args.dtype]
        torch.testing.assert_close(whole, decode, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(whole, chunked, atol=tolerance, rtol=tolerance)
        dump("logits.json", {
            "dtype": args.dtype,
            "tokens": tokens[0].tolist(),
            "whole": whole.tolist(),
            "decode": decode.tolist(),
            "chunked": chunked.tolist(),
            "max_decode_error": (whole - decode).abs().max().item(),
            "max_chunk_error": (whole - chunked).abs().max().item(),
        })
        save_file({name: value.contiguous() for name, value in model.state_dict().items()},
                  str(output_dir / "model.safetensors"))
        dump("config.json", {"model_type": "qwen4_exp", "text_config": config.to_dict()})
        if args.mtp_candidate:
            draft = {}
            for name, value in model.state_dict().items():
                if name.startswith("model.layers.3."):
                    draft[name.replace("model.layers.3.", "mtp.layers.0.", 1)] = value.clone().contiguous()
                elif name.startswith("model.hyper_connection_mixer."):
                    draft[name.replace("model.", "mtp.", 1)] = value.clone().contiguous()
            dtype = getattr(torch, args.dtype)
            hidden = config.hidden_size
            draft["mtp.fc_hidden.weight"] = torch.eye(hidden, dtype=dtype)
            draft["mtp.fc_embedding.weight"] = torch.eye(hidden, dtype=dtype) * 0.5
            draft["mtp.pre_fc_norm_hidden.weight"] = torch.zeros(hidden * config.hc_count, dtype=dtype)
            draft["mtp.pre_fc_norm_embedding.weight"] = torch.zeros(hidden, dtype=dtype)
            assert len(draft) == 31
            save_file(draft, str(output_dir / "mtp.safetensors"))
            dump("mtplx_runtime.json", {
                "schema_version": "axquant.mtp-runtime.v1",
                "mtp_norm_layout": "raw_hf_delta",
                "note": "Synthetic AX draft fixture; not an official MTP numerical oracle",
            })

    source = Path(upstream.__file__)
    dump("provenance.json", {
        "transformers_commit": TRANSFORMERS_COMMIT,
        "modeling_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "seed": 7901,
        "dtype": args.dtype,
        "component_fixture_dtype": "float32",
        "model": "synthetic four-layer Flash Next",
        "qualification": False,
        "synthetic_mtp_candidate": args.mtp_candidate,
    })
    print(f"Oracle artifacts written to {output_dir}")


def _bf16_pattern(count: int, start: int) -> list[float]:
    return [((start + i * 3) % 65 - 32) / 8.0 for i in range(count)]


def _nest(flat: list[float], shape: list[int]) -> list:
    if len(shape) == 1:
        return list(flat[: shape[0]])
    step = 1
    for dim in shape[1:]:
        step *= dim
    return [_nest(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def dump_qsa_bf16_rotary(dump, upstream, model, config) -> None:
    """Official BF16 rotary fixture. Cos/sin stay FP32 until one cast."""
    import torch

    rotary_dim = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
    head_dim = config.head_dim
    query_heads = config.num_attention_heads
    key_heads = config.num_key_value_heads
    seq = 4
    offset = 3
    queries = torch.tensor(
        _nest(_bf16_pattern(query_heads * seq * head_dim, 1), [1, query_heads, seq, head_dim]),
        dtype=torch.bfloat16,
    )
    keys = torch.tensor(
        _nest(_bf16_pattern(key_heads * seq * head_dim, 7), [1, key_heads, seq, head_dim]),
        dtype=torch.bfloat16,
    )
    position_ids = torch.arange(offset, offset + seq).reshape(1, 1, seq).expand(3, 1, seq)
    dummy = queries.reshape(1, seq, -1)
    rotary_emb = model.model.rotary_emb
    cos_fp32, sin_fp32 = rotary_emb(dummy.float(), position_ids)
    cos_bf16, sin_bf16 = rotary_emb(dummy, position_ids)
    queries_official, keys_official = upstream.apply_rotary_pos_emb(
        queries, keys, cos_bf16, sin_bf16
    )
    queries_fp32, keys_fp32 = upstream.apply_rotary_pos_emb(
        queries.float(), keys.float(), cos_fp32, sin_fp32
    )
    queries_single = queries_fp32.to(torch.bfloat16)
    keys_single = keys_fp32.to(torch.bfloat16)
    mismatch = (queries_official.float() != queries_single.float()).reshape(-1).nonzero(as_tuple=False)
    if mismatch.numel() == 0:
        raise SystemExit("qsa_bf16_rotary fixture needs an official vs single-rounding mismatch")

    def as_list(tensor) -> list:
        return tensor.detach().float().cpu().tolist()

    dump("qsa_bf16_rotary.json", {
        "source": (
            "Pinned official Qwen4ExpTextRotaryEmbedding and apply_rotary_pos_emb, "
            "CPU bfloat16"
        ),
        "transformers_commit": TRANSFORMERS_COMMIT,
        "modeling_sha256": hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "scope": (
            "Official eager rotary: FP32 inv_freq and angles, cos/sin cast once to "
            "BF16, then BF16 rotate_half products. Single-rounding is FP32 apply "
            "then one BF16 cast."
        ),
        "note": (
            "Certification notes must record this as a production-math correction "
            "with M2 evidence."
        ),
        "rope_base": float(config.rope_parameters["rope_theta"]),
        "rotary_dim": rotary_dim,
        "head_dim": head_dim,
        "query_heads": query_heads,
        "key_heads": key_heads,
        "offset": offset,
        "position_ids": list(range(offset, offset + seq)),
        "queries": as_list(queries),
        "keys": as_list(keys),
        "cos_fp32": as_list(cos_fp32[0]),
        "sin_fp32": as_list(sin_fp32[0]),
        "cos_bf16": as_list(cos_bf16[0]),
        "sin_bf16": as_list(sin_bf16[0]),
        "queries_official": as_list(queries_official),
        "keys_official": as_list(keys_official),
        "queries_single_rounding": as_list(queries_single),
        "keys_single_rounding": as_list(keys_single),
        "first_query_mismatch": int(mismatch[0].item()),
    })


if __name__ == "__main__":
    main()
