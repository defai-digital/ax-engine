#!/usr/bin/env python3
"""Assemble summary.json from the reduced tables of passes A, B and C.

usage: build_summary.py <fair_A.json> <fair_B.json> <fair_C.json> <out summary.json>
"""
import json
import sys

a, b, c = (json.load(open(p)) for p in sys.argv[1:4])
out = sys.argv[4]


def hmean(values):
    return len(values) / sum(1 / v for v in values)


def lane(suite, name):
    row = dict(a[suite][name])
    row.pop("prompt_tokens", None)
    return row


suites = ("flappy", "long_code")
lanes = [
    "ax_tp", "ax_exact", "ax_d2", "ax_d1", "ax_direct",
    "mtplx_d3", "mtplx_d2", "mtplx_d1",
    "omlx_d3", "omlx_d2", "omlx_d1",
    "mlx_lm",
]
summary = {
    "campaign": "qwen36-35b-a3b-axq-6bit-mtp-peer-fair-2026-10-22",
    "supersedes": "the 2026-09-21 Qwen 3.6 35B-A3B 6-bit rows (MTPLX 127.87, OMLX unsupported)",
    "host": {
        "descriptor": "MacBook Pro, Apple M5 Max, 128 GB unified memory, internal SSD",
        "os": "macOS 27.0.1 (26A434)",
        "power": "AC",
        "thermal": "no thermal or performance warning recorded",
        "role": "campaign host, signal only; not a certification run, not the Mac mini M4 Pro SKU",
    },
    "window_utc": {
        "pass_A_forward": "2026-10-22T19:24:49Z .. 20:19",
        "pass_B_reverse": "2026-10-22T20:19:04Z .. 20:47",
        "pass_C_controls": "2026-10-22T21:13:07Z .. 21:30",
    },
    "runtimes": {
        "ax_engine": "7.8.0, branch commit 77242419 (unmerged at recording; main db2a8b92 plus the DFlash2 branch, "
                     "which does not engage on a Qwen 3.6 pack), release build, cargo 1.99.1 (Homebrew), MLX 0.32.4",
        "mtplx": "2.12.2 (latest release 2026-10-03), sustained profile, own venv: MLX 0.32.2, mlx-lm 0.31.3",
        "omlx": "0.7.2 (latest release 2026-10-21), Lightning MTP, MLX 0.32.4, mlx-vlm 0.7.8",
        "mlx_lm": "0.32.2 (latest release 2026-10-16), direct autoregressive, MLX 0.32.4",
    },
    "model": {
        "repo": "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP",
        "revision": "6a4c220734f81112555ee8783d91e0065c54301c",
        "source_model": "Qwen/Qwen3.6-35B-A3B",
        "weights": "AXQ mixed precision: 24.3% of parameters at 6 bit, 68.7% at 4 bit, 2.0% at 8 bit, 5.0% bf16; bf16 MTP head",
        "oMLX_variant": {
            "built_by": "axquant export-runtime --target omlx --source-trunk-norm-layout mlx_multiplier "
                        "(profile qwen35-mtp-v1, AXQuant branch feat/qwen35-omlx-export 3a65347, unmerged)",
            "trunk_shards": "five shards byte-identical to the stock pack (SHA-256 checked)",
            "head": "7 RMSNorm gains shifted +1 (raw_hf_delta -> mlx_multiplier), indexed as model-mtp.safetensors; vision sidecar omitted",
            "why": "oMLX repairs only 4 of the 7 head norms of a raw-delta head: 3.8% draft acceptance on the stock pack, 97.6% on the variant",
        },
    },
    "contract": {
        "suites": {"flappy": "4 prompts, 273-432 tokens", "long_code": "4 prompts, 451-852 tokens"},
        "generated_tokens": 256, "sampling": "greedy, seed 0", "thinking": "off (chat template)",
        "warmups": 2, "measured": 5, "cooldown_s": 3, "prefix_cache": "off",
        "gate": "three consecutive CPU-idle samples >= 94% before every lane",
        "statistic": "median of 20 measured runs per suite",
        "mtp_depth": "MTPLX and OMLX at 1, 2, 3; AX default controller and fixed 2, 1",
        "decode": "rate after the first generated token",
        "prefill_engine": "AX runner prefill_tok_s; MTPLX prompt_tokens / prompt_eval_time_s; mlx-lm prompt_tps; OMLX reports none",
        "prefill_client": "prompt_tokens / client time to first token (AX over HTTP; others in process)",
    },
    "results": {},
    "pass_B_reverse_order_decode": {},
    "text_agreement_with_ax_direct_pass_B": {},
    "context_copy_control_pass_C": {},
}
for suite in suites:
    summary["results"][suite] = {name: lane(suite, name) for name in lanes}
    summary["pass_B_reverse_order_decode"][suite] = {
        name: {"A": a[suite][name]["decode"], "B": b[suite][name]["decode"]}
        for name in ("ax_tp", "mtplx_d3", "omlx_d3", "mlx_lm") if name in b[suite]
    }
    summary["text_agreement_with_ax_direct_pass_B"][suite] = {
        name: b[suite][name]["text_identical_to_ax_direct"]
        for name in b[suite] if b[suite][name].get("text_identical_to_ax_direct")
    }
    summary["context_copy_control_pass_C"][suite] = {
        name: {k: c[suite][name][k] for k in ("decode", "prefill_engine", "ttft_ms", "e2e", "accept")}
        for name in c[suite]
    }

ratios = {}
for suite in suites:
    r = summary["results"][suite]
    ratios[suite] = {
        "ax_vs_mtplx_d3_decode": round(r["ax_tp"]["decode"] / r["mtplx_d3"]["decode"], 3),
        "ax_vs_mtplx_best_depth_decode": round(
            r["ax_tp"]["decode"] / max(r[f"mtplx_d{d}"]["decode"] for d in (1, 2, 3)), 3),
        "ax_vs_mtplx_copy_off_decode": round(
            r["ax_tp"]["decode"] / summary["context_copy_control_pass_C"][suite]["mtplx_nocopy_d3"]["decode"], 3),
        "ax_vs_omlx_d3_decode": round(r["ax_tp"]["decode"] / r["omlx_d3"]["decode"], 3),
        "ax_vs_mlx_lm_decode": round(r["ax_tp"]["decode"] / r["mlx_lm"]["decode"], 3),
        "ax_mtp_vs_ax_direct_decode": round(r["ax_tp"]["decode"] / r["ax_direct"]["decode"], 3),
        "ax_vs_mtplx_prefill_engine": round(r["ax_tp"]["prefill_engine"] / r["mtplx_d3"]["prefill_engine"], 3),
        "ax_vs_omlx_prefill_client": round(r["ax_tp"]["prefill_client"] / r["omlx_d3"]["prefill_client"], 3),
        "ax_vs_mlx_lm_prefill_client": round(r["ax_tp"]["prefill_client"] / r["mlx_lm"]["prefill_client"], 3),
    }
summary["ratios"] = ratios
both = {}
for name in ("ax_tp", "mtplx_d3", "omlx_d3", "mlx_lm", "ax_direct"):
    cases = [x for s in suites for x in summary["results"][s][name]["decode_case_medians"].values()]
    both[name] = round(hmean(cases), 1)
summary["decode_harmonic_mean_8_prompts"] = both
json.dump(summary, open(out, "w"), indent=2)
print("wrote", out)
