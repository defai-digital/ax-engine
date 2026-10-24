#!/usr/bin/env python3
"""Reduce the fair-campaign lane outputs (prefill and decode for every runtime).

usage: summarize_fair.py <out-dir> <key> <pass> <tokenizer-dir> <suite> [<suite>...]
Lane files: <out-dir>/<key>_<suite>_<lane>_<pass>.json.

Per lane and suite (4 cases x 5 measured repetitions):
  decode   tokens after the first, over the time after the first token
           (AX: runner decode rate; MTPLX: decode rate; oMLX: stream timing;
           mlx-lm: stream timing)
  prefill  engine-reported prompt rate where the runtime reports one (AX runner,
           MTPLX prompt_eval_time_s, mlx-lm prompt_tps) and the client-side
           prompt_tokens / time-to-first-token where one was measured
  e2e      generated tokens over the whole request, prefill included
  text     the lane's greedy text against the AX direct decode of the same prompt
"""
import json
import pathlib
import statistics as st
import sys


def med(xs):
    xs = [x for x in xs if x]
    return st.median(xs) if xs else None


def hmean(xs):
    xs = [x for x in xs if x]
    return len(xs) / sum(1 / x for x in xs) if xs else None


def lane_ax(d, tok):
    cases = {}
    for r in d["results"]:
        trials = r["trials"]
        pt = r.get("prompt_tokens")
        gen = r.get("generation_tokens") or 256
        accept = [0, 0]
        for t in trials:
            tel = t.get("ngram_acceleration_telemetry") or {}
            accept[0] += tel.get("ax_mtp_accepted_tokens", 0)
            accept[1] += tel.get("ax_mtp_draft_tokens", 0)
        ids = trials[0].get("output_token_ids")
        cases[r["prompt_case_id"]] = dict(
            prompt_tokens=pt,
            decode=[t["decode_tok_s"] for t in trials],
            prefill_engine=[t.get("prefill_tok_s") for t in trials],
            prefill_client=[pt / (t["client_wall_ttft_ms"] / 1000) for t in trials
                            if pt and t.get("client_wall_ttft_ms")],
            ttft_ms=[t.get("client_wall_ttft_ms") for t in trials],
            e2e=[(t.get("output_tokens") or gen) / (t["client_wall_total_ms"] / 1000)
                 for t in trials if t.get("client_wall_total_ms")],
            accept=(accept[0] / accept[1]) if accept[1] else None,
            text=tok.decode(ids) if ids else None,
        )
    return cases


def lane_mtplx(d, tok):
    cases = {}
    for r in d["results"]:
        runs = [x for x in r["runs"] if x.get("measured", True)]
        pt = r.get("prompt_tokens")
        acc = [sum(x.get("accepted_drafts", 0) for x in runs), sum(x.get("drafted_tokens", 0) for x in runs)]
        cases[r["prompt_id"]] = dict(
            prompt_tokens=pt,
            decode=[x["decode_tok_s"] for x in runs],
            prefill_engine=[pt / x["prompt_eval_time_s"] for x in runs if x.get("prompt_eval_time_s") and pt],
            prefill_client=[],
            ttft_ms=[x["prompt_eval_time_s"] * 1000 for x in runs if x.get("prompt_eval_time_s")],
            e2e=[x.get("end_to_end_tok_s") for x in runs],
            accept=(acc[0] / acc[1]) if acc[1] else None,
            text=runs[0].get("text"),
        )
    return cases


def lane_omlx(d, tok):
    cases = {}
    for r in d["results"]:
        tr = r["trials"]
        pt = r.get("prompt_tokens")
        cases[r["prompt_case_id"]] = dict(
            prompt_tokens=pt,
            decode=[t["decode_only_tok_s"] for t in tr],
            prefill_engine=[t["engine_prompt_tps"] for t in tr if t.get("engine_prompt_tps")],
            prefill_client=[t["prefill_tok_s"] for t in tr],
            ttft_ms=[t["ttft_s"] * 1000 for t in tr],
            e2e=[t["decode_tok_s"] for t in tr],
            accept=None,
            text=tr[0].get("text"),
        )
    return cases


def lane_mlx_lm(d, tok):
    cases = {}
    for r in d["results"]:
        tr = r["trials"]
        pt = r.get("prompt_tokens")
        cases[r["prompt_id"]] = dict(
            prompt_tokens=pt,
            decode=[t["decode_tok_s"] for t in tr],
            prefill_engine=[t["engine_prompt_tps"] for t in tr],
            prefill_client=[t["prefill_tok_s"] for t in tr],
            ttft_ms=[t["ttft_s"] * 1000 for t in tr],
            e2e=[t["generated_tokens"] / (t["ttft_s"] + (t["generated_tokens"] - 1) / t["decode_tok_s"])
                 for t in tr if t["decode_tok_s"]],
            accept=None,
            text=tr[0].get("text"),
        )
    return cases


def loader(lane):
    if lane.startswith("ax"):
        return lane_ax
    return dict(mtplx=lane_mtplx, omlx=lane_omlx, mlx=lane_mlx_lm)[lane.split("_")[0]]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    out = pathlib.Path(sys.argv[1])
    key, label, tokdir, suites = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokdir)
    summary = {}
    for suite in suites:
        lanes = {}
        for path in sorted(out.glob(f"{key}_{suite}_*_{label}.json")):
            lane = path.name[len(f"{key}_{suite}_"):-len(f"_{label}.json")]
            try:
                lanes[lane] = loader(lane)(json.load(open(path)), tok)
            except Exception as e:  # report and keep going
                lanes[lane] = {"error": f"{type(e).__name__}: {e}"}
        ref = lanes.get("ax_direct", {})
        summary[suite] = {}
        for lane, cases in lanes.items():
            if "error" in cases:
                summary[suite][lane] = cases
                continue
            dec = [x for c in cases.values() for x in c["decode"]]
            pe = [x for c in cases.values() for x in c["prefill_engine"]]
            pc = [x for c in cases.values() for x in c["prefill_client"]]
            tt = [x for c in cases.values() for x in c["ttft_ms"]]
            e2 = [x for c in cases.values() for x in c["e2e"]]
            acc = [c["accept"] for c in cases.values() if c["accept"] is not None]
            same = div = 0
            if ref and lane != "ax_direct" and not isinstance(ref, dict) or ref:
                for cid, c in cases.items():
                    a = (ref.get(cid) or {}).get("text")
                    b = c.get("text")
                    if a is None or b is None:
                        continue
                    a, b = a.strip(), b.strip()
                    same += a == b
                    div += a != b
            summary[suite][lane] = dict(
                decode=round(med(dec), 2) if med(dec) else None,
                decode_case_medians={k: round(med(c["decode"]), 1) for k, c in cases.items()},
                prefill_engine=round(med(pe), 1) if med(pe) else None,
                prefill_client=round(med(pc), 1) if med(pc) else None,
                ttft_ms=round(med(tt), 1) if med(tt) else None,
                e2e=round(med(e2), 2) if med(e2) else None,
                accept=round(sum(acc) / len(acc), 3) if acc else None,
                text_identical_to_ax_direct=f"{same}/{same + div}" if (same + div) else None,
                prompt_tokens={k: c["prompt_tokens"] for k, c in cases.items()},
            )
    json.dump(summary, open(out / f"{key}_fair_{label}.json", "w"), indent=1)
    for suite, lanes in summary.items():
        print(f"## {key} / {suite} (pass {label})")
        print("| lane | decode tok/s | prefill tok/s (engine) | prefill tok/s (client TTFT) | TTFT ms | e2e tok/s | draft accept | text = AX direct |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for lane, v in lanes.items():
            if "error" in v:
                print(f"| {lane} | error: {v['error']} |")
                continue
            f = lambda x: "" if x is None else x
            print(f"| {lane} | {f(v['decode'])} | {f(v['prefill_engine'])} | {f(v['prefill_client'])} | {f(v['ttft_ms'])} | {f(v['e2e'])} | {f(v['accept'])} | {f(v['text_identical_to_ax_direct'])} |")
        print()


main()
