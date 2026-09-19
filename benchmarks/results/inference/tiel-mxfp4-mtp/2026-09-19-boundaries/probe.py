"""Native Tiel request-boundary probe; run only on the audited M5 Max host."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.update({
        "AX_MLX_NATIVE_CONFIRM": "1", "AX_MLX_MTP_FORCE_REQUESTED": "1",
        "AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP": "1",
        "AX_MLX_MTP_CONSERVATIVE_DEPTH": "0", "AX_ENGINE_PREFIX_REUSE_DISABLED": "1",
        "AX_MLX_PREFIX_CACHE_MAX_BYTES": "0", "AX_MLX_PREFIX_CACHE_MAX_ENTRIES": "0",
        "AX_MLX_PREFIX_CACHE_DISK_DISABLED": "1",
    })
    import _ax_engine

    coding = json.loads(Path(args.cases).read_text())[0]["token_ids"]
    prompts = {"coding": coding, "single": [42]}
    doc = {"model": Path(args.model).name, "prompts": prompts, "valid": [], "rejected": [],
           "extension_sha256": hashlib.sha256(Path(_ax_engine.__file__).read_bytes()).hexdigest()}
    session = _ax_engine.Session(model_id="tiel-boundaries", mlx=True,
                                 mlx_model_artifacts_dir=args.model)

    def generate(tokens, budget):
        emitted = []
        response = None
        batches = []
        for event in session.stream_generate(input_tokens=tokens, max_output_tokens=budget,
                temperature=0, top_p=1, top_k=0, seed=0, ignore_eos=True):
            if event["event"] == "step":
                batch = event.get("delta_tokens", [])
                emitted.extend(batch)
                batches.append(len(batch))
                assert len(emitted) <= budget, "stream exceeded output budget"
            if event["event"] == "response":
                assert response is None, "multiple terminal responses"
                response = event["response"]
        assert response is not None
        assert response["output_tokens"] == emitted and len(emitted) == budget
        assert response["finish_reason"] == "max_output_tokens"
        telemetry = response["route"]["crossover_decisions"]
        return {"tokens": emitted, "batches": batches, "finish_reason": response["finish_reason"],
                "mtp": {k: v for k, v in telemetry.items() if k.startswith("ax_mtp_source_mtp_")
                        or k in ("ax_mtp_requested", "ax_mtp_drafted_depth2")}}

    # Each rejection is immediately followed by a valid request on the same
    # session, proving that validation does not leave its worker slot wedged.
    for name, tokens, budget, error_type in [
        ("zero_budget", coding, 0, ValueError),
        ("empty_prompt", [], 1, ValueError),
        ("negative_budget", coding, -1, OverflowError),
        ("overflow_budget", coding, 2**32, OverflowError),
    ]:
        try:
            generate(tokens, budget)
        except error_type as error:
            doc["rejected"].append({"case": name, "type": type(error).__name__,
                                    "message": str(error), "recovery": generate(coding, 16)})
        else:
            raise AssertionError(f"{name} was not rejected")
    for kind, budgets in [("coding", [1, 2, 3, 4, 5, 16]), ("single", [1, 2, 4, 5])]:
        for budget in budgets:
            for repeat in range(2):
                doc["valid"].append({"prompt": kind, "budget": budget, "repeat": repeat,
                                     **generate(prompts[kind], budget)})
    session.close()
    doc["complete"] = True
    Path(args.output).write_text(json.dumps(doc, indent=2) + "\n")
    print("PASS: 20 short requests, four rejections and four same-session recoveries", flush=True)


if __name__ == "__main__":
    main()
