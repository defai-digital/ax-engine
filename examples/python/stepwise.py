import os

import ax_engine


def main() -> None:
    model_artifacts_dir = os.environ.get("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR")
    if not model_artifacts_dir:
        print(
            "skipping MLX stepwise example: "
            "set AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR to run it"
        )
        return

    with ax_engine.Session(
        model_id="qwen3_dense",
        mlx=True,
        mlx_model_artifacts_dir=model_artifacts_dir,
    ) as session:
        request_id = session.submit([1, 2, 3], max_output_tokens=2)

        while True:
            request = session.snapshot(request_id)
            if request is None:
                raise RuntimeError(f"missing request snapshot for {request_id}")
            if request.state in {"finished", "cancelled", "failed"}:
                break

            step = session.step()
            print(
                "step:",
                step.step_id,
                "scheduled_requests:",
                step.scheduled_requests,
                "scheduled_tokens:",
                step.scheduled_tokens,
            )

    print("request_id:", request.request_id)
    print("state:", request.state)
    print("output_tokens:", request.output_tokens)


if __name__ == "__main__":
    main()
