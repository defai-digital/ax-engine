import os

import ax_engine


def main() -> None:
    model_artifacts_dir = os.environ.get("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR")
    if not model_artifacts_dir:
        print(
            "skipping MLX streaming example: "
            "set AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR to run it"
        )
        return

    with ax_engine.Session(
        model_id="qwen3_dense",
        mlx=True,
        mlx_model_artifacts_dir=model_artifacts_dir,
    ) as session:
        for event in session.stream_generate([1, 2, 3], max_output_tokens=2):
            print("event:", event.event)
            if event.delta_tokens:
                print("delta_tokens:", event.delta_tokens)
            if event.request is not None:
                print("state:", event.request.state)
            if event.response is not None:
                print("output_tokens:", event.response.output_tokens)
                print("finish_reason:", event.response.finish_reason)


if __name__ == "__main__":
    main()
