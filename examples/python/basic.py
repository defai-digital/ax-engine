import os

import ax_engine


def main() -> None:
    model_artifacts_dir = os.environ.get("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR")
    if not model_artifacts_dir:
        print("skipping MLX example: set AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR to run it")
        return

    with ax_engine.Session(
        model_id="qwen3_dense",
        mlx=True,
        mlx_model_artifacts_dir=model_artifacts_dir,
    ) as session:
        runtime = session.runtime()
        result = session.generate([1, 2, 3], max_output_tokens=2)

    print("selected_backend:", runtime.selected_backend)
    print("support_tier:", runtime.support_tier)
    print("resolution_policy:", runtime.resolution_policy)
    print("output_tokens:", result.output_tokens)
    print("finish_reason:", result.finish_reason)


if __name__ == "__main__":
    main()
