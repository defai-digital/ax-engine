import ax_engine


def main() -> None:
    with ax_engine.Session(model_id="qwen3_dense") as session:
        runtime = session.runtime()
        result = session.generate([1, 2, 3], max_output_tokens=2)

    print("selected_backend:", runtime.selected_backend)
    print("support_tier:", runtime.support_tier)
    print("resolution_policy:", runtime.resolution_policy)
    print("output_tokens:", result.output_tokens)
    print("finish_reason:", result.finish_reason)


if __name__ == "__main__":
    main()
