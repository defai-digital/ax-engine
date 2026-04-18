import ax_engine


def main() -> None:
    with ax_engine.Session(
        model_id="qwen3_dense",
        support_tier="compatibility",
        compat_backend="vllm",
        compat_server_url="http://127.0.0.1:8081",
    ) as session:
        result = session.chat(
            [
                ax_engine.ChatMessage(role="system", content="You are AX."),
                {"role": "user", "content": "Say hello from the Python chat helper."},
            ],
            max_output_tokens=32,
        )

    print("prompt_text:", result.prompt_text)
    print("output_text:", result.output_text)


if __name__ == "__main__":
    main()
