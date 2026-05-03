import ax_engine


def main() -> None:
    with ax_engine.Session(model_id="qwen3_5_9b_q4", native_mode=True) as session:
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
