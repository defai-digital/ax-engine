import argparse

import ax_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic ax_engine Python example")
    parser.add_argument("model", help="Path to a GGUF model")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cpu", "metal", "hybrid", "hybrid_cpu_decode"],
        help="Backend selection",
    )
    args = parser.parse_args()

    with ax_engine.Model.load(args.model, backend=args.backend) as model:
        print("architecture:", model.architecture)
        print("vocab_size:", model.vocab_size)
        print("model_name:", model.model_name)
        print("support_note:", model.support_note)

        with model.session() as session:
            reply = session.generate(
                "Explain KV cache in one short sentence.",
                max_tokens=48,
                temperature=0.7,
            )
            print("generate:", reply)

            session.reset()
            chat_reply = session.chat(
                [{"role": "user", "content": "Say hello in five words."}],
                max_tokens=16,
                temperature=0.0,
            )
            print("chat:", chat_reply)

            session.reset()
            print("stream:", end=" ", flush=True)
            for chunk in session.stream(
                "Count from 1 to 5.",
                max_tokens=16,
                temperature=0.0,
            ):
                print(chunk, end="", flush=True)
            print()


if __name__ == "__main__":
    main()
