import argparse

import ax_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the ax_engine Python binding")
    parser.add_argument("model", help="Path to a GGUF model")
    parser.add_argument(
        "--backend",
        default="cpu",
        choices=["auto", "cpu", "metal", "hybrid", "hybrid_cpu_decode"],
        help="Backend to use for the smoke test",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run a tiny generation smoke test after load/tokenize checks",
    )
    args = parser.parse_args()

    with ax_engine.Model.load(args.model, backend=args.backend) as model:
        print("loaded:", model.model_name or "<unnamed>", "arch=", model.architecture)
        print("vocab_size:", model.vocab_size, "ctx:", model.context_length)
        print("bos/eos:", model.bos_token_id, model.eos_token_id)
        if model.support_note:
            print("support_note:", model.support_note)

        tokens = model.tokenize("hello world", add_special=False)
        print("token_count:", len(tokens))
        print("decoded:", repr(model.decode(tokens)))

        with model.session() as session:
            print("session_position:", session.position)
            if args.generate:
                text = session.generate(
                    "Say hello in two words.",
                    max_tokens=4,
                    temperature=0.0,
                )
                print("generated:", repr(text))
                print("session_position_after:", session.position)
            else:
                print("generation_skipped: pass --generate to test decode")


if __name__ == "__main__":
    main()
