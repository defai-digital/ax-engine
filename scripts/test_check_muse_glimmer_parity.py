import unittest

from scripts.check_muse_glimmer_parity import (
    common_prefix_len,
    extract_mlxcel_generated_text,
)


class MuseGlimmerParityHelpersTests(unittest.TestCase):
    def test_extracts_generated_stream_between_cli_markers(self) -> None:
        output = (
            "Runtime device: Apple GPU (Metal)\r\n"
            "Generating...\r\n"
            "prompt text and continuation\r\n\r\n"
            "[Generated 8 tokens in 0.5s = 16 tok/s]\r\n"
        )

        self.assertEqual(
            extract_mlxcel_generated_text(output),
            "prompt text and continuation",
        )

    def test_rejects_output_without_summary(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "generated-token summary"):
            extract_mlxcel_generated_text("Generating...\nprompt and continuation\n")

    def test_common_prefix_is_ordered_and_duplicate_sensitive(self) -> None:
        self.assertEqual(common_prefix_len([1, 2, 2, 3], [1, 2, 9, 2, 3]), 2)
        self.assertEqual(common_prefix_len([1, 2], [1, 2, 3]), 2)
        self.assertEqual(common_prefix_len([], [1, 2, 3]), 0)


if __name__ == "__main__":
    unittest.main()
