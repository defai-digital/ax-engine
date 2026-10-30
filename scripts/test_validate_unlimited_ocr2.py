#!/usr/bin/env python3
"""Offline assertions for the exact-text OCR functional witness."""

import unittest

from scripts.validate_unlimited_ocr2 import ocr_text_matches


class OcrWitnessTests(unittest.TestCase):
    def test_expected_substring_does_not_hide_hallucinated_text(self):
        text = "<PAGE><BLOCK>text[1,2,3,4]<FORK>HELLO 123\n<BLOCK>text[5,6,7,8]<FORK>Extra text."
        self.assertFalse(ocr_text_matches(text, ["HELLO 123"], 1)[0])

    def test_two_pages_must_have_complete_ordered_bodies(self):
        text = "<PAGE><BLOCK>title[1,2,3,4]<FORK>Hello 123\n<PAGE><BLOCK>text[1,2,3,4]<FORK>World 456"
        self.assertTrue(ocr_text_matches(text, ["HELLO 123", "WORLD 456"], 2)[0])
        self.assertFalse(ocr_text_matches(text, ["WORLD 456", "HELLO 123"], 2)[0])
        self.assertFalse(ocr_text_matches(text, ["HELLO 123", "WORLD 456"], 1)[0])

    def test_missing_layout_body_is_not_an_ocr_pass(self):
        self.assertFalse(ocr_text_matches("HELLO 123", ["HELLO 123"], 1)[0])


if __name__ == "__main__":
    unittest.main()
