#!/usr/bin/env python3
"""Unit tests for the capability-QA grader in qa/run_ds4_qa.py.

Covers the strict LINE_SET status semantics (unchanged from the campaign
baseline) and the additive detection dimension, plus the repetition
diagnostic used on long primary outputs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

from run_ds4_qa import (  # noqa: E402
    grade,
    line_set_detection,
    normalize,
    scan_repetition,
    summarize,
)


def response(answer_line: str | None, reason: str = "stop") -> dict:
    content = answer_line if answer_line is not None else "no answer here"
    return {"choices": [{"finish_reason": reason, "message": {"content": content}}]}


class NormalizeLineSetTests(unittest.TestCase):
    def test_single_line_and_explicit_set(self) -> None:
        self.assertEqual(normalize("12", "LINE_SET"), frozenset({12}))
        self.assertEqual(normalize("9,10", "LINE_SET"), frozenset({9, 10}))

    def test_range_expands_inclusively(self) -> None:
        self.assertEqual(normalize("17-20", "LINE_SET"), frozenset({17, 18, 19, 20}))

    def test_inverted_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize("20-17", "LINE_SET")

    def test_safe_marker_mixed_with_lines_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize("0,5", "LINE_SET")

    def test_garbage_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize("lines 3 and 4", "LINE_SET")


class GradeLineSetTests(unittest.TestCase):
    CASE = {"kind": "LINE_SET", "answer": "17-20", "aliases": [], "key": "case-1"}

    def test_exact_range_is_correct_with_full_span(self) -> None:
        result = grade(self.CASE, response("Answer: 17-20"))
        self.assertEqual(result["status"], "correct")
        self.assertTrue(result["detection"])
        self.assertEqual(result["span_reported"], 4)
        self.assertEqual(result["span_expected"], 4)

    def test_single_line_inside_range_is_wrong_but_detected(self) -> None:
        result = grade(self.CASE, response("Answer: 20"))
        self.assertEqual(result["status"], "wrong")
        self.assertTrue(result["detection"])
        self.assertEqual(result["span_reported"], 1)
        self.assertEqual(result["span_expected"], 4)

    def test_disjoint_lines_are_wrong_and_undetected(self) -> None:
        result = grade(self.CASE, response("Answer: 2,5"))
        self.assertEqual(result["status"], "wrong")
        self.assertFalse(result["detection"])
        # Escaping answers carry no span denominator, so downstream ratios
        # never mix definitions.
        self.assertIsNone(result["span_expected"])

    def test_alias_key_is_honored(self) -> None:
        case = {"kind": "LINE_SET", "answer": "3", "aliases": ["5-6"], "key": "case-2"}
        detected = grade(case, response("Answer: 6"))
        self.assertEqual(detected["status"], "wrong")
        self.assertTrue(detected["detection"])
        self.assertEqual(detected["span_expected"], 2)
        exact = grade(case, response("Answer: 5,6"))
        self.assertEqual(exact["status"], "correct")

    def test_safe_marker_exact_match(self) -> None:
        case = {"kind": "LINE_SET", "answer": "0", "aliases": [], "key": "case-3"}
        result = grade(case, response("Answer: 0"))
        self.assertEqual(result["status"], "correct")
        self.assertTrue(result["detection"])
        self.assertEqual(result["span_reported"], 1)

    def test_unparseable_answer_is_invalid_format_without_detection(self) -> None:
        result = grade(self.CASE, response("Answer: almost everywhere"))
        self.assertEqual(result["status"], "invalid_format")
        self.assertNotIn("detection", result)

    def test_non_line_set_kinds_carry_no_detection_keys(self) -> None:
        case = {"kind": "INTEGER", "answer": "5", "aliases": [], "key": "case-4"}
        result = grade(case, response("Answer: 5"))
        self.assertEqual(result, {"status": "correct", "answer": "5"})

    def test_strict_status_ignores_detection(self) -> None:
        # The campaign-comparable contract: detection must never flip status.
        result = grade(self.CASE, response("Answer: 18"))
        self.assertEqual(result["status"], "wrong")


class GradeFormatTests(unittest.TestCase):
    CHOICE = {"kind": "CHOICE", "answer": "D", "aliases": [], "key": "case-5"}

    def test_missing_answer_line_is_invalid_format(self) -> None:
        result = grade(self.CHOICE, response("The answer is D."))
        self.assertEqual(result["status"], "invalid_format")

    def test_duplicate_answer_lines_are_invalid_format(self) -> None:
        result = grade(self.CHOICE, response("Answer: C\nAnswer: D"))
        self.assertEqual(result["status"], "invalid_format")

    def test_truncated_is_not_graded(self) -> None:
        result = grade(self.CHOICE, response(None, reason="length"))
        self.assertEqual(result, {"status": "truncated", "answer": None})

    def test_choice_letter_case_insensitive(self) -> None:
        result = grade(self.CHOICE, response("Answer: d"))
        self.assertEqual(result["status"], "correct")

    def test_think_blocks_are_stripped_before_grading(self) -> None:
        content = "<think>secret reasoning Answer: C</think>\nAnswer: D"
        result = grade(self.CHOICE, response(content))
        self.assertEqual(result["status"], "correct")

    def test_ordered_sequence_is_order_sensitive(self) -> None:
        case = {
            "kind": "ORDERED_SEQUENCE",
            "answer": "1,3,5",
            "aliases": [],
            "key": "case-6",
        }
        self.assertEqual(grade(case, response("Answer: 1, 3, 5"))["status"], "correct")
        self.assertEqual(grade(case, response("Answer: 1, 5, 3"))["status"], "wrong")


class SummarizeDetectionTests(unittest.TestCase):
    @staticmethod
    def row(kind: str, source: str, grade_payload: dict) -> dict:
        return {
            "key": f"{source}-1",
            "source": source,
            "domain": "d",
            "suite": "s",
            "kind": kind,
            "answer": "0",
            "grade": grade_payload,
            "elapsed_seconds": 1.0,
        }

    def test_three_way_detection_tally(self) -> None:
        rows = [
            self.row("LINE_SET", "COMPSEC", {"status": "correct", "detection": True}),
            self.row("LINE_SET", "COMPSEC", {"status": "wrong", "detection": True}),
            self.row("LINE_SET", "COMPSEC", {"status": "wrong", "detection": False}),
            self.row("LINE_SET", "COMPSEC", {"status": "truncated", "answer": None}),
            self.row("INTEGER", "COMPSEC", {"status": "correct"}),
        ]
        summary = summarize(rows, 5)
        self.assertEqual(summary["detection_total"], 4)
        self.assertEqual(summary["detection_graded"], 3)
        self.assertEqual(summary["detection_correct"], 2)
        self.assertEqual(
            summary["breakdown"]["source"]["COMPSEC"]["detection"],
            {"total": 4, "graded": 3, "correct": 2},
        )
        # Strict dimension is untouched by the additive one.
        self.assertEqual(
            summary["status_counts"],
            {"correct": 2, "wrong": 2, "truncated": 1},
        )
        self.assertEqual(summary["accuracy_all_planned"], 2 / 5)

    def test_no_line_set_rows_omits_detection(self) -> None:
        rows = [self.row("INTEGER", "AIME2025", {"status": "correct"})]
        summary = summarize(rows, 1)
        self.assertNotIn("detection_total", summary)
        self.assertNotIn("detection", summary["breakdown"]["source"]["AIME2025"])

    def test_ungraded_line_set_rows_stay_in_total(self) -> None:
        rows = [self.row("LINE_SET", "COMPSEC", {"status": "protocol_error", "answer": None})]
        summary = summarize(rows, 1)
        self.assertEqual(summary["detection_total"], 1)
        self.assertEqual(summary["detection_graded"], 0)
        self.assertEqual(summary["detection_correct"], 0)


class ScanRepetitionTests(unittest.TestCase):
    def test_empty_inputs_yield_no_keys(self) -> None:
        self.assertEqual(scan_repetition(None), {})
        self.assertEqual(scan_repetition(""), {})
        self.assertEqual(scan_repetition("   \n\n  "), {})

    def test_single_paragraph(self) -> None:
        result = scan_repetition("only one block")
        self.assertEqual(
            result,
            {
                "repetition_max_repeat": 1,
                "repetition_max_run": 1,
                "repetition_max_line_run": 1,
                "loop_suspect": False,
            },
        )

    def test_interleaved_repeats_do_not_suspect(self) -> None:
        result = scan_repetition("a\n\nb\n\na\n\nb\n\na")
        self.assertEqual(result["repetition_max_repeat"], 3)
        self.assertEqual(result["repetition_max_run"], 1)
        self.assertEqual(result["repetition_max_line_run"], 1)
        self.assertFalse(result["loop_suspect"])

    def test_consecutive_run_over_threshold_suspects(self) -> None:
        loop = scan_repetition("\n\n".join(["same block"] * 354))
        self.assertEqual(loop["repetition_max_repeat"], 354)
        self.assertEqual(loop["repetition_max_run"], 354)
        self.assertEqual(loop["repetition_max_line_run"], 354)
        self.assertTrue(loop["loop_suspect"])

    def test_single_newline_loop_is_detected_at_line_granularity(self) -> None:
        # A truncated loop that repeats lines WITHOUT blank lines merges into
        # one paragraph, so only the line-level run can catch it.
        loop = scan_repetition("\n".join(["same line"] * 354))
        self.assertEqual(loop["repetition_max_repeat"], 1)
        self.assertEqual(loop["repetition_max_run"], 1)
        self.assertEqual(loop["repetition_max_line_run"], 354)
        self.assertTrue(loop["loop_suspect"])

    def test_threshold_boundary(self) -> None:
        at_limit = scan_repetition("\n\n".join(["same block"] * 20))
        self.assertFalse(at_limit["loop_suspect"])
        over = scan_repetition("\n\n".join(["same block"] * 21))
        self.assertTrue(over["loop_suspect"])
        line_at_limit = scan_repetition("\n".join(["same line"] * 20))
        self.assertFalse(line_at_limit["loop_suspect"])
        line_over = scan_repetition("\n".join(["same line"] * 21))
        self.assertTrue(line_over["loop_suspect"])

    def test_oversized_text_records_error_instead_of_raising(self) -> None:
        from run_ds4_qa import REPETITION_SCAN_CHAR_LIMIT

        result = scan_repetition("x" * (REPETITION_SCAN_CHAR_LIMIT + 1))
        self.assertIn("repetition_scan_error", result)
        self.assertTrue(result["repetition_scan_error"].startswith("skipped"))


if __name__ == "__main__":
    unittest.main()
