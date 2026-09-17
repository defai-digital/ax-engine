"""Offline regression coverage for saved retest evidence auditing."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_qa_retest import audit


class RetestAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pack").mkdir()
        self.case = {"key": "case", "kind": "LINE_SET", "answer": "3-5", "aliases": []}
        self.write("pack/questions.json", {"cases": [self.case]})
        self.write("pack/manifest.json", {"question_order": ["case"], "max_tokens": 4})
        self.write("pack_provenance.json", {"source_commit": None, "binary_sha256": "abc"})
        self.row = {
            **self.case,
            "response": {"finish_reason": "max_output_tokens", "output_tokens": [1, 2, 3, 4]},
            "primary_grade": {"status": "truncated", "answer": None},
            "decoded_response": {
                "choices": [{"finish_reason": "stop", "message": {"content": "Answer: 4"}}]
            },
            "grade": {"status": "wrong", "answer": "4"},
        }
        self.row["recovery_decoded_response"] = self.row["decoded_response"]
        self.write("pack/000.json", self.row)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_recovery_preserves_primary_and_reports_partial_recall(self):
        result = audit(self.root, ["pack"])
        pack = result["packs"]["pack"]
        self.assertEqual(pack["primary"], {"truncated": 1})
        self.assertEqual(pack["final"], {"wrong": 1})
        self.assertEqual(pack["output_cap_hits"], 1)
        self.assertEqual(pack["line_set"]["recall_mean"], 1 / 3)
        self.assertEqual(pack["line_set"]["full_recall"], 0)
        self.assertEqual(pack["source_binding"], "unknown")
        self.assertEqual(result, audit(self.root, ["pack"]))
        self.assertEqual(len(result["input_sha256"]), 4)

    def test_rejects_missing_or_duplicate_records(self):
        self.write("pack/001.json", self.row)
        with self.assertRaisesRegex(ValueError, "missing, duplicate or misordered"):
            audit(self.root, ["pack"])
        (self.root / "pack/001.json").unlink()
        (self.root / "pack/000.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing, duplicate or misordered"):
            audit(self.root, ["pack"])

    def test_rejects_tampered_grade(self):
        self.row["grade"]["status"] = "correct"
        self.write("pack/000.json", self.row)
        with self.assertRaisesRegex(ValueError, "saved grade differs"):
            audit(self.root, ["pack"])

    def test_rejects_different_question_key(self):
        self.case["answer"] = "8-9"
        self.write("pack/questions.json", {"cases": [self.case]})
        with self.assertRaisesRegex(ValueError, "answer key differs"):
            audit(self.root, ["pack"])

    def test_hash_changes_with_evidence(self):
        before = audit(self.root, ["pack"])
        self.write("pack_provenance.json", {"source_commit": "abc"})
        after = audit(self.root, ["pack"])
        self.assertNotEqual(before["input_sha256"], after["input_sha256"])
        self.assertEqual(after["packs"]["pack"]["source_binding"], "unverified")


if __name__ == "__main__":
    unittest.main()
