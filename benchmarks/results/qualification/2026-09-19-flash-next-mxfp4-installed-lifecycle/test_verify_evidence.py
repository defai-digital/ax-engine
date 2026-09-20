"""Reject changed lifecycle claims and altered raw protocol evidence."""
import copy
import unittest

import verify_evidence as verifier


class EvidenceTests(unittest.TestCase):
    def test_original_raw_reconstruction(self):
        result = verifier.verify(*verifier.load())
        self.assertTrue(result["installed_lifecycle_passed"])
        self.assertFalse(result["release_ready"])

    def test_reject_source_and_gate_promotion(self):
        for key, value in (("source_commit", "other"), ("wheel_sha256", "other"),
                           ("release_ready", True),
                           ("mtp_certification", {"MTP-S": "passed"})):
            with self.subTest(key=key):
                summary, evidence = verifier.load()
                summary[key] = value
                with self.assertRaises(ValueError):
                    verifier.verify(summary, evidence)

    def test_reject_inactive_disconnect_and_changed_recovery(self):
        for kind in ("disconnect", "recovery"):
            with self.subTest(kind=kind):
                summary, evidence = verifier.load()
                mode = evidence["records"]["output/result.json"]["modes"]["required"]
                if kind == "disconnect":
                    mode["disconnect"]["producer_active_before_close"] = False
                else:
                    mode["recovered"]["choices"][0]["text"] += "changed"
                with self.assertRaises(ValueError):
                    verifier.verify(summary, evidence)

    def test_reject_changed_payload_and_unclean_exit(self):
        for kind in ("payload", "cleanup"):
            with self.subTest(kind=kind):
                summary, evidence = verifier.load()
                record = evidence["records"]["output/result.json"]
                if kind == "payload":
                    record["pack_layout_after"] = []
                else:
                    record["modes"]["default"]["cleanup"]["forced_kill"] = True
                with self.assertRaises(ValueError):
                    verifier.verify(summary, evidence)

    def test_reject_incomplete_stream(self):
        summary, evidence = verifier.load()
        stream = evidence["records"]["output/result.json"]["modes"]["default"]["stream"]
        stream["events"] = []
        with self.assertRaises(ValueError):
            verifier.verify(summary, evidence)


if __name__ == "__main__":
    unittest.main()
