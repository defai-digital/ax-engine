"""Reject altered acceptance decisions, coverage and promotion in the published record."""
import copy
import json
import unittest

import verify_evidence as verifier


class EvidenceControls(unittest.TestCase):
    def setUp(self):
        self.summary, self.evidence = verifier.load()

    def test_original_bounded_controls_recompute(self):
        result = verifier.verify(self.summary, self.evidence)
        self.assertTrue(result["bounded_native_controls_passed"])
        self.assertFalse(result["qualification"])
        self.assertEqual((result["real_accepted"], result["real_proposed"]), (95, 117))

    def test_invalid_same_state_acceptance_is_rejected(self):
        name = "output/real_head.log"
        lines = self.evidence["records"][name].splitlines()
        marker = verifier.review_same_state.EVENT
        for index, line in enumerate(lines):
            if marker not in line:
                continue
            prefix, payload = line.split(marker, 1)
            event = json.loads(payload)
            if event["accepted"]:
                event["draft"] = event["singleton"]["token"] + 1
                lines[index] = prefix + marker + json.dumps(event)
                break
        else:
            self.fail("Missing accepted path in the real cohort")
        self.evidence["records"][name] = "\n".join(lines) + "\n"
        with self.assertRaisesRegex(ValueError, "same-state acceptance"):
            verifier.verify(self.summary, self.evidence)

    def test_missing_trace_boundary_is_rejected(self):
        name = "output/real_head.log"
        lines = self.evidence["records"][name].splitlines()
        index = next(i for i, line in enumerate(lines) if "Flash Next MTP oracle request:" in line)
        del lines[index]
        self.evidence["records"][name] = "\n".join(lines) + "\n"
        with self.assertRaisesRegex(ValueError, "Incomplete trace cohort"):
            verifier.verify(self.summary, self.evidence)

    def test_runner_state_coverage_cannot_be_promoted(self):
        self.summary["runner_decode_state_compared"] = True
        with self.assertRaisesRegex(ValueError, "runner coverage"):
            verifier.verify(self.summary, self.evidence)

    def test_partial_controls_cannot_promote_s_p_or_d(self):
        for gate in ("MTP-S", "MTP-P", "MTP-D"):
            summary = copy.deepcopy(self.summary)
            summary["mtp_certification"][gate] = "passed"
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, "gate promotion"):
                verifier.verify(summary, self.evidence)


if __name__ == "__main__":
    unittest.main()
