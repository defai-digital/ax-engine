#!/usr/bin/env python3
"""Prevent resource availability from being mistaken for peer execution evidence."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_flash_next_peer_bench_plan as mod


class PresentProbe(mod.Probe):
    def __init__(self, directory):
        self.directory = directory

    def is_apple_silicon(self):
        return True

    def memory_gib(self):
        return 192

    def pack_dir(self):
        return self.directory

    def tool(self, name):
        return '/usr/bin/true'


class PeerPreparationTest(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(mod.CONTRACT_PATH.read_text())
        self.metrics = mod.SERVER_METRICS.read_text()
        self.keys = mod.FALLBACK_KEYS.read_text()

    def problems(self, contract):
        return mod.contract_problems(contract, self.metrics, self.keys)

    def test_tracked_preparation_is_valid_but_not_executable(self):
        self.assertEqual(self.problems(self.contract), [])
        self.assertIs(self.contract['execution_ready'], False)
        self.assertTrue(all(p['command'] is None for p in self.contract['peers']))

    def test_empty_pack_and_dummy_tools_do_not_establish_readiness(self):
        with tempfile.TemporaryDirectory(prefix='ax-peer-empty-') as directory:
            missing = mod.missing_preconditions(self.contract, PresentProbe(directory))
        self.assertEqual(len(missing), 3, missing)
        for peer in self.contract['peers']:
            self.assertTrue(any(peer['id'] in message and 'not execution-ready' in message
                                for message in missing), missing)
        self.assertTrue(any('GGUF' in message for message in missing))

    def test_cli_resource_presence_still_fails_required_preconditions(self):
        with tempfile.TemporaryDirectory(prefix='ax-peer-cli-') as directory:
            probe = PresentProbe(directory)
            with mock.patch.object(mod, 'Probe', return_value=probe):
                for arguments, expected in ((['--require-preconditions'], 1), (['--dry-run'], 0)):
                    with self.subTest(arguments=arguments), mock.patch.object(sys, 'argv', ['check', *arguments]):
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            code = mod.main()
                        self.assertEqual(code, expected)
                        self.assertIn('not execution-ready', output.getvalue())

    def test_commands_and_promoted_readiness_are_rejected(self):
        for key, value in (('command', 'ds4 -m {pack} --speculative'), ('readiness', 'validated')):
            contract = copy.deepcopy(self.contract)
            contract['peers'][2][key] = value
            self.assertTrue(self.problems(contract))
        for key, value in (('execution_ready', True), ('execution_ready', 0),
                           ('comparison_only', False), ('version', 1)):
            contract = copy.deepcopy(self.contract)
            contract[key] = value
            self.assertTrue(self.problems(contract))

    def test_peer_identity_and_review_cannot_be_missing(self):
        mutations = [lambda c: c['peers'].pop(),
                     lambda c: c['peers'].append(c['peers'][0]),
                     lambda c: c['peers'][0].pop('source_review'),
                     lambda c: c['peers'][0]['source_review'].update(revision='unknown'),
                     lambda c: c['peers'][0]['source_review'].update(paths=[]),
                     lambda c: c['peers'][0].update(blocker='')]
        for mutation in mutations:
            contract = copy.deepcopy(self.contract)
            mutation(contract)
            self.assertTrue(self.problems(contract))

    def test_malformed_shapes_fail_without_exceptions(self):
        for value in (None, [], 7, 'record'):
            self.assertTrue(self.problems(value))
            for key in ('target_host', 'pack', 'ax_arm', 'peers', 'open_gates_untouched'):
                contract = copy.deepcopy(self.contract)
                contract[key] = value
                self.assertTrue(self.problems(contract))
        for key, field in (('target_host', 'min_memory_gib'), ('pack', 'published_bytes')):
            for value in (True, '192', None, -1):
                contract = copy.deepcopy(self.contract)
                contract[key][field] = value
                self.assertTrue(self.problems(contract))

    def test_unknown_metric_is_rejected(self):
        contract = copy.deepcopy(self.contract)
        contract['ax_arm']['required_metrics'].append('ax_engine_invented_metric')
        self.assertTrue(any('does not publish' in p for p in self.problems(contract)))

    def test_missing_host_and_pack_are_reported_with_peer_blockers(self):
        missing = mod.missing_preconditions(self.contract, mod.EmptyProbe())
        self.assertTrue(any('not Apple Silicon' in p for p in missing))
        self.assertTrue(any('PACK_DIR is not set' in p for p in missing))
        self.assertEqual(sum('not execution-ready' in p for p in missing), 3)


    def test_gate_names_must_match_exactly(self):
        contract = copy.deepcopy(self.contract)
        contract['open_gates_untouched'][0] = 'MTP-S-unrelated'
        self.assertTrue(any('omits MTP-S' in p for p in self.problems(contract)))


if __name__ == '__main__':
    unittest.main()
