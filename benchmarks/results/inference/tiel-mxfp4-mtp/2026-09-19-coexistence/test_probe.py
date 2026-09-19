import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("coexistence_probe", Path(__file__).with_name("probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

class PressureGuards(unittest.TestCase):
    def setUp(self):
        self.base = dict(available=80*probe.GIB, compressed=probe.GIB, swap=512*1024**2, pressure=1)

    def test_reserve_is_checked_before_allocating(self):
        current = dict(self.base, available=16*probe.GIB+probe.GIB//2)
        self.assertIsNone(probe.guard(current, self.base))
        self.assertEqual(probe.guard(current, self.base, probe.GIB), "headroom")

    def test_existing_swap_and_compression_do_not_trigger_growth_guard(self):
        self.assertIsNone(probe.guard(self.base, self.base))

    def test_warning_is_observed_but_critical_releases(self):
        self.assertIsNone(probe.guard(dict(self.base, pressure=2), self.base))
        self.assertEqual(probe.guard(dict(self.base, pressure=4), self.base), "critical_pressure")

    def test_swap_growth_aborts_with_plenty_of_headroom(self):
        current = dict(self.base, swap=self.base["swap"]+128*1024**2+1)
        self.assertEqual(probe.guard(current, self.base), "swap_growth")

    def test_compressor_growth_aborts_without_swap(self):
        current = dict(self.base, compressed=self.base["compressed"]+probe.GIB+1)
        self.assertEqual(probe.guard(current, self.base), "compressor_growth")

    def test_normal_process_exit_does_not_hide_guard_release(self):
        record = dict(exit_code=0, released=dict(reason="compressor_growth", release_started=10))
        self.assertFalse(probe.pressure_completed(record, last_request_end=9))

    def test_request_after_memory_release_is_not_a_pressure_sample(self):
        record = dict(exit_code=0, released=dict(reason="released", release_started=10))
        self.assertFalse(probe.pressure_completed(record, last_request_end=11))
        self.assertTrue(probe.pressure_completed(record, last_request_end=9))

if __name__ == "__main__":
    unittest.main()
