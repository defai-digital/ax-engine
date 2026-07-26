"""Tests for the MLX eval-site inventory guard."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_mlx_eval_sites.py"

spec = importlib.util.spec_from_file_location("check_mlx_eval_sites", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class CountingTests(unittest.TestCase):
    def _count(self, source: str) -> dict[str, int]:
        with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as handle:
            handle.write(source)
            path = pathlib.Path(handle.name)
        try:
            return guard.count_file(path)
        finally:
            path.unlink()

    def test_counts_each_eval_kind(self) -> None:
        source = """
fn run(arrays: &[&MlxArray]) {
    eval(&[a]);
    mlx_sys::async_eval(&[b]);
    let _ = try_eval(&[c]);
    let t = eval_first_u32(&d);
}
"""
        self.assertEqual(
            self._count(source),
            {"eval": 1, "async_eval": 1, "try_eval": 1, "eval_first_u32": 1},
        )

    def test_ignores_comments_and_test_modules(self) -> None:
        source = """
fn run() {
    // eval(&[a]); commented out
    eval(&[a]); // trailing comment mentioning eval(
}

#[cfg(test)]
mod tests {
    #[test]
    fn inner() {
        eval(&[a]);
        eval(&[b]);
    }
}
"""
        self.assertEqual(self._count(source), {"eval": 1})

    def test_does_not_match_unrelated_identifiers(self) -> None:
        source = """
fn run() {
    evaluate(&[a]);
    self.reeval(&[b]);
    let evaluation = 1;
}
"""
        self.assertEqual(self._count(source), {})


class BaselineTests(unittest.TestCase):
    def test_baseline_exists_and_matches_tree(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            "eval-site inventory drifted from baseline; review new sites for "
            "the fallible-step boundary (P0-C) and regenerate with --update:\n"
            f"{result.stderr}",
        )

    def test_baseline_is_sorted_json(self) -> None:
        baseline_path = ROOT / "scripts" / "mlx_eval_site_baseline.json"
        baseline = json.loads(baseline_path.read_text())
        self.assertEqual(list(baseline), sorted(baseline))
        self.assertGreater(len(baseline), 0)


if __name__ == "__main__":
    unittest.main()
