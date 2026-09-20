"""Native Python admission must reject invalid modes before loading any model."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

SOURCE_PYTHON = Path(__file__).resolve().parents[1]


def _child_environment() -> dict[str, str]:
    """Build the subprocess env without shadowing an installed wheel.

    `maturin develop` needs `python/` on PYTHONPATH so the child finds the
    editable package. Wheel smoke (`AX_ENGINE_RUN_INSTALLED_TESTS=1`) must
    import site-packages, including `_ax_engine`; pointing PYTHONPATH at
    source hides that extension.
    """
    environment = dict(os.environ)
    if os.environ.get("AX_ENGINE_RUN_INSTALLED_TESTS") == "1":
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = str(SOURCE_PYTHON)
    return environment


class StreamExpertsConfigTests(unittest.TestCase):
    def test_installed_wheel_child_does_not_put_source_on_pythonpath(self):
        previous_flag = os.environ.get("AX_ENGINE_RUN_INSTALLED_TESTS")
        previous_path = os.environ.get("PYTHONPATH")
        try:
            os.environ["AX_ENGINE_RUN_INSTALLED_TESTS"] = "1"
            os.environ["PYTHONPATH"] = str(SOURCE_PYTHON)
            environment = _child_environment()
            self.assertNotIn("PYTHONPATH", environment)
        finally:
            if previous_flag is None:
                os.environ.pop("AX_ENGINE_RUN_INSTALLED_TESTS", None)
            else:
                os.environ["AX_ENGINE_RUN_INSTALLED_TESTS"] = previous_flag
            if previous_path is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_path

    def test_source_tree_child_puts_package_root_on_pythonpath(self):
        previous_flag = os.environ.get("AX_ENGINE_RUN_INSTALLED_TESTS")
        try:
            os.environ.pop("AX_ENGINE_RUN_INSTALLED_TESTS", None)
            environment = _child_environment()
            self.assertEqual(environment["PYTHONPATH"], str(SOURCE_PYTHON))
        finally:
            if previous_flag is None:
                os.environ.pop("AX_ENGINE_RUN_INSTALLED_TESTS", None)
            else:
                os.environ["AX_ENGINE_RUN_INSTALLED_TESTS"] = previous_flag

    def test_invalid_explicit_and_environment_modes_fail_before_model_load(self):
        program = '''
from ax_engine._ax_engine import Session
import os
for explicit, environment in [("invalid", "off"), (None, "invalid"), ("", "off")]:
    os.environ["AX_STREAM_EXPERTS"] = environment
    try:
        Session(mlx=True, mlx_model_artifacts_dir="/nonexistent/ax-model",
                mlx_stream_experts=explicit)
    except ValueError as error:
        assert "invalid mlx_stream_experts" in str(error), str(error)
    else:
        raise AssertionError("invalid mode was accepted")
os.environ["AX_STREAM_EXPERTS"] = "invalid"
session = Session(mlx=False, llama_server_url="http://127.0.0.1:1")
session.close()
'''
        result = subprocess.run(
            [sys.executable, "-c", program],
            env=_child_environment(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
