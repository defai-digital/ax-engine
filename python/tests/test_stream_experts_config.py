"""Native Python admission must reject invalid modes before loading any model."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class StreamExpertsConfigTests(unittest.TestCase):
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
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
        result = subprocess.run([sys.executable, '-c', program], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
