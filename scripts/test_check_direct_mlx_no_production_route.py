import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_direct_mlx_no_production_route.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_direct_mlx_no_production_route", SCRIPT_PATH
)
assert MODULE_SPEC is not None
checker = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class DirectMlxNoProductionRouteTests(unittest.TestCase):
    def test_allows_probe_and_mlx_sys_definition_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(
                root / "crates/mlx-sys/src/ops.rs",
                "pub fn gelu_approx_quantized_ffn() {}\n",
            )
            write(
                root / "crates/mlx-sys/src/bin/direct-mlx-hotpath-probe.rs",
                "let _ = gelu_approx_mul_matmul();\n",
            )
            write(
                root / "crates/mlx-sys/native/activation.cpp",
                "extern int ax_mlx_gelu_approx_quantized_ffn();\n",
            )

            checker.check_no_production_route(root)

    def test_rejects_runtime_use_of_no_go_quantized_ffn_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(
                root / "crates/ax-engine-mlx/src/model/shared/mlp.rs",
                "use mlx_sys::gelu_approx_quantized_ffn;\n",
            )

            with self.assertRaisesRegex(
                checker.DirectMlxNoProductionRouteError,
                "gelu_approx_quantized_ffn",
            ):
                checker.check_no_production_route(root)

    def test_rejects_runtime_use_of_no_go_activation_down_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(
                root / "crates/ax-engine-server/src/main.rs",
                "let _ = mlx_sys::gelu_approx_mul_matmul(&a, &b, &c, None);\n",
            )

            with self.assertRaisesRegex(
                checker.DirectMlxNoProductionRouteError,
                "gelu_approx_mul_matmul",
            ):
                checker.check_no_production_route(root)


if __name__ == "__main__":
    unittest.main()
