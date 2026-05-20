import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_mlx_fastpath_env_controls.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_fastpath_env_controls", SCRIPT_PATH
)
assert MODULE_SPEC is not None
checker = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def owner_with_required_envs() -> str:
    return "\n".join(f'"{env_name}"' for env_name in sorted(checker.FASTPATH_CONTROL_ENVS))


class MlxFastpathEnvControlTests(unittest.TestCase):
    def test_allows_fastpath_owner_and_unrelated_envs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / checker.FASTPATH_OWNER, owner_with_required_envs())
            write(
                root / "crates/ax-engine-mlx/src/runner.rs",
                'let _ = std::env::var("AX_MLX_PREFIX_CACHE_MAX_BYTES");\n',
            )
            write(
                root / "crates/ax-engine-mlx/src/weights.rs",
                'let _ = std::env::var("AX_MMAP_WEIGHTS");\n',
            )

            checker.check_mlx_fastpath_env_controls(root)

    def test_rejects_fastpath_env_direct_parse_outside_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / checker.FASTPATH_OWNER, owner_with_required_envs())
            write(
                root / "crates/ax-engine-mlx/src/weights.rs",
                'let _ = std::env::var("AX_MLX_PACK_QKV_PROJECTIONS");\n',
            )

            with self.assertRaisesRegex(
                checker.MlxFastpathEnvControlError,
                "AX_MLX_PACK_QKV_PROJECTIONS",
            ):
                checker.check_mlx_fastpath_env_controls(root)

    def test_rejects_new_pack_env_prefix_outside_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / checker.FASTPATH_OWNER, owner_with_required_envs())
            write(
                root / "crates/ax-engine-mlx/src/model/shared/mlp.rs",
                'let _ = std::env::var("AX_MLX_PACK_NEW_TEMP_PROBE");\n',
            )

            with self.assertRaisesRegex(
                checker.MlxFastpathEnvControlError,
                "AX_MLX_PACK_NEW_TEMP_PROBE",
            ):
                checker.check_mlx_fastpath_env_controls(root)

    def test_rejects_direct_parse_of_dense_geglu_qffn_direct_env(self) -> None:
        # The dense GEGLU quantized FFN direct router is a fastpath control
        # owned by fastpath.rs, just like the AX_MLX_PACK_* family. Asserting
        # the gate explicitly here prevents future drift where the env gets
        # added to fastpath.rs but missed in FASTPATH_CONTROL_ENVS, which
        # would let other files parse it directly without being caught.
        self.assertIn("AX_MLX_DENSE_GEGLU_QFFN_DIRECT", checker.FASTPATH_CONTROL_ENVS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / checker.FASTPATH_OWNER, owner_with_required_envs())
            write(
                root / "crates/ax-engine-mlx/src/model/shared/mlp.rs",
                'let _ = std::env::var("AX_MLX_DENSE_GEGLU_QFFN_DIRECT");\n',
            )

            with self.assertRaisesRegex(
                checker.MlxFastpathEnvControlError,
                "AX_MLX_DENSE_GEGLU_QFFN_DIRECT",
            ):
                checker.check_mlx_fastpath_env_controls(root)

    def test_rejects_missing_owner_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(
                root / checker.FASTPATH_OWNER,
                owner_with_required_envs().replace('"AX_DISABLE_MLA_PREFIX_RESTORE"', ""),
            )

            with self.assertRaisesRegex(
                checker.MlxFastpathEnvControlError,
                "AX_DISABLE_MLA_PREFIX_RESTORE",
            ):
                checker.check_mlx_fastpath_env_controls(root)


if __name__ == "__main__":
    unittest.main()
