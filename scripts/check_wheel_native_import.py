#!/usr/bin/env python3
"""Load an unpacked wheel without developer packages or dylib search overrides."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_directory", type=Path)
    root = parser.parse_args().wheel_directory.resolve()
    if not (root / "ax_engine" / "__init__.py").is_file():
        raise SystemExit("unpacked wheel has no ax_engine package")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("DYLD_", "AX_ENGINE_"))
           and key not in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")}
    # This gate checks native loading. Do not rewrite the user's Metal cache
    # with paths into the temporary extraction; installed smoke checks assets.
    env["AX_ENGINE_METAL_BUILD_DIR"] = str(root / "ax_engine" / "_metal" / "build")
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", """
from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import ax_engine
from ax_engine import _ax_engine
assert Path(ax_engine.__file__).resolve().parent == root / 'ax_engine'
assert Path(_ax_engine.__file__).resolve().is_relative_to(root)
assert _ax_engine.__file__.endswith('.so')
print('verified: isolated native wheel import')
""", str(root)],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
