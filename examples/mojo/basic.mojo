"""
Basic AX Engine usage from Mojo.

Prerequisites:
    maturin develop          # build the ax_engine Python extension
    magic run mojo basic.mojo

The Session struct delegates to the ax_engine Python package via Mojo's
PythonObject interop, so no separate HTTP client is needed.
"""

from sdk.mojo.ax_engine import Session, download_model


fn main() raises:
    # Download model if not already present (skips if manifest exists)
    # var model_dir = download_model("mlx-community/Qwen3-4B-4bit")

    var session = Session(
        "qwen3_dense",
        mlx=True,
        mlx_model_artifacts_dir="/path/to/mlx-model-artifacts",
    )

    var result = session.generate("Hello from Mojo!", max_output_tokens=64)
    print(result.output_text)

    session.close()
