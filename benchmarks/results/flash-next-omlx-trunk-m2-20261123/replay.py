"""Run the oMLX SSD-backed Flash Next trunk on frozen AX natural-load inputs."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tarfile
import time

import mlx.core as mx


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import run_flash_next_reference as reference


ARCHIVE = HERE.parent / "flash-next-current-natural-load-m5-20261123" / "evidence.tar.xz"
INPUT_HASHES = {
    "mixed-corpus.jsonl": "2c0c205d803207df6316dbd0076eef97451a55c80c0413fdc6b3bee5ffef7e9a",
    "prior-output.json": "517144b516b9a3d2e36448123fbe59a9fca0ea8b34015abf924417f2aaf9f7c8",
}
PACK_HASHES = {
    "model-manifest.json": "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef",
    "model.safetensors.index.json": "37344288777e7e78b35901cb1d572dad0dc640cb9cedb90024860b932a657696",
}
OMLX = "5eeec7757f83bfc99eeb87abe49686d00a0bba1f"
MLX_VLM = "138d7617d39653b39a5cbc379b1b5907a4481d67"
LENGTHS = (512, 2048, 8192)
BUDGET = 16


def report(stage, **values):
    print(json.dumps({"stage": stage, **values}, sort_keys=True, allow_nan=False), flush=True)


def head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def pressure():
    return int(subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]))


def timeout(_signal, _frame):
    raise TimeoutError("The bounded trunk smoke exceeded its deadline")


def frozen_inputs():
    data = {}
    with tarfile.open(ARCHIVE) as archive:
        for name, digest in INPUT_HASHES.items():
            with archive.extractfile(f"bundle/{name}") as handle:
                data[name] = handle.read()
            assert sha256(data[name]).hexdigest() == digest
    corpus = [json.loads(line) for line in data["mixed-corpus.jsonl"].splitlines()]
    return corpus, json.loads(data["prior-output.json"])


def protect_pack(pack):
    index = json.loads((pack / "model.safetensors.index.json").read_bytes())["weight_map"]
    weights = {(pack / name).resolve(strict=True) for name in set(index.values())}

    def deny_writes(event, args):
        if event != "open":
            return
        filename, _, flags = args
        if isinstance(filename, (str, bytes, os.PathLike)) and flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC
        ):
            path = Path(os.fsdecode(filename)).resolve()
            if path.is_relative_to(pack) or path in weights:
                raise PermissionError("This smoke denies checkpoint writes")

    sys.addaudithook(deny_writes)


def run(pack, omlx, mlx_vlm):
    from omlx.engine import vlm
    from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime
    import mlx_vlm.utils as utils

    assert head(omlx) == OMLX and head(mlx_vlm) == MLX_VLM
    assert Path(vlm.__file__).resolve() == omlx / "omlx/engine/vlm.py"
    assert Path(utils.__file__).resolve() == mlx_vlm / "mlx_vlm/utils.py"
    assert version("mlx") == "0.32.4"
    for name, digest in PACK_HASHES.items():
        assert sha256((pack / name).read_bytes()).hexdigest() == digest
    protect_pack(pack)
    corpus, ax = frozen_inputs()
    assert pressure() == 1
    mx.set_memory_limit(104 << 30)
    mx.set_cache_limit(2 << 30)
    assert configure_qwen4_exp_runtime(pack, mode="mmap", mtp_enabled=False) == "mmap"
    started = time.monotonic()
    with vlm._force_qwen4_exp_sanitize_on_load(pack), vlm._remap_nested_visual_on_load(pack):
        model = utils.load_model(pack, lazy=True, strict=True, trust_remote_code=False)
    language = model.language_model
    assert language.ple_gathers_ahead()
    report(
        "strict_lazy_load",
        omlx_commit=OMLX,
        mlx_vlm_commit=MLX_VLM,
        mlx_version=version("mlx"),
        pack_hashes=PACK_HASHES,
        seconds=round(time.monotonic() - started, 6),
    )
    for length in LENGTHS:
        row = next(row for row in corpus if len(row["input_tokens"]) == length)
        prompt = row["input_tokens"]
        result = reference.generate(language, mx, prompt, BUDGET, stop_on_eos=False)
        expected = ax[row["id"]][:BUDGET]
        actual = result["generated_ids"]
        assert len(actual) == len(expected) == BUDGET
        shared_prefix = next(
            (i for i, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b),
            BUDGET,
        )
        assert pressure() == 1
        report(
            "greedy_case",
            id=row["id"],
            input_tokens=length,
            prompt_sha256=sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
            ax_ids=expected,
            shared_prefix=shared_prefix,
            **result,
        )
    report(
        "complete",
        peak_mlx_bytes=mx.get_peak_memory(),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        pressure=pressure(),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pack", type=Path)
    cli.add_argument("omlx", type=Path)
    cli.add_argument("mlx_vlm", type=Path)
    args = cli.parse_args()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    run(args.pack.resolve(strict=True), args.omlx.resolve(strict=True), args.mlx_vlm.resolve(strict=True))
