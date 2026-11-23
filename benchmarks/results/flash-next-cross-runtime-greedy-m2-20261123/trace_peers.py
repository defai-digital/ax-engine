"""Compare first differing next-token decisions on the M2 host."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import signal
import sys

import mlx.core as mx
import numpy as np


mode, pack_arg, native_arg, reference_arg = sys.argv[1:]
pack, native, ref_root = map(lambda s: Path(s).resolve(strict=True), (pack_arg, native_arg, reference_arg))
root = Path.cwd()
smoke = runpy.run_path(str(root / "benchmarks/results/flash-next-omlx-trunk-m2-20261123/replay.py"))
target = json.loads(native.read_bytes())
corpus, _ = smoke["frozen_inputs"]()
index = {row["id"]: row["input_tokens"] for row in corpus}


def fail_deadline(_sig, _frame):
    raise TimeoutError("The cross-runtime trace exceeded its deadline")


def check_layout(language, prompt, expected, quantum):
    cache = language.make_cache()
    for offset in range(0, len(prompt) - 1, quantum):
        part = prompt[offset:min(offset + quantum, len(prompt) - 1)]
        logits = language(mx.array([part], dtype=mx.uint32), cache=cache).logits
        mx.eval(logits)
        del logits
    token = mx.array([[prompt[-1]]], dtype=mx.uint32)
    ids = []
    split = None
    for step, ax_id in enumerate(expected):
        logits = language(token, cache=cache).logits[0, -1].astype(mx.float32)
        token = mx.argmax(logits).reshape(1, 1)
        predicted = int(token.item())
        ids.append(predicted)
        if split is None and ax_id != predicted:
            scores = np.asarray(logits)
            assert np.isfinite(scores).all()
            order = np.argsort(-scores, kind="stable")
            assert int(order[0]) == predicted
            split = {
                "index":step, "ax_m2_id":ax_id, "peer_id":predicted,
                "ax_m2_rank":int(np.where(order == ax_id)[0][0])+1,
                "peer_minus_ax_m2":float(scores[predicted]-scores[ax_id]),
                "top3":[[int(i),float(scores[i])] for i in order[:3]],
                "all_better_than_ax_m2":int(np.count_nonzero(scores > scores[ax_id])),
            }
    mx.synchronize()
    mx.clear_cache()
    return {"ids":ids,"first_split_from_ax_m2":split,
            "shared_prefix":split["index"] if split else len(expected)}


def report(**fields):
    print(json.dumps(fields, sort_keys=True, allow_nan=False), flush=True)


def load_language():
    from mlx_vlm import utils
    assert Path(utils.__file__).resolve() == ref_root / "mlx_vlm/utils.py"
    smoke["protect_pack"](pack)
    if mode == "omlx":
        from omlx.engine import vlm
        from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime
        assert configure_qwen4_exp_runtime(pack, mode="mmap", mtp_enabled=False) == "mmap"
        with vlm._force_qwen4_exp_sanitize_on_load(pack), vlm._remap_nested_visual_on_load(pack):
            language = utils.load_model(pack,lazy=True,strict=True,trust_remote_code=False).language_model
        assert language.ple_gathers_ahead()
        return language
    if mode == "bounded":
        import run_flash_next_affine8_reference as bounded
        table = bounded.Affine8Table(pack)
        return bounded.load_language_model(pack, table)
    raise ValueError(mode)


for name, digest in smoke["PACK_HASHES"].items():
    assert sha256((pack / name).read_bytes()).hexdigest() == digest
assert smoke["pressure"]() == 1
signal.signal(signal.SIGALRM, fail_deadline)
signal.alarm(420)
mx.set_memory_limit(104 << 30)
mx.set_cache_limit(2 << 30)
model = load_language()
report(stage="ready", runner=mode, mlx=smoke["version"]("mlx"), ref_revision=smoke["head"](ref_root))
for row in target["rows"]:
    ids = index[row["id"]]
    assert sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()==row["prompt_sha256"]
    for cap in ((1024, 2048) if mode == "omlx" and len(ids) == 8192 else (1024,)):
        result = check_layout(model, ids, row["m2_ids"], cap)
        assert smoke["pressure"]() == 1
        report(stage="case", id=row["id"], length=len(ids), chunk=cap, **result)
report(stage="complete", mlx_peak_bytes=mx.get_peak_memory(), pressure=smoke["pressure"]())
