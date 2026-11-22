"""Replay the latest complete native Flash Next sequence and partial reference."""

from gzip import decompress
import json
from pathlib import Path
import subprocess
import sys
from tarfile import open as open_tar
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
SCRIPTS = ROOT.parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_flash_next_release_gates as gates  # noqa: E402

DELIVERY = RESULTS / "flash-next-mxfp4-current-delivery-m5-20261122"
QA = RESULTS / "flash-next-current-qa-m5-20261122"
LIFECYCLE = RESULTS / "flash-next-current-lifecycle-m5-20261122"
MATRIX = RESULTS / "flash-next-current-matrix-m5-20261122"
REFERENCE = RESULTS / "flash-next-ordinary-reference-m5-20261121" / "partial-result.json"


def member(archive, name):
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"Expected a sealed evidence file: {name}")
    with handle:
        return handle.read()


def write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def stage(root):
    with open_tar(DELIVERY / "evidence.tar.xz") as archive:
        build = json.loads(member(archive, "build.json"))
        write(root, gates.RESULT_FILES["delivery"], member(archive, "delivery.json"))
        write(root, gates.DELIVERY_CONTROLS_FILE, member(archive, "native-support.json"))
    with open_tar(QA / "evidence.tar.xz") as archive:
        write(root, gates.RESULT_FILES["qa"], member(archive, "result.json"))
        items = gates.qa_cohort.load_items(gates.QA_ITEMS)
        for mode in gates.AX_ARMS:
            for index, item in enumerate(items):
                name = f'{mode}/{index:03d}-{item["id"]}.json'
                write(root, f"qa/{name}", member(archive, name))
    write(root, gates.RESULT_FILES["lifecycle"], (LIFECYCLE / "lifecycle.json").read_bytes())
    write(root, gates.RESULT_FILES["matrix"], decompress((MATRIX / "result.json.gz").read_bytes()))
    write(root, gates.MATRIX_REPEATS_FILE, decompress((MATRIX / "raw-runs.json.gz").read_bytes()))
    write(root, gates.RESULT_FILES["reference"], REFERENCE.read_bytes())
    return build


def replay():
    with TemporaryDirectory(prefix="ax-flash-release-board-") as directory:
        root = Path(directory)
        build = stage(root)
        report = root / "verdict.json"
        result = subprocess.run([sys.executable, "-B", str(SCRIPTS / "check_flash_next_release_gates.py"),
                                 "--evidence-dir", str(root), "--output", str(report)],
                                capture_output=True, text=True, check=False)
        if result.returncode not in (0, 1) or not report.is_file():
            raise RuntimeError(result.stderr or result.stdout)
        verdict = json.loads(report.read_bytes())
        if (result.returncode == 0) != (verdict["release_ready"] is True):
            raise ValueError("Evaluator exit status disagrees with the recorded result")
        delivery = json.loads((root / gates.RESULT_FILES["delivery"]).read_bytes())
        if delivery["server_sha256"] != build["binaries_sha256"]["ax-engine-server"]:
            raise ValueError("Native build does not match the delivery receipt")
    # The recorded evaluator source is its file hash, independent of the checkout's Git state.
    verdict.pop("source")
    return {**verdict, "native_build_commit": build["commit"], "native_server_sha256": delivery["server_sha256"],
            "evaluator_sha256": gates.sha256_file(Path(gates.__file__))}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
