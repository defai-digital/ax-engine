#!/usr/bin/env python3
"""Replay the signed Flash Next single-request QSA route comparison."""
from pathlib import Path
from subprocess import run
import sys
import tarfile
from tempfile import TemporaryDirectory


def main():
    root = Path(__file__).resolve().parent
    with TemporaryDirectory(prefix="ax-flash-qsa-http-") as name:
        work = Path(name)
        with tarfile.open(root / "evidence.tar.xz") as archive:
            archive.extractall(work, filter="data")
        run([sys.executable, "-E", "-s", "-B", str(work / "compare_http.py"), str(work)], check=True)


if __name__ == "__main__":
    main()
