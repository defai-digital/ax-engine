import os
import pathlib
import sys


def server() -> None:
    bin = pathlib.Path(__file__).parent / "_bin" / "ax-engine-server"
    os.execv(str(bin), [str(bin)] + sys.argv[1:])
