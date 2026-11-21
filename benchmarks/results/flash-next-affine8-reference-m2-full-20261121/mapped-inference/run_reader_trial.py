import mmap
from pathlib import Path
import sys

import numpy as np

import run_flash_next_affine8_reference as reference


class MappedRows:
    def __init__(self):
        self.files = {}

    def read(self, handle, start, row_count, row_bytes, indices):
        if any(index < 0 or index >= row_count for index in indices):
            raise IndexError("index outside packed row plane")
        path = Path(handle.name)
        mapping = self.files.get(path)
        if mapping is None:
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self.files[path] = mapping
        rows = np.ndarray((row_count,), dtype=f"V{row_bytes}", buffer=mapping, offset=start)
        return rows[indices].tobytes()

    def close(self):
        for mapping in self.files.values():
            mapping.close()


def check_mapping():
    from tempfile import TemporaryDirectory

    original = reference.ple.read_rows
    with TemporaryDirectory() as trial:
        path = Path(trial) / "planes"
        path.write_bytes(bytes(range(96)))
        loaded = MappedRows()
        try:
            with path.open("rb") as handle:
                for offset, count, width, selected in ((8, 8, 5, [7, 1, 5, 1]),
                                                        (48, 4, 8, [3, 2, 0])):
                    assert loaded.read(handle, offset, count, width, selected) == original(
                        handle, offset, count, width, selected
                    )
            assert len(loaded.files) == 1
        finally:
            loaded.close()


def main():
    pack, out, arm = sys.argv[1:]
    mapped = MappedRows()
    if arm == "mapped":
        reference.ple.read_rows = mapped.read
    elif arm != "positional":
        raise ValueError(f"unknown trial arm {arm}")
    try:
        reference.run(Path(pack), "bounded", Path(out), profile_row_io=True)
        print(f"trial reader {arm}, {len(mapped.files)} physical file mappings", flush=True)
    finally:
        mapped.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check_mapping()
    else:
        main()
