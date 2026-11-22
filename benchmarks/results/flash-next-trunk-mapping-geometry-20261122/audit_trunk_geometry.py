"""Audit aligned-buffer feasibility using only Flash Next safetensors headers."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

PAGE_BYTES = 16384
WIDTHS = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U32": 4, "I64": 8, "U8": 1}


def snake_case(name):
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def selection(root, loader):
    source = loader.read_text().split("fn is_trunk_resident_role(", 1)[1].split("\n}", 1)[0]
    roles = {snake_case(name) for name in re.findall(r"NativeTensorRole::(\w+)", source)}
    manifest = json.loads((root / "model-manifest.json").read_bytes())
    index = json.loads((root / "model.safetensors.index.json").read_bytes())["weight_map"]
    files = defaultdict(set)
    for tensor in manifest["tensors"]:
        if tensor["role"] not in roles:
            continue
        name, file = tensor["name"], tensor["file"]
        if index.get(name, file) != file:
            raise ValueError("Manifest and weight index disagree")
        files[file].add(name)
        if name.endswith(".weight"):
            for suffix in (".scales", ".biases", ".bias"):
                sidecar = name.removesuffix(".weight") + suffix
                files[index.get(sidecar, file)].add(sidecar)
    return files


def measure(path, names):
    size = path.stat().st_size
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 2 <= length <= min(size - 8, 100_000_000):
            raise ValueError("Invalid safetensors header length")
        header_bytes = stream.read(length)
        data_start = stream.tell()
    ranges = []
    for name, value in json.loads(header_bytes).items():
        if name == "__metadata__":
            continue
        start, end = (data_start + offset for offset in value["data_offsets"])
        if not data_start <= start <= end <= size:
            raise ValueError("An array range escaped the shard")
        ranges.append((name, start, end, value["dtype"]))
    selected = [(start, end, WIDTHS[dtype]) for name, start, end, dtype in ranges if name in names]
    if not selected:
        raise ValueError("A selected shard contains no selected arrays")
    first = min(start for start, _, _ in selected)
    last = max(end for _, end, _ in selected)
    aligned_start = first // PAGE_BYTES * PAGE_BYTES
    aligned_end = (last + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES
    skipped = [(start, end) for name, start, end, _ in ranges if name not in names]
    counts = Counter(selected_tensors=len(selected), selected_bytes=sum(end-start for start, end, _ in selected),
                     scalar_aligned=sum(start % width == 0 for start, _, width in selected),
                     scalar_aligned_bytes=sum(end-start for start, end, width in selected
                                              if start % width == 0),
                     page_aligned_starts=sum(start % PAGE_BYTES == 0 for start, _, _ in selected),
                     page_aligned_ranges=sum(start % PAGE_BYTES == 0 and end % PAGE_BYTES == 0
                                             for start, end, _ in selected),
                     enclosing_page_aligned_bytes=aligned_end - aligned_start,
                     enclosing_ranges_past_eof=int(aligned_end > size),
                     excluded_bytes_inside_enclosing_range=sum(
                         max(0, min(end, aligned_end) - max(start, aligned_start)) for start, end in skipped))
    return {"header_sha256": sha256(header_bytes).hexdigest(), "shard_bytes": size, **counts}, counts


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    args = parser.parse_args()
    files, summary = {}, Counter()
    for name, selected in sorted(selection(args.model_dir, args.loader).items()):
        files[name], counts = measure(args.model_dir / name, selected)
        summary.update(counts)
    payload = {"scope": "metadata-only resident MXFP4 trunk geometry", "page_bytes": PAGE_BYTES,
               "source_sha256": sha256(args.loader.read_bytes()).hexdigest(),
               "manifest_sha256": sha256((args.model_dir / "model-manifest.json").read_bytes()).hexdigest(),
               "index_sha256": sha256((args.model_dir / "model.safetensors.index.json").read_bytes()).hexdigest(),
               "files": files, "total": {"shards": len(files), **summary}}
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
