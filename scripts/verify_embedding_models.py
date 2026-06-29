#!/usr/bin/env python3
"""Verify ax-engine embedding output against the family reference contract.

Exit codes mirror the benchmark convention:
  0 = all checks passed
  2 = contract failure (bad model dir, dependency failure, bad response shape)
  3 = correctness failure (cosine or normalization below threshold)

Examples:
    python scripts/verify_embedding_models.py \
        --model-dir /path/to/Qwen3-Embedding-0.6B-8bit/snapshots/<sha>

    python scripts/verify_embedding_models.py \
        --model-kind embeddinggemma \
        --model-dir /path/to/embeddinggemma-300m-8bit/snapshots/<sha>

Qwen uses mlx-lm last-token pooling as the oracle. EmbeddingGemma uses
mlx-embeddings one row at a time as the oracle because its reference batch path
is not invariant for mixed-length batches.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "AX Engine runs local embeddings on Apple Silicon.",
    "What is the capital of France?",
    "A short query",
    "This is a longer passage about vector databases, retrieval augmented "
    "generation, and chunk indexing performance on a local machine.",
    "Embedding models should produce stable normalized vectors for semantically "
    "similar text.",
    "Hello world",
]

DEFAULT_COSINE_THRESHOLD = 0.9990
QWEN_8B_4BIT_COSINE_THRESHOLD = 0.9960
DEFAULT_BATCH_CONSISTENCY_THRESHOLD = 0.9990


@dataclass(frozen=True)
class EmbeddingContract:
    model_kind: str
    model_id: str
    reference: str
    pooling: str
    cosine_threshold: float
    batch_consistency_threshold: float
    reference_single_oracle: bool


@dataclass(frozen=True)
class VerificationRows:
    token_ids: list[list[int]]
    reference_single: Any
    ax_single: Any
    ax_batch: Any | None


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def ensure_port_available(port: int, host: str = "127.0.0.1") -> None:
    if port == 0:
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(
                f"ax-engine embedding verification port {host}:{port} is already in use; "
                "stop the existing ax-engine-server process, pass --skip-server "
                "to verify that running server, or choose a free --port."
            )


def allocate_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def http_post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()}") from e


def read_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"model-manifest.json not found in {model_dir}")
    return json.loads(manifest_path.read_text())


def infer_model_kind(model_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    manifest = read_manifest(model_dir)
    family = str(manifest.get("model_family", ""))
    if family == "embeddinggemma":
        return "embeddinggemma"
    return "qwen3"


def default_cosine_threshold(model_dir: Path, model_kind: str) -> float:
    if model_kind == "qwen3":
        lowered = str(model_dir).lower()
        if "8b" in lowered and ("4bit" in lowered or "dwq" in lowered):
            return QWEN_8B_4BIT_COSINE_THRESHOLD
    return DEFAULT_COSINE_THRESHOLD


def build_contract(
    model_dir: Path,
    model_kind: str,
    cosine_threshold: float | None,
    batch_consistency_threshold: float | None,
) -> EmbeddingContract:
    threshold = (
        cosine_threshold
        if cosine_threshold is not None
        else default_cosine_threshold(model_dir, model_kind)
    )
    batch_threshold = (
        batch_consistency_threshold
        if batch_consistency_threshold is not None
        else DEFAULT_BATCH_CONSISTENCY_THRESHOLD
    )
    if model_kind == "embeddinggemma":
        return EmbeddingContract(
            model_kind=model_kind,
            model_id="embeddinggemma",
            reference="mlx-embeddings",
            pooling="mean",
            cosine_threshold=threshold,
            batch_consistency_threshold=batch_threshold,
            reference_single_oracle=True,
        )
    return EmbeddingContract(
        model_kind="qwen3",
        model_id="qwen3",
        reference="mlx-lm",
        pooling="last",
        cosine_threshold=threshold,
        batch_consistency_threshold=batch_threshold,
        reference_single_oracle=True,
    )


def tokenize_sentences(model_dir: Path, contract: EmbeddingContract) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    rows: list[list[int]] = []
    for sentence in SENTENCES:
        if contract.model_kind == "embeddinggemma":
            token_ids = tokenizer.encode(sentence, add_special_tokens=True)
        else:
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("Qwen embedding tokenizer has no eos_token_id")
            token_ids.append(int(eos_id))
        if not token_ids:
            raise ValueError(f"tokenizer produced an empty row for {sentence!r}")
        rows.append([int(token_id) for token_id in token_ids])
    return rows


def np_array_from_blob(blob: bytes, batch_size: int, hidden_size: int) -> Any:
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32).reshape(batch_size, hidden_size).copy()


def row_cosines(left: Any, right: Any) -> Any:
    import numpy as np

    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / np.maximum(denom, 1e-30)


def max_norm_error(matrix: Any) -> float:
    import numpy as np

    return float(np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0)))


def compute_qwen_reference_single(model_dir: Path, token_ids: list[list[int]]) -> Any:
    import mlx.core as mx
    from mlx_lm import load  # type: ignore[import]
    import numpy as np

    print(f"  Loading mlx-lm reference: {model_dir}")
    model, _ = load(str(model_dir))
    rows = []
    for ids in token_ids:
        hidden = model.model(mx.array([ids]))
        last = hidden[0, -1, :].astype(mx.float32)
        norm = mx.sqrt(mx.sum(last * last))
        normalized = last / (norm + 1e-12)
        mx.eval(normalized)
        rows.append(np.array(normalized, dtype=np.float32, copy=True))
    return np.vstack(rows)


def compute_embeddinggemma_reference_single(model_dir: Path, token_ids: list[list[int]]) -> Any:
    import mlx.core as mx
    import mlx_embeddings
    import numpy as np

    print(f"  Loading mlx-embeddings reference: {model_dir}")
    model, _ = mlx_embeddings.load(str(model_dir))
    rows = []
    for ids in token_ids:
        attention_mask = [[1] * len(ids)]
        out = model(mx.array([ids]), attention_mask=mx.array(attention_mask))
        rows.append(np.array(out.text_embeds[0], dtype=np.float32, copy=True))
    return np.vstack(rows)


def compute_reference_single(
    model_dir: Path,
    contract: EmbeddingContract,
    token_ids: list[list[int]],
) -> Any:
    if contract.model_kind == "embeddinggemma":
        return compute_embeddinggemma_reference_single(model_dir, token_ids)
    return compute_qwen_reference_single(model_dir, token_ids)


def load_ax_session(model_dir: Path, contract: EmbeddingContract) -> Any:
    python_root = REPO_ROOT / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    import ax_engine

    print(f"  Loading ax-engine session: {model_dir}")
    return ax_engine.Session(
        model_id=contract.model_id,
        mlx=True,
        support_tier="mlx_preview",
        mlx_model_artifacts_dir=str(model_dir),
    )


def ax_batch_matrix(session: Any, token_ids: list[list[int]], contract: EmbeddingContract) -> Any:
    blob, batch_size, hidden_size = session.embed_batch_flat_bytes(
        token_ids,
        pooling=contract.pooling,
        normalize=True,
    )
    return np_array_from_blob(blob, batch_size, hidden_size)


def verify_direct(model_dir: Path, contract: EmbeddingContract) -> VerificationRows:
    token_ids = tokenize_sentences(model_dir, contract)
    reference = compute_reference_single(model_dir, contract, token_ids)
    session = load_ax_session(model_dir, contract)
    try:
        ax_single_rows = [ax_batch_matrix(session, [ids], contract)[0] for ids in token_ids]
        import numpy as np

        ax_single = np.vstack(ax_single_rows)
        ax_batch = ax_batch_matrix(session, token_ids, contract)
    finally:
        close = getattr(session, "close", None)
        if close is not None:
            close()
    return VerificationRows(token_ids, reference, ax_single, ax_batch)


def compute_ax_http_embeddings(
    server_url: str,
    token_ids: list[list[int]],
    contract: EmbeddingContract,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for ids in token_ids:
        resp = http_post(
            f"{server_url}/v1/embeddings",
            {"input": ids, "pooling": contract.pooling, "normalize": True},
        )
        embeddings.append(resp["data"][0]["embedding"])
    return embeddings


def start_server(model_dir: Path, contract: EmbeddingContract, port: int) -> subprocess.Popen:
    ensure_port_available(port)
    server_bin = REPO_ROOT / "target" / "release" / "ax-engine-server"
    if not server_bin.exists():
        server_bin = REPO_ROOT / "target" / "debug" / "ax-engine-server"
    if not server_bin.exists():
        raise FileNotFoundError(
            "ax-engine-server binary not found. Run: cargo build -p ax-engine-server"
        )

    cmd = [
        str(server_bin),
        "--model-id",
        contract.model_id,
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print(f"  Starting server: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def verify_http(
    model_dir: Path,
    contract: EmbeddingContract,
    port: int,
    skip_server: bool,
) -> VerificationRows:
    token_ids = tokenize_sentences(model_dir, contract)
    reference = compute_reference_single(model_dir, contract, token_ids)
    server_url = f"http://127.0.0.1:{port}"
    server_proc: subprocess.Popen | None = None
    try:
        if not skip_server:
            server_proc = start_server(model_dir, contract, port)
            print(f"  Waiting for server on port {port}")
            if not wait_for_port("127.0.0.1", port, timeout=120.0):
                if server_proc:
                    out, _ = server_proc.communicate(timeout=5)
                    print(out, file=sys.stderr)
                raise RuntimeError("server did not start within 120s")
        import numpy as np

        ax_single = np.array(
            compute_ax_http_embeddings(server_url, token_ids, contract),
            dtype=np.float32,
        )
        return VerificationRows(token_ids, reference, ax_single, None)
    finally:
        if server_proc is not None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                try:
                    server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass


def print_results(contract: EmbeddingContract, rows: VerificationRows) -> bool:
    ref_vs_single = row_cosines(rows.reference_single, rows.ax_single)
    single_norm_error = max_norm_error(rows.ax_single)
    passed = True

    print()
    print(
        "Contract: "
        f"kind={contract.model_kind}, reference={contract.reference}, "
        f"pooling={contract.pooling}, cosine_threshold={contract.cosine_threshold:.6f}, "
        f"batch_consistency_threshold={contract.batch_consistency_threshold:.6f}"
    )
    print(f"{'Row':>3} {'Tokens':>6} {'AX single vs ref':>18} {'Pass':>6}")
    print("-" * 43)
    for idx, (ids, cosine) in enumerate(zip(rows.token_ids, ref_vs_single)):
        ok = float(cosine) >= contract.cosine_threshold
        passed = passed and ok
        print(f"{idx:>3} {len(ids):>6} {float(cosine):>18.9f} {str(ok):>6}")

    print(f"AX single max norm error: {single_norm_error:.3e}")
    passed = passed and single_norm_error < 1e-4

    if rows.ax_batch is not None:
        batch_vs_ref = row_cosines(rows.reference_single, rows.ax_batch)
        batch_vs_single = row_cosines(rows.ax_single, rows.ax_batch)
        batch_norm_error = max_norm_error(rows.ax_batch)
        print()
        print(f"{'Row':>3} {'AX batch vs ref':>18} {'AX batch vs single':>20} {'Pass':>6}")
        print("-" * 55)
        for idx, (ref_cosine, single_cosine) in enumerate(
            zip(batch_vs_ref, batch_vs_single)
        ):
            ok = (
                float(ref_cosine) >= contract.cosine_threshold
                and float(single_cosine) >= contract.batch_consistency_threshold
            )
            passed = passed and ok
            print(
                f"{idx:>3} {float(ref_cosine):>18.9f} "
                f"{float(single_cosine):>20.9f} {str(ok):>6}"
            )
        print(f"AX batch max norm error: {batch_norm_error:.3e}")
        passed = passed and batch_norm_error < 1e-4

    print()
    if passed:
        print("PASS: embedding output matches the reference contract")
    else:
        print("FAIL: embedding output does not match the reference contract")
    return passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model directory with model-manifest.json",
    )
    parser.add_argument(
        "--model-kind",
        choices=["auto", "qwen3", "embeddinggemma"],
        default="auto",
        help="Embedding family contract. Default: infer from model-manifest.json.",
    )
    parser.add_argument(
        "--backend",
        choices=["direct", "http"],
        default="direct",
        help="AX route to verify. Direct checks batch consistency; HTTP checks endpoint output.",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        help="Override reference cosine threshold.",
    )
    parser.add_argument(
        "--batch-consistency-threshold",
        type=float,
        help="Override AX batch-vs-single cosine threshold for --backend direct.",
    )
    parser.add_argument(
        "--hf-model",
        help="Deprecated; ignored. Reference now uses --model-dir directly.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="HTTP server port. Default 0: auto-allocate when starting a server.",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="For --backend http, assume server is already running on --port.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        model_dir = args.model_dir.resolve()
        model_kind = infer_model_kind(model_dir, args.model_kind)
        contract = build_contract(
            model_dir,
            model_kind,
            args.cosine_threshold,
            args.batch_consistency_threshold,
        )
        if args.backend == "http":
            if args.port == 0:
                if args.skip_server:
                    parser.error("--skip-server requires an explicit non-zero --port")
                args.port = allocate_port()
            rows = verify_http(model_dir, contract, args.port, args.skip_server)
        else:
            rows = verify_direct(model_dir, contract)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    return 0 if print_results(contract, rows) else 3


if __name__ == "__main__":
    sys.exit(main())
