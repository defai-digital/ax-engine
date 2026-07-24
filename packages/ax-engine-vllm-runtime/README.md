# AX Engine vLLM runtime

This distribution owns the Python/OCI side of AX Engine's delegated vLLM
backend. It is intentionally independent from the `ax-engine` Python wheel and
from AX OCR.

The runtime is selected explicitly by profile. A profile validates OS,
architecture, GPU identity/compute capability, CPython, PyTorch, PyTorch CUDA,
the exact vLLM release, and the packaged dependency-lock digest before it
launches a worker. It never changes the AX↔vLLM wire contract and never falls
back to another provider.

```bash
ax-engine-vllm-runtime --list-profiles
ax-engine-vllm-runtime \
  --profile cuda-linux-x86_64-a6000-sm86 \
  --model baidu/Unlimited-OCR \
  --served-model-name baidu/Unlimited-OCR \
  --check-only
```

Release environments are installed from the architecture-specific,
SHA-256-complete lock before installing this wheel without dependencies:

```bash
UV_TORCH_BACKEND=cu130 uv pip sync \
  --python /opt/ax-engine-vllm-venv/bin/python \
  --require-hashes locks/requirements-runtime-amd64.lock
uv pip install \
  --python /opt/ax-engine-vllm-venv/bin/python \
  --no-deps dist/ax_engine_vllm_runtime-0.1.0-py3-none-any.whl
```

Use `requirements-runtime-arm64.lock` on Thor. Both locks are also embedded in
the wheel so preflight can attest the exact release closure after installation.
The profile status remains `candidate` until its native correctness,
performance, security, and soak gates have passed.

The two BF16 Unlimited-OCR profiles pin source revision
`ee63731b6461c8afcdcc7b15352e7d2ffecc2ead`. This prevents Hugging Face offline
resolution from selecting a newer, partial cache snapshot. An explicit
`--revision` is still available for a separately reviewed artifact.

Bearer credentials are accepted only through an environment variable or a
secret file. The default worker bind is loopback. A public bind requires both
`--allow-public-bind` and a configured API key.
