# vLLM OCI build

Build with `packages/ax-engine-vllm-runtime` as the context. The Dockerfile uses
digest-pinned multi-architecture CUDA and uv bases, an immutable Ubuntu
snapshot, architecture-specific hash locks, a content-addressed runtime wheel,
and a non-root runtime user.

```bash
cd packages/ax-engine-vllm-runtime
python3.12 -m build --wheel

wheel=dist/ax_engine_vllm_runtime-0.1.0-py3-none-any.whl
wheel_sha256="$(sha256sum "$wheel" | cut -d ' ' -f 1)"
source_sha256="$(
  find README.md pyproject.toml src locks -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | cut -d ' ' -f 1
)"

docker buildx build \
  --platform linux/amd64 \
  --file ../../containers/vllm/Dockerfile \
  --build-arg TARGETARCH=amd64 \
  --build-arg "RUNTIME_WHEEL=$wheel" \
  --build-arg "RUNTIME_WHEEL_SHA256=$wheel_sha256" \
  --build-arg "RELEASE_SOURCE_SHA256=$source_sha256" \
  --tag ax-engine-vllm-runtime:0.1.0-amd64 \
  --load .
```

Use `linux/arm64` and `TARGETARCH=arm64` on Thor. Production builds must add
BuildKit `mode=max` provenance and a pinned SBOM generator, publish by digest,
and preserve the image inspect, dependency-lock hashes, wheel hash, source
identity, and native acceptance receipt.

Mount model/cache directories explicitly. Run with a read-only root filesystem,
the minimum NVIDIA device allocation, loopback or private networking, and a
secret-file mount. Do not embed credentials in the image or command line.

For a read-only preflight, provide one writable cache tmpfs owned by the image
user:

```bash
docker run --rm --gpus all --read-only \
  --tmpfs /home/axvllm/.cache:rw,nosuid,nodev,uid=10001,gid=10001,mode=0700 \
  ax-engine-vllm-runtime:0.1.0-<arch> \
  --profile <profile-id> --check-only --json
```
