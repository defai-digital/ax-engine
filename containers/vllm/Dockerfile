# syntax=docker/dockerfile:1.8@sha256:e87caa74dcb7d46cd820352bfea12591f3dba3ddc4285e19c7dcd13359f7cefd

# These are multi-architecture OCI index digests. BuildKit selects the native
# amd64 or arm64 child for TARGETARCH.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.31@sha256:ecd4de2f060c64bea0ff8ecb182ddf46ba3fcccdc8a60cfdbaf20d1a047d7437
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@sha256:14d94b039cb94bbd5da559f303b46bc4b0d5d6c24ab1a9d7b186e566ed3400dc

FROM ${UV_IMAGE} AS uv
FROM ${CUDA_IMAGE}

ARG TARGETARCH
ARG UV_IMAGE
ARG CUDA_IMAGE
ARG UBUNTU_SNAPSHOT=20260722T000000Z
ARG UBUNTU_BUILD_ESSENTIAL_VERSION=12.10ubuntu1
ARG UBUNTU_CA_CERTIFICATES_VERSION=20260601~24.04.1
ARG UBUNTU_PYTHON3_VERSION=3.12.3-0ubuntu2.1
ARG UBUNTU_PYTHON3_DEV_VERSION=3.12.3-0ubuntu2.1
ARG RUNTIME_WHEEL=dist/ax_engine_vllm_runtime-0.1.0-py3-none-any.whl
ARG RUNTIME_WHEEL_SHA256
ARG RELEASE_SOURCE_SHA256

LABEL org.opencontainers.image.title="AX Engine vLLM runtime" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/defai-digital/ax-engine" \
      com.automatosx.cuda.architecture="${TARGETARCH}" \
      com.automatosx.cuda.base.uv="${UV_IMAGE}" \
      com.automatosx.cuda.base.cuda="${CUDA_IMAGE}" \
      com.automatosx.ubuntu.snapshot="${UBUNTU_SNAPSHOT}" \
      com.automatosx.release.source.sha256="${RELEASE_SOURCE_SHA256}" \
      com.automatosx.runtime-wheel.sha256="${RUNTIME_WHEEL_SHA256}" \
      com.automatosx.vllm.version="0.25.1"

# Triton compiles runtime helpers on first use, so the native runtime retains a
# compiler and Python headers. Apt is constrained to one immutable Ubuntu
# snapshot and every explicitly installed package is version-checked.
RUN printf '%s\n' \
        'Types: deb' \
        "URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}" \
        'Suites: noble noble-updates noble-backports noble-security' \
        'Components: main universe restricted multiverse' \
        'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
        > /tmp/ubuntu-snapshot.sources \
    && DEBIAN_FRONTEND=noninteractive apt-get \
        -o Dir::Etc::sourcelist=/tmp/ubuntu-snapshot.sources \
        -o Dir::Etc::sourceparts=- \
        update \
    && DEBIAN_FRONTEND=noninteractive apt-get \
        -o Dir::Etc::sourcelist=/tmp/ubuntu-snapshot.sources \
        -o Dir::Etc::sourceparts=- \
        install --yes --no-install-recommends \
        build-essential="${UBUNTU_BUILD_ESSENTIAL_VERSION}" \
        ca-certificates="${UBUNTU_CA_CERTIFICATES_VERSION}" \
        python3="${UBUNTU_PYTHON3_VERSION}" \
        python3-dev="${UBUNTU_PYTHON3_DEV_VERSION}" \
    && test "$(dpkg-query --show --showformat='${Version}' build-essential)" = "${UBUNTU_BUILD_ESSENTIAL_VERSION}" \
    && test "$(dpkg-query --show --showformat='${Version}' ca-certificates)" = "${UBUNTU_CA_CERTIFICATES_VERSION}" \
    && test "$(dpkg-query --show --showformat='${Version}' python3)" = "${UBUNTU_PYTHON3_VERSION}" \
    && test "$(dpkg-query --show --showformat='${Version}' python3-dev)" = "${UBUNTU_PYTHON3_DEV_VERSION}" \
    && rm -f /tmp/ubuntu-snapshot.sources \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/
COPY locks/requirements-runtime-amd64.lock /tmp/requirements-runtime-amd64.lock
COPY locks/requirements-runtime-arm64.lock /tmp/requirements-runtime-arm64.lock
COPY ${RUNTIME_WHEEL} /tmp/ax_engine_vllm_runtime-0.1.0-py3-none-any.whl

ENV VIRTUAL_ENV=/opt/ax-engine-vllm-venv \
    PATH=/opt/ax-engine-vllm-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN (test "${TARGETARCH}" = "amd64" && test "$(uname -m)" = "x86_64") \
        || (test "${TARGETARCH}" = "arm64" && test "$(uname -m)" = "aarch64") \
    && test "$(printf '%s' "${RUNTIME_WHEEL_SHA256}" | wc -c)" -eq 64 \
    && ! printf '%s' "${RUNTIME_WHEEL_SHA256}" | grep -q '[^0-9a-f]' \
    && test "$(sha256sum /tmp/ax_engine_vllm_runtime-0.1.0-py3-none-any.whl | cut -d ' ' -f 1)" = "${RUNTIME_WHEEL_SHA256}" \
    && test "$(printf '%s' "${RELEASE_SOURCE_SHA256}" | wc -c)" -eq 64 \
    && ! printf '%s' "${RELEASE_SOURCE_SHA256}" | grep -q '[^0-9a-f]' \
    && UV_LINK_MODE=copy UV_NO_CACHE=1 UV_TORCH_BACKEND=cu130 uv venv \
        --python /usr/bin/python3 "${VIRTUAL_ENV}" \
    && UV_LINK_MODE=copy UV_NO_CACHE=1 UV_TORCH_BACKEND=cu130 uv pip sync \
        --python "${VIRTUAL_ENV}/bin/python" --require-hashes \
        "/tmp/requirements-runtime-${TARGETARCH}.lock" \
    && UV_LINK_MODE=copy UV_NO_CACHE=1 uv pip install \
        --python "${VIRTUAL_ENV}/bin/python" --no-deps \
        /tmp/ax_engine_vllm_runtime-0.1.0-py3-none-any.whl \
    && python -c "import importlib.metadata, platform, torch, vllm; assert platform.machine() in {'x86_64', 'aarch64'}; assert torch.__version__.split('+', 1)[0] == '2.11.0'; assert torch.version.cuda == '13.0'; assert vllm.__version__.split('+', 1)[0] == '0.25.1'; assert importlib.metadata.version('ax-engine-vllm-runtime') == '0.1.0'" \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin axvllm \
    && mkdir -p /home/axvllm/.cache \
    && chown -R axvllm:axvllm /home/axvllm \
    && rm -f /tmp/ax_engine_vllm_runtime-0.1.0-py3-none-any.whl \
        /tmp/requirements-runtime-*.lock

# vLLM's bundled CUDA 12.9 ptxas cannot compile Thor sm_110a kernels. The
# pinned cu130 closure provides the validated CUDA 13.2 ptxas. The environment
# is harmless on x86_64 and profile preflight still gates exact hardware.
ENV HOME=/home/axvllm \
    HF_HOME=/home/axvllm/.cache/huggingface \
    HF_HUB_CACHE=/home/axvllm/.cache/huggingface/hub \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    PYTHONNOUSERSITE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    TRITON_PTXAS_BLACKWELL_PATH=/opt/ax-engine-vllm-venv/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas \
    LD_LIBRARY_PATH=/opt/ax-engine-vllm-venv/lib/python3.12/site-packages/nvidia/cu13/lib \
    XDG_CACHE_HOME=/home/axvllm/.cache/xdg \
    TRITON_CACHE_DIR=/home/axvllm/.cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/home/axvllm/.cache/torchinductor \
    CUDA_CACHE_PATH=/home/axvllm/.cache/cuda \
    TMPDIR=/home/axvllm/.cache \
    TMP=/home/axvllm/.cache \
    TEMP=/home/axvllm/.cache

RUN test -x "${TRITON_PTXAS_BLACKWELL_PATH}" \
    && "${TRITON_PTXAS_BLACKWELL_PATH}" --version \
    && test ! -L /etc/ld.so.preload \
    && { test ! -e /etc/ld.so.preload || test ! -s /etc/ld.so.preload; }

USER 10001:10001
WORKDIR /home/axvllm
EXPOSE 8000
ENTRYPOINT ["ax-engine-vllm-runtime"]
