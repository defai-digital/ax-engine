from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any


def _setup_bundled_metal() -> None:
    """Point AX_ENGINE_METAL_BUILD_DIR at the bundled metallib for pip installs.

    Generates a build_report.json with absolute paths in ~/.cache/ax-engine/metal-build/
    so the Rust engine can locate the pre-compiled metallib without a source checkout.
    No-ops if AX_ENGINE_METAL_BUILD_DIR is already set or the bundled assets are absent.
    """
    import json

    if "AX_ENGINE_METAL_BUILD_DIR" in os.environ:
        return

    _pkg = Path(__file__).parent
    _bundled = _pkg / "_metal"
    _manifest_path = _bundled / "metal" / "phase1-kernels.json"
    _source_path = _bundled / "metal" / "kernels" / "phase1_dense_path.metal"
    _metallib_path = _bundled / "build" / "ax_phase1_dense_path.metallib"
    _air_path = _bundled / "build" / "ax_phase1_dense_path.air"

    if not all(p.is_file() for p in [_manifest_path, _source_path, _metallib_path, _air_path]):
        return

    _source_sha256 = sha256(_source_path.read_bytes()).hexdigest()
    _metallib_sha256 = sha256(_metallib_path.read_bytes()).hexdigest()
    _air_sha256 = sha256(_air_path.read_bytes()).hexdigest()

    _cache_dir = Path.home() / ".cache" / "ax-engine" / "metal-build"
    _report_path = _cache_dir / "build_report.json"

    # Reuse cached report when paths and this release's SHA256 still match.
    if _report_path.is_file():
        try:
            _cached = json.loads(_report_path.read_bytes())
            if (
                _cached.get("manifest_path") == str(_manifest_path)
                and _cached.get("outputs", {}).get("metallib") == str(_metallib_path)
                and _cached.get("outputs", {}).get("metallib_sha256") == _metallib_sha256
            ):
                os.environ["AX_ENGINE_METAL_BUILD_DIR"] = str(_cache_dir)
                return
        except Exception:
            pass

    _manifest = json.loads(_manifest_path.read_bytes())

    _report: dict[str, Any] = {
        "schema_version": "ax.metal.build_report.v1",
        "manifest_path": str(_manifest_path),
        "source_file": str(_source_path),
        "mlx_target": _manifest["mlx_target"],
        "metal_language_standard": _manifest["metal_language_standard"],
        "library_name": _manifest["library_name"],
        "default_block_size_tokens": _manifest["default_block_size_tokens"],
        "supported_block_size_tokens": _manifest["supported_block_size_tokens"],
        "toolchain_requirements": _manifest["toolchain_requirements"],
        "doctor": {
            "status": "ready",
            "bringup_allowed": True,
            "mlx_runtime_ready": True,
            "metal_toolchain_fully_available": False,
            "host": {"os": "macos", "arch": "aarch64"},
            "metal_toolchain": {},
        },
        "kernels": _manifest["kernels"],
        "source_sha256": _source_sha256,
        "outputs": {
            "air": str(_air_path),
            "metalar": None,
            "metallib": str(_metallib_path),
            "air_sha256": _air_sha256,
            "metalar_sha256": None,
            "metallib_sha256": _metallib_sha256,
        },
        "compile_commands": [],
        "status": "compiled",
        "reason": None,
    }

    try:
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _report_path.write_text(json.dumps(_report, indent=2))
        os.environ["AX_ENGINE_METAL_BUILD_DIR"] = str(_cache_dir)
    except Exception:
        pass


_setup_bundled_metal()


def _import_native_module() -> None:
    """Import the native extension, recovering from a stale MLX rpath.

    `maturin develop` links `_ax_engine.abi3.so` against MLX resolved from the
    active venv and embeds that absolute path as an rpath. In editable installs
    the extension lives in the source tree, so a build run inside an ephemeral
    venv (e.g. scripts/check-python-preview.sh) leaves an rpath that dangles
    once that venv is deleted, and dlopen fails with
    "Library not loaded: @rpath/libmlx.dylib".

    libmlx's install name is `@rpath/libmlx.dylib`, so pre-loading the pip
    wheel's copy lets dyld satisfy the dependency by install name instead of
    the rpath search. Release wheels vendor a delocated libmlx and never take
    this path.

    libmlx.dylib itself carries no LC_RPATH of its own and links
    `@rpath/libjaccl.dylib`, so a bare `dlopen(libmlx)` on an absolute path
    resolves that dependency using whichever rpaths the *calling* process
    happens to already have (e.g. a conda-based venv's own paths) rather than
    anything relative to the mlx package — which fails outside that specific
    layout. Preload libjaccl the same way, from the same directory, before
    libmlx, so its dependency is already satisfied by install name too.
    """
    import importlib

    try:
        importlib.import_module("._ax_engine", __package__)
        return
    except ImportError as exc:
        if "libmlx" not in str(exc):
            raise

    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("mlx")
    for base in list(spec.submodule_search_locations or []) if spec else []:
        lib_dir = Path(base) / "lib"
        libjaccl = lib_dir / "libjaccl.dylib"
        if libjaccl.is_file():
            ctypes.CDLL(str(libjaccl))
        libmlx = lib_dir / "libmlx.dylib"
        if libmlx.is_file():
            ctypes.CDLL(str(libmlx))
            break
    importlib.import_module("._ax_engine", __package__)


_import_native_module()

from ._ax_engine import (
    EngineBackendError,
    EngineError,
    EngineInferenceError,
    EngineStateError,
)
from ._ax_engine import (
    Session as _Session,
)
from .gemma4_unified import (
    Gemma4UnifiedAudioRequest,
    Gemma4UnifiedImageRequest,
    Gemma4UnifiedMultimodalRequest,
    Gemma4UnifiedVideoRequest,
    prepare_gemma4_unified_audio_request,
    prepare_gemma4_unified_image_request,
    prepare_gemma4_unified_multimodal_request,
    prepare_gemma4_unified_video_request,
)
from .unlimited_ocr import (
    UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT,
    UNLIMITED_OCR_LOCAL_QUERY_GRID,
    UNLIMITED_OCR_LOCAL_TILE_SIZE,
    UNLIMITED_OCR_MAX_LOCAL_TILES,
    UnlimitedOcrImageRequest,
    prepare_unlimited_ocr_image_request,
)

_QWEN_CHATML_ASSISTANT_GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_NO_THINK = "<|im_start|>assistant\n"


@dataclass(frozen=True)
class CapabilityReport:
    text_generation: bool
    token_streaming: bool
    deterministic_mode: bool
    prefix_reuse: bool
    long_context_validation: str
    benchmark_metrics: str
    image_input: str = "unsupported"
    delegated_readiness: str = "unsupported"
    provider_extensions: str = "unsupported"


@dataclass(frozen=True)
class HostInfo:
    os: str = ""
    arch: str = ""
    detected_soc: str | None = None
    supported_mlx_runtime: bool = False
    unsupported_host_override_active: bool = False


@dataclass(frozen=True)
class ToolStatusInfo:
    available: bool = False
    version: str | None = None


@dataclass(frozen=True)
class MetalToolchainInfo:
    fully_available: bool = False
    metal: ToolStatusInfo = field(default_factory=ToolStatusInfo)
    metallib: ToolStatusInfo = field(default_factory=ToolStatusInfo)
    metal_ar: ToolStatusInfo = field(default_factory=ToolStatusInfo)


@dataclass(frozen=True)
class MlxRuntimeInfo:
    runner: str = ""
    artifacts_source: str | None = None


@dataclass(frozen=True)
class SourceQuantizationInfo:
    format: str = ""
    tensor_type_counts: dict[str, int] = field(default_factory=dict)
    quantized_tensor_count: int = 0
    contains_quantized_tensors: bool = False


@dataclass(frozen=True)
class RuntimeStatusInfo:
    ready: bool = True
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MlxModelInfo:
    artifacts_source: str | None = None
    model_family: str = ""
    tensor_format: str = ""
    source_quantization: SourceQuantizationInfo | None = None
    runtime_status: RuntimeStatusInfo = field(default_factory=RuntimeStatusInfo)
    layer_count: int = 0
    tensor_count: int = 0
    tie_word_embeddings: bool = False
    is_moe: bool = False
    is_hybrid_attention: bool = False
    hybrid_full_attention_interval: int | None = None
    mla_kv_latent_dim: int | None = None
    moe_active_experts: int | None = None
    bindings_prepared: bool = False
    buffers_bound: bool = False
    buffer_count: int = 0
    buffer_bytes: int = 0
    source_quantized_binding_count: int = 0
    source_q4_k_binding_count: int = 0
    source_q5_k_binding_count: int = 0
    source_q6_k_binding_count: int = 0
    source_q8_0_binding_count: int = 0


@dataclass(frozen=True)
class RuntimeInfo:
    selected_backend: str
    support_tier: str
    resolution_policy: str
    capabilities: CapabilityReport
    fallback_reason: str | None = None
    host: HostInfo = field(default_factory=HostInfo)
    metal_toolchain: MetalToolchainInfo = field(default_factory=MetalToolchainInfo)
    mlx_runtime: MlxRuntimeInfo | None = None
    mlx_model: MlxModelInfo | None = None


@dataclass(frozen=True)
class GenerateRoute:
    execution_plan: str | None = None
    attention_route: str | None = None
    kv_mode: str | None = None
    prefix_cache_path: str | None = None
    barrier_mode: str | None = None
    crossover_decisions: dict[str, int] | None = None


@dataclass(frozen=True)
class GenerateResult:
    request_id: int
    model_id: str
    prompt_tokens: list[int]
    prompt_text: str | None
    output_tokens: list[int]
    output_text: str | None
    status: str
    finish_reason: str | None
    step_count: int
    ttft_step: int | None
    route: GenerateRoute
    runtime: RuntimeInfo
    output_token_logprobs: list[float | None] = field(default_factory=list)


@dataclass(frozen=True)
class RequestReport:
    request_id: int
    model_id: str
    state: str
    prompt_tokens: list[int]
    processed_prompt_tokens: int
    output_tokens: list[int]
    prompt_len: int
    output_len: int
    max_output_tokens: int
    cancel_requested: bool
    execution_plan_ref: str | None
    route: GenerateRoute
    finish_reason: str | None = None
    terminal_stop_reason: str | None = None
    output_token_logprobs: list[float | None] = field(default_factory=list)


@dataclass(frozen=True)
class StepReport:
    step_id: int | None
    scheduled_requests: int
    scheduled_tokens: int
    ttft_events: int
    prefix_hits: int
    kv_usage_blocks: int
    evictions: int
    cpu_time_us: int
    runner_time_us: int
    route: GenerateRoute | None = None
    metal_dispatch: MetalDispatchInfo | None = None


@dataclass(frozen=True)
class MetalDispatchValidationInfo:
    expected_key_cache_checksum: int
    expected_attention_output_checksum: int
    expected_gather_output_checksum: int
    expected_copy_output_checksum: int
    attention_max_abs_diff_microunits: int


@dataclass(frozen=True)
class MetalDispatchNumericInfo:
    key_cache_checksum: int
    attention_output_checksum: int
    gather_output_checksum: int
    copy_output_checksum: int
    validation: MetalDispatchValidationInfo | None = None


@dataclass(frozen=True)
class MetalDispatchKernelInfo:
    function_name: str
    element_count: int
    threads_per_grid_width: int
    threads_per_threadgroup_width: int


@dataclass(frozen=True)
class MetalDispatchInfo:
    command_queue_label: str
    command_buffer_label: str
    command_buffer_status: str
    runtime_device_name: str
    runtime_required_pipeline_count: int
    runtime_max_thread_execution_width: int
    runtime_model_conditioned_inputs: bool = False
    runtime_real_model_tensor_inputs: bool = False
    runtime_complete_model_forward_supported: bool = False
    runtime_model_bindings_prepared: bool = False
    runtime_model_buffers_bound: bool = False
    runtime_model_buffer_count: int = 0
    runtime_model_buffer_bytes: int = 0
    runtime_model_family: str | None = None
    execution_direct_decode_token_count: int = 0
    execution_direct_decode_checksum_lo: int = 0
    execution_logits_output_count: int = 0
    execution_remaining_logits_handle_count: int = 0
    execution_model_bound_ffn_decode: bool = False
    execution_real_model_forward_completed: bool = False
    execution_prefix_native_dispatch_count: int = 0
    execution_prefix_cpu_reference_dispatch_count: int = 0
    execution_qkv_projection_token_count: int = 0
    execution_layer_continuation_token_count: int = 0
    execution_logits_projection_token_count: int = 0
    execution_logits_vocab_scan_row_count: int = 0
    binary_archive_state: str = ""
    binary_archive_attached_pipeline_count: int = 0
    binary_archive_serialized: bool = False
    arena_token_capacity: int = 0
    arena_slot_capacity: int = 0
    arena_attention_ref_capacity: int = 0
    arena_gather_ref_capacity: int = 0
    arena_gather_output_capacity: int = 0
    arena_copy_pair_capacity: int = 0
    arena_sequence_capacity: int = 0
    arena_reused_existing: bool = False
    arena_grew_existing: bool = False
    kernels: list[MetalDispatchKernelInfo] = field(default_factory=list)
    numeric: MetalDispatchNumericInfo | None = None


@dataclass(frozen=True)
class GenerateStreamEvent:
    event: str
    runtime: RuntimeInfo | None = None
    request: RequestReport | None = None
    step: StepReport | None = None
    delta_tokens: list[int] | None = None
    delta_token_logprobs: list[float | None] | None = None
    delta_text: str | None = None
    response: GenerateResult | None = None


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class Session:
    def __init__(
        self,
        model_id: str = "qwen3_dense",
        *,
        deterministic: bool = True,
        max_batch_tokens: int = 2048,
        cache_group_id: int = 0,
        block_size_tokens: int = 16,
        total_blocks: int = 1024,
        mlx: bool = False,
        support_tier: str = "llama_cpp",
        llama_cli_path: str = "llama-cli",
        llama_model_path: str | None = None,
        llama_server_url: str | None = None,
        mlx_lm_server_url: str | None = None,
        mlx_model_artifacts_dir: str | None = None,
        delegated_http_connect_timeout_secs: int = 30,
        delegated_http_read_timeout_secs: int = 300,
        delegated_http_write_timeout_secs: int = 300,
    ) -> None:
        if mlx and mlx_model_artifacts_dir is None:
            if not os.environ.get("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"):
                raise ValueError(
                    "mlx=True requires mlx_model_artifacts_dir or the "
                    "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR environment variable.\n\n"
                    "To download a model:\n"
                    "  from ax_engine import download_model\n"
                    "  path = download_model('mlx-community/Qwen3-4B-4bit')\n"
                    "  # then run: cargo run -p ax-engine-core --bin generate-manifest -- <path>\n"
                    "  session = Session(mlx=True, mlx_model_artifacts_dir=str(path))\n\n"
                    "Or use the download script:\n"
                    "  python scripts/download_model.py mlx-community/Qwen3-4B-4bit"
                )
        self._inner = _Session(
            model_id,
            deterministic=deterministic,
            max_batch_tokens=max_batch_tokens,
            cache_group_id=cache_group_id,
            block_size_tokens=block_size_tokens,
            total_blocks=total_blocks,
            mlx=mlx,
            support_tier=support_tier,
            llama_cli_path=llama_cli_path,
            llama_model_path=llama_model_path,
            llama_server_url=llama_server_url,
            mlx_lm_server_url=mlx_lm_server_url,
            mlx_model_artifacts_dir=mlx_model_artifacts_dir,
            delegated_http_connect_timeout_secs=delegated_http_connect_timeout_secs,
            delegated_http_read_timeout_secs=delegated_http_read_timeout_secs,
            delegated_http_write_timeout_secs=delegated_http_write_timeout_secs,
        )

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def closed(self) -> bool:
        return self._inner.closed

    def close(self) -> None:
        self._inner.close()

    def runtime(self) -> RuntimeInfo:
        return _runtime_from_dict(self._inner.runtime())

    def generate(
        self,
        input_tokens: list[int] | None = None,
        *,
        input_text: str | None = None,
        multimodal_inputs: dict[str, Any] | None = None,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: int | None = None,
        no_repeat_ngram_size: int = 0,
        ngram_window: int = 128,
        seed: int = 0,
        deterministic: bool | None = None,
        stop_sequences: list[str] | None = None,
        metadata: str | None = None,
    ) -> GenerateResult:
        return _generate_from_dict(
            self._inner.generate(
                input_tokens,
                input_text=input_text,
                multimodal_inputs=multimodal_inputs,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
                ngram_window=ngram_window,
                seed=seed,
                deterministic=deterministic,
                stop_sequences=stop_sequences,
                metadata=metadata,
            )
        )

    def submit(
        self,
        input_tokens: list[int] | None = None,
        *,
        input_text: str | None = None,
        multimodal_inputs: dict[str, Any] | None = None,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: int | None = None,
        no_repeat_ngram_size: int = 0,
        ngram_window: int = 128,
        seed: int = 0,
        deterministic: bool | None = None,
        stop_sequences: list[str] | None = None,
        metadata: str | None = None,
    ) -> int:
        return self._inner.submit(
            input_tokens,
            input_text=input_text,
            multimodal_inputs=multimodal_inputs,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ngram_window=ngram_window,
            seed=seed,
            deterministic=deterministic,
            stop_sequences=stop_sequences,
            metadata=metadata,
        )

    def step(self) -> StepReport:
        return _step_from_dict(self._inner.step())

    def snapshot(self, request_id: int) -> RequestReport | None:
        value = self._inner.snapshot(request_id)
        if value is None:
            return None
        return _request_from_dict(value)

    def cancel(self, request_id: int) -> None:
        self._inner.cancel(request_id)

    def stream_generate(
        self,
        input_tokens: list[int] | None = None,
        *,
        input_text: str | None = None,
        multimodal_inputs: dict[str, Any] | None = None,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: int | None = None,
        no_repeat_ngram_size: int = 0,
        ngram_window: int = 128,
        seed: int = 0,
        deterministic: bool | None = None,
        stop_sequences: list[str] | None = None,
        metadata: str | None = None,
    ) -> Iterator[GenerateStreamEvent]:
        for value in self._inner.stream_generate(
            input_tokens,
            input_text=input_text,
            multimodal_inputs=multimodal_inputs,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ngram_window=ngram_window,
            seed=seed,
            deterministic=deterministic,
            stop_sequences=stop_sequences,
            metadata=metadata,
        ):
            yield _stream_event_from_dict(value)

    def generate_text(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: int | None = None,
        seed: int = 0,
        deterministic: bool | None = None,
        stop_sequences: list[str] | None = None,
        metadata: str | None = None,
    ) -> GenerateResult:
        return self.generate(
            input_text=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            seed=seed,
            deterministic=deterministic,
            stop_sequences=stop_sequences,
            metadata=metadata,
        )

    def submit_text(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: int | None = None,
        seed: int = 0,
        deterministic: bool | None = None,
        stop_sequences: list[str] | None = None,
        metadata: str | None = None,
    ) -> int:
        return self.submit(
            input_text=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            seed=seed,
            deterministic=deterministic,
            stop_sequences=stop_sequences,
            metadata=metadata,
        )

    def stream_text(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: int | None = None,
        seed: int = 0,
        deterministic: bool | None = None,
        stop_sequences: list[str] | None = None,
        metadata: str | None = None,
    ) -> Iterator[GenerateStreamEvent]:
        return self.stream_generate(
            input_text=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            seed=seed,
            deterministic=deterministic,
            stop_sequences=stop_sequences,
            metadata=metadata,
        )

    def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: int | None = None,
        seed: int = 0,
        deterministic: bool | None = None,
        stop_sequences: list[str] | None = None,
        metadata: str | None = None,
    ) -> GenerateResult:
        return self.generate_text(
            _render_chat_prompt(messages, self.model_id),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            seed=seed,
            deterministic=deterministic,
            stop_sequences=stop_sequences,
            metadata=metadata,
        )

    def submit_chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: int | None = None,
        seed: int = 0,
        deterministic: bool | None = None,
        stop_sequences: list[str] | None = None,
        metadata: str | None = None,
    ) -> int:
        return self.submit_text(
            _render_chat_prompt(messages, self.model_id),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            seed=seed,
            deterministic=deterministic,
            stop_sequences=stop_sequences,
            metadata=metadata,
        )

    def stream_chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        repetition_context_size: int | None = None,
        seed: int = 0,
        deterministic: bool | None = None,
        stop_sequences: list[str] | None = None,
        metadata: str | None = None,
    ) -> Iterator[GenerateStreamEvent]:
        return self.stream_text(
            _render_chat_prompt(messages, self.model_id),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            seed=seed,
            deterministic=deterministic,
            stop_sequences=stop_sequences,
            metadata=metadata,
        )

    def embed(
        self,
        token_ids: list[int],
        *,
        pooling: str = "last",
        normalize: bool = True,
    ) -> list[float]:
        """Compute a dense embedding for the given token IDs.

        Calls directly into the MLX runner without HTTP overhead — equivalent
        call depth to mlx-lm and mlx-swift-lm for benchmarking purposes.

        Parameters
        ----------
        token_ids:
            Pre-tokenized input. For Qwen3-Embedding, append EOS before calling.
        pooling:
            Pooling strategy: ``"last"`` (default), ``"mean"``, or ``"cls"``.
        normalize:
            L2-normalize the output vector (default ``True``).

        Returns
        -------
        list[float]
            Embedding vector.
        """
        return self._inner.embed(token_ids, pooling=pooling, normalize=normalize)

    def embed_bytes(
        self,
        token_ids: list[int],
        *,
        pooling: str = "last",
        normalize: bool = True,
    ) -> bytes:
        """Like :meth:`embed` but returns the embedding as raw f32 bytes.

        The byte buffer is little-endian ``hidden_size * 4`` bytes long.
        Avoids the per-element ``PyFloat`` allocation that ``list[float]``
        incurs, which is significant for larger models.  Typical use:

        .. code-block:: python

            import numpy as np
            buf = session.embed_bytes(ids)
            vec = np.frombuffer(buf, dtype=np.float32)   # zero-copy view

        Or without numpy:

        .. code-block:: python

            import array
            vec = array.array("f", session.embed_bytes(ids))
        """
        return self._inner.embed_bytes(token_ids, pooling=pooling, normalize=normalize)

    def embed_batch(
        self,
        batch_token_ids: list[list[int]],
        *,
        pooling: str = "last",
        normalize: bool = True,
    ) -> list[list[float]]:
        """Compute dense embeddings for a batch of token ID sequences.

        Runs a single batched forward pass for all sequences, which is more
        efficient than calling :meth:`embed` once per sequence.  Sequences are
        right-padded to the longest length before the forward pass.

        Parameters
        ----------
        batch_token_ids:
            List of pre-tokenized sequences.  For Qwen3-Embedding, append EOS
            to each sequence before calling.
        pooling:
            Pooling strategy: ``"last"`` (default), ``"mean"``, or ``"cls"``.
        normalize:
            L2-normalize each output vector (default ``True``).

        Returns
        -------
        list[list[float]]
            One embedding vector per input sequence, in the same order.
        """
        return self._inner.embed_batch(batch_token_ids, pooling=pooling, normalize=normalize)

    def embed_batch_bytes(
        self,
        batch_token_ids: list[list[int]],
        *,
        pooling: str = "last",
        normalize: bool = True,
    ) -> list[bytes]:
        """Like :meth:`embed_batch` but returns one raw f32-bytes blob per
        sequence.  See :meth:`embed_bytes` for the rationale and usage.
        """
        return self._inner.embed_batch_bytes(batch_token_ids, pooling=pooling, normalize=normalize)

    def embed_batch_flat_bytes(
        self,
        batch_token_ids: list[list[int]],
        *,
        pooling: str = "last",
        normalize: bool = True,
    ) -> tuple[bytes, int, int]:
        """Batch embedding as one contiguous f32-bytes blob plus
        ``(batch_size, hidden_size)``. Saves the ``B-1`` per-sequence
        ``PyBytes`` allocations that :meth:`embed_batch_bytes` makes and
        gives the caller a single row-major ``[B, H]`` buffer suited to
        numpy / faiss / HNSW.
        """
        return self._inner.embed_batch_flat_bytes(
            batch_token_ids, pooling=pooling, normalize=normalize
        )

    def embed_batch_array(
        self,
        batch_token_ids: list[list[int]],
        *,
        pooling: str = "last",
        normalize: bool = True,
    ):
        """Batch embedding as a NumPy ``(B, H)`` ``float32`` ndarray.

        Wraps :meth:`embed_batch_flat_bytes` with ``np.frombuffer`` for a
        zero-copy view over the contiguous buffer Rust returned. Lazily
        imports numpy so callers who do not need this method do not pay
        the import cost. Returns a *read-only* view of bytes the Rust
        side owns; copy with ``.copy()`` before mutating.

        Example
        -------
        >>> arr = session.embed_batch_array([ids1, ids2, ids3])
        >>> arr.shape, arr.dtype
        ((3, 1024), dtype('float32'))
        >>> # zero-copy hand off to faiss:
        >>> # index.add(arr)
        """
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "numpy is required for embed_batch_array(). Install it with:\n"
                "  pip install numpy\n"
                "or use embed_batch_flat_bytes() for a numpy-free interface."
            ) from error

        blob, batch_size, hidden_size = self.embed_batch_flat_bytes(
            batch_token_ids, pooling=pooling, normalize=normalize
        )
        arr = np.frombuffer(blob, dtype=np.float32).reshape(batch_size, hidden_size)
        return arr

    def __enter__(self) -> Session:
        return self

    def __exit__(
        self, exc_type: object | None, exc: object | None, traceback: object | None
    ) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise


def _runtime_from_dict(value: dict[str, Any]) -> RuntimeInfo:
    return RuntimeInfo(
        selected_backend=value["selected_backend"],
        support_tier=value["support_tier"],
        resolution_policy=value["resolution_policy"],
        capabilities=CapabilityReport(**value["capabilities"]),
        fallback_reason=value.get("fallback_reason"),
        host=_host_from_dict(value.get("host", {})),
        metal_toolchain=_metal_toolchain_from_dict(value.get("metal_toolchain", {})),
        mlx_runtime=(
            _mlx_runtime_from_dict(value["mlx_runtime"])
            if value.get("mlx_runtime") is not None
            else None
        ),
        mlx_model=(
            _mlx_model_from_dict(value["mlx_model"]) if value.get("mlx_model") is not None else None
        ),
    )


def _host_from_dict(value: dict[str, Any]) -> HostInfo:
    return HostInfo(
        os=str(value.get("os", "")),
        arch=str(value.get("arch", "")),
        detected_soc=value.get("detected_soc"),
        supported_mlx_runtime=bool(value.get("supported_mlx_runtime", False)),
        unsupported_host_override_active=bool(value.get("unsupported_host_override_active", False)),
    )


def _tool_status_from_dict(value: dict[str, Any]) -> ToolStatusInfo:
    return ToolStatusInfo(
        available=bool(value.get("available", False)),
        version=value.get("version"),
    )


def _metal_toolchain_from_dict(value: dict[str, Any]) -> MetalToolchainInfo:
    return MetalToolchainInfo(
        fully_available=bool(value.get("fully_available", False)),
        metal=_tool_status_from_dict(value.get("metal", {})),
        metallib=_tool_status_from_dict(value.get("metallib", {})),
        metal_ar=_tool_status_from_dict(value.get("metal_ar", {})),
    )


def _mlx_runtime_from_dict(value: dict[str, Any]) -> MlxRuntimeInfo:
    return MlxRuntimeInfo(
        runner=str(value.get("runner", "")),
        artifacts_source=value.get("artifacts_source"),
    )


def _mlx_model_from_dict(value: dict[str, Any]) -> MlxModelInfo:
    return MlxModelInfo(
        artifacts_source=value.get("artifacts_source"),
        model_family=str(value.get("model_family", "")),
        tensor_format=str(value.get("tensor_format", "")),
        source_quantization=_source_quantization_from_dict(value["source_quantization"])
        if value.get("source_quantization") is not None
        else None,
        runtime_status=_runtime_status_from_dict(value.get("runtime_status", {})),
        layer_count=int(value.get("layer_count", 0)),
        tensor_count=int(value.get("tensor_count", 0)),
        tie_word_embeddings=bool(value.get("tie_word_embeddings", False)),
        is_moe=bool(value.get("is_moe", False)),
        is_hybrid_attention=bool(value.get("is_hybrid_attention", False)),
        hybrid_full_attention_interval=(
            int(value["hybrid_full_attention_interval"])
            if value.get("hybrid_full_attention_interval") is not None
            else None
        ),
        mla_kv_latent_dim=(
            int(value["mla_kv_latent_dim"]) if value.get("mla_kv_latent_dim") is not None else None
        ),
        moe_active_experts=(
            int(value["moe_active_experts"])
            if value.get("moe_active_experts") is not None
            else None
        ),
        bindings_prepared=bool(value.get("bindings_prepared", False)),
        buffers_bound=bool(value.get("buffers_bound", False)),
        buffer_count=int(value.get("buffer_count", 0)),
        buffer_bytes=int(value.get("buffer_bytes", 0)),
        source_quantized_binding_count=int(value.get("source_quantized_binding_count", 0)),
        source_q4_k_binding_count=int(value.get("source_q4_k_binding_count", 0)),
        source_q5_k_binding_count=int(value.get("source_q5_k_binding_count", 0)),
        source_q6_k_binding_count=int(value.get("source_q6_k_binding_count", 0)),
        source_q8_0_binding_count=int(value.get("source_q8_0_binding_count", 0)),
    )


def _source_quantization_from_dict(value: dict[str, Any]) -> SourceQuantizationInfo:
    return SourceQuantizationInfo(
        format=str(value.get("format", "")),
        tensor_type_counts={
            str(key): int(count) for key, count in dict(value.get("tensor_type_counts", {})).items()
        },
        quantized_tensor_count=int(value.get("quantized_tensor_count", 0)),
        contains_quantized_tensors=bool(value.get("contains_quantized_tensors", False)),
    )


def _runtime_status_from_dict(value: dict[str, Any]) -> RuntimeStatusInfo:
    return RuntimeStatusInfo(
        ready=bool(value.get("ready", True)),
        blockers=[str(item) for item in value.get("blockers", [])],
        notes=[str(item) for item in value.get("notes", [])],
    )


def _route_from_dict(value: dict[str, Any]) -> GenerateRoute:
    return GenerateRoute(
        execution_plan=value.get("execution_plan"),
        attention_route=value.get("attention_route"),
        kv_mode=value.get("kv_mode"),
        prefix_cache_path=value.get("prefix_cache_path"),
        barrier_mode=value.get("barrier_mode"),
        crossover_decisions=value.get("crossover_decisions"),
    )


def _generate_from_dict(value: dict[str, Any]) -> GenerateResult:
    return GenerateResult(
        request_id=value["request_id"],
        model_id=value["model_id"],
        prompt_tokens=value["prompt_tokens"],
        prompt_text=value.get("prompt_text"),
        output_tokens=value["output_tokens"],
        output_token_logprobs=list(value.get("output_token_logprobs", [])),
        output_text=value.get("output_text"),
        status=value["status"],
        finish_reason=value.get("finish_reason"),
        step_count=value["step_count"],
        ttft_step=value.get("ttft_step"),
        route=_route_from_dict(value["route"]),
        runtime=_runtime_from_dict(value["runtime"]),
    )


def _request_from_dict(value: dict[str, Any]) -> RequestReport:
    return RequestReport(
        request_id=value["request_id"],
        model_id=value["model_id"],
        state=value["state"],
        prompt_tokens=value["prompt_tokens"],
        processed_prompt_tokens=value["processed_prompt_tokens"],
        output_tokens=value["output_tokens"],
        output_token_logprobs=list(value.get("output_token_logprobs", [])),
        prompt_len=value["prompt_len"],
        output_len=value["output_len"],
        max_output_tokens=value["max_output_tokens"],
        cancel_requested=value["cancel_requested"],
        execution_plan_ref=value.get("execution_plan_ref"),
        route=_route_from_dict(value["route"]),
        finish_reason=value.get("finish_reason"),
        terminal_stop_reason=value.get("terminal_stop_reason"),
    )


def _step_from_dict(value: dict[str, Any]) -> StepReport:
    return StepReport(
        step_id=value.get("step_id"),
        scheduled_requests=value["scheduled_requests"],
        scheduled_tokens=value["scheduled_tokens"],
        ttft_events=value["ttft_events"],
        prefix_hits=value["prefix_hits"],
        kv_usage_blocks=value["kv_usage_blocks"],
        evictions=value["evictions"],
        cpu_time_us=value["cpu_time_us"],
        runner_time_us=value["runner_time_us"],
        route=_route_from_dict(value["route"]) if value.get("route") is not None else None,
        metal_dispatch=(
            _metal_dispatch_from_dict(value["metal_dispatch"])
            if value.get("metal_dispatch") is not None
            else None
        ),
    )


def _metal_dispatch_from_dict(value: dict[str, Any]) -> MetalDispatchInfo:
    return MetalDispatchInfo(
        command_queue_label=str(value["command_queue_label"]),
        command_buffer_label=str(value["command_buffer_label"]),
        command_buffer_status=str(value["command_buffer_status"]),
        runtime_device_name=str(value["runtime_device_name"]),
        runtime_required_pipeline_count=int(value["runtime_required_pipeline_count"]),
        runtime_max_thread_execution_width=int(value["runtime_max_thread_execution_width"]),
        runtime_model_conditioned_inputs=bool(value.get("runtime_model_conditioned_inputs", False)),
        runtime_real_model_tensor_inputs=bool(value.get("runtime_real_model_tensor_inputs", False)),
        runtime_complete_model_forward_supported=bool(
            value.get("runtime_complete_model_forward_supported", False)
        ),
        runtime_model_bindings_prepared=bool(value.get("runtime_model_bindings_prepared", False)),
        runtime_model_buffers_bound=bool(value.get("runtime_model_buffers_bound", False)),
        runtime_model_buffer_count=int(value.get("runtime_model_buffer_count", 0)),
        runtime_model_buffer_bytes=int(value.get("runtime_model_buffer_bytes", 0)),
        runtime_model_family=(
            str(value["runtime_model_family"])
            if value.get("runtime_model_family") is not None
            else None
        ),
        execution_direct_decode_token_count=int(
            value.get("execution_direct_decode_token_count", 0)
        ),
        execution_direct_decode_checksum_lo=int(
            value.get("execution_direct_decode_checksum_lo", 0)
        ),
        execution_logits_output_count=int(value.get("execution_logits_output_count", 0)),
        execution_remaining_logits_handle_count=int(
            value.get("execution_remaining_logits_handle_count", 0)
        ),
        execution_model_bound_ffn_decode=bool(value.get("execution_model_bound_ffn_decode", False)),
        execution_real_model_forward_completed=bool(
            value.get("execution_real_model_forward_completed", False)
        ),
        execution_prefix_native_dispatch_count=int(
            value.get("execution_prefix_native_dispatch_count", 0)
        ),
        execution_prefix_cpu_reference_dispatch_count=int(
            value.get("execution_prefix_cpu_reference_dispatch_count", 0)
        ),
        execution_qkv_projection_token_count=int(
            value.get("execution_qkv_projection_token_count", 0)
        ),
        execution_layer_continuation_token_count=int(
            value.get("execution_layer_continuation_token_count", 0)
        ),
        execution_logits_projection_token_count=int(
            value.get("execution_logits_projection_token_count", 0)
        ),
        execution_logits_vocab_scan_row_count=int(
            value.get("execution_logits_vocab_scan_row_count", 0)
        ),
        binary_archive_state=str(value["binary_archive_state"]),
        binary_archive_attached_pipeline_count=int(value["binary_archive_attached_pipeline_count"]),
        binary_archive_serialized=bool(value["binary_archive_serialized"]),
        arena_token_capacity=int(value["arena_token_capacity"]),
        arena_slot_capacity=int(value["arena_slot_capacity"]),
        arena_attention_ref_capacity=int(value["arena_attention_ref_capacity"]),
        arena_gather_ref_capacity=int(value["arena_gather_ref_capacity"]),
        arena_gather_output_capacity=int(value["arena_gather_output_capacity"]),
        arena_copy_pair_capacity=int(value["arena_copy_pair_capacity"]),
        arena_sequence_capacity=int(value["arena_sequence_capacity"]),
        arena_reused_existing=bool(value["arena_reused_existing"]),
        arena_grew_existing=bool(value["arena_grew_existing"]),
        kernels=[
            MetalDispatchKernelInfo(
                function_name=str(kernel["function_name"]),
                element_count=int(kernel["element_count"]),
                threads_per_grid_width=int(kernel["threads_per_grid_width"]),
                threads_per_threadgroup_width=int(kernel["threads_per_threadgroup_width"]),
            )
            for kernel in value.get("kernels", [])
        ],
        numeric=(
            MetalDispatchNumericInfo(
                key_cache_checksum=int(value["numeric"]["key_cache_checksum"]),
                attention_output_checksum=int(value["numeric"]["attention_output_checksum"]),
                gather_output_checksum=int(value["numeric"]["gather_output_checksum"]),
                copy_output_checksum=int(value["numeric"]["copy_output_checksum"]),
                validation=(
                    MetalDispatchValidationInfo(
                        expected_key_cache_checksum=int(
                            value["numeric"]["validation"]["expected_key_cache_checksum"]
                        ),
                        expected_attention_output_checksum=int(
                            value["numeric"]["validation"]["expected_attention_output_checksum"]
                        ),
                        expected_gather_output_checksum=int(
                            value["numeric"]["validation"]["expected_gather_output_checksum"]
                        ),
                        expected_copy_output_checksum=int(
                            value["numeric"]["validation"]["expected_copy_output_checksum"]
                        ),
                        attention_max_abs_diff_microunits=int(
                            value["numeric"]["validation"]["attention_max_abs_diff_microunits"]
                        ),
                    )
                    if value["numeric"].get("validation") is not None
                    else None
                ),
            )
            if value.get("numeric") is not None
            else None
        ),
    )


def _stream_event_from_dict(value: dict[str, Any]) -> GenerateStreamEvent:
    return GenerateStreamEvent(
        event=value["event"],
        runtime=(
            _runtime_from_dict(value["runtime"]) if value.get("runtime") is not None else None
        ),
        request=(
            _request_from_dict(value["request"]) if value.get("request") is not None else None
        ),
        step=_step_from_dict(value["step"]) if value.get("step") is not None else None,
        delta_tokens=value.get("delta_tokens"),
        delta_token_logprobs=value.get("delta_token_logprobs"),
        delta_text=value.get("delta_text"),
        response=(
            _generate_from_dict(value["response"]) if value.get("response") is not None else None
        ),
    )


def _render_chat_prompt(messages: list[ChatMessage | dict[str, str]], model_id: str) -> str:
    if not messages:
        raise ValueError("chat requires at least one message")

    template = _chat_prompt_template(model_id)
    prompt_parts: list[str] = []
    if template == "llama3":
        prompt_parts.append("<|begin_of_text|>")

    for message in messages:
        normalized = _normalize_chat_message(message)
        role = _normalize_chat_role(normalized.role)
        content = normalized.content
        if template == "qwen_chatml":
            prompt_parts.append(
                f"<|im_start|>{role}\n{_escape_qwen_chatml_content(content)}<|im_end|>\n"
            )
        elif template == "llama3":
            prompt_parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                f"{_escape_llama3_content(content)}<|eot_id|>"
            )
        else:
            safe_content = content.replace("\\", "\\\\").replace("\n", "\\n")
            prompt_parts.append(f"{role}: {safe_content}\n")

    if template == "qwen_chatml":
        prompt_parts.append(_qwen_assistant_generation_prompt(model_id))
    elif template == "llama3":
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    else:
        prompt_parts.append("assistant:")
    return "".join(prompt_parts)


def _escape_qwen_chatml_content(content: str) -> str:
    """Escape literal ChatML turn-boundary tokens inside message content.

    Mirrors ``escape_qwen_chatml_content`` in the server's chat.rs: a message
    whose content happens to contain ``<|im_start|>``/``<|im_end|>`` (pasted
    ChatML docs, or a deliberate attempt) must not be read by the model as a
    real role switch.
    """
    return content.replace("<|im_start|>", "&lt;|im_start|>").replace(
        "<|im_end|>", "&lt;|im_end|>"
    )


def _escape_llama3_content(content: str) -> str:
    """Escape literal Llama 3.x header/turn-boundary tokens inside content.

    Mirrors ``escape_llama3_content`` in the server's chat.rs, for the same
    reason as ``_escape_qwen_chatml_content``.
    """
    return (
        content.replace("<|start_header_id|>", "&lt;|start_header_id|>")
        .replace("<|end_header_id|>", "&lt;|end_header_id|>")
        .replace("<|eot_id|>", "&lt;|eot_id|>")
    )


def _chat_prompt_template(model_id: str) -> str:
    normalized = model_id.lower()
    if "qwen" in normalized:
        return "qwen_chatml"
    # Llama 3.x and Llama 4 Instruct share header/eot framing (server chat.rs).
    if (
        "llama-4" in normalized
        or "llama4" in normalized
        or "llama_4" in normalized
        or "llama-3" in normalized
        or "llama3" in normalized
        or "llama_3" in normalized
    ):
        return "llama3"
    return "plain_role_prefix"


def _qwen_assistant_generation_prompt(model_id: str) -> str:
    if _is_qwen_non_thinking_only_model(model_id):
        return _QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_NO_THINK
    return _QWEN_CHATML_ASSISTANT_GENERATION_PROMPT


def _is_qwen_non_thinking_only_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return normalized == "qwen3" or _is_qwen_coder_model(model_id)


def _is_qwen_coder_model(model_id: str) -> bool:
    normalized = _normalize_model_id_token(model_id)
    return "qwen3-coder-next" in normalized or "qwen3-coder" in normalized


def _normalize_model_id_token(model_id: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in model_id.lower())


def _normalize_chat_role(role: str) -> str:
    normalized = role.strip()
    if normalized not in {"system", "user", "assistant", "tool", "function"}:
        raise ValueError(
            "unsupported chat role; expected one of system, user, assistant, tool, function"
        )
    return normalized


def _normalize_chat_message(message: ChatMessage | dict[str, str]) -> ChatMessage:
    if isinstance(message, ChatMessage):
        return message
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("chat message dicts must contain string role and content")
        return ChatMessage(role=role, content=content)
    raise TypeError("chat messages must be ChatMessage instances or dicts with role/content")


_MODEL_MANIFEST_FILE = "model-manifest.json"
_DOWNLOAD_PROVENANCE_FILE = ".ax-engine-download.json"
_DOWNLOAD_PROVENANCE_SCHEMA_VERSION = "ax.download_provenance.v1"
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


def _safetensors_files(model_dir: Path) -> list[Path]:
    """Return main-model safetensors recursively, excluding the assistant drafter."""
    try:
        return sorted(
            path
            for path in model_dir.rglob("*.safetensors")
            if path.is_file()
            and path.relative_to(model_dir).parts[:1] != ("assistant",)
        )
    except OSError:
        return []


def _weight_tensor_names(model_dir: Path) -> set[str]:
    """Read tensor names without loading model payloads."""
    import json

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_bytes())
            weight_map = payload.get("weight_map", {})
            if isinstance(weight_map, dict):
                return {name for name in weight_map if isinstance(name, str)}
        except (OSError, ValueError, TypeError):
            return set()

    names: set[str] = set()
    for path in _safetensors_files(model_dir):
        try:
            with path.open("rb") as handle:
                header_size_bytes = handle.read(8)
                if len(header_size_bytes) != 8:
                    continue
                header_size = int.from_bytes(header_size_bytes, "little")
                if not 0 < header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
                    continue
                header = json.loads(handle.read(header_size))
                if isinstance(header, dict):
                    names.update(
                        name for name in header if isinstance(name, str) and name != "__metadata__"
                    )
        except (OSError, ValueError, TypeError):
            continue
    return names


def _manifest_needs_media_rebuild(model_dir: Path) -> bool:
    """Detect published manifests that silently omitted declared media towers."""
    import json

    try:
        config = json.loads((model_dir / "config.json").read_bytes())
        manifest = json.loads((model_dir / _MODEL_MANIFEST_FILE).read_bytes())
    except (OSError, ValueError, TypeError):
        return False
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not isinstance(config.get("vision_config"), dict):
        return False

    required_prefix_groups: tuple[tuple[str, ...], ...]
    if model_type in {
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3-vl",
        "qwen3-vl-moe",
    }:
        required_prefix_groups = (("vision_tower.", "visual.", "model.visual."),)
    elif model_type in {"gemma4", "gemma4_vl", "gemma4-vl"}:
        required_prefix_groups = (
            ("vision_tower.", "model.vision_tower."),
            ("embed_vision.", "model.embed_vision."),
        )
    else:
        return False

    source_names = _weight_tensor_names(model_dir)
    if not source_names:
        return False
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        manifest_names: set[str] = set()
    else:
        manifest_names = {
            name
            for tensor in tensors
            if isinstance(tensor, dict) and isinstance((name := tensor.get("name")), str)
        }
    return any(
        any(name.startswith(prefixes) for name in source_names)
        and not any(name.startswith(prefixes) for name in manifest_names)
        for prefixes in required_prefix_groups
    )


def _slug_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _default_mlx_lm_cache_root() -> Path:
    import os

    if hf_hub_cache := os.environ.get("HF_HUB_CACHE"):
        return Path(hf_hub_cache).expanduser()
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home).expanduser() / "hub"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache_home / "huggingface" / "hub"


def _latest_mlx_lm_snapshot(repo_id: str) -> Path | None:
    repo_cache = _default_mlx_lm_cache_root() / f"models--{_slug_repo_id(repo_id)}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        if revision:
            snapshot = repo_cache / "snapshots" / revision
            if snapshot.is_dir():
                return snapshot
    snapshots = repo_cache / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _run_hf_snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    import os

    previous = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    # huggingface_hub reads this setting during import, so it must be installed
    # before importing the module rather than immediately before the call.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is required for download_model(). Install it with:\n"
                "  pip install huggingface_hub\n"
                "or:\n"
                "  pip install 'ax-engine[download]'"
            ) from error

        # Ask the Hub client to refresh only the selected revision when forced.
        # Removing the whole repo cache would discard unrelated revisions and
        # shared blobs.
        try:
            return Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    force_download=force,
                )
            )
        except Exception as error:
            raise RuntimeError(
                f"Hugging Face Hub download failed for {repo_id}: {error}"
            ) from error
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous


def _snapshot_allowed_link_roots(snapshot: Path) -> tuple[Path, ...]:
    """Return roots from which a canonical Hub snapshot may materialize file links."""
    try:
        snapshot_root = snapshot.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"cannot resolve Hugging Face snapshot {snapshot}: {error}") from error
    if snapshot.is_symlink() or not snapshot_root.is_dir():
        raise RuntimeError(f"Hugging Face snapshot is not a real directory: {snapshot}")

    allowed_link_roots = [snapshot_root]
    if snapshot_root.parent.name == "snapshots":
        blobs = snapshot_root.parent.parent / "blobs"
        if blobs.is_dir() and not blobs.is_symlink():
            try:
                allowed_link_roots.append(blobs.resolve(strict=True))
            except OSError as error:
                raise RuntimeError(
                    f"cannot resolve Hugging Face blob directory {blobs}: {error}"
                ) from error
    return tuple(allowed_link_roots)


def _validated_snapshot_entry(
    source: Path, allowed_link_roots: tuple[Path, ...]
) -> tuple[str, Path]:
    import stat

    if source.is_symlink():
        try:
            resolved = source.resolve(strict=True)
        except OSError as error:
            raise RuntimeError(f"cannot resolve snapshot symlink {source}: {error}") from error
        contained = any(resolved == root or root in resolved.parents for root in allowed_link_roots)
        if not contained or not resolved.is_file():
            raise RuntimeError(
                f"unsafe snapshot symlink {source}: target {resolved} must be a regular "
                "file inside the snapshot or its Hugging Face blob directory"
            )
        return "file", resolved

    try:
        mode = source.stat(follow_symlinks=False).st_mode
    except OSError as error:
        raise RuntimeError(f"cannot inspect snapshot entry {source}: {error}") from error
    if stat.S_ISDIR(mode):
        return "directory", source
    if stat.S_ISREG(mode):
        return "file", source
    raise RuntimeError(
        f"unsafe snapshot entry {source}: only regular files and directories are allowed"
    )


def _validate_mlx_lm_snapshot(snapshot: Path) -> None:
    allowed_link_roots = _snapshot_allowed_link_roots(snapshot)

    def validate_entry(source: Path) -> None:
        kind, _resolved = _validated_snapshot_entry(source, allowed_link_roots)
        if kind == "directory":
            for child in source.iterdir():
                validate_entry(child)

    for path in snapshot.iterdir():
        validate_entry(path)


def _copy_mlx_lm_snapshot(snapshot: Path, dest: Path) -> None:
    import shutil

    allowed_link_roots = _snapshot_allowed_link_roots(snapshot)

    def copy_entry(source: Path, target: Path) -> None:
        kind, resolved = _validated_snapshot_entry(source, allowed_link_roots)
        if kind == "directory":
            target.mkdir()
            for child in source.iterdir():
                copy_entry(child, target / child.name)
            shutil.copystat(source, target, follow_symlinks=False)
        else:
            shutil.copy2(resolved, target, follow_symlinks=False)

    dest.mkdir(parents=True, exist_ok=True)
    for path in snapshot.iterdir():
        copy_entry(path, dest / path.name)


def _path_lexists(path: Path) -> bool:
    import os

    return os.path.lexists(path)


def _validate_model_destination(dest: Path) -> Path:
    """Reject destinations whose atomic replacement could erase broad user data."""
    dest = Path(dest).expanduser()
    try:
        cwd = Path.cwd().resolve(strict=False)
        home = Path.home().resolve(strict=False)
        resolved = (
            dest.resolve(strict=False) if dest.is_absolute() else (cwd / dest).resolve(strict=False)
        )
    except OSError as error:
        raise RuntimeError(f"cannot resolve model destination {dest}: {error}") from error

    if not dest.name or resolved.parent == resolved:
        raise RuntimeError(f"unsafe model destination {dest}: choose a dedicated model directory")
    if resolved == cwd or resolved in cwd.parents:
        raise RuntimeError(
            f"unsafe model destination {dest}: the current working directory "
            "or one of its ancestors cannot be replaced"
        )
    if resolved == home or resolved in home.parents:
        raise RuntimeError(
            f"unsafe model destination {dest}: the home directory or one of its "
            "ancestors cannot be replaced"
        )
    return dest


def _model_destination_is_empty(dest: Path, *, error_context: str) -> bool:
    try:
        next(dest.iterdir())
    except StopIteration:
        return True
    except OSError as error:
        raise RuntimeError(f"cannot inspect {error_context} {dest}: {error}") from error
    return False


def _validate_forced_model_destination(dest: Path) -> None:
    """Only replace non-empty real directories that look owned by a model download."""
    if not _path_lexists(dest) or dest.is_symlink() or not dest.is_dir():
        return
    if _model_destination_is_empty(dest, error_context="existing model destination"):
        return

    has_download_provenance = _has_valid_download_provenance(dest)
    has_manifest = (dest / _MODEL_MANIFEST_FILE).is_file()
    has_weights = bool(_safetensors_files(dest))
    if has_download_provenance or has_manifest or has_weights:
        return
    raise RuntimeError(
        f"refusing to replace non-model destination {dest}: the directory is non-empty "
        f"but contains no {_DOWNLOAD_PROVENANCE_FILE}, {_MODEL_MANIFEST_FILE}, or "
        ".safetensors files"
    )


def _read_download_provenance(dest: Path) -> object | None:
    import json

    try:
        return json.loads((dest / _DOWNLOAD_PROVENANCE_FILE).read_bytes())
    except (OSError, ValueError, TypeError):
        return None


def _has_valid_download_provenance(dest: Path) -> bool:
    from ._repo_ref import parse_repo_ref, validate_revision

    payload = _read_download_provenance(dest)
    if not isinstance(payload, dict):
        return False
    repo_id = payload.get("repo_id")
    revision = payload.get("revision")
    if not isinstance(repo_id, str) or not (revision is None or isinstance(revision, str)):
        return False
    try:
        parsed_repo_id, embedded_revision = parse_repo_ref(repo_id)
        if revision is not None:
            # The stored revision was already percent-decoded when it was
            # written; re-escape `%` so validation checks the stored value
            # itself instead of decoding it a second time (which rejects any
            # revision containing a literal percent sign).
            revision = validate_revision(revision.replace("%", "%25"), reference=repo_id)
    except ValueError:
        return False
    return (
        embedded_revision is None
        and parsed_repo_id == repo_id
        and _destination_matches(dest, repo_id, revision)
    )


def _validate_model_snapshot_destination(snapshot: Path, dest: Path) -> None:
    try:
        resolved_snapshot = snapshot.resolve(strict=True)
        resolved_dest = dest.resolve(strict=False)
    except OSError as error:
        raise RuntimeError(
            f"cannot validate model snapshot {snapshot} against destination {dest}: {error}"
        ) from error
    if (
        resolved_snapshot == resolved_dest
        or resolved_snapshot in resolved_dest.parents
        or resolved_dest in resolved_snapshot.parents
    ):
        raise RuntimeError(
            f"model destination {dest} must not overlap the Hugging Face snapshot {snapshot}"
        )


def _download_provenance(repo_id: str, revision: str | None) -> dict[str, object]:
    return {
        "schema_version": _DOWNLOAD_PROVENANCE_SCHEMA_VERSION,
        "repo_id": repo_id,
        "revision": revision,
    }


def _write_download_provenance(dest: Path, repo_id: str, revision: str | None) -> None:
    import json

    (dest / _DOWNLOAD_PROVENANCE_FILE).write_text(
        json.dumps(_download_provenance(repo_id, revision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _destination_matches(dest: Path, repo_id: str, revision: str | None) -> bool:
    return _read_download_provenance(dest) == _download_provenance(repo_id, revision)


def _validate_model_destination_before_activation(
    dest: Path,
    *,
    repo_id: str,
    revision: str | None,
    force: bool,
) -> None:
    """Revalidate the destination after staging to close download-time races."""
    if not _path_lexists(dest):
        return
    if force:
        _validate_forced_model_destination(dest)
        return
    if dest.is_symlink() or not dest.is_dir():
        raise RuntimeError(
            f"refusing to replace model destination {dest}: it appeared or changed "
            "while the model was being prepared"
        )
    if _model_destination_is_empty(dest, error_context="model destination"):
        return
    if not _destination_matches(dest, repo_id, revision):
        raise RuntimeError(
            f"refusing to replace model destination {dest}: it no longer matches "
            f"{repo_id} at revision {revision!r}"
        )
    try:
        _validate_downloaded_model_dir(dest)
    except RuntimeError:
        return
    manifest_path = dest / _MODEL_MANIFEST_FILE
    if _manifest_is_structurally_valid(
        manifest_path
    ) and not _manifest_needs_media_rebuild(dest):
        raise RuntimeError(
            f"refusing to replace model destination {dest}: another process made it "
            "ready while this model was being prepared"
        )


def _replace_with_staged_snapshot(
    snapshot: Path,
    dest: Path,
    *,
    repo_id: str,
    revision: str | None,
    force: bool,
) -> None:
    """Build and validate an explicit destination before replacing it.

    Staging and backup directories are unique siblings owned by this call. The
    previous destination remains in place through copy/manifest generation and
    is restored if the final rename fails.
    """
    import shutil
    import tempfile

    dest = _validate_model_destination(dest)
    _validate_model_snapshot_destination(snapshot, dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}.download-", dir=str(dest.parent)))
    # mkdtemp creates mode 0700; restore the umask-derived mode a plain mkdir
    # would have used so the activated destination stays readable by other
    # users and services.
    current_umask = os.umask(0)
    os.umask(current_umask)
    os.chmod(stage, 0o777 & ~current_umask)
    backup_root: Path | None = None
    backup: Path | None = None
    try:
        _copy_mlx_lm_snapshot(snapshot, stage)
        _validate_downloaded_model_dir(stage)
        _ensure_manifest(stage)
        _write_download_provenance(stage, repo_id, revision)

        try:
            _validate_model_destination_before_activation(
                dest,
                repo_id=repo_id,
                revision=revision,
                force=force,
            )
            if _path_lexists(dest):
                backup_root = Path(
                    tempfile.mkdtemp(prefix=f".{dest.name}.backup-", dir=str(dest.parent))
                )
                backup = backup_root / "previous"
                dest.rename(backup)
            stage.rename(dest)
        except BaseException as swap_error:
            if backup is not None and _path_lexists(backup):
                if not _path_lexists(dest):
                    try:
                        backup.rename(dest)
                    except BaseException as restore_error:
                        stranded_backup = backup
                        # Preserve the user's previous destination instead of
                        # deleting it in the cleanup block below.
                        backup_root = None
                        raise RuntimeError(
                            f"failed to install model at {dest} and could not restore "
                            f"the previous destination; it remains at {stranded_backup}: "
                            f"{restore_error}"
                        ) from swap_error
                if _path_lexists(backup):
                    # An unexpected destination appeared before rollback.
                    # Keep the previous contents in their unique backup, and
                    # tell the user where they went.
                    backup_root = None
                    raise RuntimeError(
                        f"failed to install model at {dest}; the previous "
                        f"destination was preserved at {backup}: {swap_error}"
                    ) from swap_error
            raise
        if backup_root is not None:
            try:
                shutil.rmtree(backup_root)
            except OSError as cleanup_error:
                stranded_backup = backup if backup is not None else backup_root
                # The new destination is ready. Preserve and report the old copy
                # rather than hiding a potentially model-sized cleanup failure.
                backup_root = None
                raise RuntimeError(
                    f"model installed at {dest}, but the previous destination could not "
                    f"be removed and remains at {stranded_backup}: {cleanup_error}"
                ) from cleanup_error
            else:
                backup_root = None
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if backup_root is not None and (backup is None or not _path_lexists(backup)):
            shutil.rmtree(backup_root, ignore_errors=True)


def _validate_downloaded_model_dir(dest: Path) -> None:
    errors = []
    if not _safetensors_files(dest):
        errors.append(f"no .safetensors files found in {dest}")
    if not (dest / "config.json").exists():
        errors.append(f"config.json missing in {dest}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _manifest_missing_required_roles(manifest: dict) -> str | None:
    """Return a reason when the manifest lacks roles the native loader requires.

    Mirrors the essential checks in ``NativeModelArtifacts::from_dir`` so the
    Python download path cannot report a model as AX-ready when it only has a
    structurally intact but role-incomplete manifest.
    """
    model_family = manifest.get("model_family")
    if not isinstance(model_family, str) or not model_family:
        return "missing model_family"

    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        return "missing tensors"

    if model_family == "whisper":
        if any(
            not isinstance(tensor, dict) or tensor.get("role") != "other"
            for tensor in tensors
        ):
            return "whisper tensors must use role=other"
        return None

    layer_count = manifest.get("layer_count")
    if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
        return "invalid layer_count"

    global_roles: set[str] = set()
    layer_roles: dict[int, set[str]] = {}
    for tensor in tensors:
        if not isinstance(tensor, dict):
            return "tensor entry is not an object"
        role = tensor.get("role")
        if not isinstance(role, str) or not role:
            return "tensor missing role"
        layer_index = tensor.get("layer_index")
        if layer_index is None:
            global_roles.add(role)
            continue
        if not isinstance(layer_index, int) or isinstance(layer_index, bool) or layer_index < 0:
            return f"invalid layer_index for role {role}"
        if layer_index >= layer_count:
            return f"layer_index {layer_index} exceeds layer_count {layer_count}"
        layer_roles.setdefault(layer_index, set()).add(role)

    if "token_embedding" not in global_roles:
        return "missing required tensor role token_embedding"
    if "final_norm" not in global_roles:
        return "missing required tensor role final_norm"

    if model_family == "embeddinggemma":
        if "embedding_dense0" not in global_roles:
            return "missing required tensor role embedding_dense0"
        if "embedding_dense1" not in global_roles:
            return "missing required tensor role embedding_dense1"
    elif model_family == "nemotron_embed":
        # Encoder-only mean-pool: no lm_head / Dense head required.
        pass
    elif not manifest.get("tie_word_embeddings", False) and "lm_head" not in global_roles:
        return "missing required tensor role lm_head"

    if model_family == "gemma4_assistant":
        if "assistant_pre_projection" not in global_roles:
            return "missing required tensor role assistant_pre_projection"
        if "assistant_post_projection" not in global_roles:
            return "missing required tensor role assistant_post_projection"

    is_nemotron_h = model_family == "nemotron_h"
    for layer_index in range(layer_count):
        roles = layer_roles.get(layer_index)
        if not roles:
            return f"missing tensors for layer {layer_index}"
        if "attention_norm" not in roles:
            return f"layer {layer_index} is missing required tensor role attention_norm"
        if is_nemotron_h:
            continue
        if "ffn_norm" not in roles and "attention_post_norm" not in roles:
            return (
                f"layer {layer_index} is missing required tensor role "
                "ffn_norm or attention_post_norm"
            )

        has_packed_gate_up = "ffn_gate_up_packed" in roles
        has_split_gate_up = "ffn_gate" in roles and "ffn_up" in roles
        has_dense_ffn = "ffn_down" in roles and (has_packed_gate_up or has_split_gate_up)
        has_shared_expert_ffn = (
            "ffn_shared_expert_gate_inp" in roles
            and "ffn_shared_expert_gate" in roles
            and "ffn_shared_expert_up" in roles
            and "ffn_shared_expert_down" in roles
        )
        has_mla_shared_expert_ffn = model_family in {
            "glm4_moe_lite",
            "deepseek_v3",
            "deepseek_v32",
            "unlimited_ocr",
        } and (
            "ffn_shared_expert_gate" in roles
            and "ffn_shared_expert_up" in roles
            and "ffn_shared_expert_down" in roles
        )
        has_gpt_oss_mxfp4_moe = model_family == "gpt_oss" and (
            "ffn_gate_up_exps_mxfp4_blocks" in roles
            and "ffn_gate_up_exps_mxfp4_scales" in roles
            and "ffn_down_exps_mxfp4_blocks" in roles
            and "ffn_down_exps_mxfp4_scales" in roles
        )
        has_moe_expert_ffn = "ffn_gate_inp" in roles and (
            has_gpt_oss_mxfp4_moe
            or (
                "ffn_down_exps" in roles
                and (
                    "ffn_gate_up_exps_packed" in roles
                    or "ffn_gate_exps" in roles
                    or "ffn_up_exps" in roles
                )
            )
        )
        if not (
            has_dense_ffn
            or has_shared_expert_ffn
            or has_mla_shared_expert_ffn
            or has_moe_expert_ffn
        ):
            return f"layer {layer_index} must provide dense FFN tensors or MoE expert tensors"

        has_any_attention = any(
            role in roles
            for role in (
                "attention_o",
                "attention_q",
                "attention_k",
                "attention_v",
                "attention_qkv_packed",
                "attention_qa",
                "attention_qb",
                "attention_kv_a",
                "attention_kv_b",
                "attention_embed_q",
                "attention_unembed_out",
            )
        )
        has_any_linear_attention = any(
            role in roles
            for role in (
                "linear_attention_in_proj_qkv",
                "linear_attention_in_proj_qkvz",
                "linear_attention_in_proj_z",
                "linear_attention_in_proj_a",
                "linear_attention_in_proj_b",
                "linear_attention_in_proj_ba",
                "linear_attention_conv1d",
                "linear_attention_dt_bias",
                "linear_attention_a_log",
                "linear_attention_norm",
                "linear_attention_out_proj",
            )
        )
        if has_any_attention:
            if "attention_o" not in roles:
                return f"layer {layer_index} is missing required tensor role attention_o"
            has_packed_qkv = "attention_qkv_packed" in roles
            has_split_qkv = (
                "attention_q" in roles
                and "attention_k" in roles
                and "attention_v" in roles
            )
            has_mla = any(
                role in roles
                for role in (
                    "attention_qa",
                    "attention_qb",
                    "attention_kv_a",
                    "attention_kv_b",
                    "attention_embed_q",
                    "attention_unembed_out",
                )
            )
            if not (has_packed_qkv or has_split_qkv or has_mla):
                return (
                    f"layer {layer_index} must provide attention_qkv_packed or "
                    "attention_q/attention_k/attention_v"
                )
        elif not has_any_linear_attention and not has_moe_expert_ffn:
            return (
                f"layer {layer_index} must provide attention, linear attention, "
                "or MoE expert tensors"
            )

    return None


def _manifest_is_structurally_valid(path: Path) -> bool:
    import json

    try:
        payload = json.loads(path.read_bytes())
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != "ax.native_model.v1":
        return False
    model_family = payload.get("model_family")
    if not isinstance(model_family, str) or not model_family.strip():
        return False
    if payload.get("tensor_format") != "safetensors":
        return False
    for field_name in (
        "layer_count",
        "hidden_size",
        "attention_head_count",
        "attention_head_dim",
        "kv_head_count",
        "vocab_size",
    ):
        value = payload.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return False

    runtime_status = payload.get("runtime_status")
    if runtime_status is not None:
        if not isinstance(runtime_status, dict):
            return False
        ready = runtime_status.get("ready", True)
        blockers = runtime_status.get("blockers", [])
        if ready is not True or not isinstance(blockers, list) or blockers:
            return False

    tensors = payload.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        return False
    for tensor in tensors:
        if not isinstance(tensor, dict):
            return False
        for field_name in ("name", "role", "dtype", "file"):
            value = tensor.get(field_name)
            if not isinstance(value, str) or not value:
                return False
        shape = tensor.get("shape")
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
                for dimension in shape
            )
            or (not shape and tensor.get("role") != "other")
        ):
            return False
        for field_name in ("offset_bytes", "length_bytes"):
            value = tensor.get(field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (1 if field_name == "length_bytes" else 0)
            ):
                return False
    # Structural fields alone are not enough for AX-ready: require the same
    # essential tensor roles the native loader enforces.
    return _manifest_missing_required_roles(payload) is None


def download_model(
    repo_id: str,
    dest: str | Path | None = None,
    *,
    force: bool = False,
    revision: str | None = None,
) -> Path:
    """Download an MLX model through Hugging Face Hub and generate its ax-engine manifest.

    Accepts a bare ``owner/repo`` id, ``owner/repo@revision``, or a full
    ``https://huggingface.co/owner/repo`` link (``/tree/<revision>`` included).
    Downloads the model, then generates ``model-manifest.json`` via the bundled
    ``ax-engine-bench`` (wheel), an ``ax-engine-bench`` on ``PATH``, or ``cargo run``
    (dev). The returned path is always AX-ready: if the manifest is missing and cannot
    be generated, this raises rather than returning a path to a non-ready model.

    Args:
        repo_id: MLX LLM repo id or Hugging Face URL, e.g.
            ``"mlx-community/Qwen3-4B-4bit"``.
        dest: Destination directory. Defaults to the Hugging Face Hub cache snapshot.
        force: Re-download the default Hugging Face Hub cache entry before
            resolving the snapshot.
        revision: Branch, tag, or commit sha to download (overrides a revision
            parsed from ``repo_id``).

    Returns:
        Path to the downloaded, AX-ready model directory (contains ``model-manifest.json``).

    Raises:
        ValueError: if ``repo_id`` or ``revision`` is not a valid Hugging Face
            model reference.
        RuntimeError: if the download is incomplete or the manifest cannot be generated.
    """
    # Single source of truth: delegate to the bundled download helper (the same
    # entry point the Rust/Python CLIs and the TUI use) so URL parsing,
    # revision pinning, disk preflight, atomic --dest copies, and manifest
    # semantics cannot diverge. The legacy in-process path below remains as a
    # fallback for stripped installs missing scripts/download_model.py.
    from ._repo_ref import parse_repo_ref, validate_revision

    input_repo_ref = repo_id
    repo_id, parsed_revision = parse_repo_ref(repo_id)
    if revision is None:
        revision = parsed_revision
    else:
        revision = validate_revision(revision, reference=input_repo_ref)
    if dest is not None:
        dest = _validate_model_destination(Path(dest))
        if force:
            _validate_forced_model_destination(dest)

    from ._cli import _download_helper_command, _find_repo_script, _parse_download_summary

    helper = _find_repo_script("download_model.py")
    if helper is not None:
        import subprocess

        command = _download_helper_command(
            helper, repo_id, revision=revision, dest=dest, force=force
        )
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except OSError as error:
            raise RuntimeError(
                f"failed to launch model download helper {helper}: {error}"
            ) from error
        summary = _parse_download_summary(result.stdout)
        if result.returncode == 0 and summary is not None and summary.get("status") == "ready":
            summary_dest = summary.get("dest")
            if isinstance(summary_dest, str) and summary_dest:
                return Path(summary_dest)
            raise RuntimeError("download helper returned ready status without a destination")
        raw_errors = (summary or {}).get("errors")
        if isinstance(raw_errors, list):
            errors = raw_errors
        elif isinstance(raw_errors, str) and raw_errors:
            errors = [raw_errors]
        else:
            errors = [result.stderr.strip() or f"download helper exited {result.returncode}"]
        raise RuntimeError("; ".join(str(error) for error in errors))

    # ---- legacy in-process fallback (no bundled helper) ----
    if dest is None:
        dest = _run_hf_snapshot_download(repo_id, revision=revision, force=force)
        _validate_mlx_lm_snapshot(dest)
        _validate_downloaded_model_dir(dest)
        _ensure_manifest(dest)
        return dest

    dest = Path(dest)
    if _path_lexists(dest) and not force:
        if not dest.is_dir():
            raise RuntimeError(f"model destination exists and is not a directory: {dest}")
        try:
            _validate_downloaded_model_dir(dest)
        except RuntimeError as error:
            if any(dest.iterdir()):
                raise RuntimeError(
                    f"{error}; refusing to merge into incomplete destination {dest}. "
                    "Pass force=True to replace it."
                ) from error
        else:
            if not _destination_matches(dest, repo_id, revision):
                raise RuntimeError(
                    f"existing model destination {dest} does not match the requested "
                    f"repository and revision ({repo_id}, revision={revision!r}); "
                    "refusing to reuse it. Pass force=True to replace it."
                )
            _ensure_manifest(dest)
            return dest

    snapshot = _run_hf_snapshot_download(repo_id, revision=revision, force=force)
    # Fail before any staging work begins; _replace_with_staged_snapshot
    # re-checks the same precondition for its own callers.
    _validate_model_snapshot_destination(snapshot, dest)

    _replace_with_staged_snapshot(
        snapshot,
        dest,
        repo_id=repo_id,
        revision=revision,
        force=force,
    )
    return dest


def _bundled_binary(name: str) -> Path | None:
    """Return the path to a binary bundled inside the installed wheel, if present.

    Release wheels stage ``ax-engine-server`` and ``ax-engine-bench`` under
    ``ax_engine/_bin/`` so they always match the installed package version. Editable
    and source-checkout installs have no ``_bin`` directory, so this returns ``None``
    and callers fall back to a PATH lookup or ``cargo run``.
    """
    candidate = Path(__file__).resolve().parent / "_bin" / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def _source_workspace_root() -> Path | None:
    """Return this package's source workspace root, if running from a checkout."""
    for parent in Path(__file__).resolve().parents:
        cargo_toml = parent / "Cargo.toml"
        if not cargo_toml.is_file():
            continue
        try:
            if "[workspace]" in cargo_toml.read_text():
                return parent
        except OSError:
            pass
    return None


def _ensure_manifest(dest: Path) -> None:
    """Ensure ``dest`` contains a model-manifest.json, generating it if necessary.

    Raises:
        RuntimeError: when the manifest is absent and cannot be generated, so callers
            of :func:`download_model` never receive a path to a model that is not
            actually AX-ready.
    """
    manifest_path = dest / _MODEL_MANIFEST_FILE
    manifest_valid = _manifest_is_structurally_valid(
        manifest_path
    ) and not _manifest_needs_media_rebuild(dest)
    if manifest_valid and _try_validate_manifest(dest):
        return
    if _try_generate_manifest(dest, force=manifest_path.exists()):
        if _manifest_is_structurally_valid(manifest_path) and not _manifest_needs_media_rebuild(
            dest
        ):
            return
        raise RuntimeError(
            f"manifest generator reported success but wrote an invalid manifest: {manifest_path}"
        )
    if manifest_path.exists():
        raise RuntimeError(
            f"invalid {_MODEL_MANIFEST_FILE} in {dest}; regeneration failed, "
            "so the model is not AX-ready"
        )
    raise RuntimeError(_manifest_failure_message(dest))


def _try_validate_manifest(dest: Path) -> bool:
    """Validate an existing manifest through the native runtime loader."""
    import shutil
    import subprocess

    if not (dest / _MODEL_MANIFEST_FILE).is_file():
        return False

    manifest_dest = os.path.abspath(dest)
    bundled = _bundled_binary("ax-engine-bench")
    if bundled is not None:
        bench = str(bundled)
        command = [bench, "generate-manifest", "--validate", manifest_dest]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            print(f"failed to launch {bench} manifest validation: {error}")
        else:
            if result.returncode == 0:
                return True
            print(f"{bench} manifest validation failed:\n{result.stderr.strip()}")
            return False

    repo_root = _source_workspace_root()
    if repo_root is not None and shutil.which("cargo"):
        try:
            result = subprocess.run(
                [
                    "cargo",
                    "run",
                    "-q",
                    "-p",
                    "ax-engine-core",
                    "--bin",
                    "generate-manifest",
                    "--",
                    "--validate",
                    manifest_dest,
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
        except OSError as error:
            print(f"failed to launch cargo manifest validation: {error}")
            return False
        return result.returncode == 0

    if shutil.which("ax-engine-bench"):
        bench = "ax-engine-bench"
        command = [bench, "generate-manifest", "--validate", manifest_dest]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            print(f"failed to launch {bench} manifest validation: {error}")
            return False
        if result.returncode == 0:
            return True
        print(f"{bench} manifest validation failed:\n{result.stderr.strip()}")
        return False

    return False


def _try_generate_manifest(dest: Path, *, force: bool = False) -> bool:
    """Try to run generate-manifest via the bundled, installed, or cargo binary.

    Returns True on success.
    """
    import shutil
    import subprocess

    manifest_path = dest / _MODEL_MANIFEST_FILE
    if manifest_path.is_symlink():
        # A cached Hub manifest may point into the shared blob store. Detach
        # the snapshot entry before invoking older external generators that
        # might otherwise open the symlink target for writing.
        manifest_path.unlink()
    # Make option-looking relative paths unambiguous without resolving cache
    # symlinks or changing the user-visible path spelling.
    manifest_dest = os.path.abspath(dest)

    # Prefer the ax-engine-bench bundled in this exact release. A bare PATH lookup can
    # resolve to a stale ax-engine-bench from an unrelated install (e.g. an old
    # cargo-installed binary) that rejects newer model types, so the bundled binary wins.
    bundled = _bundled_binary("ax-engine-bench")
    if bundled is not None:
        bench = str(bundled)
        command = [bench, "generate-manifest"]
        if force:
            command.append("--force")
        # Re-read through NativeModelArtifacts::from_dir so incomplete manifests
        # never count as generation success.
        command.append("--validate")
        command.append(manifest_dest)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            print(f"failed to launch {bench} generate-manifest: {error}")
        else:
            if result.returncode == 0:
                print(f"manifest generated: {dest / _MODEL_MANIFEST_FILE}")
                return True
            # The binary ran and rejected the model. Surface that result rather
            # than trying a different generator with potentially different
            # model support.
            print(f"{bench} generate-manifest failed:\n{result.stderr.strip()}")
            return False

    # In a source checkout, prefer the workspace's current Rust validator over
    # a potentially stale ax-engine-bench on PATH.
    repo_root = _source_workspace_root()
    if repo_root is not None and shutil.which("cargo"):
        generate_args = (
            ["--force", "--validate", manifest_dest]
            if force
            else ["--validate", manifest_dest]
        )
        try:
            result = subprocess.run(
                [
                    "cargo",
                    "run",
                    "-q",
                    "-p",
                    "ax-engine-core",
                    "--bin",
                    "generate-manifest",
                    "--",
                    *generate_args,
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
        except OSError as error:
            print(f"failed to launch cargo generate-manifest: {error}")
        else:
            if result.returncode == 0:
                print(f"manifest generated: {dest / _MODEL_MANIFEST_FILE}")
                return True
            print(f"cargo generate-manifest failed:\n{result.stderr.strip()}")
            return False

    if shutil.which("ax-engine-bench"):
        bench = "ax-engine-bench"
        command = [bench, "generate-manifest"]
        if force:
            command.append("--force")
        command.extend(["--validate", manifest_dest])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            print(f"failed to launch {bench} generate-manifest: {error}")
            return False
        if result.returncode == 0:
            print(f"manifest generated: {dest / _MODEL_MANIFEST_FILE}")
            return True
        print(f"{bench} generate-manifest failed:\n{result.stderr.strip()}")

    return False


def _manifest_failure_message(dest: Path) -> str:
    return (
        f"manifest generation failed for {dest}.\n"
        "The model weights downloaded but model-manifest.json could not be created, "
        "so the model is not AX-ready.\n"
        "Generate the ax-engine manifest manually:\n"
        f"  ax-engine-bench generate-manifest {dest}\n"
        "or (from source):\n"
        f"  cargo run -p ax-engine-core --bin generate-manifest -- {dest}\n"
        "\nThen create a Session:\n"
        f"  Session(mlx=True, mlx_model_artifacts_dir='{dest}')"
    )


__all__ = [
    "CapabilityReport",
    "ChatMessage",
    "EngineBackendError",
    "EngineError",
    "EngineInferenceError",
    "EngineStateError",
    "Gemma4UnifiedAudioRequest",
    "Gemma4UnifiedImageRequest",
    "Gemma4UnifiedMultimodalRequest",
    "Gemma4UnifiedVideoRequest",
    "GenerateResult",
    "GenerateRoute",
    "GenerateStreamEvent",
    "MetalDispatchInfo",
    "MetalDispatchKernelInfo",
    "MetalDispatchNumericInfo",
    "MetalDispatchValidationInfo",
    "MlxModelInfo",
    "MlxRuntimeInfo",
    "RequestReport",
    "RuntimeInfo",
    "RuntimeStatusInfo",
    "Session",
    "SourceQuantizationInfo",
    "StepReport",
    "UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT",
    "UNLIMITED_OCR_LOCAL_QUERY_GRID",
    "UNLIMITED_OCR_LOCAL_TILE_SIZE",
    "UNLIMITED_OCR_MAX_LOCAL_TILES",
    "UnlimitedOcrImageRequest",
    "download_model",
    "prepare_gemma4_unified_audio_request",
    "prepare_gemma4_unified_image_request",
    "prepare_gemma4_unified_multimodal_request",
    "prepare_gemma4_unified_video_request",
    "prepare_unlimited_ocr_image_request",
]
