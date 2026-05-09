from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from ._ax_engine import (
    EngineBackendError,
    EngineError,
    EngineInferenceError,
    EngineStateError,
    Session as _Session,
)


@dataclass(frozen=True)
class CapabilityReport:
    text_generation: bool
    token_streaming: bool
    deterministic_mode: bool
    prefix_reuse: bool
    long_context_validation: str
    benchmark_metrics: str


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
class MlxModelInfo:
    artifacts_source: str | None = None
    model_family: str = ""
    tensor_format: str = ""
    layer_count: int = 0
    tensor_count: int = 0
    tie_word_embeddings: bool = False
    bindings_prepared: bool = False
    buffers_bound: bool = False
    buffer_count: int = 0
    buffer_bytes: int = 0


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
    metal_dispatch: "MetalDispatchInfo | None" = None


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
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        seed: int = 0,
        deterministic: bool | None = None,
        metadata: str | None = None,
    ) -> GenerateResult:
        return _generate_from_dict(
            self._inner.generate(
                input_tokens,
                input_text=input_text,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed,
                deterministic=deterministic,
                metadata=metadata,
            )
        )

    def submit(
        self,
        input_tokens: list[int] | None = None,
        *,
        input_text: str | None = None,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        seed: int = 0,
        deterministic: bool | None = None,
        metadata: str | None = None,
    ) -> int:
        return self._inner.submit(
            input_tokens,
            input_text=input_text,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
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
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        seed: int = 0,
        deterministic: bool | None = None,
        metadata: str | None = None,
    ) -> Iterator[GenerateStreamEvent]:
        for value in self._inner.stream_generate(
            input_tokens,
            input_text=input_text,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
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
        repetition_penalty: float = 1.0,
        seed: int = 0,
        deterministic: bool | None = None,
        metadata: str | None = None,
    ) -> GenerateResult:
        return self.generate(
            input_text=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
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
        repetition_penalty: float = 1.0,
        seed: int = 0,
        deterministic: bool | None = None,
        metadata: str | None = None,
    ) -> int:
        return self.submit(
            input_text=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
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
        repetition_penalty: float = 1.0,
        seed: int = 0,
        deterministic: bool | None = None,
        metadata: str | None = None,
    ) -> Iterator[GenerateStreamEvent]:
        return self.stream_generate(
            input_text=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
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
        repetition_penalty: float = 1.0,
        seed: int = 0,
        deterministic: bool | None = None,
        metadata: str | None = None,
    ) -> GenerateResult:
        return self.generate_text(
            _render_chat_prompt(messages, self.model_id),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
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
        repetition_penalty: float = 1.0,
        seed: int = 0,
        deterministic: bool | None = None,
        metadata: str | None = None,
    ) -> int:
        return self.submit_text(
            _render_chat_prompt(messages, self.model_id),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
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
        repetition_penalty: float = 1.0,
        seed: int = 0,
        deterministic: bool | None = None,
        metadata: str | None = None,
    ) -> Iterator[GenerateStreamEvent]:
        return self.stream_text(
            _render_chat_prompt(messages, self.model_id),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            deterministic=deterministic,
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

    def __enter__(self) -> Session:
        return self

    def __exit__(self, exc_type: object | None, exc: object | None, traceback: object | None) -> None:
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
            _mlx_model_from_dict(value["mlx_model"])
            if value.get("mlx_model") is not None
            else None
        ),
    )


def _host_from_dict(value: dict[str, Any]) -> HostInfo:
    return HostInfo(
        os=str(value.get("os", "")),
        arch=str(value.get("arch", "")),
        detected_soc=value.get("detected_soc"),
        supported_mlx_runtime=bool(value.get("supported_mlx_runtime", False)),
        unsupported_host_override_active=bool(
            value.get("unsupported_host_override_active", False)
        ),
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
        layer_count=int(value.get("layer_count", 0)),
        tensor_count=int(value.get("tensor_count", 0)),
        tie_word_embeddings=bool(value.get("tie_word_embeddings", False)),
        bindings_prepared=bool(value.get("bindings_prepared", False)),
        buffers_bound=bool(value.get("buffers_bound", False)),
        buffer_count=int(value.get("buffer_count", 0)),
        buffer_bytes=int(value.get("buffer_bytes", 0)),
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
        runtime_model_conditioned_inputs=bool(
            value.get("runtime_model_conditioned_inputs", False)
        ),
        runtime_real_model_tensor_inputs=bool(
            value.get("runtime_real_model_tensor_inputs", False)
        ),
        runtime_complete_model_forward_supported=bool(
            value.get("runtime_complete_model_forward_supported", False)
        ),
        runtime_model_bindings_prepared=bool(
            value.get("runtime_model_bindings_prepared", False)
        ),
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
        execution_model_bound_ffn_decode=bool(
            value.get("execution_model_bound_ffn_decode", False)
        ),
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
        binary_archive_attached_pipeline_count=int(
            value["binary_archive_attached_pipeline_count"]
        ),
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
                            value["numeric"]["validation"][
                                "expected_attention_output_checksum"
                            ]
                        ),
                        expected_gather_output_checksum=int(
                            value["numeric"]["validation"][
                                "expected_gather_output_checksum"
                            ]
                        ),
                        expected_copy_output_checksum=int(
                            value["numeric"]["validation"][
                                "expected_copy_output_checksum"
                            ]
                        ),
                        attention_max_abs_diff_microunits=int(
                            value["numeric"]["validation"][
                                "attention_max_abs_diff_microunits"
                            ]
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
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        elif template == "llama3":
            prompt_parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        else:
            safe_content = content.replace("\\", "\\\\").replace("\n", "\\n")
            prompt_parts.append(f"{role}: {safe_content}\n")

    if template == "qwen_chatml":
        prompt_parts.append("<|im_start|>assistant\n")
    elif template == "llama3":
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    else:
        prompt_parts.append("assistant:")
    return "".join(prompt_parts)


def _chat_prompt_template(model_id: str) -> str:
    normalized = model_id.lower()
    if "qwen" in normalized:
        return "qwen_chatml"
    if "llama-3" in normalized or "llama3" in normalized or "llama_3" in normalized:
        return "llama3"
    return "plain_role_prefix"


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


__all__ = [
    "CapabilityReport",
    "ChatMessage",
    "EngineBackendError",
    "EngineError",
    "EngineInferenceError",
    "EngineStateError",
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
    "Session",
    "StepReport",
]
