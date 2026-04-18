from dataclasses import dataclass, field
from typing import Iterator


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
    os: str
    arch: str
    detected_soc: str | None
    supported_native_runtime: bool
    unsupported_host_override_active: bool


@dataclass(frozen=True)
class ToolStatusInfo:
    available: bool
    version: str | None


@dataclass(frozen=True)
class MetalToolchainInfo:
    fully_available: bool
    metal: ToolStatusInfo
    metallib: ToolStatusInfo
    metal_ar: ToolStatusInfo


@dataclass(frozen=True)
class NativeRuntimeInfo:
    runner: str
    artifacts_source: str | None


@dataclass(frozen=True)
class NativeModelInfo:
    artifacts_source: str | None
    model_family: str
    tensor_format: str
    layer_count: int
    tensor_count: int
    tie_word_embeddings: bool
    bindings_prepared: bool
    buffers_bound: bool
    buffer_count: int
    buffer_bytes: int


@dataclass(frozen=True)
class RuntimeInfo:
    selected_backend: str
    support_tier: str
    resolution_policy: str
    capabilities: CapabilityReport
    fallback_reason: str | None = None
    host: HostInfo = ...
    metal_toolchain: MetalToolchainInfo = ...
    native_runtime: NativeRuntimeInfo | None = None
    native_model: NativeModelInfo | None = None


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
    output_token_logprobs: list[float | None] = ...


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
    output_token_logprobs: list[float | None] = ...


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
        support_tier: str = "native_preview",
        compat_backend: str = "llama_cpp",
        compat_cli_path: str = "llama-cli",
        compat_model_path: str | None = None,
        compat_server_url: str | None = None,
        native_runtime_artifacts_dir: str | None = None,
        native_model_artifacts_dir: str | None = None,
    ) -> None: ...
    @property
    def model_id(self) -> str: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...
    def runtime(self) -> RuntimeInfo: ...
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
    ) -> GenerateResult: ...
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
    ) -> GenerateResult: ...
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
    ) -> int: ...
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
    ) -> int: ...
    def step(self) -> StepReport: ...
    def snapshot(self, request_id: int) -> RequestReport | None: ...
    def cancel(self, request_id: int) -> None: ...
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
    ) -> Iterator[GenerateStreamEvent]: ...
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
    ) -> Iterator[GenerateStreamEvent]: ...
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
    ) -> GenerateResult: ...
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
    ) -> int: ...
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
    ) -> Iterator[GenerateStreamEvent]: ...
    def __enter__(self) -> "Session": ...
    def __exit__(self, exc_type: object | None, exc: object | None, traceback: object | None) -> None: ...
