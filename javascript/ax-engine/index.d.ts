export interface AxEngineClientOptions {
  baseUrl?: string;
  fetch?: typeof fetch;
  headers?: HeadersInit;
}

export interface GenerateSampling {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  seed?: number;
}

export interface PreviewGenerateRequest {
  model?: string;
  input_tokens?: number[];
  input_text?: string;
  max_output_tokens?: number;
  sampling?: GenerateSampling;
  metadata?: string;
}

export interface OpenAiCompletionRequest {
  model?: string;
  prompt: string | string[] | number[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  seed?: number;
  stream?: boolean;
  metadata?: string;
}

export interface OpenAiChatMessage {
  role: string;
  content:
    | string
    | Array<{
        type: string;
        text?: string;
      }>;
}

export interface OpenAiChatCompletionRequest {
  model?: string;
  messages: OpenAiChatMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  seed?: number;
  stream?: boolean;
  metadata?: string;
}

export interface StreamEvent<T = unknown> {
  event: string;
  data: T;
}

export interface HealthResponse {
  status: string;
  service: string;
  model_id: string;
  runtime: RuntimeInfo;
}

export interface CapabilityReport {
  text_generation: boolean;
  token_streaming: boolean;
  deterministic_mode: boolean;
  prefix_reuse: boolean;
  long_context_validation: string;
  benchmark_metrics: string;
}

export interface HostInfo {
  os: string;
  arch: string;
  detected_soc?: string;
  supported_native_runtime: boolean;
  unsupported_host_override_active: boolean;
}

export interface ToolStatusInfo {
  available: boolean;
  version?: string;
}

export interface MetalToolchainInfo {
  fully_available: boolean;
  metal: ToolStatusInfo;
  metallib: ToolStatusInfo;
  metal_ar: ToolStatusInfo;
}

export interface NativeRuntimeInfo {
  runner: string;
  artifacts_source?: string;
}

export interface NativeModelInfo {
  artifacts_source?: string;
  model_family: string;
  tensor_format: string;
  layer_count: number;
  tensor_count: number;
  tie_word_embeddings: boolean;
  bindings_prepared: boolean;
  buffers_bound: boolean;
  buffer_count: number;
  buffer_bytes: number;
}

export interface RuntimeInfo {
  selected_backend: string;
  support_tier: string;
  resolution_policy: string;
  capabilities: CapabilityReport;
  fallback_reason?: string;
  host: HostInfo;
  metal_toolchain: MetalToolchainInfo;
  native_runtime?: NativeRuntimeInfo;
  native_model?: NativeModelInfo;
}

export interface ServerInfoResponse {
  service: string;
  model_id: string;
  deterministic: boolean;
  max_batch_tokens: number;
  block_size_tokens: number;
  runtime: RuntimeInfo;
}

export interface ModelCard {
  id: string;
  object: string;
  owned_by: string;
  runtime: RuntimeInfo;
}

export interface ModelsResponse {
  object: string;
  data: ModelCard[];
}

export interface GenerateRoute {
  execution_plan?: string;
  attention_route?: string;
  kv_mode?: string;
  prefix_cache_path?: string;
  barrier_mode?: string;
  crossover_decisions?: Record<string, number>;
}

export interface GenerateResponse {
  request_id: number;
  model_id: string;
  prompt_tokens: number[];
  prompt_text?: string;
  output_tokens: number[];
  output_token_logprobs?: Array<number | null>;
  output_text?: string;
  status: string;
  finish_reason?: string;
  step_count: number;
  ttft_step?: number;
  route: GenerateRoute;
  runtime: RuntimeInfo;
}

export interface RequestReport {
  request_id: number;
  model_id: string;
  state: string;
  prompt_tokens: number[];
  processed_prompt_tokens: number;
  output_tokens: number[];
  output_token_logprobs?: Array<number | null>;
  prompt_len: number;
  output_len: number;
  max_output_tokens: number;
  cancel_requested: boolean;
  execution_plan_ref?: string;
  route: GenerateRoute;
  finish_reason?: string;
  terminal_stop_reason?: string;
}

export interface MetalDispatchValidationInfo {
  expected_key_cache_checksum: number;
  expected_attention_output_checksum: number;
  expected_gather_output_checksum: number;
  expected_copy_output_checksum: number;
  attention_max_abs_diff_microunits: number;
}

export interface MetalDispatchNumericInfo {
  key_cache_checksum: number;
  attention_output_checksum: number;
  gather_output_checksum: number;
  copy_output_checksum: number;
  validation?: MetalDispatchValidationInfo;
}

export interface MetalDispatchKernelInfo {
  function_name: string;
  element_count: number;
  threads_per_grid_width: number;
  threads_per_threadgroup_width: number;
}

export interface MetalDispatchInfo {
  command_queue_label: string;
  command_buffer_label: string;
  command_buffer_status: string;
  runtime_device_name: string;
  runtime_required_pipeline_count: number;
  runtime_max_thread_execution_width: number;
  runtime_model_conditioned_inputs: boolean;
  runtime_real_model_tensor_inputs: boolean;
  runtime_complete_model_forward_supported: boolean;
  runtime_model_bindings_prepared: boolean;
  runtime_model_buffers_bound: boolean;
  runtime_model_buffer_count: number;
  runtime_model_buffer_bytes: number;
  runtime_model_family?: string;
  execution_direct_decode_token_count: number;
  execution_direct_decode_checksum_lo: number;
  execution_logits_output_count: number;
  execution_remaining_logits_handle_count: number;
  execution_model_bound_ffn_decode: boolean;
  execution_real_model_forward_completed: boolean;
  execution_prefix_native_dispatch_count: number;
  execution_prefix_cpu_reference_dispatch_count: number;
  execution_qkv_projection_token_count: number;
  execution_layer_continuation_token_count: number;
  execution_logits_projection_token_count: number;
  execution_logits_vocab_scan_row_count: number;
  binary_archive_state: string;
  binary_archive_attached_pipeline_count: number;
  binary_archive_serialized: boolean;
  arena_token_capacity: number;
  arena_slot_capacity: number;
  arena_attention_ref_capacity: number;
  arena_gather_ref_capacity: number;
  arena_gather_output_capacity: number;
  arena_copy_pair_capacity: number;
  arena_sequence_capacity: number;
  arena_reused_existing: boolean;
  arena_grew_existing: boolean;
  kernels: MetalDispatchKernelInfo[];
  numeric: MetalDispatchNumericInfo;
}

export interface StepReport {
  step_id?: number;
  scheduled_requests: number;
  scheduled_tokens: number;
  ttft_events: number;
  prefix_hits: number;
  kv_usage_blocks: number;
  evictions: number;
  cpu_time_us: number;
  runner_time_us: number;
  route?: GenerateRoute;
  metal_dispatch?: MetalDispatchInfo;
}

export interface GenerateStreamRequestEvent {
  request: RequestReport;
  runtime: RuntimeInfo;
}

export interface GenerateStreamStepEvent {
  request: RequestReport;
  step: StepReport;
  delta_tokens?: number[];
  delta_token_logprobs?: Array<number | null>;
  delta_text?: string;
}

export interface GenerateStreamResponseEvent {
  response: GenerateResponse;
}

export type PreviewGenerateStreamEvent =
  | { event: "request"; data: GenerateStreamRequestEvent }
  | { event: "step"; data: GenerateStreamStepEvent }
  | { event: "response"; data: GenerateStreamResponseEvent }
  | { event: string; data: unknown };

export class AxEngineHttpError extends Error {
  constructor(message: string, options?: { status?: number; payload?: unknown });
  status: number;
  payload: unknown;
}

export class AxEngineClient {
  constructor(options?: AxEngineClientOptions);
  readonly baseUrl: string;
  health(): Promise<HealthResponse>;
  runtime(): Promise<ServerInfoResponse>;
  models(): Promise<ModelsResponse>;
  generate(request: PreviewGenerateRequest): Promise<GenerateResponse>;
  submit(request: PreviewGenerateRequest): Promise<RequestReport>;
  requestSnapshot(requestId: number | string): Promise<RequestReport>;
  cancel(requestId: number | string): Promise<RequestReport>;
  step(): Promise<StepReport>;
  completion(request: OpenAiCompletionRequest): Promise<any>;
  chatCompletion(request: OpenAiChatCompletionRequest): Promise<any>;
  streamGenerate(
    request: PreviewGenerateRequest,
  ): AsyncGenerator<PreviewGenerateStreamEvent, void, void>;
  streamCompletion(request: OpenAiCompletionRequest): AsyncGenerator<StreamEvent, void, void>;
  streamChatCompletion(
    request: OpenAiChatCompletionRequest,
  ): AsyncGenerator<StreamEvent, void, void>;
}

export default AxEngineClient;
