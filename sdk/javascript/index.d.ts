export interface AxEngineClientOptions {
  baseUrl?: string;
  fetch?: typeof fetch;
  headers?: HeadersInit;
}

export interface GenerateSampling {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  seed?: number;
  ignore_eos?: boolean;
}

export type Gemma4UnifiedModality = "image" | "audio" | "video";

export interface Gemma4UnifiedTokenSpan {
  modality: Gemma4UnifiedModality;
  placeholder_index: number;
  replacement_start: number;
  soft_token_count: number;
  replacement_token_count: number;
}

export interface Gemma4UnifiedSoftTokenRange {
  start: number;
  soft_token_count: number;
}

export interface Gemma4UnifiedImageRuntimeInput {
  span: Gemma4UnifiedTokenSpan;
  pixel_values: number[];
  pixel_position_ids: Array<[number, number]>;
}

export interface Gemma4UnifiedAudioRuntimeInput {
  span: Gemma4UnifiedTokenSpan;
  input_features: number[];
  frame_count: number;
  feature_count: number;
}

export interface Gemma4UnifiedVideoRuntimeInput {
  span: Gemma4UnifiedTokenSpan;
  soft_token_ranges?: Gemma4UnifiedSoftTokenRange[];
  pixel_values: number[];
  pixel_position_ids: Array<[number, number]>;
  frame_count: number;
}

export interface Gemma4UnifiedRuntimeInputs {
  images?: Gemma4UnifiedImageRuntimeInput[];
  audios?: Gemma4UnifiedAudioRuntimeInput[];
  videos?: Gemma4UnifiedVideoRuntimeInput[];
}

export interface RequestMultimodalInputs {
  gemma4_unified?: Gemma4UnifiedRuntimeInputs;
}

export interface PreviewGenerateRequest {
  model?: string;
  input_tokens?: number[];
  input_text?: string;
  multimodal_inputs?: RequestMultimodalInputs;
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
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  stop?: string | string[];
  seed?: number;
  stream?: boolean;
  metadata?: string;
  multimodal_inputs?: RequestMultimodalInputs;
}

export interface OpenAiChatMessage {
  role: "system" | "user" | "assistant" | "tool" | string;
  /** Null/absent is valid for assistant messages that only carry tool_calls. */
  content?:
    | string
    | Array<OpenAiChatContentPart>
    | null;
  /** Assistant messages echoed back into the conversation after a tool turn. */
  tool_calls?: OpenAiToolCall[];
  /** Required on `role: "tool"` result messages. */
  tool_call_id?: string;
  name?: string;
}

export interface ChatToolFunction {
  name: string;
  description?: string;
  parameters?: unknown;
}

export interface ChatTool {
  type: "function";
  function: ChatToolFunction;
}

export type ChatToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; function: { name: string } };

export type ChatResponseFormat =
  | { type: "text" }
  | { type: "json_object" }
  | {
      type: "json_schema";
      json_schema: { name?: string; strict?: boolean; schema: unknown };
    };

export interface OpenAiFunctionCall {
  name: string;
  arguments: string;
}

export interface OpenAiToolCall {
  id: string;
  type: "function";
  function: OpenAiFunctionCall;
}

export interface OpenAiFunctionCallDelta {
  name?: string;
  arguments?: string;
}

export interface OpenAiToolCallDelta {
  index: number;
  id?: string;
  type?: "function";
  function?: OpenAiFunctionCallDelta;
}

export interface OpenAiChatContentPart {
  type: string;
  text?: string;
  image_url?: unknown;
  input_audio?: unknown;
  audio_url?: unknown;
  video_url?: unknown;
}

export interface OpenAiChatCompletionRequest {
  model?: string;
  messages: OpenAiChatMessage[];
  input_tokens?: number[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repetition_penalty?: number;
  stop?: string | string[];
  seed?: number;
  stream?: boolean;
  metadata?: string;
  multimodal_inputs?: RequestMultimodalInputs;
  tools?: ChatTool[];
  tool_choice?: ChatToolChoice;
  response_format?: ChatResponseFormat;
  reasoning?: unknown;
  logprobs?: boolean;
  top_logprobs?: number;
}

export interface OpenAiPromptTokensDetails {
  /** Prompt tokens served from the prefix cache. */
  cached_tokens: number;
}

export interface OpenAiUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: OpenAiPromptTokensDetails;
}

export type OpenAiFinishReason =
  | "stop"
  | "length"
  | "tool_calls"
  | "cancel"
  | "content_filter";

export interface OpenAiCompletionChoice {
  index: number;
  text: string;
  finish_reason: OpenAiFinishReason | null;
}

export interface OpenAiCompletionResponse {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: OpenAiCompletionChoice[];
  usage?: OpenAiUsage;
}

export interface OpenAiCompletionChunkChoice {
  index: number;
  text: string;
  finish_reason: OpenAiFinishReason | null;
}

export interface OpenAiCompletionChunk {
  id: string;
  object: "text_completion.chunk";
  created: number;
  model: string;
  choices: OpenAiCompletionChunkChoice[];
}

export interface OpenAiChatMessageResponse {
  role: "assistant";
  content: string | null;
  reasoning_content?: string;
  tool_calls?: OpenAiToolCall[];
}

export interface OpenAiChatCompletionChoice {
  index: number;
  message: OpenAiChatMessageResponse;
  finish_reason: OpenAiFinishReason | null;
}

export interface OpenAiChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: OpenAiChatCompletionChoice[];
  usage?: OpenAiUsage;
}

export interface OpenAiChatDelta {
  role?: string;
  content?: string;
  reasoning_content?: string;
  tool_calls?: OpenAiToolCallDelta[];
}

export interface OpenAiChatCompletionChunkChoice {
  index: number;
  delta: OpenAiChatDelta;
  finish_reason: OpenAiFinishReason | null;
}

export interface OpenAiChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: OpenAiChatCompletionChunkChoice[];
}

export interface OpenAiEmbeddingRequest {
  model?: string;
  input: number[] | number[][];
  encoding_format?: string;
  pooling?: "last" | "mean" | "cls" | string;
  normalize?: boolean;
}

export interface OpenAiEmbeddingObject {
  object: string;
  embedding: number[];
  index: number;
}

export interface OpenAiEmbeddingResponse {
  object: string;
  data: OpenAiEmbeddingObject[];
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export interface LoadModelRequest {
  model_id: string;
  model_path: string;
  load_policy?: "availability_first" | "memory_constrained";
  load_mode?: "replace" | "add";
  /**
   * Whether the loaded model becomes the default for requests that omit
   * `model`. Server default is true; only meaningful for `load_mode: "add"`
   * (a `"replace"` load rejects `false`).
   */
  make_default?: boolean;
}

export interface LoadModelResponse {
  model_id: string;
  state: "loaded" | string;
  context_length: number;
  load_policy: "availability_first" | "memory_constrained";
  load_mode: "replace" | "add";
  /** Default model after the load; absent when the server predates 6.9. */
  default_model_id?: string;
}

export interface UnloadModelRequest {
  model_id: string;
}

export interface UnloadModelResponse {
  model_id: string;
  state: "unloaded" | string;
  /**
   * Default model after the unload (reports the reassignment when the
   * unloaded model was the default); absent when the server predates 6.9.
   */
  default_model_id?: string;
}

export interface StreamEvent<T = unknown> {
  event: string;
  data: T;
}

export interface HealthResponse {
  status: string;
  service: string;
  model_id?: string;
  /** Every loaded model id (multi-model serving); absent on pre-6.9 servers. */
  models?: string[];
  runtime?: RuntimeInfo;
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
  supported_mlx_runtime: boolean;
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

export interface MlxRuntimeInfo {
  runner: string;
  artifacts_source?: string;
}

export interface MlxModelInfo {
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
  mlx_runtime?: MlxRuntimeInfo;
  mlx_model?: MlxModelInfo;
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
  capabilities: ModelCapabilities;
  limit: ModelLimit;
  context_length: number;
  max_output_tokens: number;
  ax_engine: AxEngineModelMetadata;
  runtime: RuntimeInfo;
}

export interface ModelCapabilities {
  temperature: boolean;
  reasoning: boolean;
  attachment: boolean;
  toolcall: boolean;
  input: ModelModalities;
  output: ModelModalities;
  interleaved: boolean;
}

export interface ModelModalities {
  text: boolean;
  audio: boolean;
  image: boolean;
  video: boolean;
  pdf: boolean;
}

export interface ModelLimit {
  context: number;
  output: number;
}

export interface AxEngineModelMetadata {
  native_generate_supported: boolean;
  openai_completions_supported: boolean;
  openai_chat_completions_supported: boolean;
  openai_tool_calling_supported: boolean;
  openai_text_input_supported: boolean;
  native_multimodal_input_supported: boolean;
  gemma4_unified_multimodal_input_supported: boolean;
  openai_tokenized_multimodal_input_supported: boolean;
  primary_use: "general" | "coding" | string;
  chat_default: boolean;
  coding_supported: boolean;
  coding_only: boolean;
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

export class AxEngineStreamError extends Error {
  constructor(message: string, options?: { payload?: unknown });
  payload: unknown;
}

export interface RequestOptions {
  /** Aborts the underlying fetch (including an in-flight stream). */
  signal?: AbortSignal;
}

export class AxEngineClient {
  constructor(options?: AxEngineClientOptions);
  readonly baseUrl: string;
  health(options?: RequestOptions): Promise<HealthResponse>;
  runtime(options?: RequestOptions): Promise<ServerInfoResponse>;
  models(options?: RequestOptions): Promise<ModelsResponse>;
  generate(request: PreviewGenerateRequest, options?: RequestOptions): Promise<GenerateResponse>;
  submit(request: PreviewGenerateRequest, options?: RequestOptions): Promise<RequestReport>;
  requestSnapshot(requestId: number | string, options?: RequestOptions): Promise<RequestReport>;
  cancel(requestId: number | string, options?: RequestOptions): Promise<RequestReport>;
  step(model?: string, options?: RequestOptions): Promise<StepReport>;
  completion(
    request: OpenAiCompletionRequest,
    options?: RequestOptions,
  ): Promise<OpenAiCompletionResponse>;
  chatCompletion(
    request: OpenAiChatCompletionRequest,
    options?: RequestOptions,
  ): Promise<OpenAiChatCompletionResponse>;
  embeddings(
    request: OpenAiEmbeddingRequest,
    options?: RequestOptions,
  ): Promise<OpenAiEmbeddingResponse>;
  loadModel(request: LoadModelRequest, options?: RequestOptions): Promise<LoadModelResponse>;
  unloadModel(request: UnloadModelRequest, options?: RequestOptions): Promise<UnloadModelResponse>;
  streamGenerate(
    request: PreviewGenerateRequest,
    options?: RequestOptions,
  ): AsyncGenerator<PreviewGenerateStreamEvent, void, void>;
  streamCompletion(
    request: OpenAiCompletionRequest,
    options?: RequestOptions,
  ): AsyncGenerator<StreamEvent<OpenAiCompletionChunk>, void, void>;
  streamChatCompletion(
    request: OpenAiChatCompletionRequest,
    options?: RequestOptions,
  ): AsyncGenerator<StreamEvent<OpenAiChatCompletionChunk>, void, void>;
}

export default AxEngineClient;
