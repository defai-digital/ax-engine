package axengine

// GenerateSampling holds sampling parameters shared across generation requests.
type GenerateSampling struct {
	Temperature       *float64 `json:"temperature,omitempty"`
	TopP              *float64 `json:"top_p,omitempty"`
	TopK              *int     `json:"top_k,omitempty"`
	RepetitionPenalty *float64 `json:"repetition_penalty,omitempty"`
	Seed              *int64   `json:"seed,omitempty"`
}

// Gemma4UnifiedTokenSpan describes where processed media tokens replace a prompt placeholder.
type Gemma4UnifiedTokenSpan struct {
	Modality              string `json:"modality"`
	PlaceholderIndex      int    `json:"placeholder_index"`
	ReplacementStart      int    `json:"replacement_start"`
	SoftTokenCount        int    `json:"soft_token_count"`
	ReplacementTokenCount int    `json:"replacement_token_count"`
}

// Gemma4UnifiedSoftTokenRange points at a non-contiguous video soft-token span.
type Gemma4UnifiedSoftTokenRange struct {
	Start          int `json:"start"`
	SoftTokenCount int `json:"soft_token_count"`
}

// Gemma4UnifiedImageRuntimeInput is a processed image tensor for Gemma 4 unified models.
type Gemma4UnifiedImageRuntimeInput struct {
	Span             Gemma4UnifiedTokenSpan `json:"span"`
	PixelValues      []float64              `json:"pixel_values"`
	PixelPositionIDs [][]int                `json:"pixel_position_ids"`
}

// Gemma4UnifiedAudioRuntimeInput is a processed audio feature tensor for Gemma 4 unified models.
type Gemma4UnifiedAudioRuntimeInput struct {
	Span          Gemma4UnifiedTokenSpan `json:"span"`
	InputFeatures []float64              `json:"input_features"`
	FrameCount    int                    `json:"frame_count"`
	FeatureCount  int                    `json:"feature_count"`
}

// Gemma4UnifiedVideoRuntimeInput is a processed video tensor for Gemma 4 unified models.
type Gemma4UnifiedVideoRuntimeInput struct {
	Span             Gemma4UnifiedTokenSpan        `json:"span"`
	SoftTokenRanges  []Gemma4UnifiedSoftTokenRange `json:"soft_token_ranges,omitempty"`
	PixelValues      []float64                     `json:"pixel_values"`
	PixelPositionIDs [][]int                       `json:"pixel_position_ids"`
	FrameCount       int                           `json:"frame_count"`
}

// Gemma4UnifiedRuntimeInputs carries preprocessed Gemma 4 image/audio/video tensors.
type Gemma4UnifiedRuntimeInputs struct {
	Images []Gemma4UnifiedImageRuntimeInput `json:"images,omitempty"`
	Audios []Gemma4UnifiedAudioRuntimeInput `json:"audios,omitempty"`
	Videos []Gemma4UnifiedVideoRuntimeInput `json:"videos,omitempty"`
}

// RequestMultimodalInputs carries processed multimodal inputs for native generate.
type RequestMultimodalInputs struct {
	Gemma4Unified *Gemma4UnifiedRuntimeInputs `json:"gemma4_unified,omitempty"`
}

// PreviewGenerateRequest is the ax-engine native generate request.
type PreviewGenerateRequest struct {
	Model            *string                  `json:"model,omitempty"`
	InputTokens      []int                    `json:"input_tokens,omitempty"`
	InputText        *string                  `json:"input_text,omitempty"`
	MultimodalInputs *RequestMultimodalInputs `json:"multimodal_inputs,omitempty"`
	MaxOutputTokens  *int                     `json:"max_output_tokens,omitempty"`
	Sampling         *GenerateSampling        `json:"sampling,omitempty"`
	Metadata         *string                  `json:"metadata,omitempty"`
}

// OpenAiUsage contains token usage information.
type OpenAiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// PromptTokensDetails reports prefix-cache reuse in the OpenAI
	// prompt-caching shape; nil when the server reports none.
	PromptTokensDetails *OpenAiPromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

// OpenAiPromptTokensDetails is the OpenAI prompt-caching usage breakdown.
type OpenAiPromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// OpenAiCompletionRequest is the /v1/completions request body.
type OpenAiCompletionRequest struct {
	Model             *string                  `json:"model,omitempty"`
	Prompt            interface{}              `json:"prompt"` // string | []string | []int
	MaxTokens         *int                     `json:"max_tokens,omitempty"`
	Temperature       *float64                 `json:"temperature,omitempty"`
	TopP              *float64                 `json:"top_p,omitempty"`
	TopK              *int                     `json:"top_k,omitempty"`
	MinP              *float64                 `json:"min_p,omitempty"`
	RepetitionPenalty *float64                 `json:"repetition_penalty,omitempty"`
	Stop              interface{}              `json:"stop,omitempty"` // string | []string
	Seed              *int64                   `json:"seed,omitempty"`
	Stream            *bool                    `json:"stream,omitempty"`
	Metadata          *string                  `json:"metadata,omitempty"`
	MultimodalInputs  *RequestMultimodalInputs `json:"multimodal_inputs,omitempty"`
}

// OpenAiCompletionChoice is a single completion choice.
type OpenAiCompletionChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

// OpenAiCompletionResponse is the /v1/completions response.
type OpenAiCompletionResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []OpenAiCompletionChoice `json:"choices"`
	Usage   *OpenAiUsage             `json:"usage,omitempty"`
}

// OpenAiCompletionChunkChoice is a single chunk choice for streaming completions.
type OpenAiCompletionChunkChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

// OpenAiCompletionChunk is a streaming completion chunk.
type OpenAiCompletionChunk struct {
	ID      string                        `json:"id"`
	Object  string                        `json:"object"`
	Created int64                         `json:"created"`
	Model   string                        `json:"model"`
	Choices []OpenAiCompletionChunkChoice `json:"choices"`
}

// OpenAiChatMessage is a message in a chat conversation.
type OpenAiChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content,omitempty"` // string | []map[string]interface{} | nil
	// ToolCalls carries assistant tool calls echoed back after a tool turn.
	ToolCalls []OpenAiToolCall `json:"tool_calls,omitempty"`
	// ToolCallID is required on role "tool" result messages.
	ToolCallID *string `json:"tool_call_id,omitempty"`
	Name       *string `json:"name,omitempty"`
}

// OpenAiToolCall is a completed tool call in a chat message.
type OpenAiToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function OpenAiFunctionCall `json:"function"`
}

// OpenAiFunctionCall carries a tool call's name and JSON-encoded arguments.
type OpenAiFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// OpenAiChatCompletionRequest is the /v1/chat/completions request body.
type OpenAiChatCompletionRequest struct {
	Model             *string                  `json:"model,omitempty"`
	Messages          []OpenAiChatMessage      `json:"messages"`
	InputTokens       []int                    `json:"input_tokens,omitempty"`
	MaxTokens         *int                     `json:"max_tokens,omitempty"`
	Temperature       *float64                 `json:"temperature,omitempty"`
	TopP              *float64                 `json:"top_p,omitempty"`
	TopK              *int                     `json:"top_k,omitempty"`
	MinP              *float64                 `json:"min_p,omitempty"`
	RepetitionPenalty *float64                 `json:"repetition_penalty,omitempty"`
	Stop              interface{}              `json:"stop,omitempty"`
	Seed              *int64                   `json:"seed,omitempty"`
	Stream            *bool                    `json:"stream,omitempty"`
	Metadata          *string                  `json:"metadata,omitempty"`
	MultimodalInputs  *RequestMultimodalInputs `json:"multimodal_inputs,omitempty"`
	// Tools is the OpenAI function-tool list ([]map or typed structs).
	Tools interface{} `json:"tools,omitempty"`
	// ToolChoice is "auto" | "none" | "required" | {"type":"function",...}.
	ToolChoice interface{} `json:"tool_choice,omitempty"`
	// ResponseFormat is {"type":"text"|"json_object"|"json_schema",...}.
	ResponseFormat interface{} `json:"response_format,omitempty"`
	Reasoning      interface{} `json:"reasoning,omitempty"`
	Logprobs       *bool       `json:"logprobs,omitempty"`
	TopLogprobs    *int        `json:"top_logprobs,omitempty"`
}

// OpenAiChatMessageResponse is the assistant message in a chat completion response.
type OpenAiChatMessageResponse struct {
	Role    string  `json:"role"`
	Content *string `json:"content"` // nil when tool_calls are present (server emits null)
	// ReasoningContent carries separated reasoning when the request opted in.
	ReasoningContent *string          `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAiToolCall `json:"tool_calls,omitempty"`
}

// OpenAiChatCompletionChoice is a single chat completion choice.
type OpenAiChatCompletionChoice struct {
	Index        int                       `json:"index"`
	Message      OpenAiChatMessageResponse `json:"message"`
	FinishReason *string                   `json:"finish_reason"`
}

// OpenAiChatCompletionResponse is the /v1/chat/completions response.
type OpenAiChatCompletionResponse struct {
	ID      string                       `json:"id"`
	Object  string                       `json:"object"`
	Created int64                        `json:"created"`
	Model   string                       `json:"model"`
	Choices []OpenAiChatCompletionChoice `json:"choices"`
	Usage   *OpenAiUsage                 `json:"usage,omitempty"`
}

// OpenAiChatDelta is the delta in a streaming chat completion chunk.
type OpenAiChatDelta struct {
	Role             *string               `json:"role,omitempty"`
	Content          *string               `json:"content,omitempty"`
	ReasoningContent *string               `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAiToolCallDelta `json:"tool_calls,omitempty"`
}

// OpenAiToolCallDelta is a streamed tool-call fragment with a stream-wide index.
type OpenAiToolCallDelta struct {
	Index    int                      `json:"index"`
	ID       *string                  `json:"id,omitempty"`
	Type     *string                  `json:"type,omitempty"`
	Function *OpenAiFunctionCallDelta `json:"function,omitempty"`
}

// OpenAiFunctionCallDelta is the function fragment of a streamed tool call.
type OpenAiFunctionCallDelta struct {
	Name      *string `json:"name,omitempty"`
	Arguments *string `json:"arguments,omitempty"`
}

// OpenAiChatCompletionChunkChoice is a single chunk choice for streaming chat.
type OpenAiChatCompletionChunkChoice struct {
	Index        int             `json:"index"`
	Delta        OpenAiChatDelta `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

// OpenAiChatCompletionChunk is a streaming chat completion chunk.
type OpenAiChatCompletionChunk struct {
	ID      string                            `json:"id"`
	Object  string                            `json:"object"`
	Created int64                             `json:"created"`
	Model   string                            `json:"model"`
	Choices []OpenAiChatCompletionChunkChoice `json:"choices"`
}

// OpenAiEmbeddingRequest is the /v1/embeddings request body.
type OpenAiEmbeddingRequest struct {
	Model *string `json:"model,omitempty"`
	// Input accepts []int for one sequence or [][]int for an explicit batch.
	Input          any     `json:"input"`
	EncodingFormat *string `json:"encoding_format,omitempty"`
	Pooling        *string `json:"pooling,omitempty"`
	Normalize      *bool   `json:"normalize,omitempty"`
}

// OpenAiEmbeddingObject is a single embedding result.
type OpenAiEmbeddingObject struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// OpenAiEmbeddingResponse is the /v1/embeddings response.
type OpenAiEmbeddingResponse struct {
	Object string                  `json:"object"`
	Data   []OpenAiEmbeddingObject `json:"data"`
	Model  string                  `json:"model"`
	Usage  struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// GenerateRoute holds routing metadata for a generate request.
type GenerateRoute struct {
	ExecutionPlan      *string            `json:"execution_plan,omitempty"`
	AttentionRoute     *string            `json:"attention_route,omitempty"`
	KvMode             *string            `json:"kv_mode,omitempty"`
	PrefixCachePath    *string            `json:"prefix_cache_path,omitempty"`
	BarrierMode        *string            `json:"barrier_mode,omitempty"`
	CrossoverDecisions map[string]float64 `json:"crossover_decisions,omitempty"`
}

// GenerateResponse is the response from /v1/generate.
type GenerateResponse struct {
	RequestID           int64         `json:"request_id"`
	ModelID             string        `json:"model_id"`
	PromptTokens        []int         `json:"prompt_tokens"`
	PromptText          *string       `json:"prompt_text,omitempty"`
	OutputTokens        []int         `json:"output_tokens"`
	OutputTokenLogprobs []*float64    `json:"output_token_logprobs,omitempty"`
	OutputText          *string       `json:"output_text,omitempty"`
	Status              string        `json:"status"`
	FinishReason        *string       `json:"finish_reason,omitempty"`
	StepCount           int           `json:"step_count"`
	TtftStep            *int          `json:"ttft_step,omitempty"`
	Route               GenerateRoute `json:"route"`
}

// RequestReport is a snapshot of a live or completed request.
type RequestReport struct {
	RequestID             int64         `json:"request_id"`
	ModelID               string        `json:"model_id"`
	State                 string        `json:"state"`
	PromptTokens          []int         `json:"prompt_tokens"`
	ProcessedPromptTokens int           `json:"processed_prompt_tokens"`
	OutputTokens          []int         `json:"output_tokens"`
	OutputTokenLogprobs   []*float64    `json:"output_token_logprobs,omitempty"`
	PromptLen             int           `json:"prompt_len"`
	OutputLen             int           `json:"output_len"`
	MaxOutputTokens       int           `json:"max_output_tokens"`
	CancelRequested       bool          `json:"cancel_requested"`
	ExecutionPlanRef      *string       `json:"execution_plan_ref,omitempty"`
	Route                 GenerateRoute `json:"route"`
	FinishReason          *string       `json:"finish_reason,omitempty"`
	TerminalStopReason    *string       `json:"terminal_stop_reason,omitempty"`
}

// StepReport is the result of a /v1/step call.
type StepReport struct {
	StepID            *int64         `json:"step_id,omitempty"`
	ScheduledRequests int            `json:"scheduled_requests"`
	ScheduledTokens   int            `json:"scheduled_tokens"`
	TtftEvents        int            `json:"ttft_events"`
	PrefixHits        int            `json:"prefix_hits"`
	KvUsageBlocks     int            `json:"kv_usage_blocks"`
	Evictions         int            `json:"evictions"`
	CpuTimeUs         int64          `json:"cpu_time_us"`
	RunnerTimeUs      int64          `json:"runner_time_us"`
	Route             *GenerateRoute `json:"route,omitempty"`
}

// HealthResponse is the response from /health.
type HealthResponse struct {
	Status  string `json:"status"`
	Service string `json:"service"`
	ModelID string `json:"model_id"`
	// Models lists every loaded model id (multi-model serving); empty on
	// pre-6.9 servers.
	Models  []string     `json:"models,omitempty"`
	Runtime *RuntimeInfo `json:"runtime,omitempty"`
}

// RuntimeInfo describes the resolved backend and host, as reported by
// /health, /v1/runtime, and each /v1/models card.
type RuntimeInfo struct {
	SelectedBackend  string             `json:"selected_backend"`
	SupportTier      string             `json:"support_tier"`
	ResolutionPolicy string             `json:"resolution_policy"`
	Capabilities     CapabilityReport   `json:"capabilities"`
	FallbackReason   *string            `json:"fallback_reason,omitempty"`
	Host             HostInfo           `json:"host"`
	MetalToolchain   MetalToolchainInfo `json:"metal_toolchain"`
	MlxRuntime       *MlxRuntimeInfo    `json:"mlx_runtime,omitempty"`
	MlxModel         *MlxModelInfo      `json:"mlx_model,omitempty"`
}

// CapabilityReport describes what the resolved backend supports.
type CapabilityReport struct {
	TextGeneration        bool   `json:"text_generation"`
	TokenStreaming        bool   `json:"token_streaming"`
	DeterministicMode     bool   `json:"deterministic_mode"`
	PrefixReuse           bool   `json:"prefix_reuse"`
	LongContextValidation string `json:"long_context_validation"`
	BenchmarkMetrics      string `json:"benchmark_metrics"`
}

// HostInfo describes the serving host.
type HostInfo struct {
	OS                            string  `json:"os"`
	Arch                          string  `json:"arch"`
	DetectedSoc                   *string `json:"detected_soc,omitempty"`
	SupportedMlxRuntime           bool    `json:"supported_mlx_runtime"`
	UnsupportedHostOverrideActive bool    `json:"unsupported_host_override_active"`
}

// ToolStatusInfo describes one Metal toolchain tool.
type ToolStatusInfo struct {
	Available bool    `json:"available"`
	Version   *string `json:"version,omitempty"`
}

// MetalToolchainInfo describes the Metal toolchain availability.
type MetalToolchainInfo struct {
	FullyAvailable bool           `json:"fully_available"`
	Metal          ToolStatusInfo `json:"metal"`
	Metallib       ToolStatusInfo `json:"metallib"`
	MetalAr        ToolStatusInfo `json:"metal_ar"`
}

// MlxRuntimeInfo describes the MLX runtime artifacts in use.
type MlxRuntimeInfo struct {
	Runner          string  `json:"runner"`
	ArtifactsSource *string `json:"artifacts_source,omitempty"`
}

// MlxModelInfo describes the loaded MLX model artifacts.
type MlxModelInfo struct {
	ArtifactsSource   *string `json:"artifacts_source,omitempty"`
	ModelFamily       string  `json:"model_family"`
	TensorFormat      string  `json:"tensor_format"`
	LayerCount        int     `json:"layer_count"`
	TensorCount       int     `json:"tensor_count"`
	TieWordEmbeddings bool    `json:"tie_word_embeddings"`
	BindingsPrepared  bool    `json:"bindings_prepared"`
	BuffersBound      bool    `json:"buffers_bound"`
	BufferCount       int     `json:"buffer_count"`
	BufferBytes       int64   `json:"buffer_bytes"`
}

// ServerInfoResponse is the response from GET /v1/runtime.
type ServerInfoResponse struct {
	Service         string      `json:"service"`
	ModelID         string      `json:"model_id"`
	Deterministic   bool        `json:"deterministic"`
	MaxBatchTokens  uint32      `json:"max_batch_tokens"`
	BlockSizeTokens uint32      `json:"block_size_tokens"`
	Runtime         RuntimeInfo `json:"runtime"`
}

// StreamEvent is a generic SSE event envelope.
type StreamEvent[T any] struct {
	Event string `json:"event"`
	Data  T      `json:"data"`
}

// GenerateStreamRequestEvent is the "request" event from /v1/generate/stream.
type GenerateStreamRequestEvent struct {
	Request RequestReport `json:"request"`
}

// GenerateStreamStepEvent is the "step" event from /v1/generate/stream.
type GenerateStreamStepEvent struct {
	Request            RequestReport `json:"request"`
	Step               StepReport    `json:"step"`
	DeltaTokens        []int         `json:"delta_tokens,omitempty"`
	DeltaTokenLogprobs []*float64    `json:"delta_token_logprobs,omitempty"`
	DeltaText          *string       `json:"delta_text,omitempty"`
}

// GenerateStreamResponseEvent is the "response" event from /v1/generate/stream.
type GenerateStreamResponseEvent struct {
	Response GenerateResponse `json:"response"`
}

// GenerateStreamEvent is a typed SSE event from /v1/generate/stream.
// Exactly one of Request, Step, or Response will be non-nil, depending on Event.
type GenerateStreamEvent struct {
	Event    string
	Request  *GenerateStreamRequestEvent
	Step     *GenerateStreamStepEvent
	Response *GenerateStreamResponseEvent
}

// LoadModelRequest is the body for POST /v1/model/load.
type LoadModelRequest struct {
	ModelID    string `json:"model_id"`
	ModelPath  string `json:"model_path"`
	LoadPolicy string `json:"load_policy,omitempty"`
	LoadMode   string `json:"load_mode,omitempty"`
	// MakeDefault controls whether the loaded model becomes the default for
	// requests that omit `model`. Server default is true; only meaningful
	// for load_mode "add" (a "replace" load rejects false). Pointer so an
	// explicit false survives serialization.
	MakeDefault *bool `json:"make_default,omitempty"`
}

// LoadModelResponse is the response from POST /v1/model/load.
type LoadModelResponse struct {
	ModelID       string `json:"model_id"`
	State         string `json:"state"`
	ContextLength uint32 `json:"context_length"`
	LoadPolicy    string `json:"load_policy"`
	LoadMode      string `json:"load_mode"`
	// DefaultModelID is the default model after the load; empty when the
	// server predates 6.9.
	DefaultModelID string `json:"default_model_id"`
}

// UnloadModelRequest is the body for POST /v1/model/unload.
type UnloadModelRequest struct {
	ModelID string `json:"model_id"`
}

// UnloadModelResponse is the response from POST /v1/model/unload.
type UnloadModelResponse struct {
	ModelID string `json:"model_id"`
	State   string `json:"state"`
	// DefaultModelID is the default model after the unload (reports the
	// reassignment when the unloaded model was the default); empty when the
	// server predates 6.9.
	DefaultModelID string `json:"default_model_id"`
}

// ModelCard describes a single model served by ax-engine-server.
type ModelCard struct {
	ID              string                `json:"id"`
	Object          string                `json:"object"`
	OwnedBy         string                `json:"owned_by"`
	Capabilities    ModelCapabilities     `json:"capabilities"`
	Limit           ModelLimit            `json:"limit"`
	ContextLength   uint32                `json:"context_length"`
	MaxOutputTokens uint32                `json:"max_output_tokens"`
	AxEngine        AxEngineModelMetadata `json:"ax_engine"`
	Runtime         *RuntimeInfo          `json:"runtime,omitempty"`
}

// ModelCapabilities describes what a served model supports.
type ModelCapabilities struct {
	Temperature bool            `json:"temperature"`
	Reasoning   bool            `json:"reasoning"`
	Attachment  bool            `json:"attachment"`
	Toolcall    bool            `json:"toolcall"`
	Input       ModelModalities `json:"input"`
	Output      ModelModalities `json:"output"`
	Interleaved bool            `json:"interleaved"`
}

// ModelModalities flags the media types a model accepts or emits.
type ModelModalities struct {
	Text  bool `json:"text"`
	Audio bool `json:"audio"`
	Image bool `json:"image"`
	Video bool `json:"video"`
	PDF   bool `json:"pdf"`
}

// ModelLimit is the context/output token budget for a served model.
type ModelLimit struct {
	Context uint32 `json:"context"`
	Output  uint32 `json:"output"`
}

// AxEngineModelMetadata carries ax-engine-specific model support flags.
type AxEngineModelMetadata struct {
	NativeGenerateSupported                 bool   `json:"native_generate_supported"`
	OpenaiCompletionsSupported              bool   `json:"openai_completions_supported"`
	OpenaiChatCompletionsSupported          bool   `json:"openai_chat_completions_supported"`
	OpenaiToolCallingSupported              bool   `json:"openai_tool_calling_supported"`
	OpenaiTextInputSupported                bool   `json:"openai_text_input_supported"`
	NativeMultimodalInputSupported          bool   `json:"native_multimodal_input_supported"`
	Gemma4UnifiedMultimodalInputSupported   bool   `json:"gemma4_unified_multimodal_input_supported"`
	OpenaiTokenizedMultimodalInputSupported bool   `json:"openai_tokenized_multimodal_input_supported"`
	PrimaryUse                              string `json:"primary_use"`
	ChatDefault                             bool   `json:"chat_default"`
	CodingSupported                         bool   `json:"coding_supported"`
	CodingOnly                              bool   `json:"coding_only"`
}

// ModelsResponse is the response from GET /v1/models.
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelCard `json:"data"`
}
