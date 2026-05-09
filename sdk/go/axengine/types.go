package axengine

// GenerateSampling holds sampling parameters shared across generation requests.
type GenerateSampling struct {
	Temperature       *float64 `json:"temperature,omitempty"`
	TopP              *float64 `json:"top_p,omitempty"`
	TopK              *int     `json:"top_k,omitempty"`
	RepetitionPenalty *float64 `json:"repetition_penalty,omitempty"`
	Seed              *int64   `json:"seed,omitempty"`
}

// PreviewGenerateRequest is the ax-engine native generate request.
type PreviewGenerateRequest struct {
	Model           *string          `json:"model,omitempty"`
	InputTokens     []int            `json:"input_tokens,omitempty"`
	InputText       *string          `json:"input_text,omitempty"`
	MaxOutputTokens *int             `json:"max_output_tokens,omitempty"`
	Sampling        *GenerateSampling `json:"sampling,omitempty"`
	Metadata        *string          `json:"metadata,omitempty"`
}

// OpenAiUsage contains token usage information.
type OpenAiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// OpenAiCompletionRequest is the /v1/completions request body.
type OpenAiCompletionRequest struct {
	Model             *string      `json:"model,omitempty"`
	Prompt            interface{}  `json:"prompt"` // string | []string | []int
	MaxTokens         *int         `json:"max_tokens,omitempty"`
	Temperature       *float64     `json:"temperature,omitempty"`
	TopP              *float64     `json:"top_p,omitempty"`
	TopK              *int         `json:"top_k,omitempty"`
	MinP              *float64     `json:"min_p,omitempty"`
	RepetitionPenalty *float64     `json:"repetition_penalty,omitempty"`
	Stop              interface{}  `json:"stop,omitempty"` // string | []string
	Seed              *int64       `json:"seed,omitempty"`
	Stream            *bool        `json:"stream,omitempty"`
	Metadata          *string      `json:"metadata,omitempty"`
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
	Content interface{} `json:"content"` // string | []map[string]interface{}
}

// OpenAiChatCompletionRequest is the /v1/chat/completions request body.
type OpenAiChatCompletionRequest struct {
	Model             *string             `json:"model,omitempty"`
	Messages          []OpenAiChatMessage `json:"messages"`
	MaxTokens         *int                `json:"max_tokens,omitempty"`
	Temperature       *float64            `json:"temperature,omitempty"`
	TopP              *float64            `json:"top_p,omitempty"`
	TopK              *int                `json:"top_k,omitempty"`
	MinP              *float64            `json:"min_p,omitempty"`
	RepetitionPenalty *float64            `json:"repetition_penalty,omitempty"`
	Stop              interface{}         `json:"stop,omitempty"`
	Seed              *int64              `json:"seed,omitempty"`
	Stream            *bool               `json:"stream,omitempty"`
	Metadata          *string             `json:"metadata,omitempty"`
}

// OpenAiChatMessageResponse is the assistant message in a chat completion response.
type OpenAiChatMessageResponse struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
	Role    *string `json:"role,omitempty"`
	Content *string `json:"content,omitempty"`
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
	Model          *string  `json:"model,omitempty"`
	Input          []int    `json:"input"`
	EncodingFormat *string  `json:"encoding_format,omitempty"`
	Pooling        *string  `json:"pooling,omitempty"`
	Normalize      *bool    `json:"normalize,omitempty"`
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
	ExecutionPlan      *string             `json:"execution_plan,omitempty"`
	AttentionRoute     *string             `json:"attention_route,omitempty"`
	KvMode             *string             `json:"kv_mode,omitempty"`
	PrefixCachePath    *string             `json:"prefix_cache_path,omitempty"`
	BarrierMode        *string             `json:"barrier_mode,omitempty"`
	CrossoverDecisions map[string]float64  `json:"crossover_decisions,omitempty"`
}

// GenerateResponse is the response from /v1/generate.
type GenerateResponse struct {
	RequestID           int64          `json:"request_id"`
	ModelID             string         `json:"model_id"`
	PromptTokens        []int          `json:"prompt_tokens"`
	PromptText          *string        `json:"prompt_text,omitempty"`
	OutputTokens        []int          `json:"output_tokens"`
	OutputTokenLogprobs []*float64     `json:"output_token_logprobs,omitempty"`
	OutputText          *string        `json:"output_text,omitempty"`
	Status              string         `json:"status"`
	FinishReason        *string        `json:"finish_reason,omitempty"`
	StepCount           int            `json:"step_count"`
	TtftStep            *int           `json:"ttft_step,omitempty"`
	Route               GenerateRoute  `json:"route"`
}

// RequestReport is a snapshot of a live or completed request.
type RequestReport struct {
	RequestID               int64         `json:"request_id"`
	ModelID                 string        `json:"model_id"`
	State                   string        `json:"state"`
	PromptTokens            []int         `json:"prompt_tokens"`
	ProcessedPromptTokens   int           `json:"processed_prompt_tokens"`
	OutputTokens            []int         `json:"output_tokens"`
	OutputTokenLogprobs     []*float64    `json:"output_token_logprobs,omitempty"`
	PromptLen               int           `json:"prompt_len"`
	OutputLen               int           `json:"output_len"`
	MaxOutputTokens         int           `json:"max_output_tokens"`
	CancelRequested         bool          `json:"cancel_requested"`
	ExecutionPlanRef        *string       `json:"execution_plan_ref,omitempty"`
	Route                   GenerateRoute `json:"route"`
	FinishReason            *string       `json:"finish_reason,omitempty"`
	TerminalStopReason      *string       `json:"terminal_stop_reason,omitempty"`
}

// StepReport is the result of a /v1/step call.
type StepReport struct {
	StepID             *int64        `json:"step_id,omitempty"`
	ScheduledRequests  int           `json:"scheduled_requests"`
	ScheduledTokens    int           `json:"scheduled_tokens"`
	TtftEvents         int           `json:"ttft_events"`
	PrefixHits         int           `json:"prefix_hits"`
	KvUsageBlocks      int           `json:"kv_usage_blocks"`
	Evictions          int           `json:"evictions"`
	CpuTimeUs          int64         `json:"cpu_time_us"`
	RunnerTimeUs       int64         `json:"runner_time_us"`
	Route              *GenerateRoute `json:"route,omitempty"`
}

// HealthResponse is the response from /health.
type HealthResponse struct {
	Status  string `json:"status"`
	Service string `json:"service"`
	ModelID string `json:"model_id"`
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
