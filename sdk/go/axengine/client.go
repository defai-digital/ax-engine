// Package axengine provides an HTTP client for the AX Engine v4 inference server.
//
// Connect to a running ax-engine-server instance:
//
//	client := axengine.NewClient(nil)
//
//	resp, err := client.ChatCompletion(ctx, axengine.OpenAiChatCompletionRequest{
//	    Messages: []axengine.OpenAiChatMessage{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	    MaxTokens: axengine.Ptr(256),
//	})
package axengine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const defaultBaseURL = "http://127.0.0.1:8080"

// ClientOptions configures an AxEngineClient.
type ClientOptions struct {
	// BaseURL is the root URL of the ax-engine-server instance.
	// Defaults to http://127.0.0.1:8080.
	BaseURL string

	// HTTPClient is the underlying http.Client. Defaults to http.DefaultClient.
	HTTPClient *http.Client

	// Headers are default headers added to every request.
	Headers http.Header
}

// Client is an HTTP client for ax-engine-server.
type Client struct {
	baseURL    string
	httpClient *http.Client
	headers    http.Header
}

// NewClient creates a new Client. Pass nil to use all defaults.
func NewClient(opts *ClientOptions) *Client {
	if opts == nil {
		opts = &ClientOptions{}
	}
	baseURL := strings.TrimRight(opts.BaseURL, "/")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &Client{
		baseURL:    baseURL,
		httpClient: httpClient,
		headers:    opts.Headers,
	}
}

// HTTPError is returned when the server responds with a non-2xx status.
type HTTPError struct {
	Status  int
	Message string
	Payload []byte
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("ax-engine: HTTP %d: %s", e.Status, e.Message)
}

func (c *Client) do(ctx context.Context, method, path string, body interface{}, extraHeaders http.Header) (*http.Response, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("ax-engine: marshal request: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("ax-engine: build request: %w", err)
	}

	for k, vals := range c.headers {
		for _, v := range vals {
			req.Header.Add(k, v)
		}
	}
	for k, vals := range extraHeaders {
		for _, v := range vals {
			req.Header.Add(k, v)
		}
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ax-engine: http: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		msg := fmt.Sprintf("HTTP %d", resp.StatusCode)
		var errBody struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(payload, &errBody) == nil && errBody.Error.Message != "" {
			msg = errBody.Error.Message
		}
		return nil, &HTTPError{Status: resp.StatusCode, Message: msg, Payload: payload}
	}

	return resp, nil
}

func requestJSON[T any](c *Client, ctx context.Context, method, path string, body interface{}) (T, error) {
	var zero T
	resp, err := c.do(ctx, method, path, body, nil)
	if err != nil {
		return zero, err
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(&zero); err != nil {
		return zero, fmt.Errorf("ax-engine: decode response: %w", err)
	}
	return zero, nil
}

// Health calls GET /health.
func (c *Client) Health(ctx context.Context) (HealthResponse, error) {
	return requestJSON[HealthResponse](c, ctx, http.MethodGet, "/health", nil)
}

// Generate calls POST /v1/generate (ax-engine native API).
func (c *Client) Generate(ctx context.Context, req PreviewGenerateRequest) (GenerateResponse, error) {
	return requestJSON[GenerateResponse](c, ctx, http.MethodPost, "/v1/generate", req)
}

// Submit calls POST /v1/requests to submit a request without blocking for completion.
func (c *Client) Submit(ctx context.Context, req PreviewGenerateRequest) (RequestReport, error) {
	return requestJSON[RequestReport](c, ctx, http.MethodPost, "/v1/requests", req)
}

// RequestSnapshot calls GET /v1/requests/{id} to fetch a request state snapshot.
func (c *Client) RequestSnapshot(ctx context.Context, requestID int64) (RequestReport, error) {
	return requestJSON[RequestReport](c, ctx, http.MethodGet, fmt.Sprintf("/v1/requests/%d", requestID), nil)
}

// Cancel calls POST /v1/requests/{id}/cancel.
func (c *Client) Cancel(ctx context.Context, requestID int64) (RequestReport, error) {
	return requestJSON[RequestReport](c, ctx, http.MethodPost, fmt.Sprintf("/v1/requests/%d/cancel", requestID), nil)
}

// Step calls POST /v1/step to advance the scheduler by one step.
func (c *Client) Step(ctx context.Context) (StepReport, error) {
	return requestJSON[StepReport](c, ctx, http.MethodPost, "/v1/step", nil)
}

// Completion calls POST /v1/completions (OpenAI-compat text completion).
func (c *Client) Completion(ctx context.Context, req OpenAiCompletionRequest) (OpenAiCompletionResponse, error) {
	return requestJSON[OpenAiCompletionResponse](c, ctx, http.MethodPost, "/v1/completions", req)
}

// ChatCompletion calls POST /v1/chat/completions (OpenAI-compat chat completion).
func (c *Client) ChatCompletion(ctx context.Context, req OpenAiChatCompletionRequest) (OpenAiChatCompletionResponse, error) {
	return requestJSON[OpenAiChatCompletionResponse](c, ctx, http.MethodPost, "/v1/chat/completions", req)
}

// Embeddings calls POST /v1/embeddings.
func (c *Client) Embeddings(ctx context.Context, req OpenAiEmbeddingRequest) (OpenAiEmbeddingResponse, error) {
	return requestJSON[OpenAiEmbeddingResponse](c, ctx, http.MethodPost, "/v1/embeddings", req)
}

// StreamCompletion streams POST /v1/completions with stream=true. The caller
// receives chunks over the returned channel; when the channel is closed the
// stream is done. The errCh channel delivers at most one error.
func (c *Client) StreamCompletion(ctx context.Context, req OpenAiCompletionRequest) (<-chan OpenAiCompletionChunk, <-chan error) {
	t := true
	req.Stream = &t
	ch := make(chan OpenAiCompletionChunk)
	errCh := make(chan error, 1)
	go func() {
		defer close(ch)
		defer close(errCh)
		if err := c.streamChunks(ctx, "/v1/completions", req, func(data string) error {
			v, done, err := decodeSSEData[OpenAiCompletionChunk](data)
			if err != nil || done {
				return err
			}
			select {
			case ch <- v:
			case <-ctx.Done():
				return ctx.Err()
			}
			return nil
		}); err != nil {
			errCh <- err
		}
	}()
	return ch, errCh
}

// StreamChatCompletion streams POST /v1/chat/completions with stream=true.
func (c *Client) StreamChatCompletion(ctx context.Context, req OpenAiChatCompletionRequest) (<-chan OpenAiChatCompletionChunk, <-chan error) {
	t := true
	req.Stream = &t
	ch := make(chan OpenAiChatCompletionChunk)
	errCh := make(chan error, 1)
	go func() {
		defer close(ch)
		defer close(errCh)
		if err := c.streamChunks(ctx, "/v1/chat/completions", req, func(data string) error {
			v, done, err := decodeSSEData[OpenAiChatCompletionChunk](data)
			if err != nil || done {
				return err
			}
			select {
			case ch <- v:
			case <-ctx.Done():
				return ctx.Err()
			}
			return nil
		}); err != nil {
			errCh <- err
		}
	}()
	return ch, errCh
}

// StreamGenerate streams POST /v1/generate/stream (ax-engine native SSE API).
// The channel delivers typed events: check the Event field ("request", "step",
// "response") and read the corresponding non-nil field on GenerateStreamEvent.
func (c *Client) StreamGenerate(ctx context.Context, req PreviewGenerateRequest) (<-chan GenerateStreamEvent, <-chan error) {
	ch := make(chan GenerateStreamEvent)
	errCh := make(chan error, 1)
	go func() {
		defer close(ch)
		defer close(errCh)
		if err := c.streamEvents(ctx, "/v1/generate/stream", req, func(ev *SSEEvent) error {
			var out GenerateStreamEvent
			out.Event = ev.Event
			switch ev.Event {
			case "request":
				var r GenerateStreamRequestEvent
				if err := json.Unmarshal([]byte(ev.Data), &r); err != nil {
					return fmt.Errorf("ax-engine: decode request event: %w", err)
				}
				out.Request = &r
			case "step":
				var s GenerateStreamStepEvent
				if err := json.Unmarshal([]byte(ev.Data), &s); err != nil {
					return fmt.Errorf("ax-engine: decode step event: %w", err)
				}
				out.Step = &s
			case "response":
				var r GenerateStreamResponseEvent
				if err := json.Unmarshal([]byte(ev.Data), &r); err != nil {
					return fmt.Errorf("ax-engine: decode response event: %w", err)
				}
				out.Response = &r
			}
			select {
			case ch <- out:
			case <-ctx.Done():
				return ctx.Err()
			}
			return nil
		}); err != nil {
			errCh <- err
		}
	}()
	return ch, errCh
}

func (c *Client) streamEvents(ctx context.Context, path string, body interface{}, handle func(*SSEEvent) error) error {
	extra := http.Header{"Accept": []string{"text/event-stream"}}
	resp, err := c.do(ctx, http.MethodPost, path, body, extra)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	reader := NewSSEReader(resp.Body)
	for {
		ev, ok := reader.Next()
		if !ok {
			break
		}
		if err := handle(ev); err != nil {
			return err
		}
	}
	return reader.Err()
}

func (c *Client) streamChunks(ctx context.Context, path string, body interface{}, handle func(string) error) error {
	return c.streamEvents(ctx, path, body, func(ev *SSEEvent) error {
		return handle(ev.Data)
	})
}

// Ptr is a convenience helper that returns a pointer to v.
func Ptr[T any](v T) *T {
	return &v
}
