package axengine

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"testing"
)

// startServer launches a temporary HTTP server, calls run with its base URL,
// then shuts it down.
func startServer(t *testing.T, mux *http.ServeMux, run func(baseURL string)) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln) //nolint:errcheck
	t.Cleanup(func() { srv.Close() })
	run(fmt.Sprintf("http://127.0.0.1:%d", ln.Addr().(*net.TCPAddr).Port))
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}

func TestHealthOK(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("method: got %s want GET", r.Method)
		}
		writeJSON(w, HealthResponse{Status: "ok", Service: "ax-engine-server", ModelID: "qwen3_dense"})
	})
	startServer(t, mux, func(baseURL string) {
		client := NewClient(&ClientOptions{BaseURL: baseURL})
		resp, err := client.Health(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if resp.Status != "ok" {
			t.Errorf("status: got %q want %q", resp.Status, "ok")
		}
		if resp.ModelID != "qwen3_dense" {
			t.Errorf("model_id: got %q want %q", resp.ModelID, "qwen3_dense")
		}
	})
}

func TestCompletionOK(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method: got %s want POST", r.Method)
		}
		var req OpenAiCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode body: %v", err)
		}
		writeJSON(w, OpenAiCompletionResponse{
			ID:     "cmpl-1",
			Object: "text_completion",
			Model:  "qwen3_dense",
			Choices: []OpenAiCompletionChoice{
				{Index: 0, Text: "Hello world", FinishReason: Ptr("stop")},
			},
			Usage: &OpenAiUsage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
		})
	})
	startServer(t, mux, func(baseURL string) {
		client := NewClient(&ClientOptions{BaseURL: baseURL})
		resp, err := client.Completion(context.Background(), OpenAiCompletionRequest{
			Prompt:    "Hello",
			MaxTokens: Ptr(32),
		})
		if err != nil {
			t.Fatal(err)
		}
		if resp.Object != "text_completion" {
			t.Errorf("object: got %q", resp.Object)
		}
		if resp.Choices[0].Text != "Hello world" {
			t.Errorf("text: got %q", resp.Choices[0].Text)
		}
		if resp.Usage.TotalTokens != 5 {
			t.Errorf("total_tokens: got %d want 5", resp.Usage.TotalTokens)
		}
	})
}

func TestChatCompletionOK(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req OpenAiChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode body: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Errorf("messages: got %d want 1", len(req.Messages))
		}
		writeJSON(w, OpenAiChatCompletionResponse{
			ID:     "chatcmpl-1",
			Object: "chat.completion",
			Model:  "qwen3_dense",
			Choices: []OpenAiChatCompletionChoice{
				{
					Index:        0,
					Message:      OpenAiChatMessageResponse{Role: "assistant", Content: "Hi there!"},
					FinishReason: Ptr("stop"),
				},
			},
			Usage: &OpenAiUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
		})
	})
	startServer(t, mux, func(baseURL string) {
		client := NewClient(&ClientOptions{BaseURL: baseURL})
		resp, err := client.ChatCompletion(context.Background(), OpenAiChatCompletionRequest{
			Messages:  []OpenAiChatMessage{{Role: "user", Content: "Hello!"}},
			MaxTokens: Ptr(64),
		})
		if err != nil {
			t.Fatal(err)
		}
		if resp.Choices[0].Message.Content != "Hi there!" {
			t.Errorf("content: got %q", resp.Choices[0].Message.Content)
		}
		if resp.Usage.CompletionTokens != 3 {
			t.Errorf("completion_tokens: got %d want 3", resp.Usage.CompletionTokens)
		}
	})
}

func TestHTTPErrorPropagated(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"bad request"}}`)
	})
	startServer(t, mux, func(baseURL string) {
		client := NewClient(&ClientOptions{BaseURL: baseURL})
		_, err := client.Completion(context.Background(), OpenAiCompletionRequest{Prompt: "x"})
		if err == nil {
			t.Fatal("expected error")
		}
		httpErr, ok := err.(*HTTPError)
		if !ok {
			t.Fatalf("expected *HTTPError, got %T", err)
		}
		if httpErr.Status != http.StatusBadRequest {
			t.Errorf("status: got %d want 400", httpErr.Status)
		}
		if httpErr.Message != "bad request" {
			t.Errorf("message: got %q want %q", httpErr.Message, "bad request")
		}
	})
}

func TestStreamChatCompletionOK(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("ResponseWriter does not implement http.Flusher")
			return
		}
		chunks := []string{
			`data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
			`data: {"id":"c1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}`,
			"data: [DONE]",
		}
		for _, chunk := range chunks {
			fmt.Fprintf(w, "%s\n\n", chunk)
			flusher.Flush()
		}
	})
	startServer(t, mux, func(baseURL string) {
		client := NewClient(&ClientOptions{BaseURL: baseURL})
		ch, errCh := client.StreamChatCompletion(context.Background(), OpenAiChatCompletionRequest{
			Messages:  []OpenAiChatMessage{{Role: "user", Content: "Hi"}},
			MaxTokens: Ptr(16),
		})

		var texts []string
		for chunk := range ch {
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != nil {
				texts = append(texts, *chunk.Choices[0].Delta.Content)
			}
		}
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}

		if len(texts) != 2 {
			t.Fatalf("chunks: got %d want 2", len(texts))
		}
		if texts[0] != "Hello" || texts[1] != " world" {
			t.Errorf("text: got %v", texts)
		}
	})
}

func TestStreamCompletionOK(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"text\":\"Once\",\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	})
	startServer(t, mux, func(baseURL string) {
		client := NewClient(&ClientOptions{BaseURL: baseURL})
		ch, errCh := client.StreamCompletion(context.Background(), OpenAiCompletionRequest{
			Prompt:    "Once upon",
			MaxTokens: Ptr(8),
		})
		var combined string
		for chunk := range ch {
			combined += chunk.Choices[0].Text
		}
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
		if combined != "Once" {
			t.Errorf("text: got %q want %q", combined, "Once")
		}
	})
}

func TestStreamGenerateOK(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/generate/stream", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method: got %s want POST", r.Method)
		}
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Errorf("accept: got %q want text/event-stream", r.Header.Get("Accept"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher := w.(http.Flusher)
		fmt.Fprint(w, "event: request\ndata: {\"request\":{\"request_id\":1,\"model_id\":\"qwen3_dense\",\"state\":\"active\",\"prompt_tokens\":[1,2,3],\"processed_prompt_tokens\":3,\"output_tokens\":[],\"prompt_len\":3,\"output_len\":0,\"max_output_tokens\":4,\"cancel_requested\":false,\"route\":{}}}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "event: step\ndata: {\"request\":{\"request_id\":1,\"model_id\":\"qwen3_dense\",\"state\":\"active\",\"prompt_tokens\":[1,2,3],\"processed_prompt_tokens\":3,\"output_tokens\":[42],\"prompt_len\":3,\"output_len\":1,\"max_output_tokens\":4,\"cancel_requested\":false,\"route\":{}},\"step\":{\"scheduled_requests\":1,\"scheduled_tokens\":1,\"ttft_events\":1,\"prefix_hits\":0,\"kv_usage_blocks\":1,\"evictions\":0,\"cpu_time_us\":100,\"runner_time_us\":200},\"delta_tokens\":[42],\"delta_text\":\"hello\"}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "event: response\ndata: {\"response\":{\"request_id\":1,\"model_id\":\"qwen3_dense\",\"prompt_tokens\":[1,2,3],\"output_tokens\":[42],\"status\":\"finished\",\"finish_reason\":\"stop\",\"step_count\":1,\"route\":{}}}\n\n")
		flusher.Flush()
	})
	startServer(t, mux, func(baseURL string) {
		client := NewClient(&ClientOptions{BaseURL: baseURL})
		ch, errCh := client.StreamGenerate(context.Background(), PreviewGenerateRequest{
			InputTokens:     []int{1, 2, 3},
			MaxOutputTokens: Ptr(4),
		})

		var events []GenerateStreamEvent
		for ev := range ch {
			events = append(events, ev)
		}
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}

		if len(events) != 3 {
			t.Fatalf("events: got %d want 3", len(events))
		}
		if events[0].Event != "request" || events[0].Request == nil {
			t.Errorf("event[0]: got event=%q request=%v", events[0].Event, events[0].Request)
		}
		if events[1].Event != "step" || events[1].Step == nil {
			t.Errorf("event[1]: got event=%q step=%v", events[1].Event, events[1].Step)
		}
		if events[1].Step != nil {
			step := events[1].Step
			if len(step.DeltaTokens) != 1 || step.DeltaTokens[0] != 42 {
				t.Errorf("step delta_tokens: got %v", step.DeltaTokens)
			}
			if step.DeltaText == nil || *step.DeltaText != "hello" {
				t.Errorf("step delta_text: got %v", step.DeltaText)
			}
		}
		if events[2].Event != "response" || events[2].Response == nil {
			t.Errorf("event[2]: got event=%q response=%v", events[2].Event, events[2].Response)
		}
		if events[2].Response != nil && events[2].Response.Response.FinishReason != nil {
			if *events[2].Response.Response.FinishReason != "stop" {
				t.Errorf("finish_reason: got %q want stop", *events[2].Response.Response.FinishReason)
			}
		}
	})
}

func TestContextCancellation(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// never writes — blocks until client disconnects
		<-r.Context().Done()
	})
	startServer(t, mux, func(baseURL string) {
		ctx, cancel := context.WithCancel(context.Background())
		client := NewClient(&ClientOptions{BaseURL: baseURL})
		ch, errCh := client.StreamChatCompletion(ctx, OpenAiChatCompletionRequest{
			Messages: []OpenAiChatMessage{{Role: "user", Content: "x"}},
		})
		cancel()
		// drain channels — should not block
		for range ch {
		}
		<-errCh
	})
}
