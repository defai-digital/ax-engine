// LangChain Go example using AX Engine via langchaingo.
//
// langchaingo's OpenAI provider works directly with ax-engine-server because
// it exposes an OpenAI-compatible /v1/chat/completions endpoint.
//
// Requires:
//   go get github.com/tmc/langchaingo
//   ax-engine-server running on http://127.0.0.1:8080
//
// Run:
//   cd examples/go && go run ./langchain

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

func main() {
	// Point langchaingo's OpenAI provider at ax-engine-server.
	// ax-engine-server is OpenAI-compatible, so no custom adapter is needed.
	llm, err := openai.New(
		openai.WithBaseURL("http://127.0.0.1:8080/v1"),
		openai.WithToken("not-required"), // ax-engine does not check API keys
		openai.WithModel("qwen3_dense"),
	)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// ── Blocking chat ─────────────────────────────────────────────────────────
	resp, err := llms.GenerateFromSinglePrompt(ctx, llm, "Say hello in one sentence.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("chat:", resp)

	// ── Streaming chat ────────────────────────────────────────────────────────
	fmt.Print("\n--- streaming ---\n")
	_, err = llm.GenerateContent(
		ctx,
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "Count from 1 to 5."),
		},
		llms.WithMaxTokens(64),
		llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			fmt.Print(string(chunk))
			return nil
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}
