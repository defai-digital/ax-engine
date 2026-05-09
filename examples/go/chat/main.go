// Example: chat completion with the AX Engine Go SDK.
//
// Requires a running ax-engine-server on http://127.0.0.1:8080.
//
// Run:
//
//	cd examples/go && go run ./chat
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/ax-engine/ax-engine-go"
)

func main() {
	client := axengine.NewClient(nil)

	resp, err := client.ChatCompletion(context.Background(), axengine.OpenAiChatCompletionRequest{
		Messages: []axengine.OpenAiChatMessage{
			{Role: "system", Content: "You are AX Engine."},
			{Role: "user", Content: "Say hello in one sentence."},
		},
		MaxTokens:   axengine.Ptr(64),
		Temperature: axengine.Ptr(0.7),
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Choices[0].Message.Content)
}
