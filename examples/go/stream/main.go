// Example: streaming chat completion with the AX Engine Go SDK.
//
// Requires a running ax-engine-server on http://127.0.0.1:8080.
//
// Run:
//
//	cd examples/go && go run ./stream
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/ax-engine/ax-engine-go"
)

func main() {
	client := axengine.NewClient(nil)

	ch, errCh := client.StreamChatCompletion(context.Background(), axengine.OpenAiChatCompletionRequest{
		Messages: []axengine.OpenAiChatMessage{
			{Role: "user", Content: "Count from 1 to 5."},
		},
		MaxTokens: axengine.Ptr(64),
	})

	for chunk := range ch {
		if len(chunk.Choices) > 0 {
			if text := chunk.Choices[0].Delta.Content; text != nil {
				fmt.Print(*text)
			}
		}
	}
	fmt.Println()

	if err := <-errCh; err != nil {
		log.Fatal(err)
	}
}
