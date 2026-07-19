// Package grpc provides a gRPC client for ax-engine-server.
//
// Usage:
//
//	client, err := axgrpc.New("127.0.0.1:50051")
//	if err != nil { ... }
//	defer client.Close()
//
//	resp, err := client.Health(ctx)
//	stream, err := client.StreamGenerate(ctx, &pb.GenerateRequest{...})
package grpc

import (
	"context"
	"crypto/tls"
	"io"
	"os"
	"strings"

	pb "github.com/ax-engine/sdk/go/grpc/ax_engine_v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// Client is a high-level gRPC client for ax-engine-server.
type Client struct {
	conn *grpc.ClientConn
	svc  pb.AxEngineClient
}

// New dials the gRPC server at addr (e.g. "127.0.0.1:50051") and returns a Client.
//
// Transport selection:
//   - Explicit dial options always win (callers can override credentials).
//   - When AX_ENGINE_TLS is set to a truthy value (1/true/yes), or addr looks
//     like a TLS endpoint (scheme https://, or host:port with AX_ENGINE_TLS),
//     the default is TLS with the system root CAs.
//   - Otherwise the default is plaintext (suitable for local loopback).
//
// Pass grpc.WithTransportCredentials(...) to force a specific credential set.
func New(addr string, opts ...grpc.DialOption) (*Client, error) {
	target, useTLS := resolveDialTarget(addr)
	defaults := []grpc.DialOption{defaultTransportCredentials(useTLS)}
	conn, err := grpc.NewClient(target, append(defaults, opts...)...)
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, svc: pb.NewAxEngineClient(conn)}, nil
}

// resolveDialTarget strips optional scheme prefixes and decides whether TLS
// should be the default transport for this address.
func resolveDialTarget(addr string) (target string, useTLS bool) {
	trimmed := strings.TrimSpace(addr)
	lower := strings.ToLower(trimmed)
	switch {
	case strings.HasPrefix(lower, "https://"):
		return strings.TrimPrefix(trimmed[len("https://"):], "//"), true
	case strings.HasPrefix(lower, "http://"):
		return strings.TrimPrefix(trimmed[len("http://"):], "//"), false
	case strings.HasPrefix(lower, "grpcs://"):
		return strings.TrimPrefix(trimmed[len("grpcs://"):], "//"), true
	case strings.HasPrefix(lower, "grpc://"):
		return strings.TrimPrefix(trimmed[len("grpc://"):], "//"), false
	default:
		return trimmed, envRequestsTLS()
	}
}

func envRequestsTLS() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("AX_ENGINE_TLS")))
	switch v {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func defaultTransportCredentials(useTLS bool) grpc.DialOption {
	if useTLS {
		// System root CAs; server name is taken from the dial target host.
		// Callers that need custom CA bundles should pass their own
		// TransportCredentials via opts.
		return grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{
			MinVersion: tls.VersionTLS12,
		}))
	}
	return grpc.WithTransportCredentials(insecure.NewCredentials())
}

// Close releases the underlying gRPC connection.
func (c *Client) Close() error { return c.conn.Close() }

// Health checks the server health.
func (c *Client) Health(ctx context.Context) (*pb.HealthResponse, error) {
	return c.svc.Health(ctx, &pb.HealthRequest{})
}

// Models lists available models.
func (c *Client) Models(ctx context.Context) (*pb.ModelsResponse, error) {
	return c.svc.Models(ctx, &pb.ModelsRequest{})
}

// Generate runs a native generate request (unary).
func (c *Client) Generate(ctx context.Context, req *pb.GenerateRequest) (*pb.GenerateResponse, error) {
	return c.svc.Generate(ctx, req)
}

// StreamGenerate starts a server-streaming generate and returns a channel of events.
// The errCh receives at most one error; both channels are closed when the stream ends.
func (c *Client) StreamGenerate(
	ctx context.Context,
	req *pb.GenerateRequest,
) (<-chan *pb.GenerateStreamEvent, <-chan error) {
	ch := make(chan *pb.GenerateStreamEvent)
	errCh := make(chan error, 1)

	go func() {
		defer close(ch)
		defer close(errCh)

		stream, err := c.svc.StreamGenerate(ctx, req)
		if err != nil {
			errCh <- err
			return
		}
		for {
			event, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				errCh <- err
				return
			}
			select {
			case ch <- event:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}
	}()

	return ch, errCh
}

// ChatCompletion sends a chat completion request (unary).
func (c *Client) ChatCompletion(ctx context.Context, req *pb.ChatCompletionRequest) (*pb.ChatCompletionResponse, error) {
	return c.svc.ChatCompletion(ctx, req)
}

// StreamChatCompletion starts a streaming chat completion and returns a channel of chunks.
func (c *Client) StreamChatCompletion(
	ctx context.Context,
	req *pb.ChatCompletionRequest,
) (<-chan *pb.ChatCompletionChunk, <-chan error) {
	ch := make(chan *pb.ChatCompletionChunk)
	errCh := make(chan error, 1)

	go func() {
		defer close(ch)
		defer close(errCh)

		stream, err := c.svc.StreamChatCompletion(ctx, req)
		if err != nil {
			errCh <- err
			return
		}
		for {
			chunk, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				errCh <- err
				return
			}
			select {
			case ch <- chunk:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}
	}()

	return ch, errCh
}

// Completion sends a text completion request (unary).
func (c *Client) Completion(ctx context.Context, req *pb.CompletionRequest) (*pb.CompletionResponse, error) {
	return c.svc.Completion(ctx, req)
}

// StreamCompletion starts a streaming text completion and returns a channel of chunks.
func (c *Client) StreamCompletion(
	ctx context.Context,
	req *pb.CompletionRequest,
) (<-chan *pb.CompletionChunk, <-chan error) {
	ch := make(chan *pb.CompletionChunk)
	errCh := make(chan error, 1)

	go func() {
		defer close(ch)
		defer close(errCh)

		stream, err := c.svc.StreamCompletion(ctx, req)
		if err != nil {
			errCh <- err
			return
		}
		for {
			chunk, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				errCh <- err
				return
			}
			select {
			case ch <- chunk:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}
	}()

	return ch, errCh
}

// Embeddings computes embeddings for tokenized input (unary).
func (c *Client) Embeddings(ctx context.Context, req *pb.EmbeddingsRequest) (*pb.EmbeddingsResponse, error) {
	return c.svc.Embeddings(ctx, req)
}
