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
	"io"

	pb "github.com/ax-engine/sdk/go/grpc/ax_engine_v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client is a high-level gRPC client for ax-engine-server.
type Client struct {
	conn   *grpc.ClientConn
	svc    pb.AxEngineClient
}

// New dials the gRPC server at addr (e.g. "127.0.0.1:50051") and returns a Client.
// The connection uses plain-text (no TLS); wrap with grpc.WithTransportCredentials
// for production use.
func New(addr string, opts ...grpc.DialOption) (*Client, error) {
	defaults := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	conn, err := grpc.NewClient(addr, append(defaults, opts...)...)
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, svc: pb.NewAxEngineClient(conn)}, nil
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
