module github.com/ax-engine/sdk/go/grpc

go 1.24.0

require (
	github.com/ax-engine/sdk/go/grpc/ax_engine_v1 v0.0.0
	google.golang.org/grpc v1.79.3
)

require (
	golang.org/x/net v0.48.0 // indirect
	golang.org/x/sys v0.39.0 // indirect
	golang.org/x/text v0.32.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20251202230838-ff82c1b0f217 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

replace github.com/ax-engine/sdk/go/grpc/ax_engine_v1 => ./ax_engine_v1
