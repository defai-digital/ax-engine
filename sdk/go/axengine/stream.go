package axengine

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// SSEEvent is a parsed server-sent event.
type SSEEvent struct {
	Event string
	Data  string
}

// SSEReader reads SSE events from an io.Reader.
type SSEReader struct {
	scanner *bufio.Scanner
	event   string
	dataBuf strings.Builder
}

// maxSSELineSize bounds a single SSE line. The native /v1/generate/stream
// response event carries the full response JSON (all tokens plus text) on one
// data: line, which easily exceeds bufio.Scanner's 64KiB default.
const maxSSELineSize = 16 * 1024 * 1024

// NewSSEReader creates a new SSEReader wrapping r.
func NewSSEReader(r io.Reader) *SSEReader {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), maxSSELineSize)
	return &SSEReader{scanner: scanner, event: "message"}
}

// Next advances to the next event. Returns (event, true) when an event is
// available, or ("", false) when the stream is exhausted.
func (s *SSEReader) Next() (*SSEEvent, bool) {
	for s.scanner.Scan() {
		line := s.scanner.Text()

		if line == "" {
			if s.dataBuf.Len() == 0 {
				continue
			}
			data := strings.TrimSuffix(s.dataBuf.String(), "\n")
			ev := &SSEEvent{Event: s.event, Data: data}
			s.event = "message"
			s.dataBuf.Reset()
			return ev, true
		}

		if strings.HasPrefix(line, ":") {
			continue
		}
		if after, ok := strings.CutPrefix(line, "event:"); ok {
			s.event = strings.TrimSpace(after)
			continue
		}
		if after, ok := strings.CutPrefix(line, "data:"); ok {
			s.dataBuf.WriteString(strings.TrimLeft(after, " "))
			s.dataBuf.WriteByte('\n')
		}
	}

	if s.dataBuf.Len() > 0 {
		data := strings.TrimSuffix(s.dataBuf.String(), "\n")
		ev := &SSEEvent{Event: s.event, Data: data}
		s.dataBuf.Reset()
		return ev, true
	}

	return nil, false
}

// Err returns any scanner error.
func (s *SSEReader) Err() error {
	return s.scanner.Err()
}

// decodeSSEData parses the data field of an SSE event.
// Returns (data, done, error). done=true means the stream ended with [DONE].
func decodeSSEData[T any](data string) (T, bool, error) {
	var zero T
	if data == "[DONE]" {
		return zero, true, nil
	}
	var v T
	if err := json.Unmarshal([]byte(data), &v); err != nil {
		return zero, false, fmt.Errorf("ax-engine: decode SSE data: %w", err)
	}
	return v, false, nil
}
