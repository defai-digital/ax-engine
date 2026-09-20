package axengine

import (
	"bufio"
	"bytes"
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
	scanner   *bufio.Scanner
	event     string
	dataBuf   strings.Builder
	firstLine bool
}

// maxSSELineSize bounds a single SSE line. The native /v1/generate/stream
// response event carries the full response JSON (all tokens plus text) on one
// data: line, which easily exceeds bufio.Scanner's 64KiB default.
const maxSSELineSize = 16 * 1024 * 1024

// NewSSEReader creates a new SSEReader wrapping r.
func NewSSEReader(r io.Reader) *SSEReader {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), maxSSELineSize)
	// Consume CR immediately, then skip an optional LF on the next scan.
	// Waiting for the byte after CR would stall a complete CR-ended event.
	skipLF := false
	scanner.Split(func(data []byte, atEOF bool) (int, []byte, error) {
		offset := 0
		if skipLF && len(data) > 0 {
			skipLF = false
			if data[0] == '\n' {
				offset = 1
				data = data[1:]
			}
		}
		if i := bytes.IndexAny(data, "\r\n"); i >= 0 {
			skipLF = data[i] == '\r'
			return offset + i + 1, data[:i], nil
		}
		if atEOF {
			return offset + len(data), nil, nil
		}
		return offset, nil, nil
	})
	return &SSEReader{scanner: scanner, event: "message", firstLine: true}
}

// Next advances to the next event. Returns (event, true) when an event is
// available, or ("", false) when the stream is exhausted.
func (s *SSEReader) Next() (*SSEEvent, bool) {
	for s.scanner.Scan() {
		line := s.scanner.Text()
		if s.firstLine {
			line = strings.TrimPrefix(line, "\ufeff")
			s.firstLine = false
		}

		if line == "" {
			if s.dataBuf.Len() == 0 {
				// An `event:` line with no `data:` dispatches nothing, but
				// the event type still resets per the SSE spec — otherwise
				// the stale name leaks into the next event.
				s.event = "message"
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
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			s.event = value
			if s.event == "" {
				s.event = "message"
			}
		case "data":
			s.dataBuf.WriteString(value)
			s.dataBuf.WriteByte('\n')
		}
	}

	// EOF never dispatches a partial event, including a partial [DONE].
	s.dataBuf.Reset()
	s.event = "message"

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
