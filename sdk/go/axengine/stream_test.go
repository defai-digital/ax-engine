package axengine

import (
	"io"
	"strings"
	"testing"
	"testing/iotest"
)

func TestSSEReaderBasic(t *testing.T) {
	input := "event: step\ndata: {\"a\":1}\n\nevent: response\ndata: {\"b\":2}\n\n"
	r := NewSSEReader(strings.NewReader(input))

	ev, ok := r.Next()
	if !ok {
		t.Fatal("expected first event")
	}
	if ev.Event != "step" {
		t.Errorf("event: got %q want %q", ev.Event, "step")
	}
	if ev.Data != `{"a":1}` {
		t.Errorf("data: got %q want %q", ev.Data, `{"a":1}`)
	}

	ev, ok = r.Next()
	if !ok {
		t.Fatal("expected second event")
	}
	if ev.Event != "response" {
		t.Errorf("event: got %q want %q", ev.Event, "response")
	}
	if ev.Data != `{"b":2}` {
		t.Errorf("data: got %q want %q", ev.Data, `{"b":2}`)
	}

	_, ok = r.Next()
	if ok {
		t.Error("expected stream exhausted")
	}
	if err := r.Err(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestSSEReaderDefaultEvent(t *testing.T) {
	input := "data: hello\n\n"
	r := NewSSEReader(strings.NewReader(input))
	ev, ok := r.Next()
	if !ok {
		t.Fatal("expected event")
	}
	if ev.Event != "message" {
		t.Errorf("default event: got %q want %q", ev.Event, "message")
	}
	if ev.Data != "hello" {
		t.Errorf("data: got %q want %q", ev.Data, "hello")
	}
}

func TestSSEReaderDataLessEventResetsEventType(t *testing.T) {
	// An `event:` with no `data:` dispatches nothing, but the event type
	// must reset per the SSE spec — the next data event is "message",
	// not the stale "ping".
	input := "event: ping\n\ndata: real\n\n"
	r := NewSSEReader(strings.NewReader(input))
	ev, ok := r.Next()
	if !ok {
		t.Fatal("expected event")
	}
	if ev.Event != "message" {
		t.Errorf("event after data-less reset: got %q want %q", ev.Event, "message")
	}
	if ev.Data != "real" {
		t.Errorf("data: got %q want %q", ev.Data, "real")
	}
}

func TestSSEReaderSkipsComments(t *testing.T) {
	input := ": keep-alive\n\ndata: real\n\n"
	r := NewSSEReader(strings.NewReader(input))

	ev, ok := r.Next()
	if !ok {
		t.Fatal("expected event")
	}
	if ev.Data != "real" {
		t.Errorf("data: got %q want %q", ev.Data, "real")
	}
}

func TestSSEReaderMultilineData(t *testing.T) {
	input := "data: line1\ndata: line2\n\n"
	r := NewSSEReader(strings.NewReader(input))
	ev, ok := r.Next()
	if !ok {
		t.Fatal("expected event")
	}
	if ev.Data != "line1\nline2" {
		t.Errorf("multiline data: got %q", ev.Data)
	}
}

func TestSSEReaderCRLF(t *testing.T) {
	input := "event: ping\r\ndata: ok\r\n\r\n"
	r := NewSSEReader(strings.NewReader(input))
	ev, ok := r.Next()
	if !ok {
		t.Fatal("expected event")
	}
	if ev.Event != "ping" || ev.Data != "ok" {
		t.Errorf("CRLF: event=%q data=%q", ev.Event, ev.Data)
	}
}

func TestDecodeSSEDataDone(t *testing.T) {
	type T struct{ X int }
	_, done, err := decodeSSEData[T]("[DONE]")
	if err != nil {
		t.Fatal(err)
	}
	if !done {
		t.Error("expected done=true for [DONE]")
	}
}

func TestDecodeSSEDataJSON(t *testing.T) {
	type T struct{ X int }
	v, done, err := decodeSSEData[T](`{"X":42}`)
	if err != nil {
		t.Fatal(err)
	}
	if done {
		t.Error("expected done=false")
	}
	if v.X != 42 {
		t.Errorf("X: got %d want 42", v.X)
	}
}

func TestDecodeSSEDataBadJSON(t *testing.T) {
	type T struct{ X int }
	_, _, err := decodeSSEData[T]("not-json")
	if err == nil {
		t.Error("expected error for bad JSON")
	}
}

func TestPtrHelper(t *testing.T) {
	v := 42
	p := Ptr(v)
	if *p != 42 {
		t.Errorf("Ptr: got %d want 42", *p)
	}
	s := "hello"
	ps := Ptr(s)
	if *ps != "hello" {
		t.Errorf("Ptr[string]: got %q want %q", *ps, "hello")
	}
}

func TestSSEReaderFraming(t *testing.T) {
	cases := []struct {
		name, input, event, data string
	}{
		{"bare carriage return", "event: step\rdata: hello\r\r", "step", "hello"},
		{"mixed line endings", "event: step\r\ndata: one\rdata: two\n\r\n", "step", "one\ntwo"},
		{"one optional space", "data:   indented\n\n", "message", "  indented"},
		{"literal event name", "event:  custom \ndata: x\n\n", " custom ", "x"},
		{"empty event defaults", "event:\ndata: x\n\n", "message", "x"},
		{"colonless data", "data\n\n", "message", ""},
		{"leading BOM", "\ufeffdata: x\n\n", "message", "x"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, source := range []io.Reader{strings.NewReader(tc.input), iotest.OneByteReader(strings.NewReader(tc.input))} {
				r := NewSSEReader(source)
				ev, ok := r.Next()
				if !ok || ev.Event != tc.event || ev.Data != tc.data {
					t.Fatalf("got %#v, %v; want event %q data %q", ev, ok, tc.event, tc.data)
				}
				if ev, ok := r.Next(); ok || r.Err() != nil {
					t.Fatalf("unexpected trailing event %#v or error %v", ev, r.Err())
				}
			}
		})
	}
}

func TestSSEReaderDiscardsIncompleteEvents(t *testing.T) {
	for _, input := range []string{"data: [DONE]", "data: [DONE]\n", "data: first\ndata: partial", "data: first\n"} {
		r := NewSSEReader(strings.NewReader(input))
		if ev, ok := r.Next(); ok {
			t.Errorf("incomplete input %q dispatched %#v", input, ev)
		}
	}
}
