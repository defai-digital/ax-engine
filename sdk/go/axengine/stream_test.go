package axengine

import (
	"strings"
	"testing"
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
