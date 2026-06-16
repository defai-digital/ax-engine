require "minitest/autorun"
require_relative "../lib/ax_engine"

class TestSseReader < Minitest::Test
  def reader
    AxEngine::SseReader.new
  end

  def collect(r, input)
    events = []
    r.feed(input) { |e| events << e }
    events
  end

  def test_basic_named_event
    input = "event: step\ndata: {\"a\":1}\n\n"
    events = collect(reader, input)
    assert_equal 1, events.length
    assert_equal "step", events[0]["event"]
    assert_equal({ "a" => 1 }, events[0]["data"])
  end

  def test_default_event_name
    input = "data: {\"x\":2}\n\n"
    events = collect(reader, input)
    assert_equal "message", events[0]["event"]
    assert_equal({ "x" => 2 }, events[0]["data"])
  end

  def test_multiple_events
    input = "event: request\ndata: {\"id\":1}\n\nevent: response\ndata: {\"id\":2}\n\n"
    events = collect(reader, input)
    assert_equal 2, events.length
    assert_equal "request",  events[0]["event"]
    assert_equal "response", events[1]["event"]
  end

  def test_done_sentinel_skipped
    input = "data: {\"ok\":true}\n\ndata: [DONE]\n\n"
    events = collect(reader, input)
    assert_equal 1, events.length
    assert_equal({ "ok" => true }, events[0]["data"])
  end

  def test_comment_lines_skipped
    input = ": keep-alive\n\ndata: {\"real\":1}\n\n"
    events = collect(reader, input)
    assert_equal 1, events.length
    assert_equal({ "real" => 1 }, events[0]["data"])
  end

  def test_crlf_separator
    input = "event: ping\r\ndata: {\"ok\":true}\r\n\r\n"
    events = collect(reader, input)
    assert_equal 1, events.length
    assert_equal "ping", events[0]["event"]
  end

  def test_chunked_delivery
    r = reader
    events = []
    # Feed the stream in two pieces, simulating chunked HTTP delivery.
    r.feed("event: step\ndata: {\"") { |e| events << e }
    r.feed("v\":1}\n\n")             { |e| events << e }
    assert_equal 1, events.length
    assert_equal({ "v" => 1 }, events[0]["data"])
  end

  def test_invalid_json_passed_as_string
    input = "data: not-json\n\n"
    events = collect(reader, input)
    assert_equal 1, events.length
    assert_equal "not-json", events[0]["data"]
  end

  def test_flush_yields_trailing_event_without_separator
    r = reader
    events = []
    # Feed an event without a trailing \n\n (simulates server closing connection).
    r.feed("event: response\ndata: {\"final\":true}") { |e| events << e }
    assert_equal 0, events.length  # not yet parsed
    r.flush { |e| events << e }
    assert_equal 1, events.length
    assert_equal "response", events[0]["event"]
    assert_equal({ "final" => true }, events[0]["data"])
  end

  def test_flush_empty_buffer_yields_nothing
    r = reader
    events = []
    r.flush { |e| events << e }
    assert_equal 0, events.length
  end

  def test_flush_done_sentinel_skipped
    r = reader
    events = []
    r.feed("data: [DONE]") { |e| events << e }
    r.flush { |e| events << e }
    assert_equal 0, events.length
  end
end
