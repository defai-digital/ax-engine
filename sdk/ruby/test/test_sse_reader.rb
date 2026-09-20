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

  def test_flush_discards_trailing_event_without_separator
    r = reader
    events = []
    # Feed an event without a trailing \n\n (simulates server closing connection).
    r.feed("event: response\ndata: {\"final\":true}") { |e| events << e }
    assert_equal 0, events.length  # not yet parsed
    r.flush { |e| events << e }
    assert_empty events
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

  def test_all_line_endings_with_bytewise_utf8_and_bom
    ["\r", "\r\n", "\n"].each do |ending|
      input = "\uFEFFevent: step#{ending}data: {\"text\":\"caf\u00E9\"}#{ending}#{ending}data: [DONE]#{ending}#{ending}"
      r = reader
      events = []
      input.bytes.each do |byte|
        # HTTP can split a Unicode scalar between chunks.
        r.feed(byte.chr.force_encoding(Encoding::UTF_8)) { |e| events << e }
      end
      assert_equal [{ "event" => "step", "data" => { "text" => "caf\u00E9" } }], events
      assert r.done?
    end
  end

  def test_mixed_line_endings_and_exact_field_values
    input = "event: error\r\r\nevent:  custom \ndata: first\r\ndata\rdata: third\n\r" \
      "event:\rdata: empty-name\r\rdata: [DONE]\n\n"
    r = reader
    assert_equal [
      { "event" => " custom ", "data" => "first\n\nthird" },
      { "event" => "message", "data" => "empty-name" }
    ], collect(r, input)
    assert r.done?
  end

  def test_cr_terminated_event_dispatches_immediately
    r = reader
    assert_equal [{ "event" => "message", "data" => { "ok" => true } }],
      collect(r, "data: {\"ok\":true}\r\r")
    assert_empty collect(r, "data: [DONE]\r\r")
    assert r.done?
  end

  def test_empty_data_field_dispatches_empty_string
    assert_equal [{ "event" => "message", "data" => "" }], collect(reader, "data\n\n")
  end


  def test_flush_discards_complete_lines_without_blank_line
    ["\r", "\r\n", "\n"].each do |ending|
      r = reader
      assert_empty collect(r, "event: error#{ending}data: stale#{ending}")
      r.flush
      assert_equal [{ "event" => "message", "data" => "fresh" }], collect(r, "data: fresh\n\n")
    end
  end


  def test_binary_chunks_decode_plain_text_as_utf8
    r = reader
    events = []
    "data: caf\u00E9\n\n".bytes.each do |byte|
      r.feed(byte.chr(Encoding::ASCII_8BIT)) { |event| events << event }
    end
    assert_equal [{ "event" => "message", "data" => "caf\u00E9" }], events
    assert_equal Encoding::UTF_8, events.first.fetch("data").encoding
  end

end
