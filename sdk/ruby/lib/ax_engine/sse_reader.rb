module AxEngine
  # Parses a server-sent events stream from raw bytes, yielding
  # { event:, data: } hashes for each blank-line-terminated event.
  class SseReader
    DONE_SENTINEL = "[DONE]"

    def initialize
      @buffer = +"".b
      @done = false
      @skip_lf = false
      @first_line = true
      @event = "message"
      @data_lines = []
    end

    def done?
      @done
    end

    # Decode only complete lines so a chunk may split any UTF-8 scalar.
    def feed(chunk)
      return if @done

      @buffer << chunk.b
      until @buffer.empty?
        if @skip_lf
          @skip_lf = false
          @buffer = @buffer.byteslice(1..) if @buffer.start_with?("\n")
        end
        boundary = @buffer.index(/[\r\n]/)
        break unless boundary

        # CR completes a line immediately; the optional LF may arrive later.
        @skip_lf = @buffer.getbyte(boundary) == 13
        line = @buffer.byteslice(0, boundary).force_encoding(Encoding::UTF_8).scrub
        @buffer = @buffer.byteslice((boundary + 1)..)
        if @first_line
          line = line.delete_prefix("\uFEFF")
          @first_line = false
        end
        event = process_line(line)
        if @done
          @buffer.clear
          return
        end
        yield event if event
      end
    end

    # EOF discards all partial event state without dispatching it.
    def flush(&_block)
      @buffer.clear
      @data_lines.clear
      @event = "message"
      @skip_lf = false
    end

    private

    def process_line(line)
      if line.empty?
        event = @event.empty? ? "message" : @event
        @event = "message"
        return if @data_lines.empty?

        data = @data_lines.join("\n")
        @data_lines.clear
        if data == DONE_SENTINEL && event != "error"
          @done = true
          return
        end
        parsed = begin
          JSON.parse(data)
        rescue JSON::ParserError
          data
        end
        return { "event" => event, "data" => parsed }
      end
      return if line.start_with?(":")

      field, _, value = line.partition(":")
      value = value.delete_prefix(" ")
      @event = value if field == "event"
      @data_lines << value if field == "data"
      nil
    end
  end
end
