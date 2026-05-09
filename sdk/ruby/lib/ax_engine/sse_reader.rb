module AxEngine
  # Parses a server-sent events stream from a string buffer, yielding
  # { event:, data: } hashes for each complete event block.
  class SseReader
    DONE_SENTINEL = "[DONE]"

    def initialize
      @buffer = +""
      @event  = "message"
    end

    # Feed raw bytes into the buffer. Yields parsed event hashes.
    def feed(chunk)
      @buffer << chunk
      loop do
        boundary = @buffer.index(/\r?\n\r?\n/)
        break unless boundary

        # Find the exact separator length (\n\n or \r\n\r\n).
        sep_match = @buffer[boundary..].match(/\A(\r?\n){2}/)
        sep_len   = sep_match ? sep_match[0].length : 2

        block  = @buffer[0, boundary]
        @buffer = @buffer[(boundary + sep_len)..]

        event, data = parse_block(block)
        next if data.nil?
        next if data == DONE_SENTINEL

        parsed = begin
          JSON.parse(data)
        rescue JSON::ParserError
          data
        end

        yield({ "event" => event, "data" => parsed })
      end
    end

    private

    def parse_block(block)
      event     = "message"
      data_lines = []

      block.each_line do |line|
        line = line.chomp
        next if line.empty? || line.start_with?(":")
        if line.start_with?("event:")
          event = line[6..].strip
        elsif line.start_with?("data:")
          data_lines << line[5..].lstrip
        end
      end

      return [nil, nil] if data_lines.empty?

      [event, data_lines.join("\n")]
    end
  end
end
