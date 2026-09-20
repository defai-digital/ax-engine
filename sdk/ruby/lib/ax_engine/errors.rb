module AxEngine
  # Raised when the server returns a non-2xx HTTP response.
  class HttpError < StandardError
    attr_reader :status, :payload

    def initialize(message, status: 0, payload: nil)
      super(message)
      @status  = status
      @payload = payload
    end
  end

  # Raised for a server `error` event or a truncated stream.
  class StreamError < StandardError
    attr_reader :payload

    def initialize(message, payload: nil)
      super(message)
      @payload = payload
    end
  end
end
