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
end
