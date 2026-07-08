const DEFAULT_BASE_URL = "http://127.0.0.1:8080";
const SSE_EVENT_FIELD = "event:";
const SSE_DATA_FIELD = "data:";

function trimTrailingSlash(value) {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

function normalizeHeaders(input) {
  const headers = new Headers();
  if (input == null) {
    return headers;
  }
  if (typeof input[Symbol.iterator] === "function") {
    for (const [key, value] of input) {
      if (value != null) {
        headers.append(key, value);
      }
    }
    return headers;
  }
  for (const [key, value] of Object.entries(input)) {
    if (value != null) {
      headers.append(key, value);
    }
  }
  return headers;
}

function toHeaders(baseHeaders, extraHeaders) {
  const headers = normalizeHeaders(baseHeaders);
  for (const [key, value] of normalizeHeaders(extraHeaders).entries()) {
    headers.set(key, value);
  }
  return headers;
}

async function readJsonSafely(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function ensureFetchImpl(fetchImpl) {
  if (fetchImpl) {
    return fetchImpl;
  }
  if (typeof fetch === "function") {
    return fetch.bind(globalThis);
  }
  throw new Error("A fetch implementation is required in this environment");
}

function isReadableStream(value) {
  return value && typeof value.getReader === "function";
}

function parseSseBlock(block) {
  const lines = block.split(/\r?\n/);
  let event = "message";
  const dataLines = [];
  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith(SSE_EVENT_FIELD)) {
      event = line.slice(SSE_EVENT_FIELD.length).trim();
      continue;
    }
    if (line.startsWith(SSE_DATA_FIELD)) {
      dataLines.push(line.slice(SSE_DATA_FIELD.length).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  return {
    event,
    data: dataLines.join("\n"),
  };
}

function decodeSseData(data) {
  if (data === "[DONE]") {
    return { done: true, data };
  }
  try {
    return { done: false, data: JSON.parse(data) };
  } catch {
    return { done: false, data };
  }
}

export class AxEngineHttpError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "AxEngineHttpError";
    this.status = options.status ?? 0;
    this.payload = options.payload ?? null;
  }
}

export class AxEngineStreamError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "AxEngineStreamError";
    this.payload = options.payload ?? null;
  }
}

function streamErrorFrom(data) {
  const message =
    data && typeof data === "object" && data.error && data.error.message
      ? data.error.message
      : typeof data === "string"
        ? data
        : "stream error";
  return new AxEngineStreamError(message, { payload: data });
}

export class AxEngineClient {
  constructor(options = {}) {
    const {
      baseUrl = DEFAULT_BASE_URL,
      fetch: fetchImpl,
      headers,
    } = options;
    this.baseUrl = trimTrailingSlash(baseUrl);
    this.fetch = ensureFetchImpl(fetchImpl);
    this.defaultHeaders = normalizeHeaders(headers);
  }

  async health() {
    return this.#requestJson("/health", { method: "GET" });
  }

  async runtime() {
    return this.#requestJson("/v1/runtime", { method: "GET" });
  }

  async models() {
    return this.#requestJson("/v1/models", { method: "GET" });
  }

  async generate(request) {
    return this.#requestJson("/v1/generate", {
      method: "POST",
      body: request,
    });
  }

  async submit(request) {
    return this.#requestJson("/v1/requests", {
      method: "POST",
      body: request,
    });
  }

  async requestSnapshot(requestId) {
    return this.#requestJson(`/v1/requests/${requestId}`, { method: "GET" });
  }

  async cancel(requestId) {
    return this.#requestJson(`/v1/requests/${requestId}/cancel`, {
      method: "POST",
    });
  }

  async step() {
    return this.#requestJson("/v1/step", { method: "POST" });
  }

  async completion(request) {
    return this.#requestJson("/v1/completions", {
      method: "POST",
      body: request,
    });
  }

  async chatCompletion(request) {
    return this.#requestJson("/v1/chat/completions", {
      method: "POST",
      body: request,
    });
  }

  async embeddings(request) {
    return this.#requestJson("/v1/embeddings", {
      method: "POST",
      body: request,
    });
  }

  async loadModel(request) {
    return this.#requestJson("/v1/model/load", {
      method: "POST",
      body: request,
    });
  }

  async *streamGenerate(request) {
    yield* this.#stream("/v1/generate/stream", request);
  }

  async *streamCompletion(request) {
    yield* this.#stream("/v1/completions", { ...request, stream: true });
  }

  async *streamChatCompletion(request) {
    yield* this.#stream("/v1/chat/completions", { ...request, stream: true });
  }

  async #requestJson(path, init) {
    const response = await this.#request(path, init);
    return response.json();
  }

  async #request(path, init = {}) {
    const headers = toHeaders(this.defaultHeaders, init.headers);
    let body = init.body;
    if (body !== undefined && body !== null && typeof body !== "string" && !(body instanceof Uint8Array)) {
      headers.set("content-type", "application/json");
      body = JSON.stringify(body);
    }

    const response = await this.fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers,
      body,
    });

    if (!response.ok) {
      const payload = await readJsonSafely(response);
      const message =
        payload && typeof payload === "object" && payload.error && payload.error.message
          ? payload.error.message
          : `HTTP ${response.status}`;
      throw new AxEngineHttpError(message, {
        status: response.status,
        payload,
      });
    }

    return response;
  }

  async *#stream(path, body) {
    const response = await this.#request(path, {
      method: "POST",
      body,
      headers: {
        accept: "text/event-stream",
      },
    });

    if (!isReadableStream(response.body)) {
      throw new Error("Streaming response body is not readable");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamEnded = false;

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          streamEnded = true;
          break;
        }
        buffer += decoder.decode(value, { stream: true });

        while (true) {
          const boundary = buffer.search(/\r?\n\r?\n/);
          if (boundary === -1) {
            break;
          }
          const block = buffer.slice(0, boundary);
          const separatorMatch = buffer.slice(boundary).match(/^\r?\n\r?\n/);
          const separatorLength = separatorMatch ? separatorMatch[0].length : 2;
          buffer = buffer.slice(boundary + separatorLength);

          const parsed = parseSseBlock(block);
          if (!parsed) {
            continue;
          }

          const decoded = decodeSseData(parsed.data);
          if (parsed.event === "error") {
            throw streamErrorFrom(decoded.data);
          }
          if (decoded.done) {
            return;
          }

          yield {
            event: parsed.event,
            data: decoded.data,
          };
        }
      }

      buffer += decoder.decode();
      const trailing = parseSseBlock(buffer.trim());
      if (trailing) {
        const decoded = decodeSseData(trailing.data);
        if (trailing.event === "error") {
          throw streamErrorFrom(decoded.data);
        }
        if (!decoded.done) {
          yield {
            event: trailing.event,
            data: decoded.data,
          };
        }
      }
    } finally {
      if (!streamEnded) {
        try {
          await reader.cancel();
        } catch {
          // The stream may already be closed by the server while the generator
          // is unwinding. The important part is that early consumer exits ask
          // the body to stop producing before releasing the reader lock.
        }
      }
      reader.releaseLock();
    }
  }
}

export default AxEngineClient;
