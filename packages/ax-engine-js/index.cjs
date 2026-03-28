"use strict";

class AxEngineError extends Error {
  constructor(message) {
    super(message);
    this.name = this.constructor.name;
  }
}

class AxEngineHttpError extends AxEngineError {
  constructor(message, details) {
    super(message);
    this.status = details.status;
    this.statusText = details.statusText;
    this.body = details.body;
    this.headers = details.headers;
  }
}

class AxEngineStreamError extends AxEngineError {
  constructor(message, payload) {
    super(message);
    this.payload = payload;
  }
}

class AxEngineClient {
  constructor(options = {}) {
    this.baseURL = normalizeBaseURL(options.baseURL ?? "http://127.0.0.1:3000");
    this.defaultModel = options.defaultModel ?? null;
    this.apiKey = options.apiKey ?? null;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    this.defaultHeaders = {
      ...(options.headers ?? {}),
    };

    if (typeof this.fetchImpl !== "function") {
      throw new AxEngineError(
        "fetch is not available; pass options.fetch or run on Node.js 18+ / Next.js runtime",
      );
    }

    this.models = {
      list: (options) => this.requestJson("GET", "/v1/models", undefined, options),
    };

    this.completions = {
      create: (request, options) =>
        this.requestJson(
          "POST",
          "/v1/completions",
          withStreamFlag(this.applyDefaultModel(request), false),
          options,
        ),
      stream: (request, options) =>
        streamJson(
          this,
          "/v1/completions",
          withStreamFlag(this.applyDefaultModel(request), true),
          options,
        ),
      streamText: (request, options) =>
        mapAsyncIterable(
          streamJson(
            this,
            "/v1/completions",
            withStreamFlag(this.applyDefaultModel(request), true),
            options,
          ),
          completionChunkText,
        ),
    };

    this.chat = {
      completions: {
        create: (request, options) =>
          this.requestJson(
            "POST",
            "/v1/chat/completions",
            withStreamFlag(this.applyDefaultModel(request), false),
            options,
          ),
        stream: (request, options) =>
          streamJson(
            this,
            "/v1/chat/completions",
            withStreamFlag(this.applyDefaultModel(request), true),
            options,
          ),
        streamText: (request, options) =>
          mapAsyncIterable(
            streamJson(
              this,
              "/v1/chat/completions",
              withStreamFlag(this.applyDefaultModel(request), true),
              options,
            ),
            chatCompletionChunkText,
          ),
      },
    };

    this.responses = {
      create: async (request, options) => {
        const response = await this.chat.completions.create(
          responseRequestToChatCompletionRequest(this.applyDefaultModel(request), false),
          options,
        );
        return responseFromChatCompletion(response);
      },
      stream: (request, options) =>
        responseStream(
          this,
          responseRequestToChatCompletionRequest(this.applyDefaultModel(request), true),
          options,
        ),
      streamText: (request, options) =>
        mapAsyncIterable(
          responseStream(
            this,
            responseRequestToChatCompletionRequest(this.applyDefaultModel(request), true),
            options,
          ),
          responseStreamEventText,
        ),
    };
  }

  async health(options) {
    return this.requestJson("GET", "/healthz", undefined, options);
  }

  applyDefaultModel(request) {
    if (!request || typeof request !== "object" || Array.isArray(request)) {
      return request;
    }

    if (request.model || !this.defaultModel) {
      return { ...request };
    }

    return {
      ...request,
      model: this.defaultModel,
    };
  }

  async requestJson(method, path, body, options = {}) {
    const response = await this.request(method, path, body, options);
    return parseJsonResponse(response);
  }

  async request(method, path, body, options = {}) {
    const url = new URL(path, `${this.baseURL}/`);
    const headers = buildHeaders(this.defaultHeaders, options.headers, this.apiKey);
    const init = {
      method,
      headers,
      signal: options.signal,
    };

    if (body !== undefined) {
      if (!headers.has("content-type")) {
        headers.set("content-type", "application/json");
      }
      init.body = JSON.stringify(body);
    }

    const response = await this.fetchImpl(url, init);
    if (!response.ok) {
      throw await buildHttpError(response);
    }
    return response;
  }
}

function normalizeBaseURL(baseURL) {
  return String(baseURL).replace(/\/+$/, "");
}

function buildHeaders(defaultHeaders, requestHeaders, apiKey) {
  const headers = new Headers(defaultHeaders);
  if (requestHeaders) {
    const extra = new Headers(requestHeaders);
    for (const [key, value] of extra.entries()) {
      headers.set(key, value);
    }
  }
  if (apiKey && !headers.has("authorization")) {
    headers.set("authorization", `Bearer ${apiKey}`);
  }
  return headers;
}

function withStreamFlag(request, stream) {
  return {
    ...(request ?? {}),
    stream,
  };
}

async function parseJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    throw new AxEngineError(`invalid JSON response: ${error.message}`);
  }
}

async function buildHttpError(response) {
  const body = await safeParseResponseBody(response);
  const message =
    body?.error?.message ||
    body?.message ||
    `${response.status} ${response.statusText}`.trim();
  return new AxEngineHttpError(message, {
    status: response.status,
    statusText: response.statusText,
    body,
    headers: Object.fromEntries(response.headers.entries()),
  });
}

async function safeParseResponseBody(response) {
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

async function* streamJson(client, path, body, options = {}) {
  const response = await client.request("POST", path, body, options);
  if (!response.body) {
    throw new AxEngineStreamError("stream response body is missing", null);
  }

  for await (const event of readSseData(response.body)) {
    if (event === "[DONE]") {
      return;
    }

    let payload;
    try {
      payload = JSON.parse(event);
    } catch (error) {
      throw new AxEngineStreamError(`invalid JSON SSE payload: ${error.message}`, event);
    }

    if (payload && typeof payload === "object" && payload.error) {
      throw new AxEngineStreamError(payload.error.message || "stream error", payload.error);
    }

    yield payload;
  }
}

async function* readSseData(body) {
  const decoder = new TextDecoder();
  let buffer = "";

  for await (const chunk of toAsyncIterable(body)) {
    buffer += decoder.decode(chunk, { stream: true });
    let boundaryIndex = findEventBoundary(buffer);
    while (boundaryIndex >= 0) {
      const rawEvent = buffer.slice(0, boundaryIndex);
      buffer = buffer.slice(skipBoundary(buffer, boundaryIndex));
      const data = extractEventData(rawEvent);
      if (data !== null) {
        yield data;
      }
      boundaryIndex = findEventBoundary(buffer);
    }
  }

  buffer += decoder.decode();
  const trailing = extractEventData(buffer);
  if (trailing !== null) {
    yield trailing;
  }
}

function findEventBoundary(buffer) {
  const lf = buffer.indexOf("\n\n");
  const crlf = buffer.indexOf("\r\n\r\n");
  if (lf === -1) {
    return crlf;
  }
  if (crlf === -1) {
    return lf;
  }
  return Math.min(lf, crlf);
}

function skipBoundary(buffer, index) {
  if (buffer.slice(index, index + 4) === "\r\n\r\n") {
    return index + 4;
  }
  return index + 2;
}

function extractEventData(rawEvent) {
  const dataLines = [];
  for (const line of rawEvent.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }
  return dataLines.join("\n");
}

function toAsyncIterable(body) {
  if (typeof body[Symbol.asyncIterator] === "function") {
    return body;
  }

  if (typeof body.getReader === "function") {
    return {
      async *[Symbol.asyncIterator]() {
        const reader = body.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              return;
            }
            yield value;
          }
        } finally {
          reader.releaseLock();
        }
      },
    };
  }

  throw new AxEngineStreamError("response body is not async iterable", null);
}

async function* mapAsyncIterable(iterable, mapper) {
  for await (const item of iterable) {
    const mapped = mapper(item);
    if (mapped !== "") {
      yield mapped;
    }
  }
}

function completionChunkText(chunk) {
  return chunk?.choices?.[0]?.text ?? "";
}

function chatCompletionChunkText(chunk) {
  return chunk?.choices?.[0]?.delta?.content ?? "";
}

function responseRequestToChatCompletionRequest(request, stream) {
  if (!request || typeof request !== "object" || Array.isArray(request)) {
    throw new AxEngineError("responses request must be an object");
  }

  if ("tools" in request && request.tools != null) {
    throw new AxEngineError("responses.tools is not supported by ax-engine-js");
  }

  const messages = [];
  if (request.instructions != null) {
    messages.push({
      role: "system",
      content: String(request.instructions),
    });
  }
  messages.push(...normalizeResponseInputMessages(request.input));

  if (messages.length === 0) {
    throw new AxEngineError("responses.input must not be empty");
  }

  return {
    model: request.model,
    messages,
    max_tokens: request.max_output_tokens ?? request.max_tokens,
    temperature: request.temperature,
    top_p: request.top_p,
    top_k: request.top_k,
    min_p: request.min_p,
    repeat_penalty: request.repeat_penalty,
    repeat_last_n: request.repeat_last_n,
    frequency_penalty: request.frequency_penalty,
    presence_penalty: request.presence_penalty,
    seed: request.seed,
    stop: request.stop,
    n: request.n,
    stream,
  };
}

function normalizeResponseInputMessages(input) {
  if (typeof input === "string") {
    return [{ role: "user", content: input }];
  }

  if (input == null) {
    return [];
  }

  if (Array.isArray(input)) {
    if (input.length === 0) {
      return [];
    }

    if (input.every((item) => item && typeof item === "object" && "role" in item)) {
      return input.map(normalizeResponseMessage);
    }

    return [
      {
        role: "user",
        content: normalizeResponseContent(input),
      },
    ];
  }

  if (typeof input === "object" && "role" in input) {
    return [normalizeResponseMessage(input)];
  }

  throw new AxEngineError("unsupported responses.input shape");
}

function normalizeResponseMessage(message) {
  const role = String(message.role ?? "")
    .trim()
    .toLowerCase();
  if (!["system", "developer", "user", "assistant"].includes(role)) {
    throw new AxEngineError(`unsupported responses message role '${role}'`);
  }

  return {
    role,
    content: normalizeResponseContent(message.content),
  };
}

function normalizeResponseContent(content) {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content.map(normalizeResponseContentPart);
  }

  if (content && typeof content === "object" && "type" in content) {
    return [normalizeResponseContentPart(content)];
  }

  if (content == null) {
    return "";
  }

  throw new AxEngineError("unsupported responses content shape");
}

function normalizeResponseContentPart(part) {
  const type = String(part.type ?? "").trim().toLowerCase();
  if (!["text", "input_text", "output_text"].includes(type)) {
    throw new AxEngineError(`unsupported responses content type '${type}'`);
  }

  return {
    type: "text",
    text: String(part.text ?? ""),
  };
}

function responseFromChatCompletion(response) {
  const text = response?.choices?.[0]?.message?.content ?? "";
  const finishReason = response?.choices?.[0]?.finish_reason ?? null;
  return buildResponseObject({
    id: response.id,
    createdAt: response.created,
    model: response.model,
    status: "completed",
    text,
    usage: response.usage
      ? {
          input_tokens: response.usage.prompt_tokens,
          output_tokens: response.usage.completion_tokens,
          total_tokens: response.usage.total_tokens,
        }
      : null,
    finishReason,
  });
}

async function* responseStream(client, request, options = {}) {
  const stream = client.chat.completions.stream(request, options);
  let createdEmitted = false;
  let responseId = null;
  let createdAt = null;
  let model = null;
  let text = "";

  for await (const chunk of stream) {
    if (!createdEmitted) {
      responseId = chunk.id;
      createdAt = chunk.created;
      model = chunk.model;
      yield {
        type: "response.created",
        response: buildResponseObject({
          id: responseId,
          createdAt,
          model,
          status: "in_progress",
          text: "",
          usage: null,
          finishReason: null,
        }),
      };
      createdEmitted = true;
    }

    const delta = chatCompletionChunkText(chunk);
    if (delta) {
      text += delta;
      yield {
        type: "response.output_text.delta",
        response_id: chunk.id,
        output_index: 0,
        content_index: 0,
        delta,
      };
    }

    const finishReason = chunk?.choices?.[0]?.finish_reason ?? null;
    if (finishReason) {
      yield {
        type: "response.output_text.done",
        response_id: chunk.id,
        output_index: 0,
        content_index: 0,
        text,
      };
      yield {
        type: "response.completed",
        response: buildResponseObject({
          id: chunk.id,
          createdAt: chunk.created,
          model: chunk.model,
          status: "completed",
          text,
          usage: null,
          finishReason,
        }),
      };
    }
  }
}

function buildResponseObject({ id, createdAt, model, status, text, usage, finishReason }) {
  return {
    id,
    object: "response",
    created_at: createdAt,
    status,
    model,
    output: [
      {
        id: `${id}:output:0`,
        type: "message",
        role: "assistant",
        content: [
          {
            type: "output_text",
            text,
            annotations: [],
          },
        ],
      },
    ],
    output_text: text,
    usage,
    finish_reason: finishReason,
  };
}

function responseStreamEventText(event) {
  return event?.type === "response.output_text.delta" ? event.delta ?? "" : "";
}

module.exports = {
  AxEngineClient,
  AxEngineError,
  AxEngineHttpError,
  AxEngineStreamError,
  completionChunkText,
  chatCompletionChunkText,
  responseStreamEventText,
};
