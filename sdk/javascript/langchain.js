import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { LLM } from "@langchain/core/language_models/llms";
import { AIMessage, AIMessageChunk } from "@langchain/core/messages";
import { ChatGenerationChunk, GenerationChunk } from "@langchain/core/outputs";
import { RunnableBinding } from "@langchain/core/runnables";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import AxEngineClient from "./index.js";

function firstChoice(response, endpoint) {
  const choices = response?.choices;
  if (!Array.isArray(choices) || choices.length === 0) {
    throw new Error(`ax-engine ${endpoint} response contained no choices`);
  }
  return choices[0];
}

function identifyingParams(model) {
  return {
    baseUrl: model.client.baseUrl,
    model: model.model,
    maxTokens: model.maxTokens,
    temperature: model.temperature,
    topP: model.topP,
    topK: model.topK,
    minP: model.minP,
    repetitionPenalty: model.repetitionPenalty,
    // The constructor `stop` default is not listed separately: the resolved
    // per-call `stop` in the cache key already carries the value that is sent,
    // so two models with the same effective stop share one entry.
    seed: model.seed,
    requestMetadata: model.requestMetadata,
  };
}

// Generation options whose per-call value falls back to the constructor
// default. `resolvedGenerationOptions` is the single place that applies that
// rule, so the request builders and the cache key cannot drift apart.
const GENERATION_OPTION_KEYS = [
  "maxTokens",
  "temperature",
  "topP",
  "topK",
  "minP",
  "repetitionPenalty",
  "stop",
  "seed",
];

function resolvedGenerationOptions(model, options) {
  const resolved = {};
  for (const key of GENERATION_OPTION_KEYS) {
    resolved[key] = options?.[key] ?? model[key];
  }
  return resolved;
}

// The AX request field is `metadata` (a string), but LangChain also owns a
// `metadata` key for tracing (an object). Resolve the effective value in one
// place: an explicit `requestMetadata` wins, a legacy string `metadata` is
// accepted as an alias, and the constructor default is the final fallback.
// Both request builders and the cache key use this so they cannot drift apart.
function effectiveRequestMetadata(model, options) {
  return (
    options?.requestMetadata ??
    (typeof options?.metadata === "string" ? options.metadata : undefined) ??
    model.requestMetadata
  );
}

// LangChain's config/call-option split copies `metadata` into the runnable
// config (tracing) and deletes it from the call options before any request
// builder or cache key sees it, so a legacy string `metadata` would silently
// fall back to the constructor default. Promote the string to
// `requestMetadata` before that split: an explicit `requestMetadata` wins and a
// nullish one falls back to the string, matching `effectiveRequestMetadata`.
// The string is dropped so it is not also reported as object tracing metadata;
// object metadata and any other option shape pass through untouched.
function withLegacyRequestMetadata(options) {
  if (typeof options?.metadata !== "string") return options;
  const { metadata: legacyMetadata, ...rest } = options;
  return { ...rest, requestMetadata: rest.requestMetadata ?? legacyMetadata };
}

// LangChain merges per-call options over `_identifyingParams()` and then strips
// undefined values, so `{ temperature: undefined }` would erase the constructor
// default from the cache key and let models with different defaults share one
// cached answer. Overlay the effective values first so the key describes the
// request that is actually sent, while every other call option (tools, ...)
// keeps its LangChain handling. `requestMetadata` needs the same overlay even
// though it is not a generation option: a call that leaves it undefined must
// still key on the constructor default.
function cacheKeyCallOptions(model, callOptions) {
  return {
    ...callOptions,
    ...resolvedGenerationOptions(model, callOptions),
    requestMetadata: effectiveRequestMetadata(model, callOptions),
  };
}

// `RunnableBinding` merges its bound `config`/`kwargs` into every call, and
// most entry points run LangChain's `ensureConfig` before the model's compat
// split can promote the alias. That leaves `bindTools(..., { metadata: "x" })`
// and per-call string `metadata` either spread into a tracing object (so the
// request falls back to the constructor default) or throwing in `ensureConfig`
// when a `timeout` is set. Normalize at construction and at each public
// boundary, exactly like the model classes already do for direct calls.
class AXEngineRunnableBinding extends RunnableBinding {
  constructor(fields) {
    super({
      ...fields,
      config: withLegacyRequestMetadata(fields.config),
      kwargs: withLegacyRequestMetadata(fields.kwargs),
    });
  }

  // `Runnable.withConfig` returns a plain `RunnableBinding`, whose own
  // `withConfig` cannot promote the alias again, so chained configs would lose
  // the promotion. Re-wrap to keep every later call on this subclass.
  static wrap(binding) {
    if (binding instanceof AXEngineRunnableBinding) return binding;
    return new AXEngineRunnableBinding({
      bound: binding.bound,
      config: binding.config,
      kwargs: binding.kwargs,
      configFactories: binding.configFactories,
    });
  }

  _mergeConfig(...options) {
    return super._mergeConfig(
      ...options.map((option) => withLegacyRequestMetadata(option)),
    );
  }

  invoke(input, options) {
    return super.invoke(input, withLegacyRequestMetadata(options));
  }

  batch(inputs, options, batchOptions) {
    return super.batch(
      inputs,
      Array.isArray(options)
        ? options.map((config) => withLegacyRequestMetadata(config))
        : withLegacyRequestMetadata(options),
      batchOptions,
    );
  }

  stream(input, options) {
    return super.stream(input, withLegacyRequestMetadata(options));
  }

  streamLog(input, options, streamOptions) {
    return super.streamLog(input, withLegacyRequestMetadata(options), streamOptions);
  }

  streamEvents(input, options, streamOptions) {
    return super.streamEvents(input, withLegacyRequestMetadata(options), streamOptions);
  }

  transform(generator, options) {
    return super.transform(generator, withLegacyRequestMetadata(options));
  }

  withConfig(config) {
    return AXEngineRunnableBinding.wrap(
      super.withConfig(withLegacyRequestMetadata(config)),
    );
  }

  withRetry(fields) {
    return new AXEngineRunnableBinding({
      bound: super.withRetry(fields),
      config: {},
      kwargs: {},
    });
  }
}

function messageToOpenAi(message) {
  const reportedType = message._getType ? message._getType() : message.role;
  // `ChatMessage._getType()` reports the generic marker for every role; the real
  // role only lives on `message.role`.
  const type =
    reportedType === "generic" && typeof message.role === "string"
      ? message.role
      : reportedType;
  let role;
  if (type === "human" || type === "user") {
    role = "user";
  } else if (type === "ai" || type === "assistant") {
    role = "assistant";
  } else if (type === "system") {
    role = "system";
  } else {
    role = type;
  }
  const converted = { role, content: message.content };
  if (typeof message.name === "string") {
    converted.name = message.name;
  }
  if (role === "tool" && typeof message.tool_call_id === "string") {
    converted.tool_call_id = message.tool_call_id;
  }
  if (role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    converted.tool_calls = message.tool_calls.map((toolCall) => ({
      id: toolCall.id,
      type: "function",
      function: {
        name: toolCall.name,
        arguments:
          typeof toolCall.args === "string" ? toolCall.args : JSON.stringify(toolCall.args ?? {}),
      },
    }));
  }
  if (role === "assistant" && message.additional_kwargs?.function_call) {
    converted.function_call = message.additional_kwargs.function_call;
  }
  return converted;
}

function normalizeToolChoice(toolChoice) {
  if (typeof toolChoice !== "string") return toolChoice;
  if (toolChoice === "any") return "required";
  if (["auto", "none", "required"].includes(toolChoice)) return toolChoice;
  return { type: "function", function: { name: toolChoice } };
}

// LangChain JS reads `llmOutput.tokenUsage` as camelCase counts and
// `AIMessage.usage_metadata` as input/output totals; the server's OpenAI
// snake_case usage object passed through matches neither.
function langchainTokenUsage(usage) {
  if (!usage) return undefined;
  return {
    promptTokens: usage.prompt_tokens,
    completionTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens,
  };
}

function langchainUsageMetadata(usage) {
  if (!usage) return undefined;
  const input = usage.prompt_tokens ?? 0;
  const output = usage.completion_tokens ?? 0;
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: usage.total_tokens ?? input + output,
  };
}

function messageAdditionalKwargs(message) {
  const additionalKwargs = {};
  if (Array.isArray(message?.tool_calls)) {
    additionalKwargs.tool_calls = message.tool_calls;
  }
  if (message?.function_call && typeof message.function_call === "object") {
    additionalKwargs.function_call = message.function_call;
  }
  // Thinking models stream reasoning separately from the answer; surface it
  // rather than dropping it (native `reasoning_content`, or the OpenRouter-
  // style `reasoning` alias).
  const reasoning = message?.reasoning_content ?? message?.reasoning;
  if (typeof reasoning === "string" && reasoning.length > 0) {
    additionalKwargs.reasoning_content = reasoning;
  }
  return additionalKwargs;
}

function parsedToolCalls(rawToolCalls) {
  const toolCalls = [];
  const invalidToolCalls = [];
  for (const rawToolCall of rawToolCalls ?? []) {
    const fn = rawToolCall?.function;
    if (!fn || typeof fn.name !== "string") continue;
    const rawArguments = typeof fn.arguments === "string" ? fn.arguments : "{}";
    try {
      toolCalls.push({
        name: fn.name,
        args: JSON.parse(rawArguments) ?? {},
        id: rawToolCall.id,
      });
    } catch {
      invalidToolCalls.push({
        name: fn.name,
        args: rawArguments,
        id: rawToolCall.id,
        error: "Invalid JSON arguments",
      });
    }
  }
  return { toolCalls, invalidToolCalls };
}

function toolCallChunks(message) {
  if (!Array.isArray(message?.tool_calls)) return [];
  return message.tool_calls.flatMap((rawToolCall) => {
    if (!rawToolCall || typeof rawToolCall !== "object") return [];
    const fn = rawToolCall.function ?? {};
    return [{
      name: fn.name,
      args: fn.arguments,
      id: rawToolCall.id,
      index: rawToolCall.index,
      type: "tool_call_chunk",
    }];
  });
}

export class ChatAXEngine extends BaseChatModel {
  constructor(fields = {}) {
    const legacyRequestMetadata = typeof fields.metadata === "string" ? fields.metadata : undefined;
    super(legacyRequestMetadata === undefined ? fields : { ...fields, metadata: undefined });
    const {
      baseUrl,
      fetch: fetchImpl,
      headers,
      model,
      maxTokens,
      temperature,
      topP,
      topK,
      minP,
      repetitionPenalty,
      stop,
      seed,
      requestMetadata,
    } = fields;
    this.client = new AxEngineClient({ baseUrl, fetch: fetchImpl, headers });
    this.model = model;
    this.maxTokens = maxTokens;
    this.temperature = temperature;
    this.topP = topP;
    this.topK = topK;
    this.minP = minP;
    this.repetitionPenalty = repetitionPenalty;
    this.stop = stop;
    this.seed = seed;
    // `metadata` belongs to LangChain's tracing/callback surface and is an
    // object. AX's OpenAI-compatible request field is a string, so keep the
    // two namespaces separate. Accept the historical string spelling at
    // runtime for callers that used it before `requestMetadata` was added.
    this.requestMetadata = requestMetadata ?? legacyRequestMetadata;
  }

  _llmType() {
    return "ax-engine";
  }

  _identifyingParams() {
    return identifyingParams(this);
  }

  _getSerializedCacheKeyParametersForCall(options) {
    return super._getSerializedCacheKeyParametersForCall(
      cacheKeyCallOptions(this, options ?? {}),
    );
  }

  // Cover every framework entry point (generate/invoke/stream) in one place:
  // they all split config from call options through this compat method.
  _separateRunnableConfigFromCallOptionsCompat(options) {
    return super._separateRunnableConfigFromCallOptionsCompat(
      withLegacyRequestMetadata(options),
    );
  }

  // `Runnable.stream`, `streamLog`, `streamEvents`, `batch` and `transform` all
  // run LangChain's `ensureConfig` before the compat split above, and
  // `ensureConfig` writes `metadata.timeoutMs` whenever `timeout` is set. A
  // legacy string `metadata` would therefore throw before it could be promoted,
  // so normalize at these public boundaries too. The helper is idempotent, so
  // the later compat split (and `_streamLog`'s internal `this.stream` call)
  // sees an options object it already handled.
  stream(input, options) {
    return super.stream(input, withLegacyRequestMetadata(options));
  }

  streamLog(input, options, streamOptions) {
    return super.streamLog(input, withLegacyRequestMetadata(options), streamOptions);
  }

  // `BaseChatModel.streamEvents` handles the version-less protocol itself and
  // only enters `ensureConfig` for an explicit `version`, but normalizing here
  // covers both branches with one rule.
  streamEvents(input, options, streamOptions) {
    return super.streamEvents(input, withLegacyRequestMetadata(options), streamOptions);
  }

  // `batch` takes one config or one per input, and `_getOptionsList` normalizes
  // each of them, so both shapes need the same promotion.
  batch(inputs, options, batchOptions) {
    return super.batch(
      inputs,
      Array.isArray(options)
        ? options.map((config) => withLegacyRequestMetadata(config))
        : withLegacyRequestMetadata(options),
      batchOptions,
    );
  }

  transform(generator, options) {
    return super.transform(generator, withLegacyRequestMetadata(options));
  }

  // The normalization has to live on the binding, not only here: the bound
  // metadata/kwargs are merged in before `ensureConfig` runs on each call.
  bindTools(tools, kwargs = {}) {
    return new AXEngineRunnableBinding({
      bound: this,
      config: {},
      kwargs: {
        ...kwargs,
        tools: tools.map((tool) => convertToOpenAITool(tool)),
        tool_choice: normalizeToolChoice(kwargs.tool_choice),
      },
    });
  }

  withConfig(config) {
    return AXEngineRunnableBinding.wrap(
      super.withConfig(withLegacyRequestMetadata(config)),
    );
  }

  withRetry(fields) {
    return new AXEngineRunnableBinding({
      bound: super.withRetry(fields),
      config: {},
      kwargs: {},
    });
  }

  _buildChatRequest(messages, options) {
    const generation = resolvedGenerationOptions(this, options);
    return {
      model: this.model,
      messages: messages.map(messageToOpenAi),
      max_tokens: generation.maxTokens,
      temperature: generation.temperature,
      top_p: generation.topP,
      top_k: generation.topK,
      min_p: generation.minP,
      repetition_penalty: generation.repetitionPenalty,
      stop: generation.stop,
      seed: generation.seed,
      metadata: effectiveRequestMetadata(this, options),
      tools: options?.tools,
      tool_choice: options?.tool_choice,
    };
  }

  async _generate(messages, options, _runManager) {
    const request = this._buildChatRequest(messages, options);
    const response = await this.client.chatCompletion(request, {
      signal: options?.signal,
    });
    const choice = firstChoice(response, "chat completions");
    const text = choice?.message?.content ?? "";
    const rawToolCalls = choice?.message?.tool_calls;
    const { toolCalls, invalidToolCalls } = parsedToolCalls(rawToolCalls);
    return {
      generations: [
        {
          text,
          message: new AIMessage({
            content: text,
            additional_kwargs: messageAdditionalKwargs(choice?.message),
            tool_calls: toolCalls,
            invalid_tool_calls: invalidToolCalls,
            usage_metadata: langchainUsageMetadata(response.usage),
          }),
          generationInfo: { finishReason: choice?.finish_reason },
        },
      ],
      llmOutput: { tokenUsage: langchainTokenUsage(response.usage) },
    };
  }

  async *_streamResponseChunks(messages, options, runManager) {
    const request = { ...this._buildChatRequest(messages, options), stream: true };
    for await (const event of this.client.streamChatCompletion(request, {
      signal: options?.signal,
    })) {
      const choice = event.data?.choices?.[0];
      if (!choice) continue;
      const text = choice.delta?.content ?? "";
      yield new ChatGenerationChunk({
        text,
        message: new AIMessageChunk({
          content: text,
          additional_kwargs: messageAdditionalKwargs(choice.delta),
          tool_call_chunks: toolCallChunks(choice.delta),
        }),
        generationInfo: { finishReason: choice.finish_reason },
      });
      await runManager?.handleLLMNewToken(text);
    }
  }
}

export class AXEngineLLM extends LLM {
  constructor(fields = {}) {
    const legacyRequestMetadata = typeof fields.metadata === "string" ? fields.metadata : undefined;
    super(legacyRequestMetadata === undefined ? fields : { ...fields, metadata: undefined });
    const {
      baseUrl,
      fetch: fetchImpl,
      headers,
      model,
      maxTokens,
      temperature,
      topP,
      topK,
      minP,
      repetitionPenalty,
      stop,
      seed,
      requestMetadata,
    } = fields;
    this.client = new AxEngineClient({ baseUrl, fetch: fetchImpl, headers });
    this.model = model;
    this.maxTokens = maxTokens;
    this.temperature = temperature;
    this.topP = topP;
    this.topK = topK;
    this.minP = minP;
    this.repetitionPenalty = repetitionPenalty;
    this.stop = stop;
    this.seed = seed;
    this.requestMetadata = requestMetadata ?? legacyRequestMetadata;
  }

  _llmType() {
    return "ax-engine";
  }

  _identifyingParams() {
    return identifyingParams(this);
  }

  _getSerializedCacheKeyParametersForCall(options) {
    return super._getSerializedCacheKeyParametersForCall(
      cacheKeyCallOptions(this, options ?? {}),
    );
  }

  // Same override as the chat model: every framework entry point
  // (generate/invoke/stream) splits config from call options here.
  _separateRunnableConfigFromCallOptionsCompat(options) {
    return super._separateRunnableConfigFromCallOptionsCompat(
      withLegacyRequestMetadata(options),
    );
  }

  // Same public-boundary normalization as the chat model: `ensureConfig` runs
  // before the compat split above, so a legacy string `metadata` has to be
  // promoted to `requestMetadata` before `stream`, `streamLog`, `streamEvents`,
  // `batch` or `transform` reach it.
  stream(input, options) {
    return super.stream(input, withLegacyRequestMetadata(options));
  }

  streamLog(input, options, streamOptions) {
    return super.streamLog(input, withLegacyRequestMetadata(options), streamOptions);
  }

  streamEvents(input, options, streamOptions) {
    return super.streamEvents(input, withLegacyRequestMetadata(options), streamOptions);
  }

  batch(inputs, options, batchOptions) {
    return super.batch(
      inputs,
      Array.isArray(options)
        ? options.map((config) => withLegacyRequestMetadata(config))
        : withLegacyRequestMetadata(options),
      batchOptions,
    );
  }

  transform(generator, options) {
    return super.transform(generator, withLegacyRequestMetadata(options));
  }

  withConfig(config) {
    return AXEngineRunnableBinding.wrap(
      super.withConfig(withLegacyRequestMetadata(config)),
    );
  }

  withRetry(fields) {
    return new AXEngineRunnableBinding({
      bound: super.withRetry(fields),
      config: {},
      kwargs: {},
    });
  }

  _buildCompletionRequest(prompt, options) {
    const generation = resolvedGenerationOptions(this, options);
    return {
      model: this.model,
      prompt,
      max_tokens: generation.maxTokens,
      temperature: generation.temperature,
      top_p: generation.topP,
      top_k: generation.topK,
      min_p: generation.minP,
      repetition_penalty: generation.repetitionPenalty,
      stop: generation.stop,
      seed: generation.seed,
      metadata: effectiveRequestMetadata(this, options),
    };
  }

  async _call(prompt, options, _runManager) {
    const request = this._buildCompletionRequest(prompt, options);
    const response = await this.client.completion(request, {
      signal: options?.signal,
    });
    return firstChoice(response, "completions")?.text ?? "";
  }

  async *_streamResponseChunks(prompt, options, runManager) {
    const request = { ...this._buildCompletionRequest(prompt, options), stream: true };
    for await (const event of this.client.streamCompletion(request, {
      signal: options?.signal,
    })) {
      const choice = event.data?.choices?.[0];
      if (!choice) continue;
      const text = choice.text ?? "";
      yield new GenerationChunk({
        text,
        generationInfo: { finishReason: choice.finish_reason },
      });
      await runManager?.handleLLMNewToken(text);
    }
  }
}
