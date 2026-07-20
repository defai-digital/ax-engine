import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { LLM } from "@langchain/core/language_models/llms";
import { AIMessage, AIMessageChunk } from "@langchain/core/messages";
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

function messageToOpenAi(message) {
  const type = message._getType ? message._getType() : message.role;
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
  return converted;
}

function normalizeToolChoice(toolChoice) {
  if (typeof toolChoice !== "string") return toolChoice;
  if (toolChoice === "any") return "required";
  if (["auto", "none", "required"].includes(toolChoice)) return toolChoice;
  return { type: "function", function: { name: toolChoice } };
}

function messageAdditionalKwargs(message) {
  const additionalKwargs = {};
  if (Array.isArray(message?.tool_calls)) {
    additionalKwargs.tool_calls = message.tool_calls;
  }
  if (message?.function_call && typeof message.function_call === "object") {
    additionalKwargs.function_call = message.function_call;
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

  bindTools(tools, kwargs = {}) {
    return new RunnableBinding({
      bound: this,
      config: {},
      kwargs: {
        ...kwargs,
        tools: tools.map((tool) => convertToOpenAITool(tool)),
        tool_choice: normalizeToolChoice(kwargs.tool_choice),
      },
    });
  }

  _buildChatRequest(messages, options) {
    return {
      model: this.model,
      messages: messages.map(messageToOpenAi),
      max_tokens: options?.maxTokens ?? this.maxTokens,
      temperature: options?.temperature ?? this.temperature,
      top_p: options?.topP ?? this.topP,
      top_k: options?.topK ?? this.topK,
      min_p: options?.minP ?? this.minP,
      repetition_penalty: options?.repetitionPenalty ?? this.repetitionPenalty,
      stop: options?.stop ?? this.stop,
      seed: options?.seed ?? this.seed,
      metadata:
        options?.requestMetadata ??
        (typeof options?.metadata === "string" ? options.metadata : undefined) ??
        this.requestMetadata,
      tools: options?.tools,
      tool_choice: options?.tool_choice,
    };
  }

  async _generate(messages, options, _runManager) {
    const request = this._buildChatRequest(messages, options);
    const response = await this.client.chatCompletion(request);
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
          }),
          generationInfo: { finishReason: choice?.finish_reason },
        },
      ],
      llmOutput: { tokenUsage: response.usage },
    };
  }

  async *_streamResponseChunks(messages, options, runManager) {
    const request = { ...this._buildChatRequest(messages, options), stream: true };
    for await (const event of this.client.streamChatCompletion(request)) {
      const choice = event.data?.choices?.[0];
      if (!choice) continue;
      const text = choice.delta?.content ?? "";
      yield {
        text,
        message: new AIMessageChunk({
          content: text,
          additional_kwargs: messageAdditionalKwargs(choice.delta),
          tool_call_chunks: toolCallChunks(choice.delta),
        }),
        generationInfo: { finishReason: choice.finish_reason },
      };
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

  _buildCompletionRequest(prompt, options) {
    return {
      model: this.model,
      prompt,
      max_tokens: options?.maxTokens ?? this.maxTokens,
      temperature: options?.temperature ?? this.temperature,
      top_p: options?.topP ?? this.topP,
      top_k: options?.topK ?? this.topK,
      min_p: options?.minP ?? this.minP,
      repetition_penalty: options?.repetitionPenalty ?? this.repetitionPenalty,
      stop: options?.stop ?? this.stop,
      seed: options?.seed ?? this.seed,
      metadata:
        options?.requestMetadata ??
        (typeof options?.metadata === "string" ? options.metadata : undefined) ??
        this.requestMetadata,
    };
  }

  async _call(prompt, options, _runManager) {
    const request = this._buildCompletionRequest(prompt, options);
    const response = await this.client.completion(request);
    return firstChoice(response, "completions")?.text ?? "";
  }

  async *_streamResponseChunks(prompt, options, runManager) {
    const request = { ...this._buildCompletionRequest(prompt, options), stream: true };
    for await (const event of this.client.streamCompletion(request)) {
      const choice = event.data?.choices?.[0];
      if (!choice) continue;
      const text = choice.text ?? "";
      yield { text, generationInfo: { finishReason: choice.finish_reason } };
      await runManager?.handleLLMNewToken(text);
    }
  }
}
