import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { LLM } from "@langchain/core/language_models/llms";
import { AIMessage, AIMessageChunk } from "@langchain/core/messages";
import AxEngineClient from "./index.js";

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
  return { role, content: message.content };
}

export class ChatAXEngine extends BaseChatModel {
  constructor(fields = {}) {
    super(fields);
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
      metadata,
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
    this.metadata = metadata;
  }

  _llmType() {
    return "ax-engine";
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
      seed: this.seed,
      metadata: this.metadata,
    };
  }

  async _generate(messages, options, _runManager) {
    const request = this._buildChatRequest(messages, options);
    const response = await this.client.chatCompletion(request);
    const choice = response.choices[0];
    const text = choice?.message?.content ?? "";
    return {
      generations: [
        {
          text,
          message: new AIMessage(text),
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
        message: new AIMessageChunk({ content: text }),
        generationInfo: { finishReason: choice.finish_reason },
      };
      await runManager?.handleLLMNewToken(text);
    }
  }
}

export class AXEngineLLM extends LLM {
  constructor(fields = {}) {
    super(fields);
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
      metadata,
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
    this.metadata = metadata;
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
      seed: this.seed,
      metadata: this.metadata,
    };
  }

  async _call(prompt, options, _runManager) {
    const request = this._buildCompletionRequest(prompt, options);
    const response = await this.client.completion(request);
    return response.choices[0]?.text ?? "";
  }

  async *_streamResponseChunks(prompt, options, runManager) {
    const request = { ...this._buildCompletionRequest(prompt, options), stream: true };
    for await (const event of this.client.streamCompletion(request)) {
      const text = event.data?.choices?.[0]?.text ?? "";
      yield { text, generationInfo: { finishReason: event.data?.choices?.[0]?.finish_reason } };
      await runManager?.handleLLMNewToken(text);
    }
  }
}
