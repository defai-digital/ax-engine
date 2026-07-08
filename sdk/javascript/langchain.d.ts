import { BaseChatModel, BaseChatModelParams } from "@langchain/core/language_models/chat_models";
import { LLM, BaseLLMParams } from "@langchain/core/language_models/llms";
import type { BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

export interface AXEngineLangChainParams {
  baseUrl?: string;
  fetch?: typeof fetch;
  headers?: HeadersInit;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  repetitionPenalty?: number;
  stop?: string | string[];
  seed?: number;
  metadata?: string;
}

export interface ChatAXEngineParams extends AXEngineLangChainParams, BaseChatModelParams {}

export interface AXEngineLLMParams extends AXEngineLangChainParams, BaseLLMParams {}

export declare class ChatAXEngine extends BaseChatModel {
  constructor(fields?: ChatAXEngineParams);
  _llmType(): string;
  _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult>;
  _streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<any, void, void>;
}

export declare class AXEngineLLM extends LLM {
  constructor(fields?: AXEngineLLMParams);
  _llmType(): string;
  _call(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<string>;
  _streamResponseChunks(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<any, void, void>;
}
