export interface RequestOptions {
  signal?: AbortSignal;
  headers?: HeadersInit;
}

export interface AxEngineClientOptions {
  baseURL?: string;
  defaultModel?: string;
  apiKey?: string;
  headers?: HeadersInit;
  fetch?: typeof fetch;
}

export interface HealthResponse {
  status: "ok";
  model: string;
  architecture: string;
  context_length: number;
  vocab_size: number;
  support_note: string | null;
}

export interface ModelCard {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
  root: string;
}

export interface ModelsResponse {
  object: "list";
  data: ModelCard[];
}

export type StopSequences = string | string[];

export interface CompletionRequest {
  model?: string;
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repeat_penalty?: number;
  repeat_last_n?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  seed?: number;
  stop?: StopSequences;
  n?: 1;
  stream?: boolean;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface CompletionChoice {
  index: number;
  text: string;
  finish_reason: "stop" | "length";
}

export interface CompletionResponse {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: CompletionChoice[];
  usage: Usage;
}

export interface CompletionChunkChoice {
  index: number;
  text: string;
  finish_reason: "stop" | "length" | null;
}

export interface CompletionChunk {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: CompletionChunkChoice[];
}

export interface MessageContentPart {
  type: "text";
  text: string;
}

export interface ChatCompletionMessage {
  role: "system" | "developer" | "user" | "assistant";
  content: string | MessageContentPart[];
}

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatCompletionMessage[];
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repeat_penalty?: number;
  repeat_last_n?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  seed?: number;
  stop?: StopSequences;
  n?: 1;
  stream?: boolean;
}

export interface AssistantMessage {
  role: "assistant";
  content: string;
}

export interface ChatCompletionChoice {
  index: number;
  message: AssistantMessage;
  finish_reason: "stop" | "length";
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: Usage;
}

export interface ChatDelta {
  role?: "assistant";
  content?: string;
}

export interface ChatChunkChoice {
  index: number;
  delta: ChatDelta;
  finish_reason: "stop" | "length" | null;
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChatChunkChoice[];
}

export interface ResponseInputTextPart {
  type: "text" | "input_text" | "output_text";
  text: string;
}

export interface ResponseInputMessage {
  role: "system" | "developer" | "user" | "assistant";
  content: string | ResponseInputTextPart | ResponseInputTextPart[];
}

export interface ResponsesRequest {
  model?: string;
  instructions?: string;
  input: string | ResponseInputMessage | ResponseInputMessage[] | ResponseInputTextPart[];
  max_output_tokens?: number;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  repeat_penalty?: number;
  repeat_last_n?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  seed?: number;
  stop?: StopSequences;
  n?: 1;
  stream?: boolean;
  tools?: never;
}

export interface ResponseOutputText {
  type: "output_text";
  text: string;
  annotations: unknown[];
}

export interface ResponseOutputMessage {
  id: string;
  type: "message";
  role: "assistant";
  content: ResponseOutputText[];
}

export interface ResponseUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface ResponseObject {
  id: string;
  object: "response";
  created_at: number;
  status: "in_progress" | "completed";
  model: string;
  output: ResponseOutputMessage[];
  output_text: string;
  usage: ResponseUsage | null;
  finish_reason: "stop" | "length" | null;
}

export interface ResponseCreatedEvent {
  type: "response.created";
  response: ResponseObject;
}

export interface ResponseOutputTextDeltaEvent {
  type: "response.output_text.delta";
  response_id: string;
  output_index: number;
  content_index: number;
  delta: string;
}

export interface ResponseOutputTextDoneEvent {
  type: "response.output_text.done";
  response_id: string;
  output_index: number;
  content_index: number;
  text: string;
}

export interface ResponseCompletedEvent {
  type: "response.completed";
  response: ResponseObject;
}

export type ResponseStreamEvent =
  | ResponseCreatedEvent
  | ResponseOutputTextDeltaEvent
  | ResponseOutputTextDoneEvent
  | ResponseCompletedEvent;

export class AxEngineError extends Error {}

export class AxEngineHttpError extends AxEngineError {
  status: number;
  statusText: string;
  body: unknown;
  headers: Record<string, string>;
}

export class AxEngineStreamError extends AxEngineError {
  payload: unknown;
}

export class AxEngineClient {
  constructor(options?: AxEngineClientOptions);
  readonly baseURL: string;
  readonly defaultModel: string | null;
  readonly apiKey: string | null;

  health(options?: RequestOptions): Promise<HealthResponse>;
  models: {
    list(options?: RequestOptions): Promise<ModelsResponse>;
  };
  completions: {
    create(
      request: CompletionRequest,
      options?: RequestOptions,
    ): Promise<CompletionResponse>;
    stream(
      request: CompletionRequest,
      options?: RequestOptions,
    ): AsyncIterable<CompletionChunk>;
    streamText(
      request: CompletionRequest,
      options?: RequestOptions,
    ): AsyncIterable<string>;
  };
  chat: {
    completions: {
      create(
        request: ChatCompletionRequest,
        options?: RequestOptions,
      ): Promise<ChatCompletionResponse>;
      stream(
        request: ChatCompletionRequest,
        options?: RequestOptions,
      ): AsyncIterable<ChatCompletionChunk>;
      streamText(
        request: ChatCompletionRequest,
        options?: RequestOptions,
      ): AsyncIterable<string>;
    };
  };
  responses: {
    create(request: ResponsesRequest, options?: RequestOptions): Promise<ResponseObject>;
    stream(
      request: ResponsesRequest,
      options?: RequestOptions,
    ): AsyncIterable<ResponseStreamEvent>;
    streamText(
      request: ResponsesRequest,
      options?: RequestOptions,
    ): AsyncIterable<string>;
  };
}

export function completionChunkText(chunk: CompletionChunk): string;
export function chatCompletionChunkText(chunk: ChatCompletionChunk): string;
export function responseStreamEventText(event: ResponseStreamEvent): string;

export default AxEngineClient;
