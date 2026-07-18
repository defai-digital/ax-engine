import AxEngineClient, {
  AxEngineHttpError,
  type OpenAiChatCompletionChunk,
  type OpenAiChatCompletionRequest,
  type OpenAiCompletionRequest,
  type PreviewGenerateRequest,
  type PreviewGenerateStreamEvent,
  type StreamEvent,
} from "../../index.js";

export async function smokeTypes(client = new AxEngineClient({ baseUrl: "http://127.0.0.1:8080" })) {
  const generateRequest: PreviewGenerateRequest = {
    input_text: "Hello from TypeScript",
    max_output_tokens: 16,
    sampling: {
      temperature: 0.2,
      top_p: 0.95,
    },
  };
  const generateResponse = await client.generate(generateRequest);
  const generatedText: string | undefined = generateResponse.output_text;

  const completionRequest: OpenAiCompletionRequest = {
    prompt: "Complete this sentence",
    max_tokens: 16,
  };
  const completionResponse = await client.completion(completionRequest);
  const completionText: string = completionResponse.choices[0]?.text ?? "";

  const chatRequest: OpenAiChatCompletionRequest = {
    messages: [{ role: "user", content: "Say hello." }],
    max_tokens: 16,
  };
  const chatResponse = await client.chatCompletion(chatRequest);
  const chatText: string | null = chatResponse.choices[0]?.message.content ?? null;

  const loaded = await client.loadModel({
    model_id: "qwen3.6-27b",
    model_path: "/models/qwen3.6-27b",
    load_policy: "availability_first",
    load_mode: "add",
  });
  await client.step(loaded.model_id);
  await client.unloadModel({ model_id: loaded.model_id });

  for await (const event of client.streamGenerate(generateRequest)) {
    const typedEvent: PreviewGenerateStreamEvent = event;
    if (
      typedEvent.event === "step" &&
      typeof typedEvent.data === "object" &&
      typedEvent.data !== null &&
      "delta_tokens" in typedEvent.data
    ) {
      const deltaTokens = typedEvent.data.delta_tokens as number[] | undefined;
      void deltaTokens;
    }
  }

  for await (const event of client.streamChatCompletion(chatRequest)) {
    const typedEvent: StreamEvent<OpenAiChatCompletionChunk> = event;
    const deltaText: string | undefined = typedEvent.data.choices[0]?.delta.content;
    void deltaText;
  }

  try {
    await client.health();
  } catch (error) {
    if (error instanceof AxEngineHttpError) {
      const status: number = error.status;
      const payload: unknown = error.payload;
      void status;
      void payload;
    }
  }

  return { generatedText, completionText, chatText };
}
