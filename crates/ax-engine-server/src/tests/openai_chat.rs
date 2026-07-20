use crate::chat;
use crate::openai::chat_requests::{
    ChatPromptRenderOptions, render_openai_chat_prompt, render_openai_chat_prompt_with_options,
    render_openai_chat_prompt_with_tools,
};
use crate::openai::requests::{
    DEFAULT_OPENAI_MAX_TOKENS, build_openai_chat_request,
    build_openai_chat_request_offloading_media, build_openai_llama_cpp_chat_request,
    build_openai_mlx_lm_chat_request, chat_template_kwargs_for_model_id,
    openai_chat_stop_sequences,
};
use crate::openai::schema::{OpenAiChatCompletionHttpRequest, OpenAiChatMessage, OpenAiStopInput};
use crate::openai::validation::validate_openai_request;
use crate::routes::build_router;
use ax_engine_sdk::RequestWorkloadHints;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use super::fixtures::{
    assert_invalid_request_response, gemma4_unified_artifact, json_request_body, json_response,
    llama_cpp_server_state, minimal_tokenizer_artifact, mlx_lm_delegated_state,
    native_mlx_openai_builder_state, openai_first_choice, sample_gemma4_multimodal_inputs,
    sample_openai_chat_request, sample_openai_chat_request_with_role,
    spawn_llama_cpp_completion_server, test_app_state,
};

#[test]
fn openai_chat_prompt_renderer_uses_model_family_templates() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"}
    ]))
    .expect("sample messages should deserialize");

    assert_eq!(
        render_openai_chat_prompt("qwen3", &messages).expect("qwen prompt"),
        "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        render_openai_chat_prompt("Qwen3.6-35B-A3B-4bit", &messages).expect("qwen3.6 prompt"),
        "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(
        render_openai_chat_prompt("Meta-Llama-3.1-8B-Instruct", &messages).expect("llama prompt"),
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    assert_eq!(
        render_openai_chat_prompt("gemma4-e2b", &messages).expect("gemma4 prompt"),
        "<bos><|turn>system\nBe concise.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    );
    assert_eq!(
        render_openai_chat_prompt("glm4_moe_lite", &messages).expect("glm prompt"),
        "[gMASK]<sop><|system|>Be concise.<|user|>Hello<|assistant|></think>"
    );
    assert_eq!(
        render_openai_chat_prompt("unknown-local-model", &messages).expect("plain prompt"),
        "system: Be concise.\nuser: Hello\nassistant:"
    );
}

#[test]
fn openai_chat_prompt_renderer_injects_qwen_tool_contract() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Read README.md"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt_with_tools(
        "mlx-community/Qwen3-Coder-Next-4bit",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a workspace file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }
        ])),
        Some(&json!("auto")),
    )
    .expect("qwen tool prompt should render");

    assert!(prompt.starts_with(
        "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.\n\n# Tools"
    ));
    assert!(prompt.contains("# Tools\n\nYou have access to the following tools:"));
    assert!(prompt.contains("<function>\n<name>read_file</name>"));
    assert!(prompt.contains("<description>Read a workspace file</description>"));
    assert!(prompt.contains("<parameter>\n<name>path</name>"));
    assert!(prompt.contains("<function=example_function_name>"));
    assert!(prompt.contains("<parameter=example_parameter_1>"));
    assert!(prompt.contains("If you choose to call a tool ONLY reply"));
    assert!(prompt.contains("the tool calling block MUST begin with an opening <tool_call> tag"));
    assert!(prompt.contains("<|im_start|>user\nRead README.md<|im_end|>"));
    assert!(prompt.ends_with(chat::QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_NO_THINK));
}

#[test]
fn openai_chat_prompt_renderer_injects_glm_tool_contract() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "What is the weather in Tokyo?"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt_with_tools(
        "glm4_moe_lite",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }
        ])),
        Some(&json!("auto")),
    )
    .expect("glm tool prompt should render");

    // GLM declares tool signatures in a leading <|system|> block. The tool
    // schema is emitted as compact JSON (serde sorts object keys, which is
    // immaterial to the model since JSON objects are unordered).
    assert!(prompt.starts_with("[gMASK]<sop><|system|># Tools"));
    assert!(prompt.contains("<tools>\n{"));
    assert!(prompt.contains("\"name\":\"get_weather\""));
    assert!(prompt.contains("\"description\":\"Get the current weather\""));
    assert!(prompt.contains("</tools>"));
    assert!(prompt.contains(
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
    ));
    assert!(prompt.contains("<|user|>What is the weather in Tokyo?"));
    assert!(prompt.ends_with("<|assistant|></think>"));
}

#[test]
fn openai_chat_prompt_renderer_renders_glm_assistant_tool_call_history() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Weather in Tokyo?"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp_c\":18}"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt_with_tools(
        "glm4_moe_lite",
        &messages,
        Some(&json!([
            {"type": "function", "function": {"name": "get_weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
        ])),
        Some(&json!("auto")),
    )
    .expect("glm tool history prompt should render");

    // Prior assistant call rendered in GLM arg_key/arg_value form.
    assert!(prompt.contains(
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>"
    ));
    // Tool result rendered as a GLM observation block.
    assert!(prompt.contains("<|observation|><tool_response>{\"temp_c\":18}</tool_response>"));
}

#[test]
fn openai_chat_prompt_renderer_treats_underscore_qwen_coder_as_coding_model() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Read README.md"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt_with_tools(
        "mlx-community/Qwen3_Coder_Next_4bit",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object"}
                }
            }
        ])),
        Some(&json!("auto")),
    )
    .expect("underscore qwen coder prompt should render");

    assert!(prompt.contains("# Tools\n\nYou have access to the following tools:"));
    assert!(prompt.contains("<function>\n<name>read_file</name>"));
    assert!(prompt.contains("<function=example_function_name>"));
    assert!(prompt.ends_with(chat::QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_NO_THINK));
}

#[test]
fn openai_chat_prompt_renderer_preserves_qwen_coder_user_system_with_tool_contract() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "Use the project coding conventions."},
        {"role": "user", "content": "Read README.md"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt_with_tools(
        "Qwen/Qwen3-Coder-Next-Q4_K_M",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a workspace file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }
        ])),
        Some(&json!("auto")),
    )
    .expect("qwen coder prompt should render");

    assert!(
        prompt.starts_with("<|im_start|>system\nUse the project coding conventions.\n\n# Tools")
    );
    assert!(!prompt.contains("You are Qwen, a helpful AI assistant"));
    assert!(prompt.contains("<function>\n<name>read_file</name>"));
    assert!(prompt.contains("<|im_start|>user\nRead README.md<|im_end|>"));
    assert!(prompt.ends_with(chat::QWEN_CHATML_ASSISTANT_GENERATION_PROMPT_NO_THINK));
}

#[test]
fn openai_chat_prompt_renderer_uses_qwen36_function_xml_tool_contract() {
    // AutomatosX / hub Qwen3.6 chat_template.jinja matches Qwen3.5: JSON tool
    // schemas inside <tools> and function= call format (not Coder-Next XML
    // declarations).
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Read README.md"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt_with_tools(
        "Qwen3.6-35B-A3B-4bit",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a workspace file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }
        ])),
        Some(&json!("auto")),
    )
    .expect("qwen3.6 tool prompt should render");

    assert!(
        prompt.starts_with(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:"
        )
    );
    assert!(prompt.contains("<tools>\n"));
    assert!(prompt.contains("\"name\":\"read_file\""));
    assert!(prompt.contains("\"type\":\"function\"") || prompt.contains("\"type\": \"function\""));
    // Not Coder-Next XML tool *declarations*.
    assert!(!prompt.contains("<function>\n<name>read_file</name>"));
    assert!(
        !prompt.contains("You are Qwen, a helpful AI assistant that can interact with a computer")
    );
    assert!(prompt.contains("<function=example_function_name>"));
    assert!(prompt.contains("<parameter=example_parameter_1>"));
    assert!(prompt.contains(
        "an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags"
    ));
    assert!(prompt.contains("<|im_start|>user\nRead README.md<|im_end|>"));
    assert!(prompt.ends_with(chat::QWEN_CHATML_ASSISTANT_GENERATION_PROMPT));
}

#[test]
fn openai_chat_prompt_renderer_uses_llama4_header_markers() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt("meta-llama/Llama-4-Scout-17B-16E-Instruct", &messages)
        .expect("llama4 prompt");
    assert_eq!(
        prompt,
        "<|begin_of_text|><|header_start|>system<|header_end|>\n\nBe concise.<|eot|><|header_start|>user<|header_end|>\n\nHello<|eot|><|header_start|>assistant<|header_end|>\n\n"
    );
}

#[test]
fn openai_chat_prompt_renderer_diffusiongemma_matches_hub_generation_prefill() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"}
    ]))
    .expect("sample messages should deserialize");

    let prompt =
        render_openai_chat_prompt("diffusiongemma-26B-A4B-it-4bit", &messages).expect("diffusion");
    assert_eq!(
        prompt,
        "<bos><|turn>system\nBe concise.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n"
    );
}

#[test]
fn openai_chat_prompt_renderer_ministral_available_tools_and_last_user_system() {
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }]);

    let tools_user: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Look up AX"}
    ]))
    .expect("messages");
    let prompt = render_openai_chat_prompt_with_tools(
        "ministral-8b-instruct",
        &tools_user,
        Some(&tools),
        Some(&json!("auto")),
    )
    .expect("ministral tools");
    assert!(prompt.starts_with("<s>[AVAILABLE_TOOLS] "));
    assert!(prompt.contains("[/AVAILABLE_TOOLS][INST]Look up AX[/INST]"));
    assert!(prompt.contains("\"name\":\"lookup\"") || prompt.contains("\"name\": \"lookup\""));

    let multi: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "Again"}
    ]))
    .expect("multi");
    let multi_prompt =
        render_openai_chat_prompt("ministral-8b-instruct", &multi).expect("ministral multi");
    assert_eq!(
        multi_prompt,
        "<s>[INST]Hi[/INST] Hello there</s>[INST]Be concise.\n\nAgain[/INST]"
    );
}

#[test]
fn openai_chat_prompt_renderer_ministral_tools_roundtrip_matches_hub() {
    // Hub requires 9-char tool call ids and TOOL_CALLS / TOOL_RESULTS markers.
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Look up AX"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "callABC12",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"query": "AX", "limit": 2}
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "callABC12",
            "content": "AX Engine"
        }
    ]))
    .expect("roundtrip messages");
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }]);
    let prompt = render_openai_chat_prompt_with_tools(
        "ministral-8b-instruct",
        &messages,
        Some(&tools),
        Some(&json!("auto")),
    )
    .expect("ministral tools roundtrip");

    assert!(
        prompt.starts_with("<s>[AVAILABLE_TOOLS] "),
        "tools before last user: {prompt}"
    );
    assert!(
        prompt.contains("[/AVAILABLE_TOOLS][INST]Look up AX[/INST]"),
        "{prompt}"
    );
    assert!(
        prompt.contains("[TOOL_CALLS] [")
            && prompt.contains("\"name\":\"lookup\"")
            && prompt.contains("\"arguments\":")
            && prompt.contains(", \"id\": \"callABC12\"}]</s>"),
        "TOOL_CALLS with 9-char id and function payload: {prompt}"
    );
    assert!(
        prompt.contains(
            "[TOOL_RESULTS] {\"content\": AX Engine, \"call_id\": \"callABC12\"}[/TOOL_RESULTS]"
        ),
        "TOOL_RESULTS raw content + call_id: {prompt}"
    );
    assert!(
        !prompt.contains("[INST]AX Engine[/INST]"),
        "tool results must not be framed as INST: {prompt}"
    );
}

#[test]
fn openai_chat_prompt_renderer_llama4_injects_tools_into_user_turn() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Look up AX"}
    ]))
    .expect("messages");
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }]);
    let prompt = render_openai_chat_prompt_with_tools(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        &messages,
        Some(&tools),
        Some(&json!("auto")),
    )
    .expect("llama4 tools");
    assert!(prompt.starts_with("<|begin_of_text|><|header_start|>user<|header_end|>\n\n"));
    assert!(prompt.contains("Given the following functions, please respond with a JSON"));
    assert!(prompt.contains("\"name\": \"lookup\"") || prompt.contains("\"name\":\"lookup\""));
    assert!(prompt.contains("Look up AX<|eot|>"));
    assert!(prompt.ends_with("<|header_start|>assistant<|header_end|>\n\n"));
    assert!(!prompt.contains("<|start_header_id|>"));
}

#[test]
fn openai_chat_prompt_renderer_llama4_tools_roundtrip_uses_python_and_ipython() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Look up AX"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"query": "AX", "limit": 2}
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "AX Engine"
        }
    ]))
    .expect("roundtrip messages");
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }]);
    let prompt = render_openai_chat_prompt_with_tools(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        &messages,
        Some(&tools),
        Some(&json!("auto")),
    )
    .expect("llama4 tools roundtrip");

    assert!(
        prompt.contains("<|python_start|><|python_end|>"),
        "empty assistant content still opens python wrappers: {prompt}"
    );
    assert!(
        prompt.contains("{\"name\":\"lookup\",\"parameters\":{\"limit\":2,\"query\":\"AX\"}}")
            || prompt
                .contains("{\"name\":\"lookup\",\"parameters\":{\"query\":\"AX\",\"limit\":2}}"),
        "tool call uses parameters not arguments: {prompt}"
    );
    assert!(
        prompt.contains("<|header_start|>ipython<|header_end|>\n\n\"AX Engine\"<|eot|>"),
        "tool results use ipython header + JSON string content: {prompt}"
    );
    assert!(
        !prompt.contains("<|header_start|>user<|header_end|>\n\nAX Engine"),
        "tool results must not be framed as user: {prompt}"
    );
    assert!(
        prompt.ends_with("<|header_start|>assistant<|header_end|>\n\n"),
        "generation prompt after tool result: {prompt}"
    );
}

#[test]
fn openai_chat_prompt_renderer_devstral_multi_turn_with_eos() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "Again"}
    ]))
    .expect("messages");
    let prompt = render_openai_chat_prompt("devstral-small", &messages).expect("devstral");
    assert_eq!(
        prompt,
        "<s>[SYSTEM_PROMPT]Be concise.[/SYSTEM_PROMPT][INST]Hi[/INST]Hello there</s>[INST]Again[/INST]"
    );
}

#[test]
fn openai_chat_prompt_renderer_preserves_qwen_coder_tool_schema_metadata() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Read README.md"}
    ]))
    .expect("sample messages should deserialize");
    let long_description =
        "Use this tool carefully with detailed operational guidance. ".repeat(40);

    let prompt = render_openai_chat_prompt_with_tools(
        "mlx-community/Qwen3-Coder-Next-4bit",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": long_description,
                    "parameters": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "description": long_description,
                        "default": {},
                        "examples": [{"path": "README.md"}],
                        "properties": {
                            "path": {
                                "type": "string",
                                "title": "path",
                                "description": long_description,
                                "default": "."
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["read", "write"],
                                "description": "Operation mode"
                            }
                        },
                        "required": ["path"]
                    }
                }
            }
        ])),
        Some(&json!("auto")),
    )
    .expect("qwen tool prompt should render");

    assert!(prompt.contains("<function>\n<name>read_file</name>"));
    assert!(prompt.contains("<parameter>\n<name>path</name>"));
    assert!(prompt.contains("<required>[\"path\"]</required>"));
    assert!(prompt.contains("<enum>[\"read\",\"write\"]</enum>"));
    assert!(prompt.contains("<$schema>http://json-schema.org/draft-07/schema#</$schema>"));
    assert!(prompt.contains("<default>{}</default>"));
    assert!(prompt.contains("<examples>[{\"path\":\"README.md\"}]</examples>"));
    assert!(prompt.contains("<default>.</default>"));
    assert!(prompt.contains(long_description.trim()));
}

#[test]
fn openai_chat_prompt_renderer_uses_qwen3_dense_json_tool_contract() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Read README.md"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt_with_tools(
        "mlx-community/Qwen3-4B-4bit",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a workspace file",
                    "parameters": {"type": "object"}
                }
            }
        ])),
        Some(&json!("auto")),
    )
    .expect("qwen3 dense tool prompt should render");

    assert!(prompt.contains("# Tools\n\nYou may call one or more functions"));
    assert!(
        prompt
            .contains("You are provided with function signatures within <tools></tools> XML tags:")
    );
    assert!(prompt.contains("\"name\":\"read_file\""));
    assert!(prompt.contains("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:"));
    assert!(prompt.contains("{\"name\": <function-name>, \"arguments\": <args-json-object>}"));
    assert!(!prompt.contains("<function>\n<name>read_file</name>"));
    assert!(prompt.ends_with(chat::QWEN_CHATML_ASSISTANT_GENERATION_PROMPT));
}

#[test]
fn openai_chat_prompt_renderer_uses_gemma4_ollama_tool_dsl() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Look up AX"},
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"query": "AX", "limit": 2}
                }
            }]
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "AX Engine"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt_with_tools(
        "gemma4-e2b",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup docs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ])),
        Some(&json!("auto")),
    )
    .expect("gemma4 tool prompt should render");

    // Golden string against Google Gemma 4 Canonical Chat Template (2026-07-09).
    let expected = concat!(
        "<bos><|turn>system\n",
        "<|tool>declaration:lookup{description:<|\"|>Lookup docs<|\"|>,parameters:{properties:{limit:{type:<|\"|>INTEGER<|\"|>},query:{description:<|\"|>Search query<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>query<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n",
        "<|turn>user\nLook up AX<turn|>\n",
        "<|turn>model\n",
        "<|tool_call>call:lookup{limit:2,query:<|\"|>AX<|\"|>}<tool_call|>",
        "<|tool_response>response:lookup{value:<|\"|>AX Engine<|\"|>}<tool_response|>",
    );
    assert_eq!(prompt, expected);
}

#[test]
fn openai_gemma4_keeps_tool_answer_in_same_model_turn() {
    // Official template: tool_call + tool_response + follow-up assistant text
    // stay in one <|turn>model … <turn|> (no second model open before answer).
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Look up AX"},
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"query": "AX", "limit": 2}
                }
            }]
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "AX Engine"},
        {"role": "assistant", "content": "AX Engine is a local runtime."}
    ]))
    .expect("sample messages should deserialize");

    let tools = json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }]);

    let prompt = render_openai_chat_prompt_with_tools(
        "gemma4-e2b",
        &messages,
        Some(&tools),
        Some(&json!("auto")),
    )
    .expect("gemma4 tool+answer prompt should render");

    let expected = concat!(
        "<bos><|turn>system\n",
        "<|tool>declaration:lookup{description:<|\"|>Lookup docs<|\"|>,parameters:{properties:{limit:{type:<|\"|>INTEGER<|\"|>},query:{description:<|\"|>Search query<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>query<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n",
        "<|turn>user\nLook up AX<turn|>\n",
        "<|turn>model\n",
        "<|tool_call>call:lookup{limit:2,query:<|\"|>AX<|\"|>}<tool_call|>",
        "<|tool_response>response:lookup{value:<|\"|>AX Engine<|\"|>}<tool_response|>",
        "AX Engine is a local runtime.<turn|>\n",
        "<|turn>model\n<|channel>thought\n<channel|>",
    );
    assert_eq!(prompt, expected);
    // Explicitly guard against the pre-fix double model-turn bug.
    assert!(
        !prompt.contains("<tool_response|><|turn>model\nAX Engine"),
        "follow-up answer must continue the same model turn, not open a new one"
    );
}

#[test]
fn openai_gemma4_strips_thinking_channels_from_history() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<|channel>thought\nplan\n<channel|>Hello there"},
        {"role": "user", "content": "Again"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt("gemma4-e2b", &messages).expect("gemma4 history");
    assert_eq!(
        prompt,
        "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello there<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    );
    assert!(
        !prompt.contains("<|channel>thought\nplan"),
        "prior thinking channels must be stripped from prefill"
    );
}

#[test]
fn openai_gemma4_enable_thinking_matches_canonical_template() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Hello"}
    ]))
    .expect("sample messages should deserialize");

    let prompt = render_openai_chat_prompt_with_options(
        "gemma4-e2b",
        &messages,
        None,
        None,
        ChatPromptRenderOptions {
            enable_thinking: true,
        },
    )
    .expect("thinking-on gemma4 prompt");

    assert_eq!(
        prompt,
        "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n"
    );
}

#[test]
fn openai_gemma4_pending_tool_call_opens_tool_response_slot() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Look up AX"},
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"query": "AX", "limit": 2}
                }
            }]
        }
    ]))
    .expect("sample messages should deserialize");

    let tools = json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }]);

    let prompt = render_openai_chat_prompt_with_tools(
        "gemma4-e2b",
        &messages,
        Some(&tools),
        Some(&json!("auto")),
    )
    .expect("pending tool prompt");

    assert!(prompt.ends_with(
        "<|tool_call>call:lookup{limit:2,query:<|\"|>AX<|\"|>}<tool_call|><|tool_response>"
    ));
}

#[test]
fn openai_gemma4_tool_roundtrip_with_thinking_opens_thought_channel() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Look up AX"},
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"query": "AX", "limit": 2}
                }
            }]
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "AX Engine"}
    ]))
    .expect("sample messages should deserialize");

    let tools = json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }]);

    let prompt = render_openai_chat_prompt_with_options(
        "gemma4-e2b",
        &messages,
        Some(&tools),
        Some(&json!("auto")),
        ChatPromptRenderOptions {
            enable_thinking: true,
        },
    )
    .expect("thinking tool roundtrip");

    assert!(
        prompt.ends_with(
            "<|tool_response>response:lookup{value:<|\"|>AX Engine<|\"|>}<tool_response|><|channel>thought\n"
        ),
        "after tool response with thinking on, open thought channel: {prompt}"
    );
    assert!(
        prompt.contains("<|think|>\n"),
        "thinking on injects <|think|> in the system turn"
    );
}

#[test]
fn openai_chat_prompt_renderer_replays_assistant_tool_calls() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Read README.md"},
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"README.md\"}"
                }
            }]
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "AX Engine"}
    ]))
    .expect("tool replay messages should deserialize");

    let prompt = render_openai_chat_prompt("mlx-community/Qwen3-Coder-Next-4bit", &messages)
        .expect("qwen replay prompt should render");

    assert!(prompt.contains("<|im_start|>assistant\n<tool_call>"));
    assert!(prompt.contains("<function=read_file>"));
    assert!(prompt.contains("<parameter=path>\nREADME.md\n</parameter>"));
    assert!(
        prompt
            .contains("<|im_start|>user\n<tool_response>\nAX Engine\n</tool_response>\n<|im_end|>")
    );
}

#[test]
fn openai_chat_prompt_renderer_accepts_input_text_parts() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe"},
                {"type": "text", "text": " this scene."}
            ]
        }
    ]))
    .expect("sample messages should deserialize");

    assert_eq!(
        render_openai_chat_prompt("unknown-local-model", &messages).expect("plain prompt"),
        "user: Describe this scene.\nassistant:"
    );
}

#[test]
fn openai_chat_prompt_renderer_rejects_raw_media_parts_with_tensor_route_guidance() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                    }
                }
            ]
        }
    ]))
    .expect("sample messages should deserialize");

    let error = render_openai_chat_prompt("unknown-local-model", &messages)
        .expect_err("raw OpenAI media must fail closed until preprocessing exists");

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error
            .1
            .error
            .message
            .contains("multimodal_inputs.gemma4_unified"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[test]
fn openai_chat_stop_sequences_merge_family_defaults_with_user_stop() {
    assert_eq!(
        openai_chat_stop_sequences("qwen3", None),
        vec!["<|im_end|>".to_string()]
    );
    assert_eq!(
        openai_chat_stop_sequences("Meta-Llama-3.1-8B-Instruct", None),
        vec!["<|eot_id|>".to_string()]
    );
    assert_eq!(
        openai_chat_stop_sequences("gemma4-e2b", None),
        vec!["<turn|>".to_string()]
    );
    assert_eq!(
        openai_chat_stop_sequences("mlx-community/GLM-4.7-Flash-4bit", None),
        vec![
            "<|endoftext|>".to_string(),
            "<|user|>".to_string(),
            "<|observation|>".to_string()
        ]
    );
    assert_eq!(
        openai_chat_stop_sequences(
            "gemma4-e2b",
            Some(OpenAiStopInput::Multiple(vec!["custom".to_string()]))
        ),
        vec!["custom".to_string(), "<turn|>".to_string()]
    );
    assert_eq!(
        openai_chat_stop_sequences(
            "gemma4-e2b",
            Some(OpenAiStopInput::Multiple(vec![
                "custom".to_string(),
                "<turn|>".to_string()
            ]))
        ),
        vec!["custom".to_string(), "<turn|>".to_string()]
    );
}

#[test]
fn openai_chat_prompt_renderer_rejects_known_families_without_verified_fallback() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Hello"}
    ]))
    .expect("sample messages should deserialize");

    // Llama 4 and Mistral Instruct families have verified fallbacks; keep this
    // list aligned with `ChatUnsupportedFamily` (gemma3 / mixtral / deepseek).
    for model_id in [
        "google/gemma-3-4b-it",
        "mixtral-8x7b-instruct",
        "deepseek-ai/DeepSeek-V3",
    ] {
        let error = render_openai_chat_prompt(model_id, &messages)
            .expect_err("known unsupported chat fallback should fail closed");
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(
            error.1.error.message.contains("not supported yet"),
            "unexpected error for {model_id}: {}",
            error.1.error.message
        );
    }

    // Supported secondary families must still render (not fail closed).
    for model_id in ["meta-llama/Llama-4-Scout-17B-16E-Instruct", "mistral-small"] {
        render_openai_chat_prompt(model_id, &messages)
            .unwrap_or_else(|err| panic!("{model_id} should render chat: {}", err.1.error.message));
    }
}

#[test]
fn openai_chat_template_kwargs_disable_thinking_for_qwen_and_glm() {
    assert_eq!(
        chat_template_kwargs_for_model_id("qwen3"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("mlx-community/GLM-4.7-Flash-4bit"),
        Some(json!({"enable_thinking": false}))
    );
    // Gemma 4 also defaults thinking off for delegated backends (mlx_lm).
    assert_eq!(
        chat_template_kwargs_for_model_id("gemma4-e2b"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("deepseek-ai/DeepSeek-V3"),
        None
    );
}

#[test]
fn native_chat_renderer_disables_thinking_for_qwen_reasoning_models() {
    // The native MLX path does not go through mlx_lm's chat template; it builds
    // the prompt locally. Match Qwen templates rendered with
    // enable_thinking=false so OpenAI-compatible short responses do not spend
    // the output budget on visible reasoning text.
    let messages = vec![("user".to_string(), "hi".to_string())];
    let no_thinking = chat::render_prompt("Qwen3.6-35B-A3B-4bit", &messages).expect("render");
    assert!(
        no_thinking.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "thinking-disabled suffix should pre-close the think block: {no_thinking}"
    );
}

#[test]
fn openai_chat_template_kwargs_disable_thinking_for_qwen_reasoning_models() {
    assert_eq!(
        chat_template_kwargs_for_model_id("Qwen3.6-35B-A3B-4bit"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("qwen3_6-35b-a3b-4bit"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("mlx-community/Qwen3.6-35B-A3B-4bit"),
        Some(json!({"enable_thinking": false}))
    );
    assert_eq!(
        chat_template_kwargs_for_model_id("Qwen3-Coder-Next-4bit"),
        Some(json!({"enable_thinking": false}))
    );
}

#[test]
fn openai_glm_prompt_renderer_preserves_tool_observation_shape() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "call tool"},
        {"role": "assistant", "content": "<tool_call>x</tool_call>"},
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "continue"}
    ]))
    .expect("sample messages should deserialize");

    assert_eq!(
        render_openai_chat_prompt("mlx-community/GLM-4.7-Flash-4bit", &messages)
            .expect("glm prompt"),
        "[gMASK]<sop><|user|>call tool<|assistant|></think><tool_call>x</tool_call><|observation|><tool_response>tool result</tool_response><|user|>continue<|assistant|></think>"
    );
}

#[tokio::test]
async fn openai_chat_request_applies_gemma4_default_stop_to_native_generate() {
    let state = test_app_state(|args| {
        args.model_id = "gemma4-e2b".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("chat request should build");

    assert_eq!(
        built.generate_request.stop_sequences,
        vec!["<turn|>".to_string()]
    );
}

#[tokio::test]
async fn openai_chat_request_marks_tool_and_structured_workload_metadata() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-tool-metadata");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Call a tool and return JSON"}],
        "max_tokens": 8,
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}}
            }
        ],
        "response_format": {"type": "json_object"}
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("chat request should build");
    let hints = RequestWorkloadHints::from_metadata(built.generate_request.metadata.as_deref());

    assert!(hints.tool_call);
    assert!(hints.structured_output);
    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_context_length_exceeded_before_generation() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-context-preflight");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "input_tokens": vec![1; 16 * 1024],
        "max_tokens": 1
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("context-overflowing chat request should fail before generation"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(
        error.1.0.error.code.as_deref(),
        Some("context_length_exceeded")
    );
    assert!(error.1.0.error.message.contains("16384 tokens"));
    assert!(error.1.0.error.message.contains("context length 16384"));
    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_honors_explicit_openai_max_tokens() {
    // An explicit max_tokens that fits within the model context is honored
    // verbatim — it must not be silently truncated to a fixed ceiling.
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-max-token-honor");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "input_tokens": [1, 2, 3],
        "max_tokens": 2048
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("chat request should build");

    assert_eq!(built.generate_request.max_output_tokens, 2048);
    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_prompt_plus_max_tokens_over_context() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-context-output-clamp");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "input_tokens": vec![1; 16 * 1024 - 10],
        "max_tokens": 2048
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("prompt plus max_tokens over context should fail before generation"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(
        error.1.0.error.code.as_deref(),
        Some("context_length_exceeded")
    );
    assert!(error.1.0.error.message.contains("prompt + 2048 output"));
    assert!(error.1.0.error.message.contains("context length 16384"));
    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_gemma4_tools_when_prompt_is_pre_tokenized() {
    let state = test_app_state(|args| {
        args.model_id = "gemma4-e2b".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Call a tool"}],
        "input_tokens": [1, 2, 3],
        "max_tokens": 8,
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}}
            }
        ]
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("pre-tokenized Gemma4 tools should fail closed"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(
        error.1.0.error.code.as_deref(),
        Some("unsupported_parameter")
    );
    assert!(
        error
            .1
            .0
            .error
            .message
            .contains("Gemma 4 OpenAI tool calling requires AX to render")
            && error.1.0.error.message.contains("<|tool_call>")
    );
}

#[tokio::test]
async fn openai_chat_request_rejects_top_logprobs_until_top_n_is_supported() {
    let state = test_app_state(|args| {
        args.model_id = "gemma4-e2b".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "logprobs": true,
        "top_logprobs": 5
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("top_logprobs should fail closed until top-N alternatives are available"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(
        error.1.0.error.code.as_deref(),
        Some("unsupported_parameter")
    );
}

#[tokio::test]
async fn openai_chat_request_rejects_unsupported_sampling_params() {
    let state = test_app_state(|args| {
        args.model_id = "gemma4-e2b".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let live = state.snapshot();

    for (field, value) in [
        ("n", json!(2)),
        ("frequency_penalty", json!(0.5)),
        ("presence_penalty", json!(-0.2)),
        ("logit_bias", json!({"50256": -100})),
    ] {
        let mut body = json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8
        });
        body.as_object_mut()
            .expect("chat body is an object")
            .insert(field.to_string(), value);
        let request: OpenAiChatCompletionHttpRequest =
            serde_json::from_value(body).expect("sample chat request should deserialize");
        let error = match build_openai_chat_request(&live, request) {
            Ok(_) => panic!("non-default {field} should fail closed instead of being ignored"),
            Err(error) => error,
        };
        assert_eq!(error.0, StatusCode::BAD_REQUEST, "{field}");
        assert_eq!(
            error.1.0.error.code.as_deref(),
            Some("unsupported_parameter"),
            "{field}"
        );
    }

    // OpenAI-default values stay accepted so standard clients keep working.
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "n": 1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0,
        "logit_bias": {}
    }))
    .expect("sample chat request should deserialize");
    build_openai_chat_request(&live, request)
        .expect("default-valued unsupported params should still build");
}

#[tokio::test]
async fn openai_chat_request_carries_client_stop_for_native_enforcement() {
    // The native MLX backend enforces client stops server-side over decoded
    // text (ADR-040 D2): the raw client list rides response options into the
    // stop scanner / truncation, and invalid lists fail closed up front.
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-stop-carried");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();

    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "stop": ["\n\nHuman:"]
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request)
        .expect("client stop should be accepted on the native backend");
    assert_eq!(
        built.response_options.client_stop_sequences,
        vec!["\n\nHuman:".to_string()]
    );

    // Validation bounds still fail closed: more than 4 sequences is invalid.
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "stop": ["a", "b", "c", "d", "e"]
    }))
    .expect("sample chat request should deserialize");
    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("more than 4 stop sequences must fail closed"),
        Err(error) => error,
    };
    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(error.1.0.error.code.as_deref(), Some("invalid_request"));

    // No stop at all builds with an empty enforcement list.
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");
    let built = build_openai_chat_request(&live, request)
        .expect("request without a client stop should still build on the native backend");
    assert!(built.response_options.client_stop_sequences.is_empty());

    std::fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_allows_client_stop_on_delegated_backend() {
    // Delegated backends genuinely forward `stop` upstream (llama.cpp `stop`,
    // mlx_lm `--stop`) — must not be rejected there.
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let live = state.snapshot();

    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "stop": ["\n\nHuman:"]
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request)
        .expect("client stop should be accepted on the delegated llama.cpp backend");
    assert!(
        built
            .generate_request
            .stop_sequences
            .contains(&"\n\nHuman:".to_string())
    );
}

#[tokio::test]
async fn openai_chat_request_rejects_streaming_json_object_validation() {
    let state = test_app_state(|args| {
        args.model_id = "gemma4-e2b".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Return JSON"}],
        "max_tokens": 8,
        "stream": true,
        "response_format": {"type": "json_object"}
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("streaming JSON object validation should fail closed"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(
        error.1.0.error.code.as_deref(),
        Some("unsupported_parameter")
    );
}

#[tokio::test]
async fn openai_chat_request_allows_streaming_tool_calls_for_native_qwen_buffering() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-stream-tools");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Call a tool"}],
        "max_tokens": 8,
        "stream": true,
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}}
        }]
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request)
        .expect("native Qwen streaming tool calls should build for buffered SSE");

    assert!(built.stream);
    assert!(built.response_options.parse_tool_calls);
    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_streaming_logprobs_and_reasoning() {
    // Streaming chunks do not carry logprobs or reasoning yet; the request
    // must fail closed instead of silently dropping the asked-for contract.
    let state = test_app_state(|args| {
        args.model_id = "gemma4-e2b".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let live = state.snapshot();

    for body in [
        json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "stream": true,
            "logprobs": true
        }),
        json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "stream": true,
            "reasoning": true
        }),
    ] {
        let request: OpenAiChatCompletionHttpRequest =
            serde_json::from_value(body).expect("sample chat request should deserialize");
        let error = match build_openai_chat_request(&live, request) {
            Ok(_) => panic!("streaming logprobs/reasoning should fail closed"),
            Err(error) => error,
        };
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert_eq!(
            error.1.0.error.code.as_deref(),
            Some("unsupported_parameter")
        );
    }
}

#[tokio::test]
async fn openai_chat_request_preserves_text_metadata_when_adding_workload_hints() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-text-metadata");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Call a tool"}],
        "max_tokens": 8,
        "metadata": "tenant=bench",
        "tool_choice": {"type": "function", "function": {"name": "lookup"}}
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("chat request should build");
    let metadata = built
        .generate_request
        .metadata
        .as_deref()
        .expect("metadata should include the original value and hints");
    let hints = RequestWorkloadHints::from_metadata(Some(metadata));

    assert!(metadata.contains("tenant=bench"));
    assert!(hints.tool_call);
    assert!(!hints.structured_output);
    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_unsupported_ax_rendered_family() {
    let state = test_app_state(|args| {
        args.model_id = "deepseek-ai/DeepSeek-V3".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
    });
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("AX-rendered fallback should fail closed"),
        Err(error) => error,
    };
    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(error.1.error.message.contains("deepseek"));
}

#[tokio::test]
async fn openai_chat_request_tokenizes_text_for_native_mlx_backend() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-tokenizer");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    validate_openai_request(&live, request.model.as_deref())
        .expect("native MLX OpenAI chat should pass validation");
    let built = build_openai_chat_request(&live, request).expect("chat request should build");

    assert!(
        !built.generate_request.input_tokens.is_empty(),
        "native MLX OpenAI chat prompt should be tokenized"
    );
    assert_eq!(built.generate_request.input_text, None);
    assert_eq!(built.generate_request.max_output_tokens, 8);
    assert_eq!(
        built.generate_request.stop_sequences,
        vec!["<|im_end|>".to_string()]
    );
    assert!(!built.stream);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_preserves_gemma4_multimodal_inputs_for_native_mlx_tokens() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-mm-tokenizer");
    let state = native_mlx_openai_builder_state("gemma-4-12b-it", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{"role": "user", "content": "Describe this image"}],
        "input_tokens": [10, 258880, 11],
        "max_tokens": 8,
        "multimodal_inputs": sample_gemma4_multimodal_inputs()
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("chat request should build");
    let inputs = built
        .generate_request
        .multimodal_inputs
        .gemma4_unified
        .expect("OpenAI chat should preserve processed Gemma4 media tensors");

    assert_eq!(built.generate_request.input_tokens, vec![10, 258880, 11]);
    assert_eq!(built.generate_request.input_text, None);
    assert_eq!(inputs.images.len(), 1);
    assert_eq!(inputs.images[0].pixel_values, vec![0.0, 1.0, 2.0]);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_decodes_inline_image_into_gemma4_unified_tensors() {
    use base64::Engine as _;

    let artifact_dir = gemma4_unified_artifact("native-openai-chat-inline-image");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();

    // The golden 16x16 fixture decodes to exactly 4 soft tokens under the
    // synthetic vision config baked into the artifact.
    let png = include_bytes!("fixtures/gemma4_golden/image_noresize.png");
    let data_uri = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    );
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }],
        "max_tokens": 8
    }))
    .expect("multimodal chat request should deserialize");

    let built =
        build_openai_chat_request(&live, request).expect("inline image chat request should build");

    // The image was decoded, preprocessed, and attached as Gemma4 unified tensors.
    let inputs = built
        .generate_request
        .multimodal_inputs
        .gemma4_unified
        .expect("inline image should attach Gemma4 unified tensors");
    assert_eq!(inputs.images.len(), 1);
    assert_eq!(inputs.images[0].span.soft_token_count, 4);
    assert_eq!(inputs.images[0].pixel_position_ids.len(), 4);
    assert_eq!(inputs.images[0].pixel_values.len(), 4 * 8 * 8 * 3);
    assert!(inputs.audios.is_empty());
    assert!(inputs.videos.is_empty());

    // The single <img> placeholder round-tripped and expanded into boi(102) +
    // four image soft tokens(100) + eoi(103) in the tokenized prompt.
    let tokens = &built.generate_request.input_tokens;
    assert_eq!(tokens.iter().filter(|&&token| token == 100).count(), 4);
    assert_eq!(tokens.iter().filter(|&&token| token == 102).count(), 1);
    assert_eq!(tokens.iter().filter(|&&token| token == 103).count(), 1);
    assert_eq!(built.generate_request.input_text, None);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_media_build_offloads_to_blocking_pool_and_matches_inline_build() {
    use base64::Engine as _;

    let artifact_dir = gemma4_unified_artifact("native-openai-chat-offload-media");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();

    let png = include_bytes!("fixtures/gemma4_golden/image_noresize.png");
    let data_uri = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    );
    let media_request = || -> OpenAiChatCompletionHttpRequest {
        serde_json::from_value(json!({
            "model": "qwen3",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }],
            "max_tokens": 8
        }))
        .expect("multimodal chat request should deserialize")
    };

    // Media requests run on the blocking pool and must build the same request
    // the inline (sync) build produces.
    let offloaded = build_openai_chat_request_offloading_media(&live, media_request())
        .await
        .expect("offloaded media build should succeed");
    let inline = build_openai_chat_request(&live, media_request())
        .expect("inline media build should succeed");
    assert_eq!(
        offloaded.generate_request.input_tokens,
        inline.generate_request.input_tokens
    );
    assert!(
        offloaded
            .generate_request
            .multimodal_inputs
            .gemma4_unified
            .is_some()
    );

    // Text-only requests take the inline fast path and still build.
    let text_request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8
    }))
    .expect("text chat request should deserialize");
    let built = build_openai_chat_request_offloading_media(&live, text_request)
        .await
        .expect("text-only build should succeed");
    assert!(
        built
            .generate_request
            .multimodal_inputs
            .gemma4_unified
            .is_none()
    );

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_inline_video() {
    let artifact_dir = gemma4_unified_artifact("native-openai-chat-inline-video");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
            ]
        }],
        "max_tokens": 8
    }))
    .expect("multimodal chat request should deserialize");

    let error = build_openai_chat_request(&live, request)
        .err()
        .expect("public chat surface must reject video");
    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(
        error.1.0.error.code.as_deref(),
        Some("unsupported_modality")
    );
    assert!(error.1.0.error.message.contains("video_url"));

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_video_even_with_supported_media() {
    let artifact_dir = gemma4_unified_artifact("native-openai-chat-combined");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
            ]
        }],
        "max_tokens": 8
    }))
    .expect("combined multimodal request should deserialize");

    let error = build_openai_chat_request(&live, request)
        .err()
        .expect("video must remain unsupported when image and audio are present");
    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(
        error.1.0.error.code.as_deref(),
        Some("unsupported_modality")
    );
    assert!(error.1.0.error.message.contains("video_url"));

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_decodes_mp3_input_audio() {
    use base64::Engine as _;

    let artifact_dir = gemma4_unified_artifact("native-openai-chat-mp3");
    let state = native_mlx_openai_builder_state("qwen3", &artifact_dir);
    let live = state.snapshot();

    // 0.5 s 16 kHz mono MP3 tone; decodes to ~9216 samples with encoder
    // delay/padding -> ceil(samples/640) frames.
    let mp3 = include_bytes!("fixtures/gemma4_golden/audio_tone_16k_mono.mp3");
    let audio_b64 = base64::engine::general_purpose::STANDARD.encode(mp3);

    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}}
            ]
        }],
        "max_tokens": 8
    }))
    .expect("mp3 chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("mp3 chat should build");
    let inputs = built
        .generate_request
        .multimodal_inputs
        .gemma4_unified
        .expect("mp3 audio should attach tensors");

    assert_eq!(inputs.audios.len(), 1);
    let audio = &inputs.audios[0];
    assert_eq!(audio.span.soft_token_count, audio.frame_count);
    assert!(
        (12..=18).contains(&audio.frame_count),
        "expected ~15 audio frames from the 0.5 s MP3, got {}",
        audio.frame_count
    );
    // The audio placeholder (101) expanded to one soft token per frame.
    let tokens = &built.generate_request.input_tokens;
    assert_eq!(
        tokens.iter().filter(|&&t| t == 101).count() as u32,
        audio.frame_count
    );

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_gemma4_multimodal_inputs_without_input_tokens() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-mm-no-tokens");
    let state = native_mlx_openai_builder_state("gemma-4-12b-it", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{"role": "user", "content": "Describe this image"}],
        "max_tokens": 8,
        "multimodal_inputs": sample_gemma4_multimodal_inputs()
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("chat media tensors without input_tokens should fail"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("pre-tokenized input"),
        "unexpected error: {}",
        error.1.error.message
    );

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_raw_media_parts_even_with_input_tokens() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-chat-token-raw-media");
    let state = native_mlx_openai_builder_state("gemma-4-12b-it", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
            ]
        }],
        "input_tokens": [10, 258880, 11],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("raw media parts should fail even with input_tokens"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error
            .1
            .error
            .message
            .contains("multimodal_inputs.gemma4_unified"),
        "unexpected error: {}",
        error.1.error.message
    );

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn delegated_openai_chat_rejects_input_tokens_extension() {
    let state = mlx_lm_delegated_state("http://127.0.0.1:1".to_string());
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{"role": "user", "content": "Describe this image"}],
        "input_tokens": [10, 11],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_mlx_lm_chat_request(&live, request) {
        Ok(_) => panic!("delegated text backends should reject tokenized chat"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error
            .1
            .error
            .message
            .contains("input_tokens require native MLX"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[tokio::test]
async fn delegated_openai_chat_rejects_gemma4_multimodal_inputs() {
    let state = mlx_lm_delegated_state("http://127.0.0.1:1".to_string());
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "gemma-4-12b-it",
        "messages": [{"role": "user", "content": "Describe this image"}],
        "max_tokens": 8,
        "multimodal_inputs": sample_gemma4_multimodal_inputs()
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_mlx_lm_chat_request(&live, request) {
        Ok(_) => panic!("delegated text backends should reject processed media tensors"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error
            .1
            .error
            .message
            .contains("OpenAI chat multimodal_inputs require native MLX backend"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[tokio::test]
async fn delegated_openai_chat_rejects_tools_on_mlx_lm_backend() {
    let state = mlx_lm_delegated_state("http://127.0.0.1:1".to_string());
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "glm4_moe_lite",
        "messages": [{"role": "user", "content": "Use read_file to inspect README.md"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        }],
        "tool_choice": "auto",
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_mlx_lm_chat_request(&live, request) {
        Ok(_) => panic!("delegated mlx-lm text backends should reject tool requests"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("delegated text backends"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[tokio::test]
async fn delegated_openai_chat_rejects_tools_on_llama_cpp_backend() {
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "local-gguf",
        "messages": [{"role": "user", "content": "Use read_file to inspect README.md"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        }],
        "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_llama_cpp_chat_request(&live, request) {
        Ok(_) => panic!("delegated llama.cpp text backends should reject tool requests"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("delegated text backends"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[tokio::test]
async fn delegated_openai_chat_rejects_assistant_tool_call_history() {
    let state = mlx_lm_delegated_state("http://127.0.0.1:1".to_string());
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "glm4_moe_lite",
        "messages": [
            {"role": "user", "content": "Read README.md"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                }]
            }
        ],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_mlx_lm_chat_request(&live, request) {
        Ok(_) => panic!("delegated mlx-lm text backends should reject tool-call history"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("delegated text backends"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[tokio::test]
async fn delegated_openai_chat_rejects_tool_result_history() {
    let state = llama_cpp_server_state("http://127.0.0.1:1".to_string());
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "local-gguf",
        "messages": [
            {"role": "user", "content": "Read README.md"},
            {"role": "tool", "tool_call_id": "call_0", "content": "{\"ok\":true}"}
        ],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_llama_cpp_chat_request(&live, request) {
        Ok(_) => panic!("delegated llama.cpp text backends should reject tool-result history"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("delegated text backends"),
        "unexpected error: {}",
        error.1.error.message
    );
}

#[tokio::test]
async fn openai_qwen_chat_uses_greedy_repetition_penalty_default() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-qwen-rp-tokenizer");
    let state = native_mlx_openai_builder_state("Qwen3.6-35B-A3B-4bit", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "Qwen3.6-35B-A3B-4bit",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 384,
        "temperature": 0.0
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("chat request should build");

    assert_eq!(built.generate_request.sampling.repetition_penalty, 1.1);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_qwen_chat_preserves_explicit_repetition_penalty() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-qwen-explicit-rp-tokenizer");
    let state = native_mlx_openai_builder_state("Qwen3.6-27B-4bit", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "Qwen3.6-27B-4bit",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 384,
        "temperature": 0.0,
        "repetition_penalty": 1.03
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("chat request should build");

    assert_eq!(built.generate_request.sampling.repetition_penalty, 1.03);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_glm_chat_keeps_standard_repetition_penalty_default() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-glm-tokenizer");
    let state = native_mlx_openai_builder_state("glm4_moe_lite", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "glm4_moe_lite",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 8,
        "temperature": 0.0
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("chat request should build");

    assert_eq!(built.generate_request.sampling.repetition_penalty, 1.0);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_glm_chat_preserves_explicit_repetition_penalty() {
    let artifact_dir = minimal_tokenizer_artifact("native-openai-glm-rp-tokenizer");
    let state = native_mlx_openai_builder_state("glm4_moe_lite", &artifact_dir);
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "model": "glm4_moe_lite",
        "messages": [{"role": "user", "content": "hello openai chat"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "repetition_penalty": 1.03
    }))
    .expect("sample chat request should deserialize");

    let built = build_openai_chat_request(&live, request).expect("chat request should build");

    assert_eq!(built.generate_request.sampling.repetition_penalty, 1.03);

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_request_rejects_gemma4_artifact_without_chat_template() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let artifact_dir =
        std::env::temp_dir().join(format!("ax-engine-server-gemma4-base-artifact-{unique}"));
    fs::create_dir_all(&artifact_dir).expect("artifact dir should create");

    let state = test_app_state(|args| {
        args.model_id = "gemma4".to_string();
        args.llama_server_url = Some("http://127.0.0.1:1".to_string());
        args.mlx_model_artifacts_dir = Some(artifact_dir.clone());
    });
    let live = state.snapshot();
    let request: OpenAiChatCompletionHttpRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8
    }))
    .expect("sample chat request should deserialize");

    let error = match build_openai_chat_request(&live, request) {
        Ok(_) => panic!("Gemma4 base artifact should fail closed for OpenAI chat"),
        Err(error) => error,
    };

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert!(
        error.1.error.message.contains("chat_template.jinja"),
        "unexpected error: {}",
        error.1.error.message
    );

    fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
}

#[tokio::test]
async fn openai_chat_completions_endpoint_defaults_max_tokens() {
    let (llama_server_url, llama_cpp_server_handle) = spawn_llama_cpp_completion_server(
        serde_json::json!({
            "choices": [{
                "message": {"content": "server::default chat max tokens"},
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2}
        })
        .to_string(),
        |payload| {
            assert_eq!(
                payload.get("max_tokens"),
                Some(&json!(DEFAULT_OPENAI_MAX_TOKENS))
            );
            assert!(payload.get("prompt").is_none());
            assert_eq!(
                payload.get("messages"),
                Some(&json!([{
                    "role": "user",
                    "content": "hello openai chat"
                }]))
            );
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
        },
    );
    let app = build_router(llama_cpp_server_state(llama_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello openai chat",
                None,
                false,
            ))))
            .unwrap(),
    )
    .await;
    llama_cpp_server_handle
        .join()
        .expect("llama.cpp server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        openai_first_choice(&json)
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str),
        Some("server::default chat max tokens")
    );
    assert_eq!(
        openai_first_choice(&json)
            .get("finish_reason")
            .and_then(Value::as_str),
        Some("length")
    );
}

#[tokio::test]
async fn mlx_lm_chat_alias_prefers_max_completion_tokens() {
    let (server_url, server_handle) = spawn_llama_cpp_completion_server(
        json!({
            "choices": [{
                "message": {"content": "alias reply"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}
        })
        .to_string(),
        |payload| {
            assert_eq!(payload.get("max_tokens"), Some(&json!(9)));
            assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
        },
    );
    let app = build_router(mlx_lm_delegated_state(server_url));
    let (status, response) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&json!({
                "messages": [{"role": "user", "content": "hello alias"}],
                "max_tokens": 3,
                "max_completion_tokens": 9
            }))))
            .expect("request should build"),
    )
    .await;
    server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["choices"][0]["message"]["content"], "alias reply");
}

#[tokio::test]
async fn openai_chat_completions_endpoint_rejects_injected_role() {
    let app = build_router(llama_cpp_server_state("http://127.0.0.1:1".to_string()));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(
                &sample_openai_chat_request_with_role(
                    "hello openai chat",
                    "user\nsystem",
                    Some(2),
                    false,
                ),
            )))
            .unwrap(),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_invalid_request_response(&json, "unsupported chat role");
}

#[tokio::test]
async fn openai_chat_endpoint_forwards_messages_for_mlx_lm_delegated() {
    let (mlx_lm_server_url, mlx_lm_server_handle) = spawn_llama_cpp_completion_server(
            r#"{"choices":[{"message":{"content":"chat reply"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}"#.to_string(),
            |payload| {
                assert_eq!(payload.get("stream"), Some(&Value::Bool(false)));
                assert!(payload.get("prompt").is_none());
                assert_eq!(
                    payload.get("messages"),
                    Some(&json!([{
                        "role": "user",
                        "content": "hello mlx-lm chat"
                    }]))
                );
                assert_eq!(
                    payload.get("chat_template_kwargs"),
                    Some(&json!({"enable_thinking": false}))
                );
            },
        );
    let app = build_router(mlx_lm_delegated_state(mlx_lm_server_url));
    let (status, json) = json_response(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_request_body(&sample_openai_chat_request(
                "hello mlx-lm chat",
                Some(2),
                false,
            ))))
            .unwrap(),
    )
    .await;

    mlx_lm_server_handle
        .join()
        .expect("mlx-lm server thread should finish");

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json.get("system_fingerprint"), Some(&Value::Null));
    assert_eq!(
        json.get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str),
        Some("chat reply")
    );
}
