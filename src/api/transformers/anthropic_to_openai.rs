use serde_json::{json, Value};
use tracing::debug;

/// Transforms an Anthropic Messages API request into an OpenAI-compatible request for the upstream API.
pub fn transform_request(anthropic_body: &Value) -> Value {
    let model = anthropic_body.get("model").cloned().unwrap_or(json!(""));
    let max_tokens = anthropic_body.get("max_tokens").cloned().unwrap_or(json!(1024));
    let stream = anthropic_body.get("stream").cloned().unwrap_or(json!(false));

    let mut openai_messages: Vec<Value> = Vec::new();

    // System message
    if let Some(system) = anthropic_body.get("system") {
        match system {
            Value::String(s) => {
                openai_messages.push(json!({
                    "role": "system",
                    "content": s
                }));
            }
            Value::Array(blocks) => {
                // Anthropic system can be array of content blocks
                let text: String = blocks
                    .iter()
                    .filter_map(|b| {
                        if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                            b.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    openai_messages.push(json!({
                        "role": "system",
                        "content": text
                    }));
                }
            }
            _ => {}
        }
    }

    // Convert messages
    if let Some(Value::Array(messages)) = anthropic_body.get("messages") {
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");

            match msg.get("content") {
                Some(Value::String(text)) => {
                    openai_messages.push(json!({
                        "role": role,
                        "content": text
                    }));
                }
                Some(Value::Array(blocks)) => {
                    convert_content_blocks(role, blocks, &mut openai_messages);
                }
                _ => {}
            }
        }
    }

    let mut openai_body = json!({
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": stream,
    });

    // Pass through optional parameters
    if let Some(temp) = anthropic_body.get("temperature") {
        openai_body["temperature"] = temp.clone();
    }
    if let Some(top_p) = anthropic_body.get("top_p") {
        openai_body["top_p"] = top_p.clone();
    }
    if let Some(Value::Array(stop)) = anthropic_body.get("stop_sequences") {
        openai_body["stop"] = json!(stop);
    }

    // Tools
    if let Some(Value::Array(tools)) = anthropic_body.get("tools") {
        let openai_tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").cloned().unwrap_or(json!("")),
                        "description": tool.get("description").cloned().unwrap_or(json!("")),
                        "parameters": tool.get("input_schema").cloned().unwrap_or(json!({})),
                    }
                })
            })
            .collect();
        openai_body["tools"] = json!(openai_tools);

        // Tool choice
        if let Some(tc) = anthropic_body.get("tool_choice") {
            let tc_type = tc.get("type").and_then(|t| t.as_str()).unwrap_or("auto");
            match tc_type {
                "auto" => openai_body["tool_choice"] = json!("auto"),
                "any" => openai_body["tool_choice"] = json!("required"),
                "tool" => {
                    if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                        openai_body["tool_choice"] = json!({
                            "type": "function",
                            "function": { "name": name }
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // Thinking / reasoning support
    if let Some(thinking) = anthropic_body.get("thinking") {
        if let Some(Value::Bool(true)) = thinking.get("enabled") {
            // Some models use reasoning_effort or specific thinking params
            if let Some(budget) = thinking.get("budget_tokens") {
                openai_body["reasoning"] = json!({
                    "max_tokens": budget
                });
            }
            // Some models use thinking parameter directly
            openai_body["thinking"] = thinking.clone();
        }
    }

    // Stream options
    if stream.as_bool().unwrap_or(false) {
        openai_body["stream_options"] = json!({ "include_usage": true });
    }

    debug!(openai_body = %openai_body, "Transformed Anthropic → OpenAI request");
    openai_body
}

fn convert_content_blocks(role: &str, blocks: &[Value], messages: &mut Vec<Value>) {
    match role {
        "user" => {
            let mut parts: Vec<Value> = Vec::new();
            for block in blocks {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                match block_type {
                    "text" => {
                        parts.push(json!({
                            "type": "text",
                            "text": block.get("text").cloned().unwrap_or(json!(""))
                        }));
                    }
                    "image" => {
                        if let Some(source) = block.get("source") {
                            let media_type = source.get("media_type").and_then(|m| m.as_str()).unwrap_or("image/png");
                            let data = source.get("data").and_then(|d| d.as_str()).unwrap_or("");
                            parts.push(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{};base64,{}", media_type, data)
                                }
                            }));
                        }
                    }
                    "tool_result" => {
                        // Tool results from user go as separate tool messages
                        let tool_use_id = block.get("tool_use_id").and_then(|i| i.as_str()).unwrap_or("");
                        let content = match block.get("content") {
                            Some(Value::String(s)) => s.clone(),
                            Some(Value::Array(arr)) => {
                                arr.iter()
                                    .filter_map(|b| {
                                        if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                                            b.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            }
                            _ => String::new(),
                        };

                        // Flush any accumulated parts first
                        if !parts.is_empty() {
                            messages.push(json!({
                                "role": "user",
                                "content": parts
                            }));
                            parts = Vec::new();
                        }

                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": content
                        }));
                    }
                    _ => {}
                }
            }
            if !parts.is_empty() {
                messages.push(json!({
                    "role": "user",
                    "content": parts
                }));
            }
        }
        "assistant" => {
            let mut text_content = String::new();
            let mut tool_calls: Vec<Value> = Vec::new();

            for block in blocks {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                match block_type {
                    "text" => {
                        if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                            text_content.push_str(t);
                        }
                    }
                    "tool_use" => {
                        let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                        let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                        let input = block.get("input").cloned().unwrap_or(json!({}));
                        tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": serde_json::to_string(&input).unwrap_or_default()
                            }
                        }));
                    }
                    "thinking" => {
                        // Skip thinking blocks in history to prevent the model from mimicking the format
                        // OpenAI-compatible APIs generally don't support input reasoning blocks.
                    }
                    _ => {}
                }
            }

            let mut assistant_msg = json!({ "role": "assistant" });
            if !text_content.is_empty() {
                assistant_msg["content"] = json!(text_content);
            }
            if !tool_calls.is_empty() {
                assistant_msg["tool_calls"] = json!(tool_calls);
            }
            messages.push(assistant_msg);
        }
        _ => {
            // Other roles: just pass text through
            let text: String = blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
                .collect::<Vec<_>>()
                .join("\n");
            messages.push(json!({
                "role": role,
                "content": text
            }));
        }
    }
}
