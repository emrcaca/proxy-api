use serde_json::{json, Value};
use uuid::Uuid;

/// Map OpenAI finish_reason to Anthropic stop_reason
fn map_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "content_filter" => "end_turn",
        _ => "end_turn",
    }
}

/// Transform a non-streaming OpenAI completion response into Anthropic Messages format.
///
/// This function transforms OpenAI responses, including:
/// - Regular text content
/// - Tool calls (tool_use blocks)
/// - Tool results (converted from OpenAI tool messages)
/// - Thinking/reasoning content
pub fn transform_response(openai_response: &Value, model: &str) -> Value {
    let msg_id = format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]);

    let choice = openai_response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or(json!({}));

    let message = choice.get("message").cloned().unwrap_or(json!({}));
    let finish_reason = choice.get("finish_reason").and_then(|f| f.as_str());

    let mut content_blocks: Vec<Value> = Vec::new();

    // Thinking/reasoning content
    if let Some(reasoning) = message.get("reasoning_content").and_then(|r| r.as_str()) {
        if !reasoning.is_empty() {
            content_blocks.push(json!({
                "type": "thinking",
                "thinking": reasoning
            }));
        }
    }

    // Tool calls
    if let Some(Value::Array(tool_calls)) = message.get("tool_calls") {
        // If there's text content before tool calls, add it
        if let Some(text) = message.get("content").and_then(|c| c.as_str()) {
            let filtered_text = strip_hallucinated_tags(text);
            if !filtered_text.is_empty() {
                content_blocks.push(json!({
                    "type": "text",
                    "text": filtered_text
                }));
            }
        }

        for tc in tool_calls {
            let func = tc.get("function").cloned().unwrap_or(json!({}));
            let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args_str = func.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");
            let args: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
            let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();

            content_blocks.push(json!({
                "type": "tool_use",
                "id": if id.is_empty() { format!("toolu_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]) } else { id },
                "name": name,
                "input": args
            }));
        }
    }
    // Tool results (from OpenAI tool messages)
    else if message.get("role").and_then(|r| r.as_str()) == Some("tool")
        || message.get("tool_call_id").is_some() {
        let tool_call_id = message.get("tool_call_id")
            .and_then(|i| i.as_str())
            .unwrap_or("")
            .to_string();

        let content = match message.get("content") {
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

        content_blocks.push(json!({
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
            "is_error": message.get("is_error").and_then(|e| e.as_bool()).unwrap_or(false)
        }));
    } else {
        // Regular text content
        let text = message.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let filtered_text = strip_hallucinated_tags(text);
        content_blocks.push(json!({
            "type": "text",
            "text": filtered_text
        }));
    }

    let usage = openai_response.get("usage").cloned().unwrap_or(json!({}));
    let input_tokens = usage.get("prompt_tokens").and_then(|t| t.as_u64()).unwrap_or(0);
    let output_tokens = usage.get("completion_tokens").and_then(|t| t.as_u64()).unwrap_or(0);

    let stop_reason = finish_reason.map(map_stop_reason).unwrap_or("end_turn");

    json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    })
}

/// State machine for transforming streaming OpenAI SSE events into Anthropic SSE events.
pub struct StreamTransformer {
    model: String,
    msg_id: String,
    content_index: i32,
    in_thinking: bool,
    in_tool_call: bool,
    in_tool_result: bool,
    current_tool_id: String,
    current_tool_name: String,
    current_tool_args: String,
    current_tool_result: String,
    current_tool_result_id: String,
    current_tool_result_is_error: bool,
    started: bool,
    input_tokens: u64,
    output_tokens: u64,
    last_finish_reason: Option<String>,
    tool_call_index: Option<i32>,
    in_text_block: bool,
}

impl StreamTransformer {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            msg_id: format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]),
            content_index: -1,
            in_thinking: false,
            in_tool_call: false,
            in_tool_result: false,
            current_tool_id: String::new(),
            current_tool_name: String::new(),
            current_tool_args: String::new(),
            current_tool_result: String::new(),
            current_tool_result_id: String::new(),
            current_tool_result_is_error: false,
            started: false,
            input_tokens: 0,
            output_tokens: 0,
            last_finish_reason: None,
            tool_call_index: None,
            in_text_block: false,
        }
    }

    /// Returns the initial message_start event
    pub fn start_event(&mut self) -> String {
        self.started = true;
        format_sse("message_start", &json!({
            "type": "message_start",
            "message": {
                "id": self.msg_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            }
        }))
    }

    /// Process a single OpenAI SSE delta chunk and return Anthropic SSE events
    pub fn process_chunk(&mut self, data: &str) -> Vec<String> {
        let mut events: Vec<String> = Vec::new();

        if data.trim() == "[DONE]" {
            // Close any open blocks
            events.extend(self.close_current_block());

            // Message delta with final info
            let stop_reason = self.last_finish_reason.as_deref().map(map_stop_reason).unwrap_or("end_turn");
            events.push(format_sse("message_delta", &json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": null
                },
                "usage": {
                    "output_tokens": self.output_tokens
                }
            })));

            events.push(format_sse("message_stop", &json!({
                "type": "message_stop"
            })));

            return events;
        }

        let chunk: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return events,
        };

        // Extract usage info if present
        if let Some(usage) = chunk.get("usage") {
            if let Some(pt) = usage.get("prompt_tokens").and_then(|t| t.as_u64()) {
                self.input_tokens = pt;
            }
            if let Some(ct) = usage.get("completion_tokens").and_then(|t| t.as_u64()) {
                self.output_tokens = ct;
            }
        }

        let choices = match chunk.get("choices").and_then(|c| c.as_array()) {
            Some(c) => c,
            None => return events,
        };

        for choice in choices {
            let delta = match choice.get("delta") {
                Some(d) => d,
                None => continue,
            };

            if let Some(fr) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                self.last_finish_reason = Some(fr.to_string());
            }

            // Handle reasoning/thinking content
            if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                if !reasoning.is_empty() {
                    if !self.in_thinking {
                        // Close any previous block
                        events.extend(self.close_current_block());
                        self.in_thinking = true;
                        self.content_index += 1;
                        events.push(format_sse("content_block_start", &json!({
                            "type": "content_block_start",
                            "index": self.content_index,
                            "content_block": {
                                "type": "thinking",
                                "thinking": ""
                            }
                        })));
                    }
                    events.push(format_sse("content_block_delta", &json!({
                        "type": "content_block_delta",
                        "index": self.content_index,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": reasoning
                        }
                    })));
                }
            }

            // Handle tool calls
            if let Some(Value::Array(tool_calls)) = delta.get("tool_calls") {
                for tc in tool_calls {
                    let tc_index = tc.get("index").and_then(|i| i.as_i64()).unwrap_or(0) as i32;
                    let func = tc.get("function").cloned().unwrap_or(json!({}));

                    // New tool call starting
                    if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                        // Close previous block if needed
                        events.extend(self.close_current_block());

                        self.in_tool_call = true;
                        self.tool_call_index = Some(tc_index);
                        self.current_tool_id = id.to_string();
                        self.current_tool_name = func.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                        self.current_tool_args = String::new();
                        self.content_index += 1;

                        events.push(format_sse("content_block_start", &json!({
                            "type": "content_block_start",
                            "index": self.content_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": self.current_tool_id,
                                "name": self.current_tool_name,
                                "input": {}
                            }
                        })));
                    }

                    // Tool call argument delta
                    if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                        if !args.is_empty() {
                            self.current_tool_args.push_str(args);
                            events.push(format_sse("content_block_delta", &json!({
                                "type": "content_block_delta",
                                "index": self.content_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": args
                                }
                            })));
                        }
                    }
                }
            }

            // Handle tool result (from OpenAI tool messages with tool_call_id)
            if let Some(role) = delta.get("role").and_then(|r| r.as_str()) {
                if role == "tool" || delta.get("tool_call_id").is_some() {
                    // Close any previous block
                    events.extend(self.close_current_block());

                    let tool_use_id = delta.get("tool_call_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();

                    if !tool_use_id.is_empty() && !self.in_tool_result {
                        self.in_tool_result = true;
                        self.current_tool_result_id = tool_use_id.clone();
                        self.current_tool_result = String::new();
                        self.current_tool_result_is_error = delta.get("is_error")
                            .and_then(|e| e.as_bool())
                            .unwrap_or(false);
                        self.content_index += 1;

                        events.push(format_sse("content_block_start", &json!({
                            "type": "content_block_start",
                            "index": self.content_index,
                            "content_block": {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": "",
                                "is_error": self.current_tool_result_is_error
                            }
                        })));
                    }
                }
            }

            // Handle tool result content (when it appears in content field for tool messages)
            if self.in_tool_result {
                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        self.current_tool_result.push_str(content);
                        events.push(format_sse("content_block_delta", &json!({
                            "type": "content_block_delta",
                            "index": self.content_index,
                            "delta": {
                                "type": "content_delta",
                                "partial_json": content
                            }
                        })));
                    }
                }
            }

            // Handle regular text content
            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                if !content.is_empty() && !self.in_tool_result {
                    // Start text block if not already in one
                    if !self.in_text_block {
                        let filtered_content = strip_hallucinated_tags(content);
                        
                        if !filtered_content.is_empty() {
                            events.extend(self.close_current_block());
                            self.in_text_block = true;
                            self.content_index += 1;
                            events.push(format_sse("content_block_start", &json!({
                                "type": "content_block_start",
                                "index": self.content_index,
                                "content_block": {
                                    "type": "text",
                                    "text": ""
                                }
                            })));
                            
                            events.push(format_sse("content_block_delta", &json!({
                                "type": "content_block_delta",
                                "index": self.content_index,
                                "delta": {
                                    "type": "text_delta",
                                    "text": filtered_content
                                }
                            })));
                        }
                    } else {
                        events.push(format_sse("content_block_delta", &json!({
                            "type": "content_block_delta",
                            "index": self.content_index,
                            "delta": {
                                "type": "text_delta",
                                "text": content
                            }
                        })));
                    }
                }
            }
        }

        events
    }

    fn close_current_block(&mut self) -> Vec<String> {
        let mut events = Vec::new();
        if self.in_thinking || self.in_tool_call || self.in_text_block || self.in_tool_result {
            events.push(format_sse("content_block_stop", &json!({
                "type": "content_block_stop",
                "index": self.content_index
            })));
        }
        self.in_thinking = false;
        self.in_tool_call = false;
        self.in_text_block = false;
        self.in_tool_result = false;
        events
    }
}

fn format_sse(event_type: &str, data: &Value) -> String {
    format!("event: {}\ndata: {}\n\n", event_type, serde_json::to_string(data).unwrap_or_default())
}

/// Strip leading tags that the model sometimes hallucinated at the start of its content block.
pub fn strip_hallucinated_tags(content: &str) -> String {
    let mut filtered_content = content.to_string();
    loop {
        let trimmed = filtered_content.trim_start();
        let mut changed = false;
        
        for tag in &[
            "</thinking>", "<thinking>",
            "</thought>", "<thought>",
            "</reasoning>", "<reasoning>",
            "[End of Reasoning]", "[Reasoning]:",
            "Reasoning:", "Thought:",
            "\n\n", "\n"
        ] {
            if trimmed.to_lowercase().starts_with(&tag.to_lowercase()) {
                // Remove the tag (case-insensitive check but replacen is case-sensitive, 
                // so we use the actual slice length from the original string)
                filtered_content = trimmed[tag.len()..].to_string();
                changed = true;
                break;
            }
        }
        
        if !changed {
            break;
        }
    }
    filtered_content.trim_start().to_string()
}
