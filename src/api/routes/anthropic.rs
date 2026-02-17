use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};

use crate::core::OpenAiClient;
use crate::api::transformers::{anthropic_to_openai, openai_to_anthropic};

pub async fn messages(
    State(client): State<OpenAiClient>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("unknown");
    let is_stream = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    info!(model = model, stream = is_stream, "Anthropic messages request");

    // Transform Anthropic request → OpenAI format
    let openai_body = anthropic_to_openai::transform_request(&body);

    let response = match client.chat_completion(openai_body).await {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, "OpenAI API request failed");
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": format!("OpenAI API error: {}", e)
                    }
                })),
            )
                .into_response();
        }
    };

    let status = response.status();

    if !status.is_success() {
        let status_code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let body_text = response.text().await.unwrap_or_default();
        error!(status = %status_code, body = %body_text, "OpenAI API returned error");
        
        // Try to parse and re-format as Anthropic error
        return (
            status_code,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": body_text
                }
            })),
        )
            .into_response();
    }

    if is_stream {
        // Streaming: transform OpenAI SSE → Anthropic SSE
        let model_owned = model.to_string();
        let byte_stream = OpenAiClient::stream_response(response);

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(128);

        tokio::spawn(async move {
            let mut transformer = openai_to_anthropic::StreamTransformer::new(&model_owned);
            let mut buffer = String::new();

            // Send message_start
            let start = transformer.start_event();
            if tx.send(Ok(start)).await.is_err() {
                return;
            }

            let mut stream = byte_stream;
            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        error!(error = %e, "Stream read error");
                        break;
                    }
                };

                let text = String::from_utf8_lossy(&chunk);
                buffer.push_str(&text);

                // Process complete SSE lines
                while let Some(pos) = buffer.find("\n\n") {
                    let line = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    for l in line.lines() {
                        if let Some(data) = l.strip_prefix("data: ") {
                            let events = transformer.process_chunk(data.trim());
                            for event in events {
                                if tx.send(Ok(event)).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                }
            }

            // Process any remaining buffer
            if !buffer.is_empty() {
                for l in buffer.lines() {
                    if let Some(data) = l.strip_prefix("data: ") {
                        let events = transformer.process_chunk(data.trim());
                        for event in events {
                            if tx.send(Ok(event)).await.is_err() {
                                return;
                            }
                        }
                    }
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .body(body)
            .unwrap()
    } else {
        // Non-streaming: transform response
        let body_text = match response.text().await {
            Ok(t) => t,
            Err(e) => {
                error!(error = %e, "Failed to read OpenAI response");
                return (StatusCode::BAD_GATEWAY, "Failed to read response").into_response();
            }
        };

        let openai_response: serde_json::Value = match serde_json::from_str(&body_text) {
            Ok(v) => v,
            Err(e) => {
                error!(error = %e, body = %body_text, "Failed to parse OpenAI response");
                return (StatusCode::BAD_GATEWAY, "Failed to parse response").into_response();
            }
        };

        let anthropic_response = openai_to_anthropic::transform_response(&openai_response, model);

        Json(anthropic_response).into_response()
    }
}
