use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use tracing::{error, info};

use crate::core::OpenAiClient;

pub async fn chat_completions(
    State(client): State<OpenAiClient>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let is_stream = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    info!(
        model = model,
        stream = is_stream,
        "OpenAI chat completions request"
    );

    let response = match client.chat_completion(body).await {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, "OpenAI API request failed");
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("OpenAI API error: {}", e),
                        "type": "proxy_error"
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
        return (status_code, body_text).into_response();
    }

    if is_stream {
        // Stream SSE directly from OpenAI to client
        let stream = OpenAiClient::stream_response(response);
        let body = Body::from_stream(stream);

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .body(body)
            .unwrap()
    } else {
        // Return JSON response directly
        let body_text = match response.text().await {
            Ok(t) => t,
            Err(e) => {
                error!(error = %e, "Failed to read OpenAI response");
                return (StatusCode::BAD_GATEWAY, "Failed to read response").into_response();
            }
        };

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(body_text))
            .unwrap()
    }
}
