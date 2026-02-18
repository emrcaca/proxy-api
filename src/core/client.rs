use bytes::Bytes;
use futures::Stream;
use reqwest::{Client, Response};
use std::pin::Pin;

use crate::core::Config;

#[derive(Clone)]
pub struct OpenAiClient {
    client: Client,
    config: Config,
    chat_completions_url: String,
    models_url: String,
}

impl OpenAiClient {
    pub fn new(config: Config) -> Self {
        let base = config.openai_base_url.trim_end_matches('/');
        let chat_completions_url = format!("{}/chat/completions", base);
        let models_url = format!("{}/models", base);
        Self {
            client: Client::new(),
            config,
            chat_completions_url,
            models_url,
        }
    }

    pub async fn chat_completion(
        &self,
        body: serde_json::Value,
    ) -> Result<Response, reqwest::Error> {
        self.client
            .post(&self.chat_completions_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!("Bearer {}", self.config.openai_api_key),
            )
            .json(&body)
            .send()
            .await
    }

    pub async fn check_connection(&self) -> Result<(), reqwest::Error> {
        // Try to list models as a connection check
        let response = self
            .client
            .get(&self.models_url)
            .header(
                "Authorization",
                format!("Bearer {}", self.config.openai_api_key),
            )
            .send()
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            // Even if it returns 401/404, the API is reachable.
            // Only return error if we can't reach the server at all.
            Ok(())
        }
    }

    pub fn stream_response(
        response: Response,
    ) -> Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>> {
        Box::pin(response.bytes_stream())
    }
}
