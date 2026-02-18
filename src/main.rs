mod api;
mod common;
mod core;

use axum::{
    routing::{get, post},
    Json, Router,
};
use std::env;
use tower_http::cors::CorsLayer;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::api::routes;
use crate::core::{Config, OpenAiClient};

#[cfg(windows)]
fn hide_console() {
    use windows_sys::Win32::System::Console::GetConsoleWindow;
    use windows_sys::Win32::UI::WindowsAndMessaging::{ShowWindow, SW_HIDE};
    let window = unsafe { GetConsoleWindow() };
    if window != std::ptr::null_mut() {
        unsafe { ShowWindow(window, SW_HIDE) };
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let is_debug = args.iter().any(|arg| arg == "-debug");

    if !is_debug {
        // Enable ANSI support first so if it's visible it works,
        // but then hide it if not in debug mode
        #[cfg(windows)]
        let _ = nu_ansi_term::enable_ansi_support();

        #[cfg(windows)]
        hide_console();
    } else {
        // Enable ANSI support on Windows for debug mode
        #[cfg(windows)]
        let _ = nu_ansi_term::enable_ansi_support();
    }

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = Config::load();
    let client = OpenAiClient::new(config.clone());

    info!(
        port = config.port,
        base_url = %config.openai_base_url,
        config_path = %Config::get_config_path().display(),
        "Starting Proxy API"
    );

    // Initial connection check
    if let Err(e) = client.check_connection().await {
        tracing::warn!("⚠️  COULD NOT CONNECT TO API: {}", e);
        eprintln!("\n**************************************************");
        eprintln!("WARNING: Could not connect to the OpenAI API!");
        eprintln!("Error: {}", e);
        eprintln!("Please check your internet connection or base_url.");
        eprintln!("**************************************************\n");
    } else {
        info!("Successfully connected to Upstream API");
    }

    let app = Router::new()
        // Health check
        .route("/health", get(health))
        // OpenAI-compatible endpoint
        .route(
            "/v1/chat/completions",
            post(routes::openai::chat_completions),
        )
        // Anthropic-compatible endpoint
        .route("/v1/messages", post(routes::anthropic::messages))
        .layer(CorsLayer::permissive())
        .with_state(client);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", config.port))
        .await
        .expect("Failed to bind port");

    info!("Listening on 0.0.0.0:{}", config.port);

    axum::serve(listener, app).await.expect("Server error");
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "proxy-api"
    }))
}
