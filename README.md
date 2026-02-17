# proxy-api

Rust tabanlı, istemciler ve yukarı akış LLM API'ları arasında çalışan bir API çeviri proksisidir. Hem OpenAI hem de Anthropic API formatları için uyumluluk katmanları sağlayarak, herhangi bir SDK kullanan istemcilerin yukarı akış servisiyle iletişim kurmasını sağlar.

A Rust-based API translation proxy that sits between clients and upstream LLM APIs. It provides compatibility layers for both OpenAI and Anthropic API formats, allowing clients using either SDK to communicate with the upstream service.

## Özellikler / Features

- OpenAI API uyumlu passthrough endpoint / OpenAI API compatibility passthrough endpoint
- Otomatik format dönüşümü ile Anthropic API uyumluluğu / Anthropic API compatibility with automatic format transformation
- Her iki endpoint için SSE (Server-Sent Events) akış desteği / Server-Sent Events (SSE) streaming support for both endpoints
- YAML dosyası üzerinden yapılandırma yönetimi / Configuration management via YAML file
- Çoklu platform desteği (Windows, macOS, Linux) / Cross-platform support (Windows, macOS, Linux)
- Sağlık kontrol endpoint'i / Health check endpoint
- CORS etkin / CORS enabled
- Başlangıçta bağlantı doğrulama / Connection validation on startup

## Mimari / Architecture

Proksi katmanlı bir mimarisi izler / The proxy follows a layered architecture:

```
Clients -> Routes Layer -> Transform Layer -> HTTP Client -> Upstream LLM API
```

### Çekirdek Modüller / Core Modules

| Modül / Module | Amaç / Purpose |
|----------------|----------------|
| `src/core/config.rs` | `{User Documents}/proxy-api/config.yaml` konumundan YAML yapılandırmasını yükler. Eksikse varsayılan oluşturur. / Loads YAML config from `{User Documents}/proxy-api/config.yaml`. Creates default if missing. |
| `src/core/client.rs` | Reqwest Client'ı yapılandırma ile sarar. Akışlı ve akışsız yukarı akış çağrılarını işler. / Wraps reqwest Client with config. Handles streaming and non-streaming upstream calls. |
| `src/api/routes/` | Axum route işleyicileri: `/health`, `/v1/chat/completions` (OpenAI passthrough), `/v1/messages` (Anthropic) / Axum route handlers: `/health`, `/v1/chat/completions` (OpenAI passthrough), `/v1/messages` (Anthropic) |
| `src/api/transformers/` | API formatları arasında dönüşüm yapar: `anthropic_to_openai.rs` (istek), `openai_to_anthropic.rs` (yanıt) / Converts between API formats: `anthropic_to_openai.rs` (request), `openai_to_anthropic.rs` (response) |

## Kurulum / Installation

### Ön Koşullar / Prerequisites

- Rust 2021 edition veya üstü / Rust 2021 edition or later
- Cargo paket yöneticisi / Cargo package manager

### Derleme / Build

```bash
# Debug build / Hata ayıklama derlemesi
cargo build

# Release build / Sürüm derlemesi
cargo build --release
```

## Yapılandırma / Configuration

Yapılandırma dosyası otomatik olarak `{User Documents}/proxy-api/config.yaml` konumunda oluşturulur (çapraz platform yolu `directories` crate'i ile otomatik algılanır).

Config file is created automatically at `{User Documents}/proxy-api/config.yaml` (cross-platform path automatically detected via `directories` crate).

Varsayılan yapılandırma dizini / Default config directory:
- **Windows**: `C:\Users\<kullanıcı>\Documents\proxy-api\` / `C:\Users\<user>\Documents\proxy-api\`
- **macOS**: `/Users/<kullanıcı>/Documents/proxy-api/` / `/Users/<user>/Documents/proxy-api/`
- **Linux**: `/home/<kullanıcı>/Documents/proxy-api/` / `/home/<user>/Documents/proxy-api/`

Yapılandırma dosyası yoksa, uygulama çıkış yapacak ve yapılandırma dosyası konumunu yazdıracaktır.

If the config file doesn't exist, the app will exit and print the config file location.

### Yapılandırma Dosyası Formatı / Config File Format

```yaml
openai:
  api_key: "api-anahtarınız-buraya" # "your-api-key-here"
  base_url: "https://integrate.api.nvidia.com/v1"
port: 3000
```

## Kullanım / Usage

### Proksiyi Çalıştırma / Running the Proxy

```bash
# Debug modunda çalıştır / Run in debug mode
cargo run

# Release modunda çalıştır / Run in release mode
cargo run --release

# Konsol görünür şekilde çalıştır (Windows) / Run with console visible (Windows)
cargo run -- -debug
```

### Endpoint'ler / Endpoints

#### Sağlık Kontrolü / Health Check
```
GET /health
```

Döndürür / Returns:
```json
{
  "status": "ok",
  "service": "proxy-api"
}
```

#### OpenAI Uyumlu Endpoint / OpenAI-Compatible Endpoint
```
POST /v1/chat/completions
```

Doğrudan yukarı akış OpenAI chat completions endpoint'ine passthrough yapar. Hem akışlı hem de akışsız yanıtları destekler.

Passthrough direct to upstream OpenAI chat completions endpoint. Supports both streaming and non-streaming responses.

#### Anthropic Uyumlu Endpoint / Anthropic-Compatible Endpoint
```
POST /v1/messages
```

İstekleri ve yanıtları Anthropic ve OpenAI formatları arasında otomatik olarak dönüştürür. Destekler:
- Sistem mesajları dönüşümü / System messages transformation
- İçerik blokları dönüşümü / Content blocks conversion
- Araçlar ve tool_choice işleme / Tools and tool_choice handling
- Uygun olay formatı dönüşümü ile SSE akışı / SSE streaming with proper event format conversion

Automatically transforms requests and responses between Anthropic and OpenAI formats. Supports:
- System messages transformation
- Content blocks conversion
- Tools and tool_choice handling
- SSE streaming with proper event format conversion

## İstek Akışı / Request Flow

1. **OpenAI istemcisi** (`/v1/chat/completions`): Doğrudan yukarı akışa passthrough / Passthrough directly to upstream
2. **Anthropic istemcisi** (`/v1/messages`):
   - `anthropic_to_openai.rs` isteği dönüştürür (sistem mesajları, içerik blokları, araçlar, tool_choice)
   - `POST /v1/chat/completions` yukarı akışına iletilir / Forwarded to upstream
   - `openai_to_anthropic.rs` yanıtı dönüştürür (finish_reason -> stop_reason, içerik blokları, SSE olayları)

## Akış Mimarisi / Streaming Architecture

Her iki endpoint de SSE akışını destekler. Akış uygulaması farklılık gösterir / Both endpoints support SSE streaming. The streaming implementation differs:

### OpenAI Yolu / OpenAI Route
Dönüştürme olmadan doğrudan yukarı akıştan istemciye akış yapar / Streams directly from upstream to client without transformation.

### Anthropic Yolu / Anthropic Route
Bir tokio worker görevi ve mpsc kanalı kullanarak akışı anında dönüştürür / Transforms stream on-the-fly using a tokio worker task and mpsc channel.

Anthropic dönüştürücüsü, OpenAI akış olaylarını Anthropic'in olay formatına dönüştürmek için bir durum makinesi (`StreamTransformer`) uygular / The Anthropic transformer implements a state machine (`StreamTransformer`) to convert OpenAI streaming events to Anthropic's event format:

- `message_start` - Initial message metadata / İlk mesaj meta verileri
- `content_block_start` - Start of a content block (type: `thinking`, `tool_use`, or `text`) / İçerik bloğunun başlangıcı (tür: `thinking`, `tool_use` veya `text`)
- `content_block_delta` - Incremental content updates / Artımlı içerik güncellemeleri
- `content_block_stop` - End of current content block / Mevcut içerik bloğunun sonu
- `message_delta` - Final metadata (stop_reason, usage) / Final meta veriler (stop_reason, usage)
- `message_stop` - Stream completion / Akış tamamlanması

Dönüştürücü birden çok içerik türü için durum korur / The transformer maintains state for multiple content types:
- reasoning_content -> `thinking` blokları / `thinking` blocks
- tool_calls -> `tool_use` blokları / `tool_use` blocks
- regular content -> `text` blokları / `text` blocks

## Hata İşleme / Error Handling

Proksi tutarlı hata işleme kalıplarını izler / The proxy follows consistent error handling patterns:
- Yukarı akış HTTP hataları uygun API yanıt formatlarına yeniden formatlanır / Upstream HTTP errors are re-formatted to appropriate API response formats
- Akış hataları kaydedilir ancak bağlantı nazikçe ele alınır / Stream errors are logged but the connection is gracefully handled
- Tüm hatalar uygun HTTP durum kodlarını döndürür / All errors return appropriate HTTP status codes

## Geliştirme / Development

### Komutlar / Commands

```bash
cargo build              # Debug build / Hata ayıklama derlemesi
cargo build --release    # Release build / Sürüm derlemesi
cargo run                # Run in debug mode / Debug modunda çalıştır
cargo run -r             # Run in release mode / Release modunda çalıştır
cargo run -- -debug      # Run with console visible (Windows)
cargo fmt                # Format code / Kodları formatla
cargo clippy             # Run linter / Linter çalıştır
cargo check              # Quick compile check / Hızlı derleme kontrolü
cargo test               # Run tests / Testleri çalıştır
```

## Lisans / License

Bu proje proxy API çeviri amacıyla olduğu gibi sağlanmıştır. / This project is provided as-is for proxy API translation purposes.
