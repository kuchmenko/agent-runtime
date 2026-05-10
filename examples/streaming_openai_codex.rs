//! Streaming example against the ChatGPT subscription Codex backend.
//!
//! ## Credential ownership
//!
//! `OpenAICodex` does not read environment variables, refresh tokens,
//! or talk to a keyring. **This example reads from environment
//! variables for demo purposes only** — production callers wrap their
//! own OAuth client / token cache and pass it as
//! [`CodexCredentialsProvider`].
//!
//! Env knobs (example-side, not provider behavior):
//!   OPENAI_CODEX_ACCESS_TOKEN=...     # required
//!   OPENAI_CODEX_ACCOUNT_ID=...       # required
//!   OPENAI_CODEX_BASE_URL=https://chatgpt.com/backend-api  # optional
//!   OPENAI_CODEX_MODEL=gpt-5-codex    # optional
//!   OPENAI_CODEX_ORIGINATOR=tkach     # optional
//!
//! Run: `cargo run --example streaming_openai_codex`

use std::io::Write;

use futures::StreamExt;
use tkach::{
    Agent, CancellationToken, Message, ProviderError, StreamEvent,
    providers::{CodexCredentials, CodexCredentialsProvider, OpenAICodex},
};

/// Minimal credential provider that captures token + account once at
/// startup and returns clones on every call. Real consumers would call
/// their token cache here and refresh on expiry.
struct EnvCredentials {
    creds: CodexCredentials,
}

#[async_trait::async_trait]
impl CodexCredentialsProvider for EnvCredentials {
    async fn credentials(&self) -> Result<CodexCredentials, ProviderError> {
        Ok(self.creds.clone())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let access_token = std::env::var("OPENAI_CODEX_ACCESS_TOKEN").unwrap_or_default();
    let account_id = std::env::var("OPENAI_CODEX_ACCOUNT_ID").unwrap_or_default();
    if access_token.is_empty() || account_id.is_empty() {
        eprintln!(
            "skipping: OPENAI_CODEX_ACCESS_TOKEN / OPENAI_CODEX_ACCOUNT_ID not set. \
             This example needs a ChatGPT subscription bearer token + account id."
        );
        return Ok(());
    }

    let base_url = std::env::var("OPENAI_CODEX_BASE_URL")
        .unwrap_or_else(|_| "https://chatgpt.com/backend-api".to_string());
    let model = std::env::var("OPENAI_CODEX_MODEL")
        .unwrap_or_else(|_| tkach::model::gpt::FIVE_CODEX.to_string());
    let originator = std::env::var("OPENAI_CODEX_ORIGINATOR").unwrap_or_else(|_| "tkach".into());

    eprintln!("[model: {model}]  [base: {base_url}]  [originator: {originator}]");
    eprintln!();

    let provider = OpenAICodex::new(EnvCredentials {
        creds: CodexCredentials::new(access_token, account_id),
    })
    .with_base_url(base_url)
    .with_originator(originator);

    let agent = Agent::builder()
        .provider(provider)
        .model(model)
        .system("Answer in one short sentence.")
        .max_turns(1)
        .max_tokens(512)
        .build()
        .unwrap();

    let mut stream = agent.stream(
        vec![Message::user_text("What is 2+2?")],
        CancellationToken::new(),
    );

    print!("> ");
    std::io::stdout().flush()?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            StreamEvent::ThinkingDelta { text } => {
                eprint!("\n[thinking] {text}");
                std::io::stderr().flush()?;
            }
            StreamEvent::ThinkingBlock { text, .. } => {
                eprintln!("\n[thinking block: {} chars]", text.chars().count());
            }
            _ => {}
        }
    }
    println!();

    let result = stream.into_result().await?;
    eprintln!();
    eprintln!(
        "tokens: {} in / {} out",
        result.usage.input_tokens, result.usage.output_tokens
    );
    Ok(())
}
