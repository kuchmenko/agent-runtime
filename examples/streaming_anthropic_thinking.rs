//! Real Anthropic streaming with positive thinking coverage.
//!
//! This example enables Anthropic extended thinking and asserts that the
//! stream yields provider-returned thinking text plus a finalized
//! `ThinkingBlock` with replay metadata.
//!
//! Env knobs:
//!   ANTHROPIC_API_KEY=sk-ant-...
//!   ANTHROPIC_THINKING_MODEL=claude-sonnet-4-6
//!   ANTHROPIC_THINKING_BUDGET=1024
//!
//! Run: `cargo run --example streaming_anthropic_thinking`

use std::io::Write;

use futures::StreamExt;
use tkach::{
    Agent, CancellationToken, Content, Message, StreamEvent, ThinkingMetadata, ThinkingProvider,
    providers::Anthropic,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    if api_key.is_empty() || api_key.starts_with("sk-ant-...") {
        eprintln!("skipping: ANTHROPIC_API_KEY missing, empty, or still the placeholder.");
        return Ok(());
    }

    let model = std::env::var("ANTHROPIC_THINKING_MODEL")
        .unwrap_or_else(|_| tkach::model::claude::SONNET.into());
    let budget = std::env::var("ANTHROPIC_THINKING_BUDGET")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(1024);
    let max_tokens = budget.saturating_add(1024);

    eprintln!("[model: {model}]  [thinking budget: {budget}]  [max tokens: {max_tokens}]");
    eprintln!();

    let provider = Anthropic::new(api_key).with_thinking_budget(budget);

    let agent = Agent::builder()
        .provider(provider)
        .model(model)
        .system(
            "Answer the final question in one short sentence. Do not put reasoning in \
             the final answer; the API is configured to return thinking separately.",
        )
        .max_turns(1)
        .max_tokens(max_tokens)
        .build();

    let mut stream = agent.stream(
        vec![Message::user_text(
            "Solve carefully: A box has 3 red balls and 2 blue balls. \
             Without replacement, what is the probability that two draws are both red?",
        )],
        CancellationToken::new(),
    );

    print!("> ");
    std::io::stdout().flush()?;

    let mut thinking_delta_chars = 0usize;
    let mut thinking_block_chars = 0usize;
    let mut thinking_blocks = 0usize;
    let mut signed_blocks = 0usize;
    let mut redacted_blocks = 0usize;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            StreamEvent::ThinkingDelta { text } => {
                thinking_delta_chars += text.chars().count();
                eprint!("\n[thinking] {text}");
                std::io::stderr().flush()?;
            }
            StreamEvent::ThinkingBlock {
                text,
                provider,
                metadata,
            } => {
                thinking_blocks += 1;
                thinking_block_chars += text.chars().count();
                match metadata {
                    ThinkingMetadata::Anthropic { signature: Some(_) } => signed_blocks += 1,
                    ThinkingMetadata::AnthropicRedacted { .. } => redacted_blocks += 1,
                    _ => {}
                }
                eprintln!(
                    "\n[thinking block: {provider:?}, {} chars; metadata preserved]",
                    text.chars().count()
                );
            }
            StreamEvent::ToolUse { name, .. } => {
                eprintln!("\n[unexpected tool: {name}]");
            }
            _ => {}
        }
    }
    println!();

    let result = stream.into_result().await?;
    eprintln!();
    eprintln!("--- summary ---");
    eprintln!("thinking deltas : {thinking_delta_chars} chars");
    eprintln!("thinking blocks : {thinking_blocks} blocks / {thinking_block_chars} chars");
    eprintln!("signed blocks   : {signed_blocks}");
    eprintln!("redacted blocks : {redacted_blocks}");
    eprintln!(
        "tokens          : {} in / {} out",
        result.usage.input_tokens, result.usage.output_tokens
    );

    assert!(
        thinking_blocks > 0,
        "expected at least one Anthropic thinking block; did you use a thinking-capable model?"
    );
    assert!(
        thinking_delta_chars > 0 || thinking_block_chars > 0,
        "expected non-empty Anthropic thinking text; provider returned only redacted/empty blocks"
    );
    assert!(
        signed_blocks + redacted_blocks > 0,
        "Anthropic thinking blocks should preserve signature or redacted replay metadata"
    );
    assert!(
        !result.text.trim().is_empty(),
        "final answer should be visible text"
    );
    assert!(
        result.new_messages.iter().any(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    Content::Thinking {
                        provider: ThinkingProvider::Anthropic,
                        ..
                    }
                )
            })
        }),
        "AgentResult history should preserve the finalized Anthropic thinking block"
    );

    eprintln!("✓ Anthropic thinking assertions passed");
    Ok(())
}
