//! Real Anthropic adaptive-thinking streaming with positive thinking coverage.
//!
//! Adaptive mode is Anthropic's recommended path for Claude Sonnet 4.6,
//! Opus 4.6, and Opus 4.7+. It has no fixed `budget_tokens`; Claude
//! decides whether/how much to think from the task and optional effort.
//! This example requests summarized display and asserts that thinking
//! events arrive.
//!
//! Env knobs:
//!   ANTHROPIC_API_KEY=sk-ant-...
//!   ANTHROPIC_ADAPTIVE_THINKING_MODEL=claude-sonnet-4-6
//!   ANTHROPIC_ADAPTIVE_THINKING_EFFORT=high
//!   ANTHROPIC_ADAPTIVE_THINKING_MAX_TOKENS=4096
//!
//! Run: `cargo run --example streaming_anthropic_adaptive_thinking`

use std::io::Write;

use futures::StreamExt;
use tkach::{
    Agent, CancellationToken, Content, Message, StreamEvent, ThinkingMetadata, ThinkingProvider,
    providers::anthropic::{Anthropic, AnthropicEffort},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    if api_key.is_empty() || api_key.starts_with("sk-ant-...") {
        eprintln!("skipping: ANTHROPIC_API_KEY missing, empty, or still the placeholder.");
        return Ok(());
    }

    let model = std::env::var("ANTHROPIC_ADAPTIVE_THINKING_MODEL")
        .unwrap_or_else(|_| tkach::model::claude::SONNET.into());
    let effort: AnthropicEffort = std::env::var("ANTHROPIC_ADAPTIVE_THINKING_EFFORT")
        .map(Into::into)
        .unwrap_or(AnthropicEffort::High);
    let max_tokens = std::env::var("ANTHROPIC_ADAPTIVE_THINKING_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(4096);

    eprintln!(
        "[model: {model}]  [adaptive effort: {}]  [max tokens: {max_tokens}]",
        effort.as_wire()
    );
    eprintln!();

    let provider = Anthropic::new(api_key).with_adaptive_thinking_effort(effort);

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
            "Solve carefully: Prove briefly that the square root of 2 is irrational, \
             then state the core contradiction in one sentence.",
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
        "expected at least one Anthropic adaptive-thinking block; use high/max effort for this positive smoke"
    );
    assert!(
        thinking_delta_chars > 0 || thinking_block_chars > 0,
        "expected non-empty adaptive-thinking summary text"
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

    eprintln!("✓ Anthropic adaptive-thinking assertions passed");
    Ok(())
}
