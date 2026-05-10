//! Real OpenAI Responses streaming with positive thinking coverage.
//!
//! This example is intentionally different from `streaming_openai_tools`:
//! it uses `/responses`, requests `reasoning.summary`, and asserts that
//! at least one provider-returned reasoning summary block arrives.
//!
//! Env knobs:
//!   OPENAI_RESPONSES_API_KEY=sk-...          # falls back to OPENAI_API_KEY
//!   OPENAI_RESPONSES_BASE_URL=https://api.openai.com/v1
//!   OPENAI_RESPONSES_MODEL=gpt-5
//!   OPENAI_RESPONSES_REASONING_EFFORT=medium
//!   OPENAI_RESPONSES_REASONING_SUMMARY=detailed
//!
//! If targeting a compatible proxy that implements `/responses`, set
//! `OPENAI_RESPONSES_BASE_URL` and `OPENAI_RESPONSES_MODEL` explicitly.
//!
//! Run: `cargo run --example streaming_openai_responses_thinking`

use std::io::Write;

use futures::StreamExt;
use tkach::{
    Agent, CancellationToken, Content, Message, StreamEvent, ThinkingMetadata, ThinkingProvider,
    providers::{OpenAIEffort, OpenAIResponses, OpenAISummary},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let api_key = std::env::var("OPENAI_RESPONSES_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_default();
    if api_key.is_empty() || api_key.starts_with("sk-...") {
        eprintln!(
            "skipping: OPENAI_RESPONSES_API_KEY or OPENAI_API_KEY missing, empty, \
             or still the placeholder."
        );
        return Ok(());
    }

    let base_url = std::env::var("OPENAI_RESPONSES_BASE_URL")
        .or_else(|_| std::env::var("OPENAI_BASE_URL"))
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let model = std::env::var("OPENAI_RESPONSES_MODEL").unwrap_or_else(|_| {
        if base_url.contains("openrouter.ai") {
            std::env::var("OPENAI_SMOKE_MODEL")
                .unwrap_or_else(|_| tkach::model::openrouter::OPENAI_GPT_5_5.to_string())
        } else {
            tkach::model::gpt::FIVE.to_string()
        }
    });
    let effort: OpenAIEffort = std::env::var("OPENAI_RESPONSES_REASONING_EFFORT")
        .map(Into::into)
        .unwrap_or(OpenAIEffort::Medium);
    let summary: OpenAISummary = std::env::var("OPENAI_RESPONSES_REASONING_SUMMARY")
        .map(Into::into)
        .unwrap_or(OpenAISummary::Detailed);

    eprintln!(
        "[model: {model}]  [base: {base_url}]  [reasoning: {}/{}]",
        effort.as_wire(),
        summary.as_wire()
    );
    eprintln!();

    let provider = OpenAIResponses::new(api_key)
        .with_base_url(base_url)
        .with_reasoning(effort, summary);

    let agent = Agent::builder()
        .provider(provider)
        .model(model)
        .system(
            "Answer the final question in one short sentence. Do not print your reasoning; \
             the API stream is configured to return a separate reasoning summary.",
        )
        .max_turns(1)
        .max_tokens(1024)
        .build();

    let mut stream = agent.stream(
        vec![Message::user_text(
            "Solve this carefully: A box has 3 red balls and 2 blue balls. \
             Without replacement, what is the probability that two draws are both red?",
        )],
        CancellationToken::new(),
    );

    print!("> ");
    std::io::stdout().flush()?;

    let mut thinking_delta_chars = 0usize;
    let mut thinking_block_chars = 0usize;
    let mut thinking_blocks = 0usize;
    let mut encrypted_blocks = 0usize;

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
                if matches!(
                    metadata,
                    ThinkingMetadata::OpenAIResponses {
                        encrypted_content: Some(_),
                        ..
                    }
                ) {
                    encrypted_blocks += 1;
                }
                eprintln!(
                    "\n[thinking block: {provider:?}, {} chars; replay metadata preserved]",
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
    eprintln!("encrypted blocks: {encrypted_blocks}");
    eprintln!(
        "tokens          : {} in / {} out",
        result.usage.input_tokens, result.usage.output_tokens
    );

    assert!(
        thinking_blocks > 0,
        "expected at least one OpenAI Responses reasoning summary block; \
         Chat Completions cannot satisfy this test"
    );
    assert!(
        thinking_delta_chars > 0 || thinking_block_chars > 0,
        "expected non-empty thinking summary text"
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
                        provider: ThinkingProvider::OpenAIResponses,
                        ..
                    }
                )
            })
        }),
        "AgentResult history should preserve the finalized OpenAI reasoning block"
    );

    eprintln!("✓ OpenAI Responses thinking assertions passed");
    Ok(())
}
