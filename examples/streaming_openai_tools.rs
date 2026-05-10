//! Real OpenAI-compatible streaming with a tool call.
//!
//! The smoke we shipped earlier only covered text round-trip (PING/PONG).
//! This example exercises the trickier path: the model decides to call
//! a tool, the SSE stream delivers `tool_calls.function.arguments` as a
//! sequence of partial-JSON fragments, our parser accumulates them
//! across chunks, and on `[DONE]` (or `finish_reason: tool_calls`) we
//! emit one atomic `StreamEvent::ToolUse` with parsed input.
//!
//! Chat Completions has no standard thinking block. The consumer still
//! handles `ThinkingDelta` / `ThinkingBlock` so UI code can share one
//! event loop across providers, then asserts this provider does not leak
//! non-standard reasoning fields by default.
//!
//! Defaults to OpenRouter; override base URL + model via env. Examples:
//!
//!   OPENAI_BASE_URL=https://openrouter.ai/api/v1
//!   OPENAI_SMOKE_MODEL=openai/gpt-5.5
//!   OPENAI_SMOKE_MODEL=anthropic/claude-haiku-4-5
//!   OPENAI_SMOKE_MODEL=moonshotai/kimi-k2.6     # works, may need verbose system
//!
//! Run:  `cargo run --example streaming_openai_tools`

use std::io::Write;

use futures::StreamExt;
use tkach::message::Content;
use tkach::{Agent, CancellationToken, Message, StreamEvent, providers::OpenAICompatible};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `dotenv_override` (not `dotenv`) so .env beats any pre-existing
    // shell vars. Otherwise an empty `OPENAI_API_KEY=` left over from
    // a previous session silently masks the real key in .env.
    let _ = dotenvy::dotenv_override();

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if api_key.is_empty() || api_key.starts_with("sk-...") {
        eprintln!(
            "skipping: OPENAI_API_KEY missing, empty, or still the placeholder. \
             set it in .env to enable this example."
        );
        return Ok(());
    }

    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
    let model = std::env::var("OPENAI_SMOKE_MODEL")
        .unwrap_or_else(|_| tkach::model::openrouter::OPENAI_GPT_5_5.to_string());

    eprintln!("[model: {model}]  [base: {base_url}]");
    eprintln!();

    let dir = std::env::temp_dir().join("tkach_streaming_openai");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let provider = OpenAICompatible::new(api_key).with_base_url(base_url);

    let agent = Agent::builder()
        .provider(provider)
        .model(model)
        .system(
            "You are concise. When asked to run a shell command, call the \
             `bash` tool with that exact command. After receiving the \
             output, report it back briefly.",
        )
        .tools(tkach::tools::defaults())
        .max_turns(5)
        .max_tokens(512)
        .working_dir(&dir)
        .build()
        .unwrap();

    let mut stream = agent.stream(
        vec![Message::user_text(
            "Run the shell command `echo openrouter_streaming_works` and \
             tell me what it printed.",
        )],
        CancellationToken::new(),
    );

    print!("> ");
    std::io::stdout().flush()?;

    let mut tools_called = Vec::new();
    let mut tool_inputs: Vec<String> = Vec::new();
    let mut delta_count = 0usize;
    let mut thinking_delta_chars = 0usize;
    let mut thinking_blocks = 0usize;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) => {
                delta_count += 1;
                print!("{text}");
                std::io::stdout().flush()?;
            }
            StreamEvent::ThinkingDelta { text } => {
                thinking_delta_chars += text.chars().count();
                eprintln!("\n[thinking delta: {} chars]", text.chars().count());
            }
            StreamEvent::ThinkingBlock { text, provider, .. } => {
                thinking_blocks += 1;
                eprintln!(
                    "\n[thinking block: {provider:?}, {} chars; metadata preserved]",
                    text.chars().count()
                );
            }
            StreamEvent::ToolUse { name, input, .. } => {
                eprintln!("\n[tool: {name}  args: {input}]");
                tools_called.push(name);
                tool_inputs.push(input.to_string());
            }
            _ => {}
        }
    }
    println!();

    let result = stream.into_result().await?;

    eprintln!();
    eprintln!("--- summary ---");
    eprintln!("tools called : {tools_called:?}");
    eprintln!("tool inputs  : {tool_inputs:?}");
    eprintln!("delta count  : {delta_count}");
    eprintln!("thinking     : {thinking_delta_chars} chars / {thinking_blocks} blocks");
    eprintln!(
        "tokens       : {} in / {} out",
        result.usage.input_tokens, result.usage.output_tokens
    );
    eprintln!();

    // --- assertions ---

    assert!(
        tools_called.iter().any(|t| t == "bash"),
        "expected `bash` tool call from streaming OpenAI-compat provider, \
         got: {tools_called:?}"
    );
    assert_eq!(
        thinking_delta_chars, 0,
        "OpenAI-compatible Chat Completions should not expose non-standard reasoning by default"
    );
    assert_eq!(
        thinking_blocks, 0,
        "OpenAI-compatible Chat Completions should not finalize thinking blocks by default"
    );

    // The atomic ToolUse must have parsed arguments from the SSE
    // delta-stream correctly into a real JSON object containing the
    // command. This is the assertion that proves accumulation across
    // chunked SSE works.
    let saw_correct_input = result.new_messages.iter().any(|m| {
        m.content.iter().any(|c| {
            matches!(c, Content::ToolUse { name, input, .. }
                if name == "bash"
                && input.get("command")
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| s.contains("openrouter_streaming_works")))
        })
    });
    assert!(
        saw_correct_input,
        "atomic ToolUse should have a parsed input.command containing \
         'openrouter_streaming_works' — proves SSE arg-fragment accumulation \
         is correct end-to-end"
    );

    // Final text should report the echoed value (the bash output came
    // back as a tool_result, then the model echoed it).
    assert!(
        result.text.contains("openrouter_streaming_works"),
        "final text should echo the bash output. got: {:?}",
        result.text
    );

    eprintln!("✓ all assertions passed");
    Ok(())
}
