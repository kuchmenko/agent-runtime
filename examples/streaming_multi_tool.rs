//! Multi-turn tool chain via `Agent::stream`.
//!
//! Drives a 3-turn workflow against the real Anthropic API:
//!
//!   turn 1: agent calls `write` to create config.toml
//!   turn 2: agent calls `edit` to change a value inside it
//!   turn 3: agent calls `read` to verify and produce final text
//!
//! Live observability: every text token prints as it arrives; every
//! tool invocation announces itself with `[tool: NAME]` on stderr.
//!
//! Built-in assertions verify the complete chain — file shape after
//! each step, tool sequence, final text content. The example crashes
//! loudly (non-zero exit) if any check fails.
//!
//! Run:  `cargo run --example streaming_multi_tool`
//!       (loads ANTHROPIC_API_KEY from .env or env)

use std::io::Write;

use futures::StreamExt;
use tkach::message::Content;
use tkach::{Agent, CancellationToken, Message, StreamEvent, providers::Anthropic};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    // Scratch dir for the file the agent creates.
    let dir = std::env::temp_dir().join("tkach_streaming_multi_tool");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let target = dir.join("config.toml");

    let agent = Agent::builder()
        .provider(Anthropic::from_env())
        .model(tkach::model::claude::HAIKU_20251001)
        .system(
            "You are a concise assistant. Use tools when needed. \
             Never fabricate file contents — always verify by reading.",
        )
        .tools(tkach::tools::defaults())
        .max_turns(8)
        .max_tokens(2048)
        .working_dir(&dir)
        .build()
        .unwrap();

    let mut stream = agent.stream(
        vec![Message::user_text(format!(
            "Do these three steps using the tools available:\n\
             1. Use the write tool to create a file at {} containing exactly: \
                [server]\\nport = 8080\\n\
             2. Use the edit tool to change 8080 to 9090.\n\
             3. Use the read tool to read the resulting file, then tell me what port it shows.",
            target.display()
        ))],
        CancellationToken::new(),
    );

    print!("> ");
    std::io::stdout().flush()?;

    let mut tools_called = Vec::new();
    let mut delta_count = 0usize;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) => {
                delta_count += 1;
                print!("{text}");
                std::io::stdout().flush()?;
            }
            StreamEvent::ToolUse { name, .. } => {
                eprintln!("\n[tool: {name}]");
                tools_called.push(name);
            }
            _ => {}
        }
    }
    println!();

    let result = stream.into_result().await?;

    eprintln!();
    eprintln!("--- summary ---");
    eprintln!("tools called : {tools_called:?}");
    eprintln!("delta count  : {delta_count}");
    eprintln!(
        "tokens       : {} in / {} out",
        result.usage.input_tokens, result.usage.output_tokens
    );
    eprintln!("stop reason  : {:?}", result.stop_reason);
    eprintln!("turns (delta msgs / 2): {}", result.new_messages.len() / 2);
    eprintln!();

    // --- assertions ---

    assert!(
        delta_count >= 1,
        "expected at least one ContentDelta, got 0"
    );
    assert!(
        tools_called.iter().any(|t| t == "write"),
        "expected `write` tool call, got: {tools_called:?}"
    );
    assert!(
        tools_called.iter().any(|t| t == "edit"),
        "expected `edit` tool call, got: {tools_called:?}"
    );
    assert!(
        tools_called.iter().any(|t| t == "read"),
        "expected `read` tool call, got: {tools_called:?}"
    );

    // Final state of the file: should have port = 9090
    let on_disk = std::fs::read_to_string(&target)?;
    assert!(
        on_disk.contains("9090"),
        "config.toml should contain '9090' after edit, got:\n{on_disk}"
    );
    assert!(
        !on_disk.contains("8080"),
        "config.toml should NOT contain '8080' after edit, got:\n{on_disk}"
    );

    // Final assistant text: should mention 9090
    assert!(
        result.text.contains("9090"),
        "final text should report port 9090, got: {:?}",
        result.text
    );

    // History shape: depends on whether the model batches all three
    // tool calls into one turn (assistant[3 tool_use] + user[3
    // tool_result] + final = 3 messages) or chains them
    // sequentially (3 × 2 + 1 = 7). Both behaviours are valid; the
    // important invariant is that we got tool calls AND a final
    // assistant turn, i.e. at least 3 delta messages.
    assert!(
        result.new_messages.len() >= 3,
        "expected at least 3 delta messages (≥1 tool round + final), got {}",
        result.new_messages.len()
    );

    // Bonus: confirm the assistant turn that contained the edit
    // tool_use carried the right input (proves atomic ToolUse was
    // wired correctly through the stream into history).
    let saw_edit_tool_use_with_9090 = result.new_messages.iter().any(|m| {
        m.content.iter().any(|c| {
            matches!(c, Content::ToolUse { name, input, .. }
                if name == "edit" && input.to_string().contains("9090"))
        })
    });
    assert!(
        saw_edit_tool_use_with_9090,
        "edit ToolUse should have been recorded in history with 9090 in input"
    );

    eprintln!("✓ all assertions passed");
    Ok(())
}
