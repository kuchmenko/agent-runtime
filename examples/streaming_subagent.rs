//! Streaming parent + sub-agent delegation.
//!
//! Sonnet streams its top-level reasoning, decides to delegate to a
//! Haiku sub-agent for a focused task, awaits the sub-agent's
//! (buffered, since `SubAgent::execute` calls `agent.run()`) result,
//! then continues streaming the wrap-up.
//!
//! This validates the architecturally tricky path:
//!
//!   parent (streaming) ─┐
//!     turn 1: stream tokens, emit ToolUse{name:"agent", ...}
//!                      │
//!                      └─→ executor.execute_batch
//!                           └─→ SubAgent::execute
//!                                └─→ child agent.run() (BUFFERED)
//!                                     └─→ provider.complete()
//!                                          ├─ Haiku reads file
//!                                          └─ returns full text
//!     turn 2: stream final tokens incorporating sub-agent output
//!
//! From the consumer's POV: parent stream pauses during the sub-agent
//! call, then resumes. Sub-agent itself does not stream (atomic from
//! parent's perspective).
//!
//! Run:  `cargo run --example streaming_subagent`
//!       (loads ANTHROPIC_API_KEY from .env or env)

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use tkach::tools::SubAgent;
use tkach::{Agent, CancellationToken, LlmProvider, Message, StreamEvent, providers::Anthropic};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let dir = std::env::temp_dir().join("tkach_streaming_subagent");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("data.txt"),
        "The quarterly figure is 4.7 billion dollars.",
    )?;

    // Sonnet for the streaming parent, Haiku for the sub-agent —
    // demonstrates the typical "expensive parent, cheap delegate"
    // pattern.
    let provider: Arc<dyn LlmProvider> = Arc::new(Anthropic::from_env());

    let sub_agent = SubAgent::new(Arc::clone(&provider), tkach::model::claude::HAIKU_20251001)
        .system("You are a focused researcher. Read the file and report the figure verbatim.")
        .max_turns(5)
        .max_tokens(1024);

    let agent = Agent::builder()
        .provider_arc(Arc::clone(&provider))
        .model(tkach::model::claude::SONNET)
        .system(
            "You are a concise senior analyst. When the user asks a question \
             that requires reading a file, delegate the read to a sub-agent \
             via the `agent` tool, then synthesise the answer briefly.",
        )
        .tools(tkach::tools::defaults())
        .tool(sub_agent)
        .max_depth(3)
        .max_turns(8)
        .max_tokens(2048)
        .working_dir(&dir)
        .build()
        .unwrap();

    let started = Instant::now();
    let mut stream = agent.stream(
        vec![Message::user_text(
            "There's a file called data.txt in the working directory. \
             Use a sub-agent (the `agent` tool) to read it and report the \
             quarterly figure. Then briefly state the figure in your own words.",
        )],
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
    let elapsed = started.elapsed();

    eprintln!();
    eprintln!("--- summary ---");
    eprintln!("tools called  : {tools_called:?}");
    eprintln!("delta count   : {delta_count}");
    eprintln!(
        "tokens (parent only): {} in / {} out",
        result.usage.input_tokens, result.usage.output_tokens
    );
    eprintln!("stop reason   : {:?}", result.stop_reason);
    eprintln!("delta msgs    : {}", result.new_messages.len());
    eprintln!("wall time     : {elapsed:?}");
    eprintln!();

    // --- assertions ---

    assert!(delta_count >= 1, "expected at least one ContentDelta");
    assert!(
        tools_called.iter().any(|t| t == "agent"),
        "expected `agent` tool to be called (sub-agent delegation), got: {tools_called:?}"
    );
    // Final answer must reference the figure the sub-agent extracted.
    // Models phrase numbers differently — accept any of these forms.
    let text_lower = result.text.to_lowercase();
    let mentions_figure = text_lower.contains("4.7 billion")
        || text_lower.contains("4.7b")
        || text_lower.contains("$4.7")
        || text_lower.contains("4,700,000,000")
        || text_lower.contains("4.7");
    assert!(
        mentions_figure,
        "final text should mention the quarterly figure (4.7 billion). got: {:?}",
        result.text
    );

    // Successful EndTurn after the sub-agent returned.
    assert_eq!(
        result.stop_reason,
        tkach::StopReason::EndTurn,
        "should have terminated cleanly"
    );

    eprintln!("✓ all assertions passed");
    Ok(())
}
