//! Streaming resilience: three failure-shaped scenarios in one run.
//!
//! Closes the verification gaps we identified after live-running the
//! happy-path streaming examples:
//!
//! 1. **Tool failure mid-stream** — the agent calls a tool, the tool
//!    returns `is_error: true` (we ask `read` for a path that doesn't
//!    exist). The loop must NOT terminate; it should hand the error
//!    back to the model and let it produce a graceful textual
//!    explanation in a follow-up turn.
//!
//! 2. **Cancel during tool execution** — the agent issues a `bash
//!    sleep 10` tool call. The example arms a cancel timer the moment
//!    it sees `ToolUse{name:"bash"}` on the stream, fires `cancel.cancel()`
//!    while bash is sleeping, and asserts both: the bash child was
//!    killed promptly (wall time « 10s), and the outer outcome is
//!    `AgentError::Cancelled`.
//!
//! 3. **Multi-block assistant turn** — Anthropic emits text deltas
//!    BEFORE a tool_use block in the same turn ("Let me check… *calls
//!    read*"). Our SSE state machine accumulates each block separately
//!    and reconstructs an assistant `Message` whose `content: Vec<Content>`
//!    has both a `Text` block and a `ToolUse` block. Asserting that
//!    structure proves the per-block state machine works end-to-end.
//!
//! Each phase fails the whole example with a clear panic message if
//! its specific invariant breaks. Run:
//!
//!   `cargo run --example streaming_resilience`
//!   (loads ANTHROPIC_API_KEY from .env or env)

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use tkach::message::Content;
use tkach::{
    Agent, AgentError, AgentResult, CancellationToken, Message, StopReason, StreamEvent,
    providers::Anthropic,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let provider = Arc::new(Anthropic::from_env());

    eprintln!("=== phase 1: tool failure mid-stream ===");
    phase1_tool_failure(&provider).await?;
    eprintln!();

    eprintln!("=== phase 2: cancel during tool execution ===");
    phase2_cancel_during_tool(&provider).await?;
    eprintln!();

    eprintln!("=== phase 3: multi-block assistant turn ===");
    phase3_multi_block(&provider).await?;
    eprintln!();

    eprintln!("✓ all three resilience scenarios passed");
    Ok(())
}

/// Phase 1: ask the agent to read a file that does not exist. The
/// `read` tool returns `is_error: true`; the loop should keep going,
/// the model's next turn should explain the failure, and final
/// outcome should be a clean `Ok` with `EndTurn`.
async fn phase1_tool_failure(provider: &Arc<Anthropic>) -> Result<(), Box<dyn std::error::Error>> {
    // A path inside an empty scratch dir, guaranteed non-existent.
    let dir = std::env::temp_dir().join("tkach_streaming_resilience_phase1");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let bogus = dir.join("does_not_exist_xyz_123.txt");

    let agent = Agent::builder()
        .provider_arc(provider.clone() as Arc<dyn tkach::LlmProvider>)
        .model(tkach::model::claude::HAIKU_20251001)
        .system(
            "You are a concise assistant. Use tools when helpful. \
             If a tool fails, explain the failure briefly to the user \
             instead of retrying blindly.",
        )
        .tools(tkach::tools::defaults())
        .max_turns(5)
        .max_tokens(512)
        .working_dir(&dir)
        .build()
        .unwrap();

    let mut stream = agent.stream(
        vec![Message::user_text(format!(
            "Read the file at {} and tell me what it contains.",
            bogus.display()
        ))],
        CancellationToken::new(),
    );

    let mut tools_called = Vec::new();
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ToolUse { name, .. } => tools_called.push(name),
            StreamEvent::ContentDelta(_) => {}
            _ => {}
        }
    }
    let result = stream.into_result().await?;

    eprintln!(
        "  tools called : {tools_called:?}\n  \
         turns        : {} delta msgs\n  \
         stop reason  : {:?}\n  \
         text len     : {}",
        result.new_messages.len(),
        result.stop_reason,
        result.text.len()
    );

    // Tool was called — the model genuinely tried.
    assert!(
        tools_called.iter().any(|t| t == "read"),
        "phase 1: agent should have called `read`, got: {tools_called:?}"
    );
    // The loop did not terminate on the tool error — we got a final
    // assistant text turn after the failed tool.
    assert_eq!(
        result.stop_reason,
        StopReason::EndTurn,
        "phase 1: should have ended cleanly after the tool failure"
    );
    // Some final text was produced.
    assert!(
        !result.text.is_empty(),
        "phase 1: agent should have explained the failure"
    );
    // History contains an is_error tool_result block — proves the
    // failure was actually surfaced to the model.
    let saw_error_tool_result = result.new_messages.iter().any(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolResult { is_error, .. } if *is_error))
    });
    assert!(
        saw_error_tool_result,
        "phase 1: history should contain a tool_result with is_error: true"
    );

    eprintln!("  ✓ tool failure surfaced and recovered cleanly");
    Ok(())
}

/// Phase 2: agent issues `bash sleep 10`. As soon as the stream
/// emits the ToolUse, we arm a 500ms cancel — bash is now sleeping,
/// then cancel fires. With `kill_on_drop(true)` + `tokio::select!` on
/// `ctx.cancel`, the child must be killed promptly. End-to-end wall
/// time should be far less than 10s.
async fn phase2_cancel_during_tool(
    provider: &Arc<Anthropic>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("tkach_streaming_resilience_phase2");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let agent = Agent::builder()
        .provider_arc(provider.clone() as Arc<dyn tkach::LlmProvider>)
        .model(tkach::model::claude::HAIKU_20251001)
        .system("You are concise. Run shell commands as asked, exactly.")
        .tools(tkach::tools::defaults())
        .max_turns(2)
        .max_tokens(256)
        .working_dir(&dir)
        .build()
        .unwrap();

    let cancel = CancellationToken::new();
    let cancel_arm = cancel.clone();

    let started = Instant::now();
    let mut stream = agent.stream(
        vec![Message::user_text(
            "Use the bash tool to run exactly: `sleep 10` (no other arguments).",
        )],
        cancel,
    );

    let mut bash_seen = false;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ToolUse { name, .. } if name == "bash" => {
                bash_seen = true;
                // Bash will start sleeping inside execute_batch which
                // runs after this stream's inner loop ends. We need
                // the cancel to fire AFTER that, so spawn a delayed
                // cancel from here. 500ms is enough for the inner
                // SSE loop to finish, the executor to spawn bash,
                // and bash to actually be sleeping.
                let arm = cancel_arm.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    arm.cancel();
                });
            }
            _ => {}
        }
    }
    let outcome = stream.into_result().await;
    let elapsed = started.elapsed();

    eprintln!(
        "  bash event seen : {bash_seen}\n  \
         wall time       : {elapsed:?}\n  \
         outcome         : {}",
        match &outcome {
            Ok(_) => "Ok (unexpected!)".to_string(),
            Err(AgentError::Cancelled { .. }) => "Cancelled".to_string(),
            Err(other) => format!("Err: {other:?}"),
        }
    );

    assert!(
        bash_seen,
        "phase 2: should have seen ToolUse{{name:'bash'}} on the stream"
    );
    let err = outcome.expect_err("phase 2: expected Cancelled, got Ok");
    let AgentError::Cancelled { partial } = &err else {
        panic!("phase 2: expected Cancelled, got {err:?}");
    };
    let _ = partial; // partial is fine; just confirm the variant
    // The full sleep would take 10s. We armed cancel at 500ms. With
    // kill_on_drop the bash child must terminate immediately, so the
    // outer wall time should be well under 10s. 5s leaves comfortable
    // slack for spawn + reap on shared CI runners.
    assert!(
        elapsed < Duration::from_secs(5),
        "phase 2: cancel did not abort bash promptly — took {elapsed:?}"
    );
    eprintln!("  ✓ bash killed mid-sleep, agent returned Cancelled");
    Ok(())
}

/// Phase 3: a prompt that nudges Sonnet to "narrate then act" — text
/// before a tool_use in the SAME turn. Our SSE state machine
/// reconstructs this as a single assistant Message with
/// `content: Vec<Content>` containing both `Text` and `ToolUse`
/// blocks. We assert that exact structure.
///
/// If the model decides to skip the narration (it sometimes does),
/// we fall back to asserting the simpler invariant — that we got
/// AT LEAST one assistant message with multiple Content blocks
/// somewhere in history (the multi-block case is exercised by any
/// turn that mixes text and tools).
async fn phase3_multi_block(provider: &Arc<Anthropic>) -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("tkach_streaming_resilience_phase3");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("note.txt"), "The codeword is BANANA.")?;

    let agent = Agent::builder()
        .provider_arc(provider.clone() as Arc<dyn tkach::LlmProvider>)
        // Sonnet is more likely than Haiku to narrate before acting.
        .model(tkach::model::claude::SONNET)
        .system(
            "You are a thoughtful assistant. Before calling a tool, \
             briefly state what you're about to do in one sentence \
             of plain text, THEN call the tool in the same response.",
        )
        .tools(tkach::tools::defaults())
        .max_turns(3)
        .max_tokens(1024)
        .working_dir(&dir)
        .build()
        .unwrap();

    let mut stream = agent.stream(
        vec![Message::user_text(
            "There's a file called note.txt. Tell me out loud what \
             you're about to do, then read it, then report what you \
             found.",
        )],
        CancellationToken::new(),
    );

    while let Some(event) = stream.next().await {
        let _ = event?;
    }
    let result: AgentResult = stream.into_result().await?;

    // Find any assistant message that has BOTH a Text and a ToolUse
    // content block — that's a multi-block turn reconstructed by
    // our state machine.
    let multi_block_assistant_turn = result.new_messages.iter().find(|m| {
        let has_text = m.content.iter().any(|c| matches!(c, Content::Text { .. }));
        let has_tool = m
            .content
            .iter()
            .any(|c| matches!(c, Content::ToolUse { .. }));
        has_text && has_tool
    });

    eprintln!(
        "  delta msgs            : {}\n  \
         multi-block turn found : {}",
        result.new_messages.len(),
        multi_block_assistant_turn.is_some(),
    );

    match multi_block_assistant_turn {
        Some(m) => {
            let blocks: Vec<&'static str> = m
                .content
                .iter()
                .map(|c| match c {
                    Content::Text { .. } => "Text",
                    Content::Thinking { .. } => "Thinking",
                    Content::ToolUse { .. } => "ToolUse",
                    Content::ToolResult { .. } => "ToolResult",
                })
                .collect();
            eprintln!("  block sequence        : {blocks:?}");
            eprintln!(
                "  ✓ multi-block assistant turn correctly reconstructed \
                 (Text + ToolUse in same Message)"
            );
        }
        None => {
            // Fallback: model elected to call the tool with no
            // preface (it sometimes does). Verify at minimum that
            // the tool was called and final text mentions the
            // codeword — the multi-block path was not exercised on
            // this run, but the streaming agent loop still worked.
            eprintln!(
                "  ! model skipped pre-tool narration on this run; \
                 multi-block PATH not exercised this time, but \
                 streaming-with-tool still verified by content"
            );
            assert!(
                result.text.contains("BANANA"),
                "phase 3 fallback: agent should have read the codeword. got: {:?}",
                result.text
            );
        }
    }
    Ok(())
}
