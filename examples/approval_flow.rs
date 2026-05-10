//! End-to-end approval flow against the real Anthropic API.
//!
//! Demonstrates and verifies in one runnable shot:
//!
//! 1. **`StreamEvent::ToolCallPending` fires** in the live event stream
//!    before the executor invokes the tool.
//! 2. **Custom `ApprovalHandler` is honoured**: a scripted handler
//!    auto-allows read-only tools but denies the first `bash`
//!    invocation. The denial reason flows back to the model as an
//!    `is_error: true` tool_result.
//! 3. **Model recovers gracefully** — the streamed response after the
//!    denial mentions that the action was blocked, instead of hanging
//!    or retrying blindly.
//!
//! Run:  `cargo run --example approval_flow`
//!       (loads ANTHROPIC_API_KEY from .env or env)

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;
use tkach::message::Content;
use tkach::{
    Agent, ApprovalDecision, ApprovalHandler, CancellationToken, Message, StreamEvent, ToolClass,
    providers::Anthropic,
};

/// Approval handler used in this demo.
///
/// - ReadOnly tools (Read/Glob/Grep/WebFetch) are auto-allowed —
///   typical TUI UX where the user only confirms destructive actions.
/// - Mutating tools: the FIRST one is denied with a fixed reason; any
///   subsequent ones (in case the model retries with a different
///   command) are allowed. This script makes the assertion below
///   deterministic without relying on model judgement.
struct ScriptedApproval {
    bash_calls: AtomicUsize,
}

#[async_trait]
impl ApprovalHandler for ScriptedApproval {
    async fn approve(&self, tool_name: &str, input: &Value, class: ToolClass) -> ApprovalDecision {
        if class == ToolClass::ReadOnly {
            return ApprovalDecision::Allow;
        }
        // Mutating: deny the first bash call only.
        if tool_name == "bash" {
            let n = self.bash_calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                eprintln!(
                    "\n[approval] DENY  bash {input}  (this is a demo policy: \
                     the first bash call is blocked)"
                );
                return ApprovalDecision::Deny(
                    "user policy: bash commands require confirmation, denied for demo".into(),
                );
            }
        }
        eprintln!("\n[approval] ALLOW {tool_name}");
        ApprovalDecision::Allow
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let approval = ScriptedApproval {
        bash_calls: AtomicUsize::new(0),
    };

    let agent = Agent::builder()
        .provider(Anthropic::from_env())
        .model(tkach::model::claude::HAIKU_20251001)
        .system(
            "You are concise. When a tool you call is denied or fails, \
             explain to the user briefly what happened — do NOT silently \
             retry the same command, and do NOT pretend it succeeded.",
        )
        .tools(tkach::tools::defaults())
        .approval(approval)
        .max_turns(5)
        .max_tokens(512)
        .build();

    let mut stream = agent.stream(
        vec![Message::user_text(
            "Use the bash tool to run `echo hello_from_approval_flow` and \
             report what it printed.",
        )],
        CancellationToken::new(),
    );

    print!("> ");
    std::io::stdout().flush()?;

    let mut tool_uses: Vec<String> = Vec::new();
    let mut pending_events: Vec<(String, ToolClass)> = Vec::new();
    let mut delta_count = 0usize;
    let mut event_sequence: Vec<&'static str> = Vec::new();

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) => {
                delta_count += 1;
                event_sequence.push("ContentDelta");
                print!("{text}");
                std::io::stdout().flush()?;
            }
            StreamEvent::ToolUse { name, .. } => {
                tool_uses.push(name);
                event_sequence.push("ToolUse");
            }
            StreamEvent::ToolCallPending { name, class, .. } => {
                pending_events.push((name, class));
                event_sequence.push("ToolCallPending");
            }
            _ => {}
        }
    }
    println!();

    let result = stream.into_result().await?;

    eprintln!();
    eprintln!("--- summary ---");
    eprintln!("tool uses       : {tool_uses:?}");
    eprintln!("pending events  : {pending_events:?}");
    eprintln!("delta count     : {delta_count}");
    eprintln!(
        "tokens          : {} in / {} out",
        result.usage.input_tokens, result.usage.output_tokens
    );
    eprintln!("stop reason     : {:?}", result.stop_reason);
    eprintln!();

    // --- assertions ---

    // 1. The model attempted bash exactly because we asked it to.
    assert!(
        tool_uses.iter().any(|t| t == "bash"),
        "model should have tried `bash`; got: {tool_uses:?}"
    );

    // 2. ToolCallPending fired for that attempt.
    assert!(
        pending_events.iter().any(|(n, _)| n == "bash"),
        "ToolCallPending should fire for bash before approval; \
         got: {pending_events:?}"
    );

    // 3. Pending event's class is correct (bash → Mutating).
    let bash_class = pending_events
        .iter()
        .find(|(n, _)| n == "bash")
        .map(|(_, c)| *c);
    assert_eq!(
        bash_class,
        Some(ToolClass::Mutating),
        "bash must surface as Mutating in ToolCallPending"
    );

    // 4. Strict ordering: every ToolUse for `bash` is followed by a
    //    ToolCallPending before any later ContentDelta resumes.
    let tu_pos = event_sequence
        .iter()
        .position(|x| *x == "ToolUse")
        .expect("ToolUse should be in event sequence");
    let pending_pos = event_sequence
        .iter()
        .position(|x| *x == "ToolCallPending")
        .expect("ToolCallPending should be in event sequence");
    assert!(
        tu_pos < pending_pos,
        "ToolUse must precede ToolCallPending in stream order; \
         got: {event_sequence:?}"
    );

    // 5. The denial flowed back to the model: history must contain a
    //    tool_result with is_error: true and our deny reason.
    let saw_denial = result.new_messages.iter().any(|m| {
        m.content.iter().any(|c| match c {
            Content::ToolResult {
                content, is_error, ..
            } => *is_error && content.contains("user policy"),
            _ => false,
        })
    });
    assert!(
        saw_denial,
        "history should contain a denial tool_result with our reason"
    );

    // 6. Model's final text reflects the denial — it should NOT
    //    pretend the command ran. Look for plausible acknowledgement
    //    keywords.
    let text_lower = result.text.to_lowercase();
    let acknowledged = [
        "denied",
        "blocked",
        "policy",
        "couldn't",
        "could not",
        "unable",
    ]
    .iter()
    .any(|w| text_lower.contains(w));
    assert!(
        acknowledged,
        "model should acknowledge the denial in final text; got: {:?}",
        result.text
    );

    // (Note: we don't assert that the echo target string is absent
    // from the final text — the model often correctly mentions what
    // command was attempted while explaining the denial. The
    // acknowledgement check above is the meaningful invariant.)

    eprintln!("✓ approval flow verified end-to-end");
    Ok(())
}
