//! Three specialised SubAgent profiles registered against one parent.
//!
//! Demonstrates issue #40 Phase 5 acceptance: a read-only research
//! SubAgent + an autonomous reasoning SubAgent (with `AutoApprove` to
//! bypass parent approval prompts) + a mutating writer SubAgent with a
//! `trace_hook` for per-turn observability — the safety pattern Cognition
//! AI's "Share full agent traces" principle calls for on mutating
//! children.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use tkach::message::{Content, StopReason, Usage};
use tkach::provider::Response;
use tkach::providers::Mock;
use tkach::tools::SubAgent;
use tkach::{Agent, AutoApprove, CancellationToken, Message, ThinkingConfig, ThinkingEffort};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- Child providers (one per SubAgent profile) -------------------

    // (1) Research child: assert that per-call thinking propagated to the
    // child Request. Returns text after observing the thinking field.
    let research_thinking_seen = Arc::new(AtomicUsize::new(0));
    let research_thinking_clone = Arc::clone(&research_thinking_seen);
    let research_child = Arc::new(Mock::new(move |req| {
        if matches!(
            req.thinking,
            Some(ThinkingConfig::Effort(ThinkingEffort::High))
        ) {
            research_thinking_clone.fetch_add(1, Ordering::SeqCst);
        }
        Ok(Response {
            content: vec![Content::text("research summary")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }));

    // (2) Autonomous reasoning child: returns a single text answer.
    let reasoning_child = Arc::new(Mock::with_text("reasoning answer"));

    // (3) Mutating writer child: returns text. The trace_hook on the
    // parent observes the child's StreamEvents.
    let writer_child = Arc::new(Mock::with_text("file written"));

    // ---- Parent loop: invoke research, then reasoning, then writer ----

    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);
    let parent = Mock::new(move |_req| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        match turn {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "call-research".into(),
                    name: "research".into(),
                    input: json!({"prompt": "Summarise the repository"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            1 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "call-reasoning".into(),
                    name: "reasoning".into(),
                    input: json!({"prompt": "Decide next step"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            2 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "call-writer".into(),
                    name: "writer".into(),
                    input: json!({"prompt": "Write the result"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    // ---- SubAgent profiles --------------------------------------------

    // Profile 1: read-only research. tools_allow restricts to read-class
    // tools; per-call thinking lifts effort to High for this child only.
    let research = SubAgent::new(research_child, "mock-haiku")
        .name("research")
        .description("Read-only repository research helper")
        .tools_allow(["read", "glob", "grep", "web_fetch"])
        .filter_tool_definitions(true)
        .thinking(ThinkingConfig::Effort(ThinkingEffort::High));

    // Profile 2: autonomous reasoning. approval_handler(AutoApprove)
    // bypasses any interactive approval the parent installs — this child
    // ships with a narrowed safety guarantee by design.
    let reasoning = SubAgent::new(reasoning_child, "mock-sonnet")
        .name("reasoning")
        .description("Autonomous reasoning, no human-in-the-loop prompts")
        .approval_handler(Arc::new(AutoApprove));

    // Profile 3: mutating writer. trace_hook captures every child
    // StreamEvent so an audit sink can record per-turn decisions —
    // Cognition's "Share full agent traces" requirement for mutating
    // children that touch user data.
    let trace_count = Arc::new(AtomicUsize::new(0));
    let trace_count_clone = Arc::clone(&trace_count);
    let writer = SubAgent::new(writer_child, "mock-sonnet")
        .name("writer")
        .description("Mutating writer with full trace observability")
        .tools_allow(["read", "edit", "write", "bash"])
        .trace_hook(move |_ev| {
            trace_count_clone.fetch_add(1, Ordering::SeqCst);
        });

    // ---- Parent agent registration ------------------------------------

    let agent = Agent::builder()
        .provider(parent)
        .model("mock-parent")
        .tools(tkach::tools::defaults())
        .tool(research)
        .tool(reasoning)
        .tool(writer)
        .working_dir(std::env::current_dir()?)
        .build()?;

    let result = agent
        .run(
            vec![Message::user_text("delegate to all three subagents")],
            CancellationToken::new(),
        )
        .await?;

    // ---- Assertions ---------------------------------------------------

    assert_eq!(result.text, "done");
    assert_eq!(parent_turn.load(Ordering::SeqCst), 4);
    assert_eq!(
        research_thinking_seen.load(Ordering::SeqCst),
        1,
        "research child must observe Request.thinking = Effort(High)"
    );
    assert!(
        trace_count.load(Ordering::SeqCst) >= 4,
        "writer trace_hook must fire for ContentDelta + MessageDelta + Usage + Done; got {}",
        trace_count.load(Ordering::SeqCst)
    );

    println!(
        "specialised sub-agents completed: result={}, trace_events={}",
        result.text,
        trace_count.load(Ordering::SeqCst)
    );
    Ok(())
}
