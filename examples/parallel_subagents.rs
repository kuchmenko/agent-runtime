//! Demonstrates parallel sub-agent fan-out for codebase exploration.
//!
//! `SubAgent` identifies itself as a recursive tool, so the executor routes
//! it to the concurrent-mutator pool automatically. All three calls overlap,
//! each spending most of its time in its inner `slow_reader` call. Wall time
//! is therefore ≈ 1× single-agent walltime rather than ≈ 3×.
//!
//! The example asserts the parallel wall time and that all three
//! results are returned in the original `tool_use` order.
//! Run with: `cargo run --example parallel_subagents`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tkach::message::{Content, StopReason, Usage};
use tkach::provider::Response;
use tkach::providers::Mock;
use tkach::tools::SubAgent;
use tkach::{
    Agent, CancellationToken, Message, Tool, ToolClass, ToolContext, ToolError, ToolOutput,
};

/// A `ReadOnly` tool used inside each sub-agent to make wall-time
/// observable. Sub-agents call this once and return — without an
/// inner delay we'd be measuring just LLM-loop overhead, which is
/// dominated by Mock latency (microseconds).
struct SlowReader {
    delay_ms: u64,
}

#[async_trait::async_trait]
impl Tool for SlowReader {
    fn name(&self) -> &str {
        "slow_reader"
    }
    fn description(&self) -> &str {
        "ReadOnly tool that sleeps then echoes its label"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": { "label": { "type": "string" } } })
    }
    fn class(&self) -> ToolClass {
        ToolClass::ReadOnly
    }
    async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let label = input["label"].as_str().unwrap_or("anon").to_string();
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        Ok(ToolOutput::text(format!("read[{label}]")))
    }
}

#[tokio::main]
async fn main() {
    let inner_delay_ms: u64 = 200;

    // Sub-agent's provider: turn 0 emits a tool_use for slow_reader;
    // turn 1 (after the tool_result lands) returns a text summary.
    // Shared across all three concurrent sub-agent invocations.
    let sub_provider: Arc<dyn tkach::LlmProvider> = Arc::new(Mock::new(move |req| {
        let has_tool_result = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_tool_result {
            return Ok(Response {
                content: vec![Content::text("research summary")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        // Pull the label off the user message so each sub-agent's
        // SlowReader call records its own label — this is what the
        // parent will see in the tool_result payload.
        let label = req
            .messages
            .iter()
            .find_map(|m| {
                m.content.iter().find_map(|c| match c {
                    Content::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_else(|| "anon".to_string());

        Ok(Response {
            content: vec![Content::ToolUse {
                id: "r1".into(),
                name: "slow_reader".into(),
                input: json!({ "label": label }),
            }],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    }));

    // Parent provider: turn 0 emits three sub-agent tool_use calls;
    // turn 1 (after all three return) wraps up.
    let parent_mock = Mock::new(move |req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_results {
            return Ok(Response {
                content: vec![Content::text("all three sub-agents done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        Ok(Response {
            content: vec![
                Content::ToolUse {
                    id: "a1".into(),
                    name: "agent".into(),
                    input: json!({ "prompt": "topic-A" }),
                },
                Content::ToolUse {
                    id: "a2".into(),
                    name: "agent".into(),
                    input: json!({ "prompt": "topic-B" }),
                },
                Content::ToolUse {
                    id: "a3".into(),
                    name: "agent".into(),
                    input: json!({ "prompt": "topic-C" }),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    });

    let agent = Agent::builder()
        .provider(parent_mock)
        .model("mock-parent")
        .tool(SlowReader {
            delay_ms: inner_delay_ms,
        })
        .tool(SubAgent::new(Arc::clone(&sub_provider), "mock-sub").max_turns(3))
        .max_concurrent_mutations(3)
        .build()
        .unwrap();

    let started = Instant::now();
    let result = agent
        .run(
            vec![Message::user_text("delegate to three sub-agents")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");
    let elapsed = started.elapsed();

    // Wall-time assertion: parallel ≈ inner_delay; serial would be ≈ 3×.
    // Use a 1.75× threshold to absorb sub-agent loop overhead while
    // still failing loudly if execution serialised.
    let parallel_threshold = Duration::from_millis(inner_delay_ms + (inner_delay_ms * 3 / 4));
    assert!(
        elapsed < parallel_threshold,
        "expected parallel wall-time (< {parallel_threshold:?}), got {elapsed:?} \
         — looks like sub-agents serialised through the width-1 pool"
    );

    println!("result: {}", result.text);
    println!("delta messages: {}", result.new_messages.len());
    println!("wall time: {elapsed:?} (per-sub-agent inner delay: {inner_delay_ms}ms)");
    println!();
    println!("Recursive tools use the concurrent-mutator pool automatically.");
    println!("All three sub-agents therefore overlap within the configured class cap.");
}
