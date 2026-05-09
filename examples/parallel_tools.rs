//! Demonstrates parallel execution of read-only tools in a single batch.
//!
//! Three custom tools are registered:
//! - `fetch_a`, `fetch_b` — `ReadOnly`, each sleeps 200ms
//! - `save`              — `Mutating` (default)
//!
//! Concurrency model: `execute_batch` partitions a batch into
//! contiguous runs of the same routing class and dispatches each run
//! concurrently via `FuturesUnordered`. Runs are awaited
//! sequentially so the LLM-emitted order across class boundaries is
//! preserved — important when a `ReadOnly` after a `Mutating` would
//! otherwise race the side effect.
//!
//! Turn 1: mock emits `[fetch_a, save, fetch_b]` — three partitioned
//! runs. Run 1 (`fetch_a`, RO) ≈ 200ms, then Run 2 (`save`, default
//! Mutating, width-1 serial pool) ≈ 0ms, then Run 3 (`fetch_b`, RO)
//! ≈ 200ms — total ≈ 400ms.
//!
//! Turn 2: mock emits `[fetch_a, fetch_b]` — one RO run, both share
//! the `read` pool; wall-time ≈ 200ms.
//!
//! Total wall time ≈ 600ms (vs ≈ 800ms strictly serial). To make
//! sibling mutators overlap (e.g. writes to independent files,
//! multiple sub-agent fan-out), see the `parallel_writes` and
//! `parallel_subagents` examples — they show the
//! `tool_concurrency(name, ToolConcurrency::on())` opt-in. SubAgent
//! is special: it overrides `is_recursive() -> true` so the executor
//! routes it through the concurrent-mutator pool by default,
//! avoiding the held-permit-during-nested-execute deadlock that
//! shared `serial_mut` would otherwise create.
//!
//! Run with: `cargo run --example parallel_tools`

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use serde_json::{Value, json};
use tkach::message::{Content, StopReason, Usage};
use tkach::provider::Response;
use tkach::providers::Mock;
use tkach::{
    Agent, CancellationToken, Message, Tool, ToolClass, ToolContext, ToolError, ToolOutput,
};

/// A read-only tool that sleeps then echoes its name.
struct SlowReader {
    label: &'static str,
    delay_ms: u64,
}

#[async_trait::async_trait]
impl Tool for SlowReader {
    fn name(&self) -> &str {
        self.label
    }
    fn description(&self) -> &str {
        "Read-only tool that simulates slow I/O"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn class(&self) -> ToolClass {
        ToolClass::ReadOnly
    }
    async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
        Ok(ToolOutput::text(format!("{} done", self.label)))
    }
}

/// A mutating tool — forced sequential.
struct Save;

#[async_trait::async_trait]
impl Tool for Save {
    fn name(&self) -> &str {
        "save"
    }
    fn description(&self) -> &str {
        "Mutating tool — must run sequentially"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    // class() defaults to Mutating.
    async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text("saved"))
    }
}

#[tokio::main]
async fn main() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    // Mock provider scripts two turns of tool use, then a final text answer.
    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            // Turn 1: mixed batch [RO, Mut, RO] — partitioned into three runs.
            0 => Ok(Response {
                content: vec![
                    tool_use("t1", "fetch_a"),
                    tool_use("t2", "save"),
                    tool_use("t3", "fetch_b"),
                ],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            // Turn 2: pure RO batch [RO, RO] — one parallel run.
            1 => Ok(Response {
                content: vec![tool_use("t4", "fetch_a"), tool_use("t5", "fetch_b")],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            // Turn 3: done.
            _ => Ok(Response {
                content: vec![Content::text("All fetches complete.")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(SlowReader {
            label: "fetch_a",
            delay_ms: 200,
        })
        .tool(SlowReader {
            label: "fetch_b",
            delay_ms: 200,
        })
        .tool(Save)
        .build();

    let started = Instant::now();
    let result = agent
        .run(
            vec![Message::user_text("fetch some stuff")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");
    let elapsed = started.elapsed();

    println!("result: {}", result.text);
    println!("delta messages: {}", result.new_messages.len());
    println!("wall time: {elapsed:?}");
    println!();
    println!("Turn 1 batch [RO, Mut, RO] — 3 partitioned runs ≈ 400ms");
    println!("Turn 2 batch [RO, RO]      — 1 RO run, both parallel ≈ 200ms");
    println!("Total: ~600ms (vs. ~800ms if everything were sequential)");
}

fn tool_use(id: &str, name: &str) -> Content {
    Content::ToolUse {
        id: id.into(),
        name: name.into(),
        input: json!({}),
    }
}
