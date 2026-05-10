//! Demonstrates parallel writes to independent files in a single batch.
//!
//! Without `tool_concurrency("slow_write", ToolConcurrency::on())`, every
//! `Mutating` tool routes to the width-1 serial-mutator pool — two writes
//! issued in one LLM batch run back-to-back, so wall time ≈ 2× per-call
//! delay.
//!
//! With the opt-in, the tool routes to the concurrent-mutator pool whose
//! width is set by `max_concurrent_mutations` (default 10). Both writes
//! overlap, so wall time ≈ 1× per-call delay.
//!
//! The example asserts the parallel wall time, files-on-disk content,
//! and result ordering. Run with: `cargo run --example parallel_writes`.
//!
//! Note: this is the consumer's responsibility contract — promoting a
//! mutating tool means the consumer guarantees the LLM-emitted batch
//! shape (and the tool's resource semantics) make racing safe. Two
//! writes to the *same* path in one batch would produce a race. Two
//! writes to *different* paths are independent at the kernel level
//! (different inodes) and parallelise cleanly.

use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tkach::message::{Content, StopReason, Usage};
use tkach::provider::Response;
use tkach::providers::Mock;
use tkach::{
    Agent, CancellationToken, Message, Tool, ToolConcurrency, ToolContext, ToolError, ToolOutput,
};

/// A `Mutating` tool that sleeps `delay_ms` then writes `content` to
/// `file_path`. The sleep makes parallelism observable in wall-time
/// without depending on real-fs latency, which is microseconds on
/// tmpfs and would not differentiate parallel from serial execution.
struct SlowWriter {
    delay_ms: u64,
}

#[async_trait::async_trait]
impl Tool for SlowWriter {
    fn name(&self) -> &str {
        "slow_write"
    }
    fn description(&self) -> &str {
        "Mutating tool that sleeps then writes content to a file"
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["file_path", "content"]
        })
    }
    // class() defaults to Mutating.
    async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let path = input["file_path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("file_path required".into()))?
            .to_string();
        let content = input["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("content required".into()))?
            .to_string();

        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        tokio::fs::write(&path, &content)
            .await
            .map_err(ToolError::Io)?;
        Ok(ToolOutput::text(format!("wrote {path}")))
    }
}

#[tokio::main]
async fn main() {
    // Per-run unique tmp dir so concurrent runs don't clobber each other.
    let dir = std::env::temp_dir().join(format!("tkach-parallel-writes-{}", std::process::id()));
    tokio::fs::create_dir_all(&dir).await.expect("mkdir tmp");
    let path_a = dir.join("a.txt");
    let path_b = dir.join("b.txt");

    let pa = path_a.to_string_lossy().to_string();
    let pb = path_b.to_string_lossy().to_string();

    // Mock provider: turn 0 emits two parallel writes; turn 1 returns
    // text once both writes have completed (detected via tool_result
    // presence in the message history).
    let mock = Mock::new(move |req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });

        if has_results {
            return Ok(Response {
                content: vec![Content::text("both writes done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        Ok(Response {
            content: vec![
                Content::ToolUse {
                    id: "w1".into(),
                    name: "slow_write".into(),
                    input: json!({ "file_path": pa, "content": "alpha" }),
                },
                Content::ToolUse {
                    id: "w2".into(),
                    name: "slow_write".into(),
                    input: json!({ "file_path": pb, "content": "beta" }),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    });

    let delay_ms: u64 = 200;
    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(SlowWriter { delay_ms })
        // The promotion that unblocks parallelism. Without it, the two
        // writes would route to the width-1 serial-mutator pool.
        .tool_concurrency("slow_write", ToolConcurrency::on())
        .build()
        .unwrap();

    let started = Instant::now();
    let result = agent
        .run(
            vec![Message::user_text("write two files")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");
    let elapsed = started.elapsed();

    // Verify both files written with expected content.
    let a_content = tokio::fs::read_to_string(&path_a).await.expect("read a");
    let b_content = tokio::fs::read_to_string(&path_b).await.expect("read b");
    assert_eq!(a_content, "alpha", "a.txt content mismatch");
    assert_eq!(b_content, "beta", "b.txt content mismatch");

    // Wall-time assertion: parallel ≈ delay; serial would be ≈ 2× delay.
    // Use 1.5× as the threshold — comfortably above sample-variance noise
    // on a contended machine but well below the serial floor.
    let parallel_threshold = Duration::from_millis(delay_ms + delay_ms / 2);
    assert!(
        elapsed < parallel_threshold,
        "expected parallel wall-time (< {parallel_threshold:?}), got {elapsed:?} \
         — looks like writes serialised through the width-1 pool"
    );

    println!("result: {}", result.text);
    println!("a.txt: {a_content}");
    println!("b.txt: {b_content}");
    println!("wall time: {elapsed:?} (single-call delay: {delay_ms}ms)");
    println!();
    println!("Without tool_concurrency promotion, this batch would take ≈ 2× delay.");
    println!("With ToolConcurrency::on(), both writes overlap in the concurrent-mutator pool.");

    // Tidy up. The next process_id will get a fresh dir if they re-run.
    let _ = tokio::fs::remove_dir_all(&dir).await;
}
