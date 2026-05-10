//! End-to-end concurrency tests.
//!
//! Unit tests in `src/executor.rs` exercise the executor in isolation
//! against a synthetic `ToolContext`. These integration tests build a
//! real `Agent` from a `Mock` provider and assert behaviour at the
//! public-API surface — the path consumers actually take.
//!
//! Each parallel-case test ships with a paired serial-baseline test
//! (without `tool_concurrency` promotion) so a regression that
//! re-serialises the executor would fail the parallel test loudly,
//! and an accidental "always-parallel" change would fail the serial
//! baseline. Without the negative case the parallel assertion alone
//! could pass spuriously on a machine fast enough that even sequential
//! 200ms sleeps fit under the threshold.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tkach::message::{Content, Message, StopReason, Usage};
use tkach::provider::Response;
use tkach::providers::Mock;
use tkach::tools::SubAgent;
use tkach::{
    Agent, CancellationToken, LlmProvider, Tool, ToolClass, ToolConcurrency, ToolContext,
    ToolError, ToolOutput,
};

// --- Test fixtures ---------------------------------------------------

/// A `Mutating` tool that sleeps `delay_ms` then writes `content` to
/// `file_path`. Used to make wall-time differences observable across
/// parallel and serial runs.
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
                "content":   { "type": "string" }
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

/// A `ReadOnly` tool that sleeps then echoes its label. Used inside
/// sub-agents so wall-time is dominated by the sleep, not LLM-loop
/// overhead.
struct SlowReader {
    delay_ms: u64,
}

#[async_trait::async_trait]
impl Tool for SlowReader {
    fn name(&self) -> &str {
        "slow_reader"
    }
    fn description(&self) -> &str {
        "ReadOnly tool that sleeps then echoes a label"
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

/// Per-test scratch directory. Process id alone is not enough — the
/// same process runs many tests, and they could overlap on a single
/// `tkach-tests-<pid>` dir if invoked concurrently. Combine pid +
/// nanos so overlapping tests still get fresh paths.
fn fresh_tmp_dir(label: &str) -> std::path::PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "tkach-tests-{label}-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Builds a Mock that emits two SlowWrite calls on turn 0, then text
/// on turn 1 (after both tool_results land in the message history).
fn writes_mock(path_a: String, path_b: String) -> Mock {
    Mock::new(move |req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_results {
            return Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        Ok(Response {
            content: vec![
                Content::ToolUse {
                    id: "w1".into(),
                    name: "slow_write".into(),
                    input: json!({ "file_path": path_a, "content": "alpha" }),
                },
                Content::ToolUse {
                    id: "w2".into(),
                    name: "slow_write".into(),
                    input: json!({ "file_path": path_b, "content": "beta" }),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    })
}

/// Builds a Mock that emits N agent tool_use blocks on turn 0, then
/// text on turn 1.
fn fanout_mock(prompts: Vec<&'static str>) -> Mock {
    Mock::new(move |req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_results {
            return Ok(Response {
                content: vec![Content::text("all done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        let content: Vec<Content> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| Content::ToolUse {
                id: format!("a{i}"),
                name: "agent".into(),
                input: json!({ "prompt": *p }),
            })
            .collect();
        Ok(Response {
            content,
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    })
}

/// Mock for sub-agents: one slow_reader call on turn 0, text on turn 1.
fn sub_provider() -> Arc<dyn LlmProvider> {
    Arc::new(Mock::new(|req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_results {
            return Ok(Response {
                content: vec![Content::text("sub done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        let label = req
            .messages
            .iter()
            .find_map(|m| {
                m.content.iter().find_map(|c| match c {
                    Content::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_else(|| "anon".into());
        Ok(Response {
            content: vec![Content::ToolUse {
                id: "r".into(),
                name: "slow_reader".into(),
                input: json!({ "label": label }),
            }],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    }))
}

// --- Parallel writes (positive + negative) ---------------------------

const WRITE_DELAY_MS: u64 = 200;

#[tokio::test]
async fn parallel_writes_with_promotion_overlap() {
    let dir = fresh_tmp_dir("parallel-writes");
    let pa = dir.join("a.txt");
    let pb = dir.join("b.txt");

    let mock = writes_mock(
        pa.to_string_lossy().into_owned(),
        pb.to_string_lossy().into_owned(),
    );

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(SlowWriter {
            delay_ms: WRITE_DELAY_MS,
        })
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

    assert_eq!(
        tokio::fs::read_to_string(&pa).await.unwrap(),
        "alpha",
        "a.txt content mismatch"
    );
    assert_eq!(
        tokio::fs::read_to_string(&pb).await.unwrap(),
        "beta",
        "b.txt content mismatch"
    );
    assert_eq!(result.text, "done");

    // Parallel ≈ 1× delay; a 1.5× ceiling absorbs scheduler noise but
    // fails loudly if the executor regresses to serial.
    let ceiling = Duration::from_millis(WRITE_DELAY_MS + WRITE_DELAY_MS / 2);
    assert!(
        elapsed < ceiling,
        "parallel writes took {elapsed:?}, expected < {ceiling:?} \
         (regressed to serial-mut pool?)"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn writes_without_promotion_serialise() {
    // Negative baseline: WITHOUT tool_concurrency promotion, two
    // SlowWrite calls must serialise via the width-1 serial-mutator
    // pool. If this test passes the parallel one above, the parallel
    // test is meaningful. If it fails, the executor is parallelising
    // by default — which would mean the consumer-opt-in contract is
    // broken.
    let dir = fresh_tmp_dir("serial-writes");
    let pa = dir.join("a.txt");
    let pb = dir.join("b.txt");

    let mock = writes_mock(
        pa.to_string_lossy().into_owned(),
        pb.to_string_lossy().into_owned(),
    );

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(SlowWriter {
            delay_ms: WRITE_DELAY_MS,
        })
        // NO tool_concurrency call — slow_write stays in serial_mut.
        .build()
        .unwrap();

    let started = Instant::now();
    let _ = agent
        .run(
            vec![Message::user_text("write two files")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");
    let elapsed = started.elapsed();

    // Serial ≈ 2× delay. Lower bound: must be at least 1.7× to prove
    // serialisation. Higher than that allows scheduler slack.
    let lower_bound = Duration::from_millis(WRITE_DELAY_MS * 17 / 10);
    assert!(
        elapsed >= lower_bound,
        "serial writes finished too fast ({elapsed:?}, expected >= {lower_bound:?}) \
         — promotion-by-default would break the consumer-opt-in contract"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

// --- Parallel sub-agents (positive + negative) -----------------------

const SUB_DELAY_MS: u64 = 200;

#[tokio::test]
async fn parallel_subagents_with_promotion_overlap() {
    let parent_mock = fanout_mock(vec!["topic-A", "topic-B", "topic-C"]);

    let agent = Agent::builder()
        .provider(parent_mock)
        .model("mock-parent")
        .tool(SlowReader {
            delay_ms: SUB_DELAY_MS,
        })
        .tool(SubAgent::new(sub_provider(), "mock-sub").max_turns(3))
        .tool_concurrency("agent", ToolConcurrency::on())
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

    assert_eq!(result.text, "all done");
    // Parallel ≈ 1× inner delay. 1.75× ceiling absorbs sub-agent loop
    // overhead while still failing loud on serial regression.
    let ceiling = Duration::from_millis(SUB_DELAY_MS + (SUB_DELAY_MS * 3 / 4));
    assert!(
        elapsed < ceiling,
        "parallel sub-agents took {elapsed:?}, expected < {ceiling:?} \
         (regressed to serial-mut pool?)"
    );
}

#[tokio::test]
async fn subagents_overlap_without_explicit_promotion() {
    // Replaces the prior `subagents_without_promotion_serialise`
    // negative baseline. Tool::is_recursive (true on SubAgent)
    // routes recursive tools through `concurrent_mut` regardless of
    // explicit `tool_concurrency` promotion — the consumer-opt-in
    // contract only applies to *non-recursive* tools (write/edit/
    // bash). Recursive tools must bypass `serial_mut` to avoid the
    // permit-held-during-nested-execute deadlock that would
    // otherwise apply when a parent holds the permit for the
    // duration of `SubAgent::execute` and a child's tool also needs
    // a permit from the same shared pool.
    //
    // Default behaviour for SubAgent is therefore "parallel up to
    // `max_concurrent_mutations`". To force serial execution of
    // `agent`, the consumer can install
    // `tool_concurrency("agent", ToolConcurrency::on().max(1))`.
    let parent_mock = fanout_mock(vec!["topic-A", "topic-B", "topic-C"]);

    let agent = Agent::builder()
        .provider(parent_mock)
        .model("mock-parent")
        .tool(SlowReader {
            delay_ms: SUB_DELAY_MS,
        })
        .tool(SubAgent::new(sub_provider(), "mock-sub").max_turns(3))
        // No `tool_concurrency` call — but is_recursive admits
        // through concurrent_mut anyway.
        .build()
        .unwrap();

    let started = Instant::now();
    let _ = agent
        .run(
            vec![Message::user_text("delegate to three sub-agents")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");
    let elapsed = started.elapsed();

    // Parallel ≈ 1× inner delay. 1.75× ceiling absorbs sub-agent
    // loop overhead while still failing loud if execution fell back
    // to serial_mut.
    let ceiling = Duration::from_millis(SUB_DELAY_MS + (SUB_DELAY_MS * 3 / 4));
    assert!(
        elapsed < ceiling,
        "is_recursive routing should overlap sub-agents by default \
         ({elapsed:?}, expected < {ceiling:?})"
    );
}

#[tokio::test]
async fn siblings_serialise_on_default_mutating_tools() {
    // Codex P1 #3 fix verification: ConcurrencyConfig::fork() shares
    // the `serial_mut` pool across sub-agent boundaries so that
    // default-Mutating built-in tools (write, edit, bash) keep their
    // "no opt-in needed for safe serialisation" contract even when
    // siblings invoke them concurrently. Without the shared pool,
    // two siblings editing the same file could race despite the
    // user opting into none of the concurrency knobs.
    //
    // Setup: 2 sub-agents in parallel (admitted via the per-level-
    // forked concurrent_mut). Each sub-agent runs a default-Mutating
    // SlowWriter (which goes through the SHARED serial_mut, not the
    // per-level-forked one). Two siblings + width-1 serial_mut means
    // wall time for the writes ≈ 2× single-write delay even though
    // sub-agents themselves overlap.
    let dir = std::sync::Arc::new(fresh_tmp_dir("siblings-serialise"));

    let dir_for_sub = std::sync::Arc::clone(&dir);
    let sub_provider: Arc<dyn LlmProvider> = Arc::new(Mock::new(move |req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_results {
            return Ok(Response {
                content: vec![Content::text("sub done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        let label = req
            .messages
            .iter()
            .find_map(|m| {
                m.content.iter().find_map(|c| match c {
                    Content::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_else(|| "anon".into());
        let path = dir_for_sub.join(format!("{label}.txt"));
        Ok(Response {
            content: vec![Content::ToolUse {
                id: "w".into(),
                name: "slow_write".into(),
                input: json!({
                    "file_path": path.to_string_lossy().into_owned(),
                    "content":   "child write"
                }),
            }],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    }));

    let parent_mock = fanout_mock(vec!["topic-A", "topic-B"]);

    // Use 200ms write delay so the timing assertion has room.
    let agent = Agent::builder()
        .provider(parent_mock)
        .model("mock-parent")
        .tool(SlowWriter { delay_ms: 200 })
        .tool(SubAgent::new(Arc::clone(&sub_provider), "mock-sub").max_turns(3))
        // No `tool_concurrency` for "slow_write" — it stays
        // default-Mutating, must serialise via shared serial_mut
        // even though the two parent sub-agents run in parallel.
        .build()
        .unwrap();

    let started = Instant::now();
    let _ = agent
        .run(
            vec![Message::user_text("delegate")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");
    let elapsed = started.elapsed();

    // Two siblings each holding a 200ms slow_write that share a
    // width-1 serial_mut pool: total ≥ 350ms (2 × 200ms minus some
    // sub-agent loop overhead). If serial_mut is forked per level,
    // the two writes overlap and total drops to ~200ms.
    let lower_bound = Duration::from_millis(350);
    assert!(
        elapsed >= lower_bound,
        "siblings must serialise on default-Mutating tools via shared serial_mut \
         (got {elapsed:?}, expected >= {lower_bound:?}) — fork() must NOT reset \
         the serial_mut pool"
    );

    let _ = std::fs::remove_dir_all(dir.as_path());
}

// --- Result ordering -------------------------------------------------

#[tokio::test]
async fn parallel_writes_preserve_input_order() {
    // Three writes with descending delays — the slowest is FIRST in
    // input order. With parallel execution, fast writes finish before
    // the slow first one, but tool_results in the assistant-tool_result
    // round-trip must come back in input order to keep the agent loop
    // invariant intact.
    let dir = fresh_tmp_dir("order");
    let p1 = dir.join("1.txt");
    let p2 = dir.join("2.txt");
    let p3 = dir.join("3.txt");

    let p1s = p1.to_string_lossy().into_owned();
    let p2s = p2.to_string_lossy().into_owned();
    let p3s = p3.to_string_lossy().into_owned();

    let mock = Mock::new(move |req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_results {
            return Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        Ok(Response {
            content: vec![
                Content::ToolUse {
                    id: "first".into(),
                    name: "slow_write".into(),
                    input: json!({ "file_path": p1s, "content": "1" }),
                },
                Content::ToolUse {
                    id: "second".into(),
                    name: "slow_write".into(),
                    input: json!({ "file_path": p2s, "content": "2" }),
                },
                Content::ToolUse {
                    id: "third".into(),
                    name: "slow_write".into(),
                    input: json!({ "file_path": p3s, "content": "3" }),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        // First call is the slowest so it definitely finishes after
        // the others — proves the result vector is reordered to match
        // input order rather than completion order.
        .tool(SlowWriter { delay_ms: 150 })
        .tool_concurrency("slow_write", ToolConcurrency::on())
        .build()
        .unwrap();

    let result = agent
        .run(
            vec![Message::user_text("three ordered writes")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");

    // Walk the assistant-emitted tool_use IDs and the user-side
    // tool_result tool_use_ids; pair them up and assert order matches.
    let assistant_msg = result
        .new_messages
        .iter()
        .find(|m| matches!(m.role, tkach::Role::Assistant))
        .expect("assistant message");
    let tool_use_ids: Vec<&str> = assistant_msg
        .content
        .iter()
        .filter_map(|c| match c {
            Content::ToolUse { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_use_ids, vec!["first", "second", "third"]);

    let user_msg = result
        .new_messages
        .iter()
        .find(|m| matches!(m.role, tkach::Role::User))
        .expect("user tool_result message");
    let tool_result_ids: Vec<&str> = user_msg
        .content
        .iter()
        .filter_map(|c| match c {
            Content::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_result_ids,
        vec!["first", "second", "third"],
        "tool_result order must match tool_use order"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

// --- Cancel during parallel batch -----------------------------------

#[tokio::test]
async fn cancel_during_parallel_batch_short_circuits_pending_calls() {
    // Cap parallelism at 1 to force queueing, then fire cancel while
    // the first call is mid-execution. The second call must short-
    // circuit with cancelled-before-execution rather than running.
    let dir = fresh_tmp_dir("cancel");
    let pa = dir.join("a.txt");
    let pb = dir.join("b.txt");

    let mock = writes_mock(
        pa.to_string_lossy().into_owned(),
        pb.to_string_lossy().into_owned(),
    );

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        // Long delay so cancel definitely fires while first call is
        // still sleeping.
        .tool(SlowWriter { delay_ms: 500 })
        // Cap at 1 to force queueing.
        .tool_concurrency("slow_write", ToolConcurrency::on().max(1))
        .build()
        .unwrap();

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_clone.cancel();
    });

    let result = agent
        .run(vec![Message::user_text("two writes")], cancel)
        .await;

    // The agent run is expected to error with Cancelled; partial
    // results may be present. The second file MUST NOT exist —
    // cancel hit before its execute() body could run.
    assert!(
        matches!(result, Err(tkach::AgentError::Cancelled { .. })),
        "expected Cancelled, got {result:?}"
    );
    assert!(
        !pb.exists(),
        "second file should not have been written — cancel must short-circuit \
         the queued call before its execute() body"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

// --- Defaults preserved when no concurrency methods called -----------

// --- Cross-class ordering barrier (Codex P1 fix) ---------------------

#[tokio::test]
async fn read_after_write_observes_the_write() {
    // Codex P1: a [Write A, Read A] batch must serialise the Read
    // after the Write so the Read observes the side effect. The
    // executor doesn't know which path each tool touches, so it
    // falls back on the LLM-emitted order across class boundaries.
    let dir = fresh_tmp_dir("ordering");
    let path = dir.join("a.txt");
    let path_str = path.to_string_lossy().into_owned();

    let mock = Mock::new(move |req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_results {
            return Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        Ok(Response {
            content: vec![
                Content::ToolUse {
                    id: "w".into(),
                    name: "slow_write".into(),
                    input: json!({ "file_path": path_str, "content": "expected" }),
                },
                Content::ToolUse {
                    id: "r".into(),
                    name: "read_file".into(),
                    input: json!({ "file_path": path_str }),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    });

    /// Minimal ReadOnly file reader for this test.
    struct ReadFile;
    #[async_trait::async_trait]
    impl Tool for ReadFile {
        fn name(&self) -> &str {
            "read_file"
        }
        fn description(&self) -> &str {
            "read a file"
        }
        fn input_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "file_path": { "type": "string" } },
                "required": ["file_path"]
            })
        }
        fn class(&self) -> ToolClass {
            ToolClass::ReadOnly
        }
        async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
            let p = input["file_path"]
                .as_str()
                .ok_or_else(|| ToolError::InvalidInput("file_path required".into()))?;
            // Read may race the Write — so missing file becomes
            // empty content rather than an error, surfacing exactly
            // what the executor enabled the read to observe.
            let content = tokio::fs::read_to_string(p).await.unwrap_or_default();
            Ok(ToolOutput::text(content))
        }
    }

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(SlowWriter { delay_ms: 100 })
        .tool(ReadFile)
        // Promote the writer so it goes into concurrent_mut. The
        // ordering invariant must still hold even with promotion —
        // the cross-class barrier between [promoted-Mut, RO] is
        // independent of within-class concurrency.
        .tool_concurrency("slow_write", ToolConcurrency::on())
        .build()
        .unwrap();

    let result = agent
        .run(
            vec![Message::user_text("write then read")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");

    // Walk the new_messages to find the read_file's tool_result body.
    let user_msg = result
        .new_messages
        .iter()
        .find(|m| matches!(m.role, tkach::Role::User))
        .expect("tool_result message");
    let read_result = user_msg
        .content
        .iter()
        .find_map(|c| match c {
            Content::ToolResult {
                tool_use_id,
                content,
                ..
            } if tool_use_id == "r" => Some(content.clone()),
            _ => None,
        })
        .expect("read tool_result");

    assert_eq!(
        read_result, "expected",
        "read after write must observe the write — got {read_result:?}"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

// --- Nested deadlock (Codex P1 fix) ----------------------------------

#[tokio::test]
async fn nested_promoted_fanout_does_not_deadlock_when_parent_saturates_pool() {
    // Codex P1: a parent batch that saturates `concurrent_mut` with
    // promoted `agent` calls would deadlock if children competed for
    // the same pool. The fix forks ConcurrencyConfig per nesting level
    // so children have independent permit accounting.
    //
    // Setup: cap = 2 concurrent mutations. Parent emits 2 sub-agent
    // calls (both filling the cap). Each sub-agent then internally
    // emits 2 promoted-mutator slow_write calls. With shared pool:
    // parent holds 2 permits, all children block forever. With fork:
    // each sub-agent has its own pool of cap 2 and completes.

    let dir = std::sync::Arc::new(fresh_tmp_dir("nested-deadlock"));

    // Sub-agent's provider: emit 2 slow_writes; on turn 2 return text.
    let dir_for_sub = std::sync::Arc::clone(&dir);
    let sub_provider: Arc<dyn LlmProvider> = Arc::new(Mock::new(move |req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_results {
            return Ok(Response {
                content: vec![Content::text("sub done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        // Pull a unique label off the user prompt so each sub-agent
        // writes to its own paths.
        let label = req
            .messages
            .iter()
            .find_map(|m| {
                m.content.iter().find_map(|c| match c {
                    Content::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_else(|| "anon".into());
        let p1 = dir_for_sub.join(format!("{label}-1.txt"));
        let p2 = dir_for_sub.join(format!("{label}-2.txt"));
        Ok(Response {
            content: vec![
                Content::ToolUse {
                    id: "c1".into(),
                    name: "slow_write".into(),
                    input: json!({
                        "file_path": p1.to_string_lossy().into_owned(),
                        "content":   "child-write-1"
                    }),
                },
                Content::ToolUse {
                    id: "c2".into(),
                    name: "slow_write".into(),
                    input: json!({
                        "file_path": p2.to_string_lossy().into_owned(),
                        "content":   "child-write-2"
                    }),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    }));

    let parent_mock = fanout_mock(vec!["topic-A", "topic-B"]);

    let agent = Agent::builder()
        .provider(parent_mock)
        .model("mock-parent")
        .tool(SlowWriter { delay_ms: 50 })
        .tool(SubAgent::new(Arc::clone(&sub_provider), "mock-sub").max_turns(3))
        .max_concurrent_mutations(2)
        // Both `agent` and `slow_write` promoted into the
        // concurrent_mut pool of width 2.
        .tool_concurrency("agent", ToolConcurrency::on())
        .tool_concurrency("slow_write", ToolConcurrency::on())
        .build()
        .unwrap();

    // The whole flow should complete without timing out. If the
    // forking is broken and deadlock occurs, this future stalls
    // forever; we wrap in a timeout that's generous enough to
    // accommodate scheduler noise but tight enough to fail loudly
    // on a real deadlock (~5x the expected wallclock).
    let runner = agent.run(
        vec![Message::user_text("delegate")],
        CancellationToken::new(),
    );
    let result = tokio::time::timeout(Duration::from_secs(10), runner)
        .await
        .expect("nested fan-out should not deadlock — Codex P1 review on PR #41")
        .expect("agent run");

    assert_eq!(result.text, "all done");

    // Confirm all four child writes landed on disk.
    let entries: Vec<_> = std::fs::read_dir(dir.as_path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        entries.len(),
        4,
        "expected 4 files (2 sub-agents × 2 writes each), got {entries:?}"
    );

    let _ = std::fs::remove_dir_all(dir.as_path());
}

#[tokio::test]
async fn agent_built_without_concurrency_methods_matches_pre_feature_behaviour() {
    // Sanity check: an Agent built without any new builder method
    // call must behave identically to pre-feature behaviour. Two
    // ReadOnly tools in one batch parallelise (read pool, default
    // cap 20), and the absence of any Mutating call means no serial
    // pool involvement.
    struct SlowReadOnly {
        delay_ms: u64,
    }
    #[async_trait::async_trait]
    impl Tool for SlowReadOnly {
        fn name(&self) -> &str {
            "slow_ro"
        }
        fn description(&self) -> &str {
            "ro"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        fn class(&self) -> ToolClass {
            ToolClass::ReadOnly
        }
        async fn execute(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(ToolOutput::text("ok"))
        }
    }

    let mock = Mock::new(|req| {
        let has_results = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|c| matches!(c, Content::ToolResult { .. }))
        });
        if has_results {
            return Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        Ok(Response {
            content: vec![
                Content::ToolUse {
                    id: "r1".into(),
                    name: "slow_ro".into(),
                    input: json!({}),
                },
                Content::ToolUse {
                    id: "r2".into(),
                    name: "slow_ro".into(),
                    input: json!({}),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(SlowReadOnly { delay_ms: 150 })
        .build()
        .unwrap();

    let started = Instant::now();
    let result = agent
        .run(
            vec![Message::user_text("read both")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");
    let elapsed = started.elapsed();

    assert_eq!(result.text, "done");
    // Two RO calls in parallel ≈ 150ms; serial would be ≈ 300ms.
    let ceiling = Duration::from_millis(225);
    assert!(
        elapsed < ceiling,
        "default-built RO calls should parallelise, took {elapsed:?}"
    );
}
