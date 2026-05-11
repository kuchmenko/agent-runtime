use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use tkach::message::{Content, Message, StopReason, Usage};
use tkach::provider::Response;
use tkach::providers::Mock;
use tkach::{
    Agent, CancellationToken, InterruptOutcome, InterruptTarget, StreamEvent, Tool, ToolClass,
    ToolContext, ToolError, ToolOutput, TurnId,
};
use tokio::sync::Notify;

fn test_dir() -> std::path::PathBuf {
    std::env::current_dir().unwrap()
}

fn prompt(text: &str) -> Vec<Message> {
    vec![Message::user_text(text)]
}

struct SlowTool {
    started: Arc<Notify>,
    delay: Duration,
}

#[async_trait]
impl Tool for SlowTool {
    fn name(&self) -> &str {
        "slow"
    }

    fn description(&self) -> &str {
        "slow test tool"
    }

    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }

    fn class(&self) -> ToolClass {
        ToolClass::ReadOnly
    }

    async fn execute(&self, _input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        self.started.notify_waiters();
        tokio::select! {
            _ = ctx.cancel.cancelled() => Err(ToolError::Cancelled),
            _ = tokio::time::sleep(self.delay) => Ok(ToolOutput::text("slow done")),
        }
    }
}

#[tokio::test]
async fn queued_user_message_drains_before_next_provider_request() {
    let started = Arc::new(Notify::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);

    let mock = Mock::new(move |req| {
        let n = calls_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "slow-1".into(),
                    name: "slow".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                let saw_queued = req
                    .messages
                    .iter()
                    .any(|m| m.text().contains("queued fact"));
                assert!(
                    saw_queued,
                    "queued steering message was not sent to provider"
                );
                Ok(Response {
                    content: vec![Content::text("final")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tool(SlowTool {
            started: Arc::clone(&started),
            delay: Duration::from_millis(50),
        })
        .working_dir(test_dir())
        .build()
        .unwrap();

    let (future, handle) = agent.run_with_handle(prompt("start"), CancellationToken::new());
    let task = tokio::spawn(future);

    started.notified().await;
    let turn_id = handle
        .queue_user_message("queued fact", handle.current_turn_id())
        .unwrap();
    assert_eq!(Some(turn_id), handle.current_turn_id());

    let result = task.await.unwrap().unwrap();
    assert_eq!(result.text, "final");
    assert!(
        result
            .new_messages
            .iter()
            .any(|m| m.text() == "queued fact")
    );
}

#[tokio::test]
async fn queue_rejects_mismatched_turn_id() {
    let started = Arc::new(Notify::new());
    let mock = Mock::new(|_| {
        Ok(Response {
            content: vec![Content::ToolUse {
                id: "slow-1".into(),
                name: "slow".into(),
                input: json!({}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tool(SlowTool {
            started: Arc::clone(&started),
            delay: Duration::from_millis(200),
        })
        .max_turns(1)
        .working_dir(test_dir())
        .build()
        .unwrap();

    let cancel = CancellationToken::new();
    let (future, handle) = agent.run_with_handle(prompt("start"), cancel.clone());
    let task = tokio::spawn(future);

    started.notified().await;
    let err = handle
        .queue_user_message("wrong turn", Some(TurnId::from("turn_wrong".to_string())))
        .unwrap_err();
    assert!(matches!(
        err,
        tkach::SteerError::ExpectedTurnMismatch { .. }
    ));
    cancel.cancel();
    let _ = task.await;
}

#[tokio::test]
async fn interrupt_tool_cancels_only_that_tool_and_turn_continues() {
    let started = Arc::new(Notify::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);

    let mock = Mock::new(move |req| {
        let n = calls_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "slow-1".into(),
                    name: "slow".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                let saw_cancelled_result = req.messages.iter().any(|m| {
                    m.content.iter().any(|c| match c {
                        Content::ToolResult {
                            content, is_error, ..
                        } => *is_error && content.contains("cancelled"),
                        _ => false,
                    })
                });
                assert!(
                    saw_cancelled_result,
                    "tool interrupt was not returned to the model"
                );
                Ok(Response {
                    content: vec![Content::text("recovered")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tool(SlowTool {
            started: Arc::clone(&started),
            delay: Duration::from_secs(30),
        })
        .working_dir(test_dir())
        .build()
        .unwrap();

    let (future, handle) = agent.run_with_handle(prompt("start"), CancellationToken::new());
    let task = tokio::spawn(future);

    started.notified().await;
    let outcome = handle
        .interrupt(InterruptTarget::Tool {
            tool_call_id: "slow-1".into(),
        })
        .unwrap();
    assert_eq!(outcome, InterruptOutcome::Cancelled);

    let result = task.await.unwrap().unwrap();
    assert_eq!(result.text, "recovered");

    let after_done = handle
        .interrupt(InterruptTarget::Tool {
            tool_call_id: "slow-1".into(),
        })
        .unwrap();
    assert_eq!(after_done, InterruptOutcome::AlreadyDone);
}

#[tokio::test]
async fn stream_with_handle_emits_turn_started() {
    let agent = Agent::builder()
        .provider(Mock::with_text("hello"))
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let (mut stream, handle) = agent.stream_with_handle(prompt("hi"), CancellationToken::new());
    let first = stream.next().await.unwrap().unwrap();
    let StreamEvent::TurnStarted { turn_id } = first else {
        panic!("expected TurnStarted");
    };
    assert!(turn_id.as_str().starts_with("turn_"));
    drop(handle);

    let result = stream.collect_result().await.unwrap();
    assert_eq!(result.text, "hello");
}
