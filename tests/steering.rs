use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use tkach::message::{Content, Message, StopReason, Usage};
use tkach::provider::{LlmProvider, Request, Response};
use tkach::providers::Mock;
use tkach::stream::ProviderEventStream;
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
        self.started.notify_one();
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
async fn queue_rejects_non_user_content() {
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
        .queue_user_message(
            vec![Content::ToolResult {
                tool_use_id: "t1".into(),
                content: "not user content".into(),
                is_error: false,
                cache_control: None,
            }],
            handle.current_turn_id(),
        )
        .unwrap_err();
    assert!(matches!(err, tkach::SteerError::InvalidContent));
    cancel.cancel();
    let _ = task.await;
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

struct MutatingProbe;

#[async_trait]
impl Tool for MutatingProbe {
    fn name(&self) -> &str {
        "mutate"
    }
    fn description(&self) -> &str {
        "mutate"
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn class(&self) -> ToolClass {
        ToolClass::Mutating
    }
    async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text("mutated"))
    }
}

#[tokio::test]
async fn plan_mode_denies_mutating_tool_without_approval() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);
    let mock = Mock::new(move |req| {
        let n = calls_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "m1".into(),
                    name: "mutate".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                let denied = req.messages.iter().any(|m| {
                    m.content.iter().any(|c| match c {
                        Content::ToolResult {
                            content, is_error, ..
                        } => *is_error && content.contains("mode denied"),
                        _ => false,
                    })
                });
                assert!(denied, "PlanMode did not deny mutating tool");
                Ok(Response {
                    content: vec![Content::text("planned")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
    });
    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tool(MutatingProbe)
        .working_dir(test_dir())
        .build()
        .unwrap();

    let (future, handle) = agent.run_with_handle(prompt("start"), CancellationToken::new());
    handle
        .set_mode(Box::new(tkach::PlanMode), tkach::ModeAuthority::Operator)
        .unwrap();
    let result = future.await.unwrap();
    assert_eq!(result.text, "planned");
}

struct TestBridge;

#[async_trait]
impl tkach::UserInputBridge for TestBridge {
    async fn collect(
        &self,
        _questions: &tkach::QuestionSet,
    ) -> Result<tkach::UserInputResponse, tkach::BridgeError> {
        Ok(tkach::UserInputResponse::Cancelled)
    }
}

struct GatedStreamProvider {
    release: Arc<Notify>,
}

#[async_trait]
impl LlmProvider for GatedStreamProvider {
    async fn complete(&self, _request: Request) -> Result<Response, tkach::ProviderError> {
        Ok(Response {
            content: vec![Content::text("hello")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }

    async fn stream(&self, _request: Request) -> Result<ProviderEventStream, tkach::ProviderError> {
        self.release.notified().await;
        Ok(Box::pin(futures::stream::iter([
            Ok(StreamEvent::ContentDelta("hello".into())),
            Ok(StreamEvent::MessageDelta {
                stop_reason: StopReason::EndTurn,
            }),
            Ok(StreamEvent::Usage(Usage::default())),
            Ok(StreamEvent::Done),
        ])))
    }
}

#[tokio::test]
async fn operator_mode_change_emits_before_single_turn_stream_finishes() {
    let release = Arc::new(Notify::new());
    let agent = Agent::builder()
        .provider(GatedStreamProvider {
            release: Arc::clone(&release),
        })
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let (mut stream, handle) = agent.stream_with_handle(prompt("hi"), CancellationToken::new());
    let first = stream.next().await.unwrap().unwrap();
    assert!(matches!(first, StreamEvent::TurnStarted { .. }));

    handle
        .set_mode(Box::new(tkach::PlanMode), tkach::ModeAuthority::Operator)
        .unwrap();
    release.notify_one();

    let mut saw_changed = false;
    while let Some(event) = stream.next().await {
        if let StreamEvent::ModeChanged {
            from,
            to,
            authority,
        } = event.unwrap()
        {
            saw_changed = from == "default" && to == "plan";
            assert_eq!(authority, tkach::ModeAuthority::Operator);
        }
    }

    assert!(saw_changed, "operator mode change was not emitted");
    let result = stream.into_result().await.unwrap();
    assert_eq!(result.text, "hello");
}

#[tokio::test]
async fn agent_mode_request_event_can_be_cancelled_before_apply() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);
    let mock = Mock::new(move |_| {
        let n = calls_clone.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "slow-1".into(),
                    name: "slow".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });
    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tool(SlowTool {
            started: Arc::new(Notify::new()),
            delay: Duration::from_millis(25),
        })
        .working_dir(test_dir())
        .build()
        .unwrap();

    let (mut stream, handle) = agent.stream_with_handle(prompt("start"), CancellationToken::new());
    while let Some(event) = stream.next().await {
        if matches!(event.unwrap(), StreamEvent::ToolCallPending { .. }) {
            break;
        }
    }
    handle
        .set_mode(Box::new(tkach::PlanMode), tkach::ModeAuthority::Agent)
        .unwrap();

    let mut saw_requested = false;
    let mut saw_changed = false;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::ModeChangeRequested { from, to, .. } => {
                saw_requested = from == "default" && to == "plan";
                handle.cancel_pending_mode_change().unwrap();
            }
            StreamEvent::ModeChanged { .. } => saw_changed = true,
            StreamEvent::ContentDelta(_) => break,
            _ => {}
        }
    }

    assert!(saw_requested, "agent mode request was not emitted");
    assert!(!saw_changed, "cancelled pending mode still applied");
    let result = stream.into_result().await.unwrap();
    assert_eq!(result.text, "done");
}

#[tokio::test]
async fn root_handle_can_ask_user_via_bridge() {
    let agent = Agent::builder()
        .provider(Mock::with_text("hello"))
        .model("test")
        .user_input_bridge(TestBridge)
        .working_dir(test_dir())
        .build()
        .unwrap();
    let (_future, handle) = agent.run_with_handle(prompt("hi"), CancellationToken::new());
    let response = handle
        .ask_user(
            tkach::QuestionSet {
                questions: Vec::new(),
            },
            Duration::from_secs(1),
        )
        .await
        .unwrap();
    assert_eq!(response, tkach::UserInputResponse::Cancelled);
}

#[tokio::test]
async fn continuation_guard_rejects_unwired_operator_command_escape() {
    let agent = Agent::builder()
        .provider(Mock::with_text("hello"))
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();
    let (_future, handle) = agent.run_with_handle(prompt("hi"), CancellationToken::new());

    let err = handle
        .install_continuation_guard(tkach::ContinuationGuard {
            name: "unbounded".into(),
            trigger: tkach::GuardTrigger::OnTurnEnd,
            predicate: Box::new(|_| tkach::GuardDecision::Continue),
            continuation_prompt: "continue".into(),
            max_iterations: None,
            escape: tkach::GuardEscape::OperatorCommand("stop".into()),
        })
        .unwrap_err();
    assert!(matches!(err, tkach::GuardError::NoEscapeMechanism));
}

#[tokio::test]
async fn continuation_guard_abort_after_tool_result_stops_run() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);
    let mock = Mock::new(move |_| {
        let n = calls_clone.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "m1".into(),
                    name: "mutate".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            panic!("guard abort should stop before the next provider turn");
        }
    });
    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tool(MutatingProbe)
        .working_dir(test_dir())
        .build()
        .unwrap();
    let (future, handle) = agent.run_with_handle(prompt("start"), CancellationToken::new());
    handle
        .install_continuation_guard(tkach::ContinuationGuard {
            name: "abort".into(),
            trigger: tkach::GuardTrigger::OnTurnEnd,
            predicate: Box::new(|snapshot| {
                assert_eq!(snapshot.recent_tool_calls, vec!["mutate:m1".to_string()]);
                tkach::GuardDecision::Abort {
                    reason: "stop now".into(),
                }
            }),
            continuation_prompt: "unused".into(),
            max_iterations: Some(1),
            escape: tkach::GuardEscape::MaxIterations,
        })
        .unwrap();

    let result = future.await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.stop_reason, StopReason::ToolUse);
}

#[tokio::test]
async fn continuation_guard_injects_until_predicate_stops() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);
    let mock = Mock::new(move |req| {
        let n = calls_clone.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Ok(Response {
                content: vec![Content::text("first")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        } else {
            assert!(
                req.messages
                    .iter()
                    .any(|m| m.text().contains("continue please"))
            );
            Ok(Response {
                content: vec![Content::text("second")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });
    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();
    let (future, handle) = agent.run_with_handle(prompt("hi"), CancellationToken::new());
    handle
        .install_continuation_guard(tkach::ContinuationGuard {
            name: "once".into(),
            trigger: tkach::GuardTrigger::OnTurnEnd,
            predicate: Box::new(|snapshot| {
                if snapshot.turn_count == 1 {
                    tkach::GuardDecision::Continue
                } else {
                    tkach::GuardDecision::Stop
                }
            }),
            continuation_prompt: "continue please".into(),
            max_iterations: Some(1),
            escape: tkach::GuardEscape::MaxIterations,
        })
        .unwrap();

    let result = future.await.unwrap();
    assert_eq!(result.text, "second");
}
