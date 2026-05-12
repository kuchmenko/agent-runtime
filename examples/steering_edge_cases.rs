//! Deterministic end-to-end steering checks with a mock provider.
//!
//! Run with:
//!
//!   `cargo run --example steering_edge_cases`
//!
//! This does not call a real LLM. The mock provider asserts the exact
//! request shape at each provider boundary, which makes the steering
//! invariants reproducible in CI and on a laptop.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tkach::message::{Content, Message, StopReason, Usage};
use tkach::provider::Response;
use tkach::providers::Mock;
use tkach::{
    Agent, CancellationToken, InterruptOutcome, InterruptTarget, PolicyScope, PolicyTrigger,
    PromptPolicy, Tool, ToolClass, ToolContext, ToolError, ToolOutput,
};
use tokio::sync::Notify;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    queue_and_prompt_policy_apply_at_provider_boundary().await?;
    interrupt_cancels_one_tool_and_the_turn_continues().await?;
    plan_mode_denies_mutating_tools_before_execution().await?;

    eprintln!("✓ steering edge cases passed");
    Ok(())
}

async fn queue_and_prompt_policy_apply_at_provider_boundary()
-> Result<(), Box<dyn std::error::Error>> {
    let started = Arc::new(Notify::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);

    let mock = Mock::new(
        move |req| match calls_clone.fetch_add(1, Ordering::SeqCst) {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "slow-policy".into(),
                    name: "slow".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                assert!(
                    req.messages
                        .iter()
                        .any(|message| message.text() == "queued fact"),
                    "queued user content must land before the second provider request"
                );
                let system = req.system.as_ref().expect("policy creates system blocks");
                assert!(
                    system.iter().any(|block| block.text.contains(
                        "<!-- runtime policy: diagnose-first -->\nPrefer diagnosis before code."
                    )),
                    "prompt policy must be injected as a traceable system block"
                );
                Ok(Response {
                    content: vec![Content::text("policy observed")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        },
    );

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(SlowTool {
            started: Arc::clone(&started),
            delay: Duration::from_millis(50),
        })
        .working_dir(std::env::current_dir()?)
        .build()?;

    let (future, handle) =
        agent.run_with_handle(vec![Message::user_text("start")], CancellationToken::new());
    let task = tokio::spawn(future);

    started.notified().await;
    handle.queue_user_message("queued fact", handle.current_turn_id())?;
    handle.install_prompt_policy(PromptPolicy {
        name: "diagnose-first".into(),
        scope: PolicyScope::NextTurn,
        content: "Prefer diagnosis before code.".into(),
        precedence: 10,
        trigger: PolicyTrigger::Always,
    })?;

    let result = task.await??;
    assert_eq!(result.text, "policy observed");
    assert!(
        handle.list_prompt_policies().is_empty(),
        "NextTurn policy must remove itself after it applies"
    );
    eprintln!("✓ queued content + prompt policy boundary");
    Ok(())
}

async fn interrupt_cancels_one_tool_and_the_turn_continues()
-> Result<(), Box<dyn std::error::Error>> {
    let started = Arc::new(Notify::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);

    let mock = Mock::new(
        move |req| match calls_clone.fetch_add(1, Ordering::SeqCst) {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "slow-interrupt".into(),
                    name: "slow".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                assert!(
                    req.messages.iter().any(|message| {
                        message.content.iter().any(|content| {
                            matches!(
                                content,
                                Content::ToolResult { is_error: true, content, .. }
                                    if content.contains("cancel")
                            )
                        })
                    }),
                    "interrupted tool must be returned to the model as an error tool result"
                );
                Ok(Response {
                    content: vec![Content::text("turn recovered")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        },
    );

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(SlowTool {
            started: Arc::clone(&started),
            delay: Duration::from_secs(5),
        })
        .working_dir(std::env::current_dir()?)
        .build()?;

    let (future, handle) =
        agent.run_with_handle(vec![Message::user_text("start")], CancellationToken::new());
    let task = tokio::spawn(future);

    started.notified().await;
    let outcome = handle.interrupt(InterruptTarget::Tool {
        tool_call_id: "slow-interrupt".into(),
    })?;
    assert_eq!(outcome, InterruptOutcome::Cancelled);

    let result = task.await??;
    assert_eq!(result.text, "turn recovered");
    eprintln!("✓ tool interrupt + turn continuation");
    Ok(())
}

async fn plan_mode_denies_mutating_tools_before_execution() -> Result<(), Box<dyn std::error::Error>>
{
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);

    let mock = Mock::new(
        move |req| match calls_clone.fetch_add(1, Ordering::SeqCst) {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "mutate-1".into(),
                    name: "mutate".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                assert!(
                    req.messages.iter().any(|message| {
                        message.content.iter().any(|content| {
                            matches!(
                                content,
                                Content::ToolResult { is_error: true, content, .. }
                                    if content.contains("mode denied")
                            )
                        })
                    }),
                    "PlanMode must deny mutating tools before execution"
                );
                Ok(Response {
                    content: vec![Content::text("plan protected")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        },
    );

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(MutatingProbe)
        .working_dir(std::env::current_dir()?)
        .build()?;

    let (future, handle) =
        agent.run_with_handle(vec![Message::user_text("start")], CancellationToken::new());
    handle.set_mode(Box::new(tkach::PlanMode), tkach::ModeAuthority::Operator)?;

    let result = future.await?;
    assert_eq!(result.text, "plan protected");
    assert_eq!(MUTATION_EXECUTIONS.load(Ordering::SeqCst), 0);
    eprintln!("✓ mode gate denies mutation before execution");
    Ok(())
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

static MUTATION_EXECUTIONS: AtomicUsize = AtomicUsize::new(0);

struct MutatingProbe;

#[async_trait]
impl Tool for MutatingProbe {
    fn name(&self) -> &str {
        "mutate"
    }

    fn description(&self) -> &str {
        "mutating probe"
    }

    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }

    fn class(&self) -> ToolClass {
        ToolClass::Mutating
    }

    async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        MUTATION_EXECUTIONS.fetch_add(1, Ordering::SeqCst);
        Ok(ToolOutput::text("mutated"))
    }
}
