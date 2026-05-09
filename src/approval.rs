//! User-approval gate for tool calls.
//!
//! Where [`ToolPolicy`](crate::ToolPolicy) is the *static* gate (which
//! tools are permitted at all), [`ApprovalHandler`] is the *dynamic*,
//! per-call gate. It runs after policy and before the tool actually
//! executes; the model never sees the call directly — only the
//! resulting tool_result, success or denial.
//!
//! ## Decision shape
//!
//! [`ApprovalDecision::Deny`] is a *recoverable* outcome: it surfaces
//! to the model as `is_error: true` `tool_result` so the LLM can pick
//! a different approach. It is **not** an `AgentError` — the loop
//! continues. This is consistent with how policy denials and missing
//! tools are handled.
//!
//! ## How consumers use it
//!
//! ```ignore
//! struct MyTuiApproval { tx: mpsc::Sender<PendingDecision> }
//!
//! #[async_trait]
//! impl ApprovalHandler for MyTuiApproval {
//!     async fn approve(&self, name: &str, input: &Value, class: ToolClass)
//!         -> ApprovalDecision
//!     {
//!         // Cheap path: blanket-allow read-only tools without a prompt.
//!         if class == ToolClass::ReadOnly { return ApprovalDecision::Allow; }
//!         // Otherwise hand off to the UI thread and await its answer.
//!         let (tx, rx) = oneshot::channel();
//!         self.tx.send(PendingDecision { name, input, reply: tx }).await.ok();
//!         rx.await.unwrap_or(ApprovalDecision::Deny("UI gone".into()))
//!     }
//! }
//! ```
//!
//! ## Cancellation
//!
//! The executor wraps `approve()` in `tokio::select!` against
//! `ctx.cancel.cancelled()`. If the caller cancels while the handler
//! is awaiting a user click, the executor short-circuits with a
//! cancelled tool_result rather than waiting forever.
//!
//! ## Concurrent invocation
//!
//! When the consumer opts a tool into the concurrent-mutator pool via
//! [`crate::AgentBuilder::tool_concurrency`], a single LLM batch may
//! produce multiple promoted-mutator calls executing in parallel.
//! Each task invokes `approve()` independently, so an
//! [`ApprovalHandler`] implementation may be called from several
//! tasks at once. The trait already requires `Send + Sync`, so type
//! safety is unconditional. UI-bound implementations (TUI modal, web
//! confirmation dialog, etc.) should serialise their UI access
//! internally — typically by funnelling all incoming
//! `PendingDecision`s through a single `mpsc::channel` into the UI
//! thread, which the example pattern below already shows. Stateless
//! handlers (such as [`AutoApprove`] or any pure-policy decision
//! function) are concurrent-safe out of the box.

use async_trait::async_trait;
use serde_json::Value;

use crate::tool::ToolClass;

/// What an [`ApprovalHandler`] returns for a given tool call.
///
/// `Deny(reason)` is recoverable — the runtime emits an
/// `is_error: true` tool_result containing the reason and lets the
/// model adapt. It is **not** an `AgentError`.
#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    Allow,
    Deny(String),
}

/// Gate that decides per-call whether a tool may run.
///
/// Implementations are typically UI-bound (TUI modal, web confirmation
/// dialog, Slack interactive button, etc.) and may block until the
/// human responds. The runtime races the wait against the agent's
/// cancellation token — the handler does not need to plumb cancel
/// itself, but it *should* take care to avoid leaking resources if
/// dropped mid-await.
///
/// The `class` parameter is provided so handlers can fast-path
/// read-only tools without bothering the user.
///
/// Concurrent invocation: when the consumer enables concurrent
/// mutators via [`crate::AgentBuilder::tool_concurrency`], `approve`
/// may be called from several executor tasks at once. UI-bound
/// implementations should serialise their UI access internally; see
/// the module-level "Concurrent invocation" section for the
/// recommended `mpsc`-into-the-UI-thread pattern.
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    async fn approve(&self, tool_name: &str, input: &Value, class: ToolClass) -> ApprovalDecision;
}

/// Default handler: allow every tool unconditionally.
///
/// Used when `AgentBuilder::approval(...)` was not called — preserves
/// pre-approval-flow behaviour exactly. Headless / batch / CI agents
/// typically want this; interactive consumers replace it with their
/// own implementation.
pub struct AutoApprove;

#[async_trait]
impl ApprovalHandler for AutoApprove {
    async fn approve(
        &self,
        _tool_name: &str,
        _input: &Value,
        _class: ToolClass,
    ) -> ApprovalDecision {
        ApprovalDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn auto_approve_allows_everything() {
        let h = AutoApprove;
        let d = h
            .approve("bash", &json!({"command": "rm -rf /"}), ToolClass::Mutating)
            .await;
        assert!(matches!(d, ApprovalDecision::Allow));
    }

    struct DenyAll(&'static str);
    #[async_trait]
    impl ApprovalHandler for DenyAll {
        async fn approve(&self, _: &str, _: &Value, _: ToolClass) -> ApprovalDecision {
            ApprovalDecision::Deny(self.0.to_string())
        }
    }

    #[tokio::test]
    async fn custom_handler_denies_with_reason() {
        let h = DenyAll("test policy");
        let d = h.approve("read", &json!({}), ToolClass::ReadOnly).await;
        let ApprovalDecision::Deny(reason) = d else {
            panic!("expected Deny");
        };
        assert_eq!(reason, "test policy");
    }
}
