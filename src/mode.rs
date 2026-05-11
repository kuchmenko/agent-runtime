//! Runtime mode gates for steering-aware agents.

use std::borrow::Cow;
use std::fmt::Debug;

use crate::tool::ToolClass;

pub trait AgentMode: Send + Sync + Debug + 'static {
    fn name(&self) -> Cow<'static, str>;

    fn allows_mutating_tools(&self) -> bool {
        true
    }

    fn tool_gate(&self, _tool_name: &str, _class: ToolClass) -> Option<ModeDecision> {
        None
    }

    fn system_prompt_addendum(&self) -> Cow<'static, str> {
        "".into()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModeDecision {
    Allow,
    Deny { reason: Cow<'static, str> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeAuthority {
    Operator,
    Agent,
}

#[derive(Debug, thiserror::Error)]
pub enum ModeError {
    #[error("no pending mode change")]
    NoPendingModeChange,
}

#[derive(Debug)]
pub struct DefaultMode;

impl AgentMode for DefaultMode {
    fn name(&self) -> Cow<'static, str> {
        "default".into()
    }
}

#[derive(Debug)]
pub struct PlanMode;

impl AgentMode for PlanMode {
    fn name(&self) -> Cow<'static, str> {
        "plan".into()
    }

    fn allows_mutating_tools(&self) -> bool {
        false
    }

    fn system_prompt_addendum(&self) -> Cow<'static, str> {
        "You are in Plan mode. Outline the approach; do not edit, write, or run mutating commands."
            .into()
    }
}

#[derive(Debug)]
pub struct AcceptEditsMode;

impl AgentMode for AcceptEditsMode {
    fn name(&self) -> Cow<'static, str> {
        "accept-edits".into()
    }
}
