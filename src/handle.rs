//! Caller-facing handle for steering a running agent.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use crate::guard::{ContinuationGuard, GuardError, GuardId};
use crate::mode::{AgentMode, ModeAuthority, ModeError};
use crate::steering::{
    AgentHandleInner, InterruptError, InterruptOutcome, InterruptTarget, IntoQueueContent,
    PendingModeChange, SteerCommand, SteerError, TurnId,
};
use crate::user_input::{self, AskUserError, QuestionSet, UserInputResponse};

/// Cheap cloneable control handle for a running agent.
#[derive(Clone)]
pub struct AgentHandle {
    pub(crate) inner: Arc<AgentHandleInner>,
}

impl AgentHandle {
    /// Queue user content for delivery at the next safe provider-call boundary.
    pub fn queue_user_message(
        &self,
        content: impl IntoQueueContent,
        expected_turn_id: Option<TurnId>,
    ) -> Result<TurnId, SteerError> {
        let content = content.into_queue_content();
        if content.is_empty() {
            return Err(SteerError::EmptyContent);
        }

        let active = self
            .inner
            .active_turn
            .read()
            .expect("agent handle turn lock poisoned")
            .clone()
            .ok_or(SteerError::NoActiveTurn)?;

        if let Some(expected) = expected_turn_id {
            if expected != active.id {
                return Err(SteerError::ExpectedTurnMismatch {
                    expected,
                    actual: active.id,
                });
            }
        }

        let turn_id = active.id;
        self.inner
            .steer_tx
            .send(SteerCommand::Append {
                turn_id: turn_id.clone(),
                content,
            })
            .map_err(|_| SteerError::ChannelClosed)?;
        Ok(turn_id)
    }

    /// Interrupt a tool, turn, or the whole session.
    pub fn interrupt(&self, target: InterruptTarget) -> Result<InterruptOutcome, InterruptError> {
        match target {
            InterruptTarget::Tool { tool_call_id } => {
                Ok(self.inner.tool_runs.interrupt(&tool_call_id))
            }
            InterruptTarget::Turn { turn_id, .. } => {
                let active = self
                    .inner
                    .active_turn
                    .read()
                    .expect("agent handle turn lock poisoned")
                    .clone();
                let Some(active) = active else {
                    return Ok(InterruptOutcome::NotInFlight);
                };
                if active.id != turn_id {
                    return Err(InterruptError::UnknownTurn(turn_id));
                }
                active.cancel.cancel();
                Ok(InterruptOutcome::Cancelled)
            }
            InterruptTarget::Session => {
                self.inner.cancel.cancel();
                Ok(InterruptOutcome::Cancelled)
            }
        }
    }

    /// Current active turn, if the agent is inside a turn.
    pub fn current_turn_id(&self) -> Option<TurnId> {
        self.inner
            .active_turn
            .read()
            .expect("agent handle turn lock poisoned")
            .as_ref()
            .map(|turn| turn.id.clone())
    }

    pub fn set_mode(
        &self,
        mode: Box<dyn AgentMode>,
        authority: ModeAuthority,
    ) -> Result<(), ModeError> {
        let mode: Arc<dyn AgentMode> = Arc::from(mode);
        match authority {
            ModeAuthority::Operator => {
                *self.inner.mode.write().expect("agent mode lock poisoned") = mode;
                *self
                    .inner
                    .pending_mode
                    .write()
                    .expect("pending mode lock poisoned") = None;
            }
            ModeAuthority::Agent => {
                *self
                    .inner
                    .pending_mode
                    .write()
                    .expect("pending mode lock poisoned") =
                    Some(PendingModeChange { mode, authority });
            }
        }
        Ok(())
    }

    pub fn current_mode(&self) -> Cow<'static, str> {
        self.inner
            .mode
            .read()
            .expect("agent mode lock poisoned")
            .name()
    }

    pub fn cancel_pending_mode_change(&self) -> Result<(), ModeError> {
        let mut pending = self
            .inner
            .pending_mode
            .write()
            .expect("pending mode lock poisoned");
        if pending.take().is_some() {
            Ok(())
        } else {
            Err(ModeError::NoPendingModeChange)
        }
    }

    pub async fn ask_user(
        &self,
        questions: QuestionSet,
        timeout: Duration,
    ) -> Result<UserInputResponse, AskUserError> {
        if !self.inner.is_root_thread {
            return Err(AskUserError::NotRootThread);
        }
        let Some(bridge) = &self.inner.user_input_bridge else {
            return Err(AskUserError::NoUserInputBridge);
        };
        Ok(user_input::collect_with_timeout(bridge.as_ref(), &questions, timeout).await?)
    }

    pub fn install_continuation_guard(
        &self,
        guard: ContinuationGuard,
    ) -> Result<GuardId, GuardError> {
        self.inner
            .guards
            .write()
            .expect("continuation guard lock poisoned")
            .install(guard)
    }

    pub fn remove_continuation_guard(&self, id: GuardId) -> Result<(), GuardError> {
        self.inner
            .guards
            .write()
            .expect("continuation guard lock poisoned")
            .remove(id)
    }
}
