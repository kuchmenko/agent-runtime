//! Caller-facing handle for steering a running agent.

use std::sync::Arc;

use crate::steering::{
    AgentHandleInner, InterruptError, InterruptOutcome, InterruptTarget, IntoQueueContent,
    SteerCommand, SteerError, TurnId,
};

/// Cheap cloneable control handle for a running agent.
#[derive(Clone, Debug)]
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
            if expected != active {
                return Err(SteerError::ExpectedTurnMismatch {
                    expected,
                    actual: active,
                });
            }
        }

        self.inner
            .steer_tx
            .send(SteerCommand::Append { content })
            .map_err(|_| SteerError::ChannelClosed)?;
        Ok(active)
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
                if active.as_ref() != Some(&turn_id) {
                    return Err(InterruptError::UnknownTurn(turn_id));
                }
                self.inner.cancel.cancel();
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
            .clone()
    }
}
