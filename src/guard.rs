//! Continuation guards for predicate-driven agent loops.

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;

use crate::message::Message;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GuardId(pub(crate) u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardTrigger {
    OnTurnEnd,
    OnSessionStop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardDecision {
    Continue,
    Stop,
    Abort { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardEscape {
    OperatorCommand(String),
    MaxIterations,
    PredicateAbort,
}

pub struct ContinuationGuard {
    pub name: String,
    pub trigger: GuardTrigger,
    pub predicate: Box<dyn Fn(&AgentSnapshot) -> GuardDecision + Send + Sync>,
    pub continuation_prompt: String,
    pub max_iterations: Option<u32>,
    pub escape: GuardEscape,
}

#[derive(Debug, Clone)]
pub struct AgentSnapshot {
    pub turn_count: usize,
    pub last_assistant_message: Option<Message>,
    pub recent_tool_calls: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum GuardError {
    #[error("guard must have max_iterations")]
    NoEscapeMechanism,
    #[error("guard not found")]
    NotFound,
}

pub(crate) struct GuardEntry {
    pub id: GuardId,
    pub guard: ContinuationGuard,
    pub iterations: u32,
}

#[derive(Default)]
pub(crate) struct GuardSet {
    next_id: u64,
    guards: VecDeque<GuardEntry>,
}

pub(crate) enum GuardEval {
    Continue {
        guard_name: String,
        prompt: String,
        iteration: u32,
    },
    Abort {
        guard_name: String,
        reason: String,
    },
    Stop,
}

impl GuardSet {
    pub(crate) fn install(&mut self, guard: ContinuationGuard) -> Result<GuardId, GuardError> {
        if guard.max_iterations.is_none() {
            return Err(GuardError::NoEscapeMechanism);
        }
        let id = GuardId(self.next_id);
        self.next_id += 1;
        self.guards.push_back(GuardEntry {
            id,
            guard,
            iterations: 0,
        });
        Ok(id)
    }

    pub(crate) fn remove(&mut self, id: GuardId) -> Result<(), GuardError> {
        let Some(idx) = self.guards.iter().position(|entry| entry.id == id) else {
            return Err(GuardError::NotFound);
        };
        self.guards.remove(idx);
        Ok(())
    }

    pub(crate) fn evaluate(
        &mut self,
        trigger: GuardTrigger,
        snapshot: &AgentSnapshot,
    ) -> GuardEval {
        let mut idx = 0;
        while idx < self.guards.len() {
            if self.guards[idx].guard.trigger != trigger {
                idx += 1;
                continue;
            }
            let decision = std::panic::catch_unwind(AssertUnwindSafe(|| {
                (self.guards[idx].guard.predicate)(snapshot)
            }));
            let decision = match decision {
                Ok(decision) => decision,
                Err(_) => {
                    let entry = self.guards.remove(idx).expect("guard index exists");
                    return GuardEval::Abort {
                        guard_name: entry.guard.name,
                        reason: "predicate panic".into(),
                    };
                }
            };
            match decision {
                GuardDecision::Stop => idx += 1,
                GuardDecision::Abort { reason } => {
                    let entry = self.guards.remove(idx).expect("guard index exists");
                    return GuardEval::Abort {
                        guard_name: entry.guard.name,
                        reason,
                    };
                }
                GuardDecision::Continue => {
                    let entry = &mut self.guards[idx];
                    entry.iterations += 1;
                    if let Some(max) = entry.guard.max_iterations {
                        if entry.iterations > max {
                            let entry = self.guards.remove(idx).expect("guard index exists");
                            return GuardEval::Abort {
                                guard_name: entry.guard.name,
                                reason: "max iterations reached".into(),
                            };
                        }
                    }
                    return GuardEval::Continue {
                        guard_name: entry.guard.name.clone(),
                        prompt: entry.guard.continuation_prompt.clone(),
                        iteration: entry.iterations,
                    };
                }
            }
        }
        GuardEval::Stop
    }
}
