//! Runtime prompt policies installed through [`crate::AgentHandle`].
//!
//! A prompt policy is a scoped system-prompt addendum. Unlike agent
//! modes, it does not change tool-dispatch authority; it changes the
//! instructions visible to the model when the next provider request is
//! built.

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;

use crate::guard::AgentSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolicyId(pub(crate) u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyScope {
    /// Apply once to the next provider request whose trigger matches.
    NextTurn,
    /// Apply to every matching provider request until removed.
    EveryTurnUntilRemoved,
    /// Apply for the lifetime of this [`crate::AgentHandle`].
    Persistent,
}

pub trait IntentMatcher: Send + Sync {
    fn matches(&self, snapshot: &AgentSnapshot) -> bool;
}

impl<F> IntentMatcher for F
where
    F: Fn(&AgentSnapshot) -> bool + Send + Sync,
{
    fn matches(&self, snapshot: &AgentSnapshot) -> bool {
        self(snapshot)
    }
}

pub enum PolicyTrigger {
    Always,
    OnIntentMatch(Box<dyn IntentMatcher>),
}

pub struct PromptPolicy {
    pub name: String,
    pub scope: PolicyScope,
    pub content: String,
    /// Lower numbers are applied first.
    pub precedence: u8,
    pub trigger: PolicyTrigger,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyMetadata {
    pub name: String,
    pub scope: PolicyScope,
    pub precedence: u8,
    pub trigger: PolicyTriggerMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyTriggerMetadata {
    Always,
    IntentMatcher,
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("prompt policy name is empty")]
    EmptyName,
    #[error("prompt policy content is empty")]
    EmptyContent,
    #[error("prompt policy precedence {precedence} is already used by policy {existing}")]
    DuplicatePrecedence { precedence: u8, existing: String },
    #[error("prompt policy not found")]
    NotFound,
}

struct PolicyEntry {
    id: PolicyId,
    policy: PromptPolicy,
}

#[derive(Default)]
pub(crate) struct PromptPolicySet {
    next_id: u64,
    policies: VecDeque<PolicyEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppliedPromptPolicy {
    pub id: PolicyId,
    pub name: String,
    pub content: String,
    pub precedence: u8,
}

impl PromptPolicySet {
    pub(crate) fn install(&mut self, policy: PromptPolicy) -> Result<PolicyId, PolicyError> {
        if policy.name.trim().is_empty() {
            return Err(PolicyError::EmptyName);
        }
        if policy.content.trim().is_empty() {
            return Err(PolicyError::EmptyContent);
        }
        if let Some(existing) = self
            .policies
            .iter()
            .find(|entry| entry.policy.precedence == policy.precedence)
        {
            return Err(PolicyError::DuplicatePrecedence {
                precedence: policy.precedence,
                existing: existing.policy.name.clone(),
            });
        }

        let id = PolicyId(self.next_id);
        self.next_id += 1;
        self.policies.push_back(PolicyEntry { id, policy });
        Ok(id)
    }

    pub(crate) fn remove(&mut self, id: PolicyId) -> Result<(), PolicyError> {
        let Some(idx) = self.policies.iter().position(|entry| entry.id == id) else {
            return Err(PolicyError::NotFound);
        };
        self.policies.remove(idx);
        Ok(())
    }

    pub(crate) fn list(&self) -> Vec<(PolicyId, PolicyMetadata)> {
        let mut policies: Vec<_> = self
            .policies
            .iter()
            .map(|entry| (entry.id, metadata(&entry.policy)))
            .collect();
        policies.sort_by_key(|(_, metadata)| metadata.precedence);
        policies
    }

    pub(crate) fn apply(&mut self, snapshot: &AgentSnapshot) -> Vec<AppliedPromptPolicy> {
        let mut applied = Vec::new();
        let mut idx = 0;
        while idx < self.policies.len() {
            let is_match = std::panic::catch_unwind(AssertUnwindSafe(|| {
                matches_trigger(&self.policies[idx].policy.trigger, snapshot)
            }))
            .unwrap_or(false);

            if is_match {
                let entry = &self.policies[idx];
                applied.push(AppliedPromptPolicy {
                    id: entry.id,
                    name: entry.policy.name.clone(),
                    content: entry.policy.content.clone(),
                    precedence: entry.policy.precedence,
                });
            }

            if is_match && self.policies[idx].policy.scope == PolicyScope::NextTurn {
                self.policies.remove(idx);
            } else {
                idx += 1;
            }
        }
        applied.sort_by_key(|policy| policy.precedence);
        applied
    }
}

fn metadata(policy: &PromptPolicy) -> PolicyMetadata {
    PolicyMetadata {
        name: policy.name.clone(),
        scope: policy.scope,
        precedence: policy.precedence,
        trigger: match policy.trigger {
            PolicyTrigger::Always => PolicyTriggerMetadata::Always,
            PolicyTrigger::OnIntentMatch(_) => PolicyTriggerMetadata::IntentMatcher,
        },
    }
}

fn matches_trigger(trigger: &PolicyTrigger, snapshot: &AgentSnapshot) -> bool {
    match trigger {
        PolicyTrigger::Always => true,
        PolicyTrigger::OnIntentMatch(matcher) => matcher.matches(snapshot),
    }
}
