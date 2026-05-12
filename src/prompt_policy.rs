//! Runtime prompt policies installed through [`crate::AgentHandle`].
//!
//! A prompt policy is a scoped system-prompt addendum. Unlike agent
//! modes, it does not change tool-dispatch authority; it changes the
//! instructions visible to the model when the next provider request is
//! built.

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use crate::guard::AgentSnapshot;

/// Opaque identifier for a runtime prompt policy installed on an [`crate::AgentHandle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolicyId(pub(crate) u64);

/// Lifetime of an installed prompt policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyScope {
    /// Apply once to the next provider request whose trigger matches.
    NextTurn,
    /// Apply to every matching provider request until removed.
    EveryTurnUntilRemoved,
    /// Apply for the lifetime of this [`crate::AgentHandle`], unless removed explicitly.
    Persistent,
}

/// Predicate used by [`PolicyTrigger::OnIntentMatch`] to decide whether a policy applies.
///
/// The matcher receives the same lightweight [`AgentSnapshot`] shape used by continuation
/// guards. It must be fast and side-effect-light: it runs while the agent is assembling the
/// next provider request.
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

/// Condition that decides whether an installed prompt policy is appended to a provider request.
pub enum PolicyTrigger {
    /// Apply whenever the policy scope allows it.
    Always,
    /// Apply when the matcher returns true for the current agent snapshot.
    OnIntentMatch(Box<dyn IntentMatcher>),
}

/// Runtime system-prompt addendum installed through [`crate::AgentHandle::install_prompt_policy`].
pub struct PromptPolicy {
    /// Human-readable name included in the traceability marker added to the system prompt.
    pub name: String,
    pub scope: PolicyScope,
    pub content: String,
    /// Lower numbers are applied first.
    pub precedence: u8,
    pub trigger: PolicyTrigger,
}

/// Public metadata returned by [`crate::AgentHandle::list_prompt_policies`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyMetadata {
    pub name: String,
    pub scope: PolicyScope,
    pub precedence: u8,
    pub trigger: PolicyTriggerMetadata,
}

/// Serializable-ish trigger shape for policy listing; matchers are opaque user code.
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

#[derive(Clone)]
enum StoredTrigger {
    Always,
    OnIntentMatch(Arc<dyn IntentMatcher>),
}

struct PolicyEntry {
    id: PolicyId,
    name: String,
    scope: PolicyScope,
    content: String,
    precedence: u8,
    trigger: StoredTrigger,
}

#[derive(Default)]
pub(crate) struct PromptPolicySet {
    next_id: u64,
    policies: VecDeque<PolicyEntry>,
}

#[derive(Clone)]
pub(crate) struct PromptPolicyCandidate {
    id: PolicyId,
    trigger: StoredTrigger,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppliedPromptPolicy {
    pub id: PolicyId,
    pub name: String,
    pub content: String,
    pub precedence: u8,
}

impl PromptPolicyCandidate {
    pub(crate) fn id(&self) -> PolicyId {
        self.id
    }

    pub(crate) fn matches(&self, snapshot: &AgentSnapshot) -> bool {
        std::panic::catch_unwind(AssertUnwindSafe(|| match &self.trigger {
            StoredTrigger::Always => true,
            StoredTrigger::OnIntentMatch(matcher) => matcher.matches(snapshot),
        }))
        .unwrap_or(false)
    }
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
            .find(|entry| entry.precedence == policy.precedence)
        {
            return Err(PolicyError::DuplicatePrecedence {
                precedence: policy.precedence,
                existing: existing.name.clone(),
            });
        }

        let trigger = match policy.trigger {
            PolicyTrigger::Always => StoredTrigger::Always,
            PolicyTrigger::OnIntentMatch(matcher) => {
                StoredTrigger::OnIntentMatch(Arc::from(matcher))
            }
        };
        let id = PolicyId(self.next_id);
        self.next_id += 1;
        self.policies.push_back(PolicyEntry {
            id,
            name: policy.name,
            scope: policy.scope,
            content: policy.content,
            precedence: policy.precedence,
            trigger,
        });
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
            .map(|entry| (entry.id, metadata(entry)))
            .collect();
        policies.sort_by_key(|(_, metadata)| metadata.precedence);
        policies
    }

    pub(crate) fn candidates(&self) -> Vec<PromptPolicyCandidate> {
        self.policies
            .iter()
            .map(|entry| PromptPolicyCandidate {
                id: entry.id,
                trigger: entry.trigger.clone(),
            })
            .collect()
    }

    pub(crate) fn apply_matches(&mut self, matched_ids: &[PolicyId]) -> Vec<AppliedPromptPolicy> {
        let mut applied = Vec::new();
        let mut idx = 0;
        while idx < self.policies.len() {
            let is_match = matched_ids.contains(&self.policies[idx].id);
            if is_match {
                let entry = &self.policies[idx];
                applied.push(AppliedPromptPolicy {
                    id: entry.id,
                    name: entry.name.clone(),
                    content: entry.content.clone(),
                    precedence: entry.precedence,
                });
            }

            if is_match && self.policies[idx].scope == PolicyScope::NextTurn {
                self.policies.remove(idx);
            } else {
                idx += 1;
            }
        }
        applied.sort_by_key(|policy| policy.precedence);
        applied
    }
}

fn metadata(policy: &PolicyEntry) -> PolicyMetadata {
    PolicyMetadata {
        name: policy.name.clone(),
        scope: policy.scope,
        precedence: policy.precedence,
        trigger: match policy.trigger {
            StoredTrigger::Always => PolicyTriggerMetadata::Always,
            StoredTrigger::OnIntentMatch(_) => PolicyTriggerMetadata::IntentMatcher,
        },
    }
}
