//! Tool-policy implementations and safe composition helpers.
//!
//! Scoping is deny-monotonic: a child scope can only remove capability
//! from its parent, never re-add capability the parent denied. This rules
//! out last-match-wins permission lists, where a later allow can silently
//! override an earlier deny under composition.

use std::collections::HashSet;
use std::sync::Arc;

use crate::executor::ToolPolicy;

/// Default policy: every tool is allowed.
pub use crate::executor::AllowAll;

/// Allow only tools whose name appears in the set.
pub struct AllowList(HashSet<String>);

impl AllowList {
    pub fn new<I, S>(tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(tools.into_iter().map(Into::into).collect())
    }
}

impl ToolPolicy for AllowList {
    fn is_allowed(&self, name: &str) -> bool {
        self.0.contains(name)
    }
}

/// Compose two policies by intersection: both must allow the tool.
pub struct IntersectPolicy {
    pub left: Arc<dyn ToolPolicy>,
    pub right: Arc<dyn ToolPolicy>,
}

impl ToolPolicy for IntersectPolicy {
    fn is_allowed(&self, name: &str) -> bool {
        self.left.is_allowed(name) && self.right.is_allowed(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_list_only_allows_named_tools() {
        let policy = AllowList::new(["read", "grep"]);
        assert!(policy.is_allowed("read"));
        assert!(!policy.is_allowed("write"));
    }

    #[test]
    fn intersect_policy_is_deny_monotonic() {
        let parent: Arc<dyn ToolPolicy> = Arc::new(AllowList::new(["read", "write"]));
        let child: Arc<dyn ToolPolicy> = Arc::new(AllowList::new(["read", "grep"]));
        let policy = IntersectPolicy {
            left: parent,
            right: child,
        };
        assert!(policy.is_allowed("read"));
        assert!(!policy.is_allowed("write"));
        assert!(!policy.is_allowed("grep"));
    }
}
