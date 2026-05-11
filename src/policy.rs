//! Tool-policy implementations and safe composition helpers.
//!
//! Scoping is deny-monotonic: a child scope can only remove capability
//! from its parent, never re-add capability the parent denied. This rules
//! out last-match-wins permission lists, where a later allow can silently
//! override an earlier deny under composition.
//!
//! Anti-pattern callout: opencode's `permission/evaluate.ts:11` uses
//! `rules.findLast(...)` — ergonomic for users, but later rules
//! silently override earlier denies, breaking deny-monotonicity under
//! composition. tkach chose [`IntersectPolicy`] (safety) over
//! last-match-wins (ergonomics) deliberately.

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

    /// Test-only policy that denies tools by name. Mirrors the
    /// `DenyNamed`-style policy referenced in issue #40 Phase 1
    /// acceptance criterion for `IntersectPolicy` composition.
    struct DenyNamed(HashSet<String>);

    impl DenyNamed {
        fn new<I, S>(names: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            Self(names.into_iter().map(Into::into).collect())
        }
    }

    impl ToolPolicy for DenyNamed {
        fn is_allowed(&self, name: &str) -> bool {
            !self.0.contains(name)
        }
    }

    #[test]
    fn allow_list_only_allows_named_tools() {
        let policy = AllowList::new(["read", "grep"]);
        assert!(policy.is_allowed("read"));
        assert!(!policy.is_allowed("write"));
    }

    #[test]
    fn allow_list_empty_denies_everything() {
        let policy = AllowList::new(std::iter::empty::<String>());
        assert!(!policy.is_allowed("read"));
        assert!(!policy.is_allowed(""));
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

    #[test]
    fn intersect_with_allow_all_reduces_to_other_side() {
        // AllowAll ∩ AllowList(set) = AllowList(set). Useful when the
        // parent had no policy installed (defaults to AllowAll) and the
        // child specifies a tools_allow.
        let parent: Arc<dyn ToolPolicy> = Arc::new(AllowAll);
        let child: Arc<dyn ToolPolicy> = Arc::new(AllowList::new(["read"]));
        let policy = IntersectPolicy {
            left: parent,
            right: child,
        };
        assert!(policy.is_allowed("read"));
        assert!(!policy.is_allowed("write"));
        assert!(!policy.is_allowed("anything-else"));
    }

    #[test]
    fn intersect_with_deny_named_subtracts_from_allow_list() {
        // AllowList(["read","write","bash"]) ∩ DenyNamed(["bash"]) =
        // AllowList(["read","write"]). Verifies deny-monotonicity holds
        // when one side is an explicit deny shape.
        let parent: Arc<dyn ToolPolicy> = Arc::new(AllowList::new(["read", "write", "bash"]));
        let child: Arc<dyn ToolPolicy> = Arc::new(DenyNamed::new(["bash"]));
        let policy = IntersectPolicy {
            left: parent,
            right: child,
        };
        assert!(policy.is_allowed("read"));
        assert!(policy.is_allowed("write"));
        assert!(!policy.is_allowed("bash"));
        assert!(!policy.is_allowed("grep"));
    }
}
