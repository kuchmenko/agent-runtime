use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::error::ToolError;
use crate::executor::ToolExecutor;

/// Context passed to every tool execution.
///
/// Intentionally slim — holds only primitives the runtime actually owns
/// and wants to share with tools:
///
/// - `working_dir`: file-system base for path resolution.
/// - `cancel`: cooperative cancellation. Long-running tools should
///   `tokio::select!` on `cancel.cancelled()` and return
///   [`ToolError::Cancelled`] promptly.
/// - `depth` / `max_depth`: current nesting level of the agent; used by
///   `SubAgent` to prevent unbounded recursion.
/// - `executor`: the parent agent's [`ToolExecutor`], letting tools that
///   spawn nested work (e.g. `SubAgent`) inherit the full toolset
///   automatically — no explicit layering required.
pub struct ToolContext {
    pub working_dir: PathBuf,
    pub cancel: CancellationToken,
    pub depth: usize,
    pub max_depth: usize,
    pub executor: Arc<ToolExecutor>,
}

/// Result of a tool execution.
pub enum ToolOutput {
    Text(String),
    Error(String),
}

impl ToolOutput {
    pub fn text(s: impl Into<String>) -> Self {
        ToolOutput::Text(s.into())
    }

    pub fn error(s: impl Into<String>) -> Self {
        ToolOutput::Error(s.into())
    }

    pub fn is_error(&self) -> bool {
        matches!(self, ToolOutput::Error(_))
    }

    pub fn content(&self) -> &str {
        match self {
            ToolOutput::Text(s) | ToolOutput::Error(s) => s,
        }
    }
}

/// Side-effect class of a tool.
///
/// Used by the executor to safely parallelise consecutive read-only calls
/// in a single batch while keeping mutating calls sequential. Mutating is
/// the default because misclassifying a side-effectful tool as `ReadOnly`
/// can lead to subtle ordering bugs (two "mutating" writes racing against
/// each other); the reverse is merely a missed optimisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolClass {
    /// No observable side effects. Safe to run concurrently with other
    /// `ReadOnly` tools. Examples: `Read`, `Glob`, `Grep`, `WebFetch`.
    ReadOnly,
    /// Changes state — file system, external services, processes, or
    /// nested agents. Must run sequentially to preserve ordering.
    /// Examples: `Write`, `Edit`, `Bash`, `SubAgent`.
    Mutating,
}

/// Trait for tools that the agent can use.
///
/// Implement this trait to create custom tools.
///
/// # Example
///
/// ```ignore
/// use tkach::{Tool, ToolContext, ToolOutput, ToolError};
/// use serde_json::{json, Value};
///
/// struct MyTool;
///
/// #[async_trait::async_trait]
/// impl Tool for MyTool {
///     fn name(&self) -> &str { "my_tool" }
///     fn description(&self) -> &str { "Does something useful" }
///     fn input_schema(&self) -> Value {
///         json!({
///             "type": "object",
///             "properties": {
///                 "query": { "type": "string" }
///             },
///             "required": ["query"]
///         })
///     }
///     async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
///         let query = input["query"].as_str().unwrap_or_default();
///         Ok(ToolOutput::text(format!("Result for: {query}")))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique name of the tool (used by the LLM to invoke it).
    fn name(&self) -> &str;

    /// Human-readable description of what the tool does.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Side-effect class. Defaults to `Mutating` — the safe choice for
    /// tools that are not explicitly marked read-only. Override to return
    /// [`ToolClass::ReadOnly`] only when you are certain the tool has no
    /// observable side effects.
    fn class(&self) -> ToolClass {
        ToolClass::Mutating
    }

    /// Whether this tool itself drives nested executor work — i.e.
    /// the tool's `execute` body holds the executor task open while
    /// running another `Agent::run` underneath. The canonical example
    /// is `SubAgent`.
    ///
    /// The executor uses this to admit recursive tools through the
    /// concurrent-mutator pool *regardless of explicit promotion*,
    /// because non-recursive admission classes (the width-1
    /// `serial_mut` pool, in particular) are shared across the
    /// agent tree and would deadlock when a parent's permit is
    /// pinned during the child's nested `execute`. Routing recursive
    /// tools through `concurrent_mut` — which the executor forks
    /// per nesting level — keeps nested fan-out free of pool
    /// contention while still bounding it by the
    /// `max_concurrent_mutations` cap of the *current* level.
    ///
    /// Defaults to `false`. Override to `true` only for tools whose
    /// `execute` body invokes `ctx.executor` recursively (directly
    /// or via an `Agent` it constructs from
    /// `ctx.executor.fork_for_subagent()`).
    fn is_recursive(&self) -> bool {
        false
    }

    /// Execute the tool with the given input.
    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>;
}
