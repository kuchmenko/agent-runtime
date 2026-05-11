//! Tool dispatch: registry, policy, executor, and concurrency configuration.
//!
//! The executor is the single entry point through which the agent loop
//! invokes tools. It handles three failure modes uniformly — returning
//! a `tool_result` `Content` with `is_error: true` so the LLM can observe
//! and adapt, rather than terminating the loop:
//!
//! 1. **Policy denial** — `ToolPolicy::is_allowed` returned false.
//! 2. **Missing tool** — the LLM invoked a name that is not in the registry.
//! 3. **Tool error** — the tool itself returned `Err(ToolError)`.
//!
//! Separating dispatch from the agent loop lets sub-agents (and future
//! orchestration tools) share the same registry via [`ToolContext`].
//!
//! ## Concurrency model
//!
//! `execute_batch` partitions the LLM-issued batch into contiguous
//! *runs* of the same routing class — `ReadOnly`, `ConcurrentMut`,
//! `SerialMut`, or `ShortCircuit`. Each run dispatches concurrently
//! via [`futures::stream::FuturesUnordered`]; runs are sequentialised
//! against each other. The barrier between class boundaries preserves
//! the LLM's intended ordering across mixed batches — `[Read A,
//! Write A, Read A]` always observes the write before the trailing
//! read, because the executor cannot know which path each tool
//! touches and falls back on the LLM's emitted order.
//!
//! Admission control is layered: each call acquires permits from at
//! most two semaphores before invoking the tool — an optional
//! per-tool semaphore (cap configured by the consumer for that
//! specific tool name) and a mandatory class semaphore (read pool,
//! serial-mutator pool, or concurrent-mutator pool, chosen by the
//! tool's [`ToolClass`], `is_recursive` flag, and whether it was
//! promoted via [`crate::AgentBuilder::tool_concurrency`]).
//!
//! Default-mutating tools that haven't been promoted route to the
//! serial-mutator pool of width 1 — preserving the pre-concurrency
//! "strict serial" semantics as a special case of the new model.
//! Recursive tools (per [`crate::Tool::is_recursive`]) route to the
//! concurrent-mutator pool regardless of explicit promotion, because
//! their `execute` body parks the dispatcher task while running a
//! nested `Agent::run`; routing them through the per-level-forked
//! `concurrent_mut` pool prevents permit-held-during-nested-execute
//! deadlock without breaking the global serialisation contract.
//!
//! Sub-agents do **not** simply share the parent's executor. They
//! call [`ToolExecutor::fork_for_subagent`] to obtain a fresh
//! executor with the same registry / policy / approval but with the
//! deadlock-prone pools (`concurrent_mut`, per-tool) forked to fresh
//! `Arc<Semaphore>`s — the non-deadlock-prone pools (`serial_mut`,
//! `read`) stay shared so default-Mutating tools still serialise
//! globally across the agent tree. See
//! [`ConcurrencyConfig::fork`] for the exact policy.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::Value;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::warn;

use crate::approval::{ApprovalDecision, ApprovalHandler, AutoApprove};
use crate::message::Content;
use crate::mode::{AgentMode, ModeDecision};
use crate::steering::ToolRunTracker;
use crate::tool::{Tool, ToolClass, ToolContext};

/// A single tool invocation decoded from an LLM `tool_use` block.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// Name-keyed collection of tools. Construction is one-shot from a
/// `Vec<Arc<dyn Tool>>` — swap the whole registry if you need to
/// reconfigure.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Build a registry from a tool list. If two tools share a `name()`,
    /// the later registration wins (consistent with `HashMap::insert`)
    /// and a `tracing::warn!` records the collision so silent shadowing
    /// — e.g. a custom tool accidentally masking a built-in — is at
    /// least visible in logs.
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut map: HashMap<String, Arc<dyn Tool>> = HashMap::with_capacity(tools.len());
        for t in tools {
            let name = t.name().to_string();
            if map.insert(name.clone(), t).is_some() {
                warn!(
                    tool = %name,
                    "duplicate tool name in registry; later registration overrode earlier"
                );
            }
        }
        Self { tools: map }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.values()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Decides whether the agent may invoke a given tool.
///
/// The loop treats a denial as a non-fatal `is_error: true` tool_result,
/// so the LLM can observe the block and try an alternative path. This is
/// deliberately consistent with how "tool not found" is handled —
/// guardrails that fail loudly inside the conversation are easier to
/// reason about than ones that explode outward.
pub trait ToolPolicy: Send + Sync {
    fn is_allowed(&self, tool_name: &str) -> bool;
}

/// Default policy: every tool is allowed.
pub struct AllowAll;

impl ToolPolicy for AllowAll {
    fn is_allowed(&self, _tool_name: &str) -> bool {
        true
    }
}

/// Per-tool concurrency configuration.
///
/// Set via [`crate::AgentBuilder::tool_concurrency`] to opt a default-
/// `Mutating` tool into the concurrent-mutator pool. Default-mutating
/// tools without an explicit opt-in continue to serialise globally
/// through a width-1 semaphore (matching pre-concurrency-feature
/// behaviour); promoting a tool moves it into the wider concurrent-
/// mutator pool whose width is set by
/// [`crate::AgentBuilder::max_concurrent_mutations`].
///
/// Promotion is a *consumer's responsibility* contract: the framework
/// can no longer prevent racing calls to the same tool with conflicting
/// inputs (e.g. two writes to the same path). Promote only when the
/// LLM-emitted batch shape — and the tool's own resource semantics —
/// make racing safe.
#[derive(Debug, Clone, Default)]
pub struct ToolConcurrency {
    enabled: bool,
    per_tool_cap: Option<usize>,
}

impl ToolConcurrency {
    /// Promote this tool into the concurrent-mutator class. The class
    /// cap from [`crate::AgentBuilder::max_concurrent_mutations`] still
    /// applies. `ReadOnly` tools are already concurrent and don't need
    /// promotion — calling `on()` on a read-only tool's name is a
    /// no-op for class routing but still installs a per-tool cap if
    /// `.max(n)` was chained.
    pub fn on() -> Self {
        Self {
            enabled: true,
            per_tool_cap: None,
        }
    }

    /// Explicit "do not promote" — equivalent to omitting the call.
    /// Useful when iterating from configuration data where every entry
    /// must produce a `ToolConcurrency` value.
    pub fn off() -> Self {
        Self {
            enabled: false,
            per_tool_cap: None,
        }
    }

    /// Apply a per-tool cap on parallelism. Combines with the class cap;
    /// effective concurrent count for this tool is
    /// `min(per_tool_cap, class_cap_remaining)`.
    ///
    /// # Panics
    /// Panics if `n == 0` to fail loud rather than ship a hung executor.
    #[must_use]
    pub fn max(mut self, n: usize) -> Self {
        assert!(n > 0, "ToolConcurrency::max requires n > 0");
        self.per_tool_cap = Some(n);
        self
    }

    /// Whether this configuration promotes the tool into the concurrent-
    /// mutator pool.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The per-tool cap, if one was set via [`ToolConcurrency::max`].
    pub fn per_tool_cap(&self) -> Option<usize> {
        self.per_tool_cap
    }
}

/// Internal concurrency configuration owned by [`ToolExecutor`].
///
/// Three class semaphores plus a per-tool override map. Constructed by
/// [`AgentBuilder`](crate::AgentBuilder)'s build step from the
/// `max_concurrent_*` and `tool_concurrency` settings, or by calling
/// [`ConcurrencyConfig::new`] directly when bypassing the builder.
///
/// Caps are stored alongside their `Arc<Semaphore>`s so
/// [`ConcurrencyConfig::fork`] can rebuild fresh semaphores with the
/// same caps for nested sub-agent execution. Without forking, a parent
/// holding a `concurrent_mut` permit during a `SubAgent::execute` call
/// would compete with its own children for permits drawn from the same
/// pool — a deadlock when the parent batch saturates the cap. Each
/// nesting level therefore gets its own independent permit accounting
/// with the same numerical cap.
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    read_cap: usize,
    mut_cap: usize,
    per_tool_caps: HashMap<String, usize>,
    promoted: HashSet<String>,

    read: Arc<Semaphore>,
    serial_mut: Arc<Semaphore>,
    concurrent_mut: Arc<Semaphore>,
    per_tool: HashMap<String, Arc<Semaphore>>,
}

impl Default for ConcurrencyConfig {
    /// Default caps: 20 concurrent reads, 10 concurrent promoted
    /// mutators, 1 serial-mutator slot. No tools promoted, no per-tool
    /// overrides. Together with no consumer-side opt-in, this preserves
    /// pre-concurrency-feature behaviour for any `Agent` built without
    /// the new builder methods.
    fn default() -> Self {
        Self {
            read_cap: 20,
            mut_cap: 10,
            per_tool_caps: HashMap::new(),
            promoted: HashSet::new(),
            read: Arc::new(Semaphore::new(20)),
            serial_mut: Arc::new(Semaphore::new(1)),
            concurrent_mut: Arc::new(Semaphore::new(10)),
            per_tool: HashMap::new(),
        }
    }
}

impl ConcurrencyConfig {
    /// Build a configuration with the given class caps and per-tool
    /// settings.
    ///
    /// `tool_settings` is a flat sequence of `(name, ToolConcurrency)`
    /// pairs. Later entries with the same name override earlier ones.
    ///
    /// # Panics
    /// Panics if either cap is `0` — a 0-permit semaphore would block
    /// the entire pool forever.
    pub fn new(
        max_concurrent_reads: usize,
        max_concurrent_mutations: usize,
        tool_settings: impl IntoIterator<Item = (String, ToolConcurrency)>,
    ) -> Self {
        assert!(
            max_concurrent_reads > 0,
            "max_concurrent_reads requires n > 0"
        );
        assert!(
            max_concurrent_mutations > 0,
            "max_concurrent_mutations requires n > 0"
        );

        let mut per_tool_caps: HashMap<String, usize> = HashMap::new();
        let mut promoted: HashSet<String> = HashSet::new();
        for (name, cfg) in tool_settings {
            if cfg.is_enabled() {
                promoted.insert(name.clone());
            } else {
                // Explicit off() removes any prior promotion; useful when
                // the consumer iterates configuration data where the same
                // name might appear twice.
                promoted.remove(&name);
            }
            match cfg.per_tool_cap() {
                Some(cap) => {
                    per_tool_caps.insert(name, cap);
                }
                None => {
                    per_tool_caps.remove(&name);
                }
            }
        }

        let per_tool: HashMap<String, Arc<Semaphore>> = per_tool_caps
            .iter()
            .map(|(name, &cap)| (name.clone(), Arc::new(Semaphore::new(cap))))
            .collect();

        Self {
            read_cap: max_concurrent_reads,
            mut_cap: max_concurrent_mutations,
            per_tool_caps,
            promoted,
            read: Arc::new(Semaphore::new(max_concurrent_reads)),
            serial_mut: Arc::new(Semaphore::new(1)),
            concurrent_mut: Arc::new(Semaphore::new(max_concurrent_mutations)),
            per_tool,
        }
    }

    /// Build a [`ConcurrencyConfig`] for nested sub-agent execution.
    /// Forks **only** the pools that would otherwise deadlock when a
    /// parent permit is held across the child's execution; shares the
    /// rest so global invariants survive nesting:
    ///
    /// - `concurrent_mut` — **forked**. A parent saturating this pool
    ///   with promoted `agent` calls (or any other promoted mutator)
    ///   would deadlock children that needed their own
    ///   `concurrent_mut` permit. Each nesting level gets its own
    ///   pool with the same cap.
    /// - `per_tool` — **forked**. User-configured per-tool caps would
    ///   deadlock for the same reason if shared (parent holds N
    ///   permits during nested run, children can't acquire). Each
    ///   nesting level gets fresh per-tool semaphores with the same
    ///   caps.
    /// - `serial_mut` — **shared**. The width-1 default-mutator pool
    ///   keeps the "non-promoted `Mutating` tools serialise globally
    ///   across the agent tree" contract that consumers rely on for
    ///   correctness — two sibling sub-agents writing the same file
    ///   via the built-in `write` tool must not race even when the
    ///   user opted into none of the concurrency knobs. Shared
    ///   `serial_mut` is safe because recursive tools (which would
    ///   otherwise deadlock here) are routed through `concurrent_mut`
    ///   by [`ToolExecutor::routing_class`] — non-recursive
    ///   default-Mutating tools have short execute bodies that
    ///   release the permit before any sibling could need it.
    /// - `read` — **shared**. The read pool is a throughput throttle,
    ///   not a correctness mechanism; sharing keeps a global limit on
    ///   parallel reads across the tree.
    ///
    /// Trade-off on the forked pools: the per-call concurrent-mutator
    /// cap is per-sibling, not tree-wide. With
    /// `max_concurrent_mutations(10)` and 10 promoted `agent` calls in
    /// a parent batch, each spawning 10 promoted children, the tree
    /// can have 100 concurrent promoted-mutator calls in flight — the
    /// cap binds at each fork, not globally. Globally-bounded
    /// admission across nesting would require a fundamentally
    /// different primitive (release-on-await rather than
    /// release-on-Drop) and is out of scope.
    #[must_use]
    pub fn fork(&self) -> Self {
        let per_tool: HashMap<String, Arc<Semaphore>> = self
            .per_tool_caps
            .iter()
            .map(|(name, &cap)| (name.clone(), Arc::new(Semaphore::new(cap))))
            .collect();
        Self {
            read_cap: self.read_cap,
            mut_cap: self.mut_cap,
            per_tool_caps: self.per_tool_caps.clone(),
            promoted: self.promoted.clone(),
            // Shared pools: same Arc, same permit accounting.
            read: Arc::clone(&self.read),
            serial_mut: Arc::clone(&self.serial_mut),
            // Forked pools: fresh permit accounting at this level.
            concurrent_mut: Arc::new(Semaphore::new(self.mut_cap)),
            per_tool,
        }
    }
}

/// Dispatches tool calls against a registry, gated by a policy and an
/// approval handler, with admission control via [`ConcurrencyConfig`].
///
/// Two gates run before every tool invocation:
///
/// 1. [`ToolPolicy::is_allowed`] — *static* gate. Synchronous, no UI
///    interaction; decides whether the tool may run at all based on
///    its name. Denial here surfaces as `is_error: true` tool_result.
/// 2. [`ApprovalHandler::approve`] — *dynamic* gate. Async, may block
///    on a UI prompt. Decides whether *this specific call* with
///    *these specific arguments* may run. Denial also surfaces as
///    `is_error: true` tool_result so the model can adapt.
///
/// The approval call is raced against `ctx.cancel.cancelled()`, so an
/// outer cancel always wins over a hung UI.
///
/// When `execute_batch` runs multiple promoted-mutator calls
/// concurrently, [`ApprovalHandler::approve`] may be invoked from
/// several tasks at once. The trait already requires `Send + Sync`,
/// so type-system safety is unconditional; human-UI implementations
/// should serialise their UI access internally.
///
/// Cloning `Arc<ToolExecutor>` is cheap and intended: sub-agents share
/// the same executor with their parent so nested agents automatically
/// inherit the same registry, policy, approval handler, AND
/// concurrency configuration (Model 3).
pub struct ToolExecutor {
    registry: Arc<ToolRegistry>,
    policy: Arc<dyn ToolPolicy>,
    approval: Arc<dyn ApprovalHandler>,
    concurrency: ConcurrencyConfig,
}

impl ToolExecutor {
    /// Construct an executor with the default `AutoApprove` handler and
    /// default [`ConcurrencyConfig`] (20 reads, 10 promoted mutators,
    /// 1 serial-mutator slot, no promotions, no per-tool overrides).
    pub fn new(registry: Arc<ToolRegistry>, policy: Arc<dyn ToolPolicy>) -> Self {
        Self::with_approval_and_concurrency(
            registry,
            policy,
            Arc::new(AutoApprove),
            ConcurrencyConfig::default(),
        )
    }

    /// Construct an executor with an explicit approval handler and
    /// default [`ConcurrencyConfig`].
    pub fn with_approval(
        registry: Arc<ToolRegistry>,
        policy: Arc<dyn ToolPolicy>,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Self {
        Self::with_approval_and_concurrency(
            registry,
            policy,
            approval,
            ConcurrencyConfig::default(),
        )
    }

    /// Construct an executor with explicit approval handler AND explicit
    /// concurrency configuration. Used by `AgentBuilder::build()` when
    /// any of the `max_concurrent_*` / `tool_concurrency` methods were
    /// called; consumers can also construct directly when bypassing the
    /// builder.
    pub fn with_approval_and_concurrency(
        registry: Arc<ToolRegistry>,
        policy: Arc<dyn ToolPolicy>,
        approval: Arc<dyn ApprovalHandler>,
        concurrency: ConcurrencyConfig,
    ) -> Self {
        Self {
            registry,
            policy,
            approval,
            concurrency,
        }
    }

    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }

    pub(crate) fn policy_arc_for_fork(&self) -> Arc<dyn ToolPolicy> {
        Arc::clone(&self.policy)
    }

    /// Build an executor for a nested sub-agent that shares this
    /// executor's registry, policy, and approval handler — but has
    /// independent concurrency permit accounting.
    ///
    /// Without this fork, a sub-agent invoked from a tool admitted via
    /// the parent's `concurrent_mut` pool would compete for permits
    /// from that same already-saturated pool whenever the child needed
    /// its own promoted-mutator slot. With enough fan-out the parent
    /// holds N permits across `SubAgent::execute` futures, leaving
    /// 0 permits for any of those children to start their own
    /// promoted work — a permanent stall.
    ///
    /// Forking the [`ConcurrencyConfig`] (same caps, fresh semaphores)
    /// gives each nesting level its own admission pool. The trade-off
    /// is documented on [`ConcurrencyConfig::fork`]: total in-flight
    /// work is no longer globally bounded, only per-level bounded.
    #[must_use]
    pub fn fork_for_subagent(&self) -> Arc<Self> {
        self.fork_for_subagent_with(None, None)
    }

    /// Like [`fork_for_subagent`](Self::fork_for_subagent), but lets the caller
    /// override the child policy and/or approval handler while keeping the
    /// shared registry and fresh per-level concurrency permits.
    #[must_use]
    pub fn fork_for_subagent_with(
        &self,
        policy_override: Option<Arc<dyn ToolPolicy>>,
        approval_override: Option<Arc<dyn ApprovalHandler>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            registry: Arc::clone(&self.registry),
            policy: policy_override.unwrap_or_else(|| Arc::clone(&self.policy)),
            approval: approval_override.unwrap_or_else(|| Arc::clone(&self.approval)),
            concurrency: self.concurrency.fork(),
        })
    }

    /// Execute a single tool call. Always returns a `tool_result` `Content`
    /// block — even on policy denial, approval denial, missing tool, or
    /// tool error (with `is_error: true`). The loop never aborts on a
    /// tool problem; the LLM sees the error and may adapt.
    pub async fn execute_one(&self, call: ToolCall, ctx: &ToolContext) -> Content {
        if !self.policy.is_allowed(&call.name) {
            return Content::tool_result(
                &call.id,
                format!("Error: tool '{}' is not allowed by policy", call.name),
                true,
            );
        }

        let Some(tool) = self.registry.get(&call.name) else {
            return Content::tool_result(
                &call.id,
                format!("Error: tool '{}' not found", call.name),
                true,
            );
        };

        // Dynamic gate: ask the approval handler. Race against the
        // outer cancellation token so a hung UI cannot deadlock the
        // agent indefinitely — `cancel.cancel()` always wins.
        let class = tool.class();
        let decision = tokio::select! {
            biased;
            _ = ctx.cancel.cancelled() => {
                return Content::tool_result(
                    &call.id,
                    "Error: cancelled while awaiting approval",
                    true,
                );
            }
            d = self.approval.approve(&call.name, &call.input, class) => d,
        };
        if let ApprovalDecision::Deny(reason) = decision {
            return Content::tool_result(
                &call.id,
                format!("Error: approval denied — {reason}"),
                true,
            );
        }

        match tool.execute(call.input, ctx).await {
            Ok(output) => Content::tool_result(&call.id, output.content(), output.is_error()),
            Err(e) => Content::tool_result(&call.id, format!("Error: {e}"), true),
        }
    }

    /// Execute a batch of tool calls in the LLM-issued order.
    ///
    /// Calls are partitioned into contiguous **runs** of the same routing
    /// class:
    ///
    /// - `ReadOnly` runs — every read-only tool from the registry.
    /// - `ConcurrentMut` runs — `Mutating` tools the consumer promoted
    ///   via [`ToolConcurrency::on`].
    /// - `SerialMut` runs — `Mutating` tools without opt-in.
    ///
    /// Within a run, calls dispatch concurrently into a
    /// [`FuturesUnordered`]. Between runs, the executor `await`s the
    /// previous run before starting the next — so an LLM-issued
    /// `[Read A, Write A, Read A]` always observes the write before
    /// the trailing read. This boundary preserves the ordering
    /// semantics consumers (and LLMs) reasonably assume across
    /// class boundaries; the executor cannot know which file/path
    /// each tool touches, so it falls back on the LLM's emitted
    /// order as the source of truth.
    ///
    /// Each task acquires admission permits before invoking
    /// [`Self::execute_one`]:
    ///
    /// - Optional **per-tool** permit (only if the consumer set
    ///   [`ToolConcurrency::max`] for this tool name).
    /// - Mandatory **class** permit from one of three pools:
    ///     - `read` (default cap 20) — every `ReadOnly` tool.
    ///     - `concurrent_mut` (default cap 10) — promoted `Mutating`
    ///       tools.
    ///     - `serial_mut` (cap 1, fixed) — non-promoted `Mutating`
    ///       tools. The width-1 pool is what makes a non-promoted
    ///       mutator a single-call run by itself.
    ///
    /// Acquisition order is per-tool first, class second, so the shared
    /// class permit is held for the shortest time when multiple tasks
    /// contend on the same per-tool cap.
    ///
    /// Both acquisitions race against [`ToolContext::cancel`] via a
    /// `biased` `tokio::select!`: cancel always wins ties and produces
    /// a synthetic `is_error: true` `tool_result` carrying
    /// `"cancelled before execution"` without invoking the tool.
    ///
    /// Results are returned in **input order** regardless of within-run
    /// completion order. Pre-cancel before any task starts short-
    /// circuits all calls with the same synthetic error so the 1:1
    /// tool_use→tool_result invariant the agent loop relies on is
    /// preserved.
    pub async fn execute_batch(&self, calls: Vec<ToolCall>, ctx: &ToolContext) -> Vec<Content> {
        self.execute_batch_with_tracker(calls, ctx, None, None)
            .await
    }

    pub(crate) async fn execute_batch_with_tracker(
        &self,
        calls: Vec<ToolCall>,
        ctx: &ToolContext,
        tracker: Option<ToolRunTracker>,
        mode: Option<Arc<dyn AgentMode>>,
    ) -> Vec<Content> {
        if calls.is_empty() {
            return Vec::new();
        }
        if ctx.cancel.is_cancelled() {
            return all_cancelled_before_execution(calls);
        }

        let control = DispatchControl { tracker, mode };
        let n = calls.len();
        let routings: Vec<RoutingClass> = calls.iter().map(|c| self.routing_class(c)).collect();
        let mut calls: Vec<Option<ToolCall>> = calls.into_iter().map(Some).collect();
        let mut slots: Vec<Option<Content>> = (0..n).map(|_| None).collect();

        let mut i = 0;
        while i < n {
            if ctx.cancel.is_cancelled() {
                fill_cancelled_tail(&mut calls, &mut slots, i);
                break;
            }
            let j = same_class_run_end(&routings, i);
            self.dispatch_run(
                &mut calls,
                &mut slots,
                &routings,
                i..j,
                ctx,
                control.clone(),
            )
            .await;
            i = j;
        }

        slots
            .into_iter()
            .map(|o| o.expect("every slot filled by dispatch or cancel short-circuit"))
            .collect()
    }

    /// Dispatch one contiguous run of same-routing-class calls in
    /// parallel via [`FuturesUnordered`], placing results into their
    /// input-ordered slots as each task completes.
    async fn dispatch_run(
        &self,
        calls: &mut [Option<ToolCall>],
        slots: &mut [Option<Content>],
        routings: &[RoutingClass],
        range: std::ops::Range<usize>,
        ctx: &ToolContext,
        control: DispatchControl,
    ) {
        let mut futs = FuturesUnordered::new();
        for k in range {
            let call = calls[k]
                .take()
                .expect("each slot taken exactly once during dispatch");
            futs.push(self.dispatch_one(k, call, routings[k], ctx, control.clone()));
        }
        while let Some((idx, content)) = futs.next().await {
            slots[idx] = Some(content);
        }
    }

    /// Pre-classify a call into one of four routing buckets so
    /// `execute_batch` can group contiguous same-class calls into a
    /// parallel run.
    ///
    /// Recursive tools (per [`Tool::is_recursive`]) are admitted
    /// through `ConcurrentMut` regardless of explicit promotion,
    /// because their `execute` body parks the executor task while
    /// driving a nested `Agent::run` — and the non-recursive pools
    /// (`serial_mut` and `read`) are shared across the agent tree.
    /// Routing recursive tools through the per-level-forked
    /// `concurrent_mut` pool prevents the permit-held-during-nested-
    /// execute deadlock without breaking the global serialisation
    /// contract that consumers (and the LLM) rely on for
    /// non-promoted default-mutators.
    fn routing_class(&self, call: &ToolCall) -> RoutingClass {
        if !self.policy.is_allowed(&call.name) {
            return RoutingClass::ShortCircuit;
        }
        let Some(tool) = self.registry.get(&call.name) else {
            return RoutingClass::ShortCircuit;
        };
        if tool.is_recursive() {
            return RoutingClass::ConcurrentMut;
        }
        match tool.class() {
            ToolClass::ReadOnly => RoutingClass::ReadOnly,
            ToolClass::Mutating if self.concurrency.promoted.contains(&call.name) => {
                RoutingClass::ConcurrentMut
            }
            ToolClass::Mutating => RoutingClass::SerialMut,
        }
    }

    /// Drive a single call through the admission-control pipeline. Returns
    /// `(idx, Content)` so the caller in `execute_batch` can place the
    /// result into the input-ordered output slot.
    async fn dispatch_one(
        &self,
        idx: usize,
        call: ToolCall,
        routing: RoutingClass,
        ctx: &ToolContext,
        control: DispatchControl,
    ) -> (usize, Content) {
        if matches!(routing, RoutingClass::ShortCircuit) {
            return (idx, self.short_circuit_result(&call));
        }

        let call_id = call.id.clone();
        let child_cancel = ctx.cancel.child_token();
        if let Some(tracker) = &control.tracker {
            tracker.register(&call_id, child_cancel.clone());
        }
        let child_ctx = ctx.with_cancel(child_cancel);

        let class_sem = self.class_semaphore_for(routing);
        let per_tool_sem = self.concurrency.per_tool.get(&call.name).cloned();

        let _permits = match acquire_admission(per_tool_sem, class_sem, &child_ctx).await {
            Some(permits) => permits,
            None => {
                if let Some(tracker) = &control.tracker {
                    tracker.mark_done(&call_id);
                }
                return (idx, cancelled_before_execution(&call_id));
            }
        };

        if let Some(denial) = self.mode_denial(&call, control.mode.as_deref()) {
            if let Some(tracker) = &control.tracker {
                tracker.mark_done(&call_id);
            }
            return (idx, denial);
        }

        let content = self.execute_one(call, &child_ctx).await;
        if let Some(tracker) = &control.tracker {
            tracker.mark_done(&call_id);
        }
        (idx, content)
    }

    fn mode_denial(&self, call: &ToolCall, mode: Option<&dyn AgentMode>) -> Option<Content> {
        let mode = mode?;
        let tool = self.registry.get(&call.name)?;
        let class = tool.class();
        let decision = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mode.tool_gate(&call.name, class).unwrap_or_else(|| {
                if class == ToolClass::Mutating && !mode.allows_mutating_tools() {
                    ModeDecision::Deny {
                        reason: format!("mode '{}' denies mutating tools", mode.name()).into(),
                    }
                } else {
                    ModeDecision::Allow
                }
            })
        }));
        match decision {
            Ok(ModeDecision::Allow) => None,
            Err(_) => Some(Content::tool_result(
                &call.id,
                format!("Error: mode gate panicked for tool '{}'", call.name),
                true,
            )),
            Ok(ModeDecision::Deny { reason }) => Some(Content::tool_result(
                &call.id,
                format!("Error: mode denied tool '{}' — {reason}", call.name),
                true,
            )),
        }
    }

    /// Build the error `tool_result` for a denied or missing tool —
    /// the only outcome of a `RoutingClass::ShortCircuit` dispatch.
    fn short_circuit_result(&self, call: &ToolCall) -> Content {
        let body = if !self.policy.is_allowed(&call.name) {
            format!("Error: tool '{}' is not allowed by policy", call.name)
        } else {
            format!("Error: tool '{}' not found", call.name)
        };
        Content::tool_result(&call.id, body, true)
    }

    /// Pick the class-pool semaphore for an admissible routing class.
    /// `ShortCircuit` is handled by its own fast-path before this is
    /// called; reaching it here is a programming error.
    fn class_semaphore_for(&self, routing: RoutingClass) -> Arc<Semaphore> {
        match routing {
            RoutingClass::ReadOnly => Arc::clone(&self.concurrency.read),
            RoutingClass::ConcurrentMut => Arc::clone(&self.concurrency.concurrent_mut),
            RoutingClass::SerialMut => Arc::clone(&self.concurrency.serial_mut),
            RoutingClass::ShortCircuit => {
                unreachable!("ShortCircuit takes the early-return path before reaching here")
            }
        }
    }
}

#[derive(Clone)]
struct DispatchControl {
    tracker: Option<ToolRunTracker>,
    mode: Option<Arc<dyn AgentMode>>,
}

enum AcquireOutcome {
    Permit(OwnedSemaphorePermit),
    Cancelled,
}

/// Acquire both the optional per-tool permit and the mandatory class
/// permit for a single dispatch. Returns `None` if cancellation
/// resolved before either acquire completed; both permits drop on
/// `None` return so no semaphore is left held.
///
/// Per-tool acquired first so the shared class permit is held for the
/// shortest time when multiple tasks contend on the same per-tool
/// cap. The returned tuple keeps both permits alive for the caller's
/// scope; dropping the tuple at the end of `dispatch_one` releases
/// permits in RAII order.
async fn acquire_admission(
    per_tool_sem: Option<Arc<Semaphore>>,
    class_sem: Arc<Semaphore>,
    ctx: &ToolContext,
) -> Option<(Option<OwnedSemaphorePermit>, OwnedSemaphorePermit)> {
    let per_tool_permit = match per_tool_sem {
        Some(sem) => match acquire_or_cancel(sem, ctx).await {
            AcquireOutcome::Permit(p) => Some(p),
            AcquireOutcome::Cancelled => return None,
        },
        None => None,
    };
    let class_permit = match acquire_or_cancel(class_sem, ctx).await {
        AcquireOutcome::Permit(p) => p,
        AcquireOutcome::Cancelled => return None,
    };
    Some((per_tool_permit, class_permit))
}

/// Pre-classified routing for a single call. Used by `execute_batch`
/// to group contiguous same-class calls into a parallel run while
/// keeping inter-class barriers — so a `[Read, Write, Read]` batch
/// always observes the write before the trailing read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutingClass {
    /// `ReadOnly` tool — admitted via the `read` pool.
    ReadOnly,
    /// `Mutating` tool promoted via `tool_concurrency(name, on())` —
    /// admitted via the `concurrent_mut` pool. Multiple promoted
    /// mutators in one run overlap up to the class cap.
    ConcurrentMut,
    /// `Mutating` tool without opt-in — admitted via the width-1
    /// `serial_mut` pool. A single such call forms its own run.
    SerialMut,
    /// Policy-denied or registry-missing tool — emits an error
    /// `tool_result` immediately, without acquiring any semaphore.
    ShortCircuit,
}

/// Race a semaphore acquire against the context's cancellation token.
/// `biased` keeps cancel as the priority branch when both are
/// immediately ready, matching the existing approval-acquire pattern.
async fn acquire_or_cancel(sem: Arc<Semaphore>, ctx: &ToolContext) -> AcquireOutcome {
    tokio::select! {
        biased;
        _ = ctx.cancel.cancelled() => AcquireOutcome::Cancelled,
        p = sem.acquire_owned() => AcquireOutcome::Permit(
            p.expect("semaphore not closed — executor never closes its own semaphores"),
        ),
    }
}

fn cancelled_before_execution(call_id: &str) -> Content {
    Content::tool_result(
        call_id,
        "Error: cancelled before execution".to_string(),
        true,
    )
}

/// Build cancelled-tool_result content for every call without spawning
/// any task. Used by `execute_batch` when cancel fires before any
/// dispatch.
fn all_cancelled_before_execution(calls: Vec<ToolCall>) -> Vec<Content> {
    calls
        .into_iter()
        .map(|c| cancelled_before_execution(&c.id))
        .collect()
}

/// Drain remaining calls from `start..` into cancelled slots without
/// spawning any task. Used by `execute_batch` when cancel fires
/// between runs.
fn fill_cancelled_tail(
    calls: &mut [Option<ToolCall>],
    slots: &mut [Option<Content>],
    start: usize,
) {
    for k in start..calls.len() {
        if let Some(call) = calls[k].take() {
            slots[k] = Some(cancelled_before_execution(&call.id));
        }
    }
}

/// Find the end (exclusive) of the contiguous run of routing-class-
/// identical calls starting at `start`.
///
/// `SerialMut` runs are forced to length 1: the width-1 `serial_mut`
/// semaphore prevents overlap but does NOT guarantee FIFO acquisition
/// order — `FuturesUnordered` can poll futures in any order and a
/// later non-promoted mutator can acquire before an earlier one,
/// re-ordering side effects relative to the LLM's emitted batch.
/// Forcing single-call runs for `SerialMut` makes adjacent default-
/// `Mutating` tools (e.g. `[Write A, Edit A]`) execute strictly in
/// input order, matching the pre-concurrency-feature behaviour the
/// LLM relies on for chained file edits.
fn same_class_run_end(routings: &[RoutingClass], start: usize) -> usize {
    if matches!(routings[start], RoutingClass::SerialMut) {
        return start + 1;
    }
    let n = routings.len();
    let mut end = start + 1;
    while end < n && routings[end] == routings[start] {
        end += 1;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ToolError;
    use crate::tool::{ToolClass, ToolOutput};
    use async_trait::async_trait;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    struct Echo;
    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echo"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        fn class(&self) -> ToolClass {
            ToolClass::ReadOnly
        }
        async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text(input["msg"].as_str().unwrap_or("")))
        }
    }

    fn empty_executor() -> Arc<ToolExecutor> {
        Arc::new(ToolExecutor::new(
            Arc::new(ToolRegistry::new(vec![])),
            Arc::new(AllowAll),
        ))
    }

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: PathBuf::from("/tmp"),
            cancel: tokio_util::sync::CancellationToken::new(),
            depth: 0,
            max_depth: 1,
            executor: empty_executor(),
        }
    }

    fn call(name: &str, input: Value) -> ToolCall {
        ToolCall {
            id: "id".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn allow_all_runs_tool() {
        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(Echo)]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let res = exec
            .execute_one(call("echo", json!({"msg": "hi"})), &ctx())
            .await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(!is_error);
        assert_eq!(content, "hi");
    }

    #[tokio::test]
    async fn missing_tool_returns_error_result() {
        let reg = Arc::new(ToolRegistry::new(vec![]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let res = exec.execute_one(call("ghost", json!({})), &ctx()).await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(is_error);
        assert!(content.contains("not found"));
    }

    struct DenyNamed(&'static str);
    impl ToolPolicy for DenyNamed {
        fn is_allowed(&self, name: &str) -> bool {
            name != self.0
        }
    }

    #[tokio::test]
    async fn policy_denial_returns_error_result() {
        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(Echo)]));
        let exec = ToolExecutor::new(reg, Arc::new(DenyNamed("echo")));
        let res = exec
            .execute_one(call("echo", json!({"msg": "hi"})), &ctx())
            .await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(is_error);
        assert!(content.contains("not allowed"));
    }

    /// A read-only tool that sleeps for `delay_ms` then echoes its label.
    struct SlowRO {
        label: String,
    }
    #[async_trait]
    impl Tool for SlowRO {
        fn name(&self) -> &str {
            &self.label
        }
        fn description(&self) -> &str {
            "slow"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        fn class(&self) -> ToolClass {
            ToolClass::ReadOnly
        }
        async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
            let delay_ms = input["delay_ms"].as_u64().unwrap_or(0);
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            Ok(ToolOutput::text(self.label.clone()))
        }
    }

    /// A mutating tool that records its invocation order in a shared counter.
    struct OrderingMut {
        label: String,
    }
    #[async_trait]
    impl Tool for OrderingMut {
        fn name(&self) -> &str {
            &self.label
        }
        fn description(&self) -> &str {
            "mut"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        // class() defaults to Mutating.
        async fn execute(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text(self.label.clone()))
        }
    }

    fn extract_text(c: &Content) -> &str {
        match c {
            Content::ToolResult { content, .. } => content.as_str(),
            _ => panic!("expected tool_result"),
        }
    }

    #[tokio::test]
    async fn batch_preserves_order_despite_parallel_ro() {
        // Put a slow RO tool BEFORE a fast RO tool. If RO runs are truly
        // parallel and results are not re-ordered, "b" finishes first but
        // appears second in the output.
        let reg = Arc::new(ToolRegistry::new(vec![
            Arc::new(SlowRO { label: "a".into() }),
            Arc::new(SlowRO { label: "b".into() }),
        ]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "a".into(),
                input: json!({"delay_ms": 50}),
            },
            ToolCall {
                id: "2".into(),
                name: "b".into(),
                input: json!({"delay_ms": 0}),
            },
        ];

        let start = std::time::Instant::now();
        let results = exec.execute_batch(calls, &ctx()).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 2);
        assert_eq!(extract_text(&results[0]), "a"); // original order kept
        assert_eq!(extract_text(&results[1]), "b");
        // If RO ran sequentially, elapsed would be ≥ 50ms. Parallel ⇒ ~50ms.
        // Sequential would be ~50ms + 0ms = 50ms too, so timing alone is not
        // a strict test; ordering IS the invariant we care about. Keep the
        // timing check loose — just assert it didn't balloon to double.
        assert!(
            elapsed < Duration::from_millis(150),
            "unexpected slowdown: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn batch_partitions_ro_and_mut_runs() {
        // Pattern: [RO a, RO b, MUT m, RO c] — without any concurrency
        // promotion, the default Mutating m routes to the width-1
        // serial-mutator pool. Each call returns; results in input order.
        let reg = Arc::new(ToolRegistry::new(vec![
            Arc::new(SlowRO { label: "a".into() }),
            Arc::new(SlowRO { label: "b".into() }),
            Arc::new(OrderingMut { label: "m".into() }),
            Arc::new(SlowRO { label: "c".into() }),
        ]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "a".into(),
                input: json!({"delay_ms": 10}),
            },
            ToolCall {
                id: "2".into(),
                name: "b".into(),
                input: json!({"delay_ms": 10}),
            },
            ToolCall {
                id: "3".into(),
                name: "m".into(),
                input: json!({}),
            },
            ToolCall {
                id: "4".into(),
                name: "c".into(),
                input: json!({"delay_ms": 10}),
            },
        ];

        let results = exec.execute_batch(calls, &ctx()).await;
        assert_eq!(results.len(), 4);
        assert_eq!(extract_text(&results[0]), "a");
        assert_eq!(extract_text(&results[1]), "b");
        assert_eq!(extract_text(&results[2]), "m");
        assert_eq!(extract_text(&results[3]), "c");
    }

    /// A mutating tool that toggles a shared flag — used to detect whether
    /// a tool ran. If `execute_batch` correctly stops dispatching after
    /// cancel fires, the second mutating tool's flag stays unset.
    struct FlagSetter(Arc<AtomicBool>, &'static str);
    #[async_trait]
    impl Tool for FlagSetter {
        fn name(&self) -> &str {
            self.1
        }
        fn description(&self) -> &str {
            "flag"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        async fn execute(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            self.0.store(true, Ordering::SeqCst);
            Ok(ToolOutput::text("ran"))
        }
    }

    #[tokio::test]
    async fn batch_stops_dispatching_after_cancel() {
        let m1_ran = Arc::new(AtomicBool::new(false));
        let m2_ran = Arc::new(AtomicBool::new(false));

        let reg = Arc::new(ToolRegistry::new(vec![
            Arc::new(FlagSetter(Arc::clone(&m1_ran), "m1")),
            Arc::new(FlagSetter(Arc::clone(&m2_ran), "m2")),
        ]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext {
            working_dir: PathBuf::from("/tmp"),
            cancel: cancel.clone(),
            depth: 0,
            max_depth: 1,
            executor: empty_executor(),
        };

        // Pre-cancel before invocation: NEITHER tool should run, both
        // should produce synthetic cancelled errors.
        cancel.cancel();
        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "m1".into(),
                input: json!({}),
            },
            ToolCall {
                id: "2".into(),
                name: "m2".into(),
                input: json!({}),
            },
        ];
        let results = exec.execute_batch(calls, &ctx).await;

        assert_eq!(results.len(), 2, "result count must match input count");
        for r in &results {
            let Content::ToolResult {
                content, is_error, ..
            } = r
            else {
                panic!("expected tool_result");
            };
            assert!(*is_error, "cancelled-before-execution should be is_error");
            assert!(
                content.contains("cancelled before execution"),
                "got: {content}"
            );
        }
        assert!(
            !m1_ran.load(Ordering::SeqCst),
            "m1 must not have run after cancel"
        );
        assert!(
            !m2_ran.load(Ordering::SeqCst),
            "m2 must not have run after cancel"
        );
    }

    // --- Approval-gate tests -----------------------------------------------

    /// Approval handler that always denies with a fixed reason.
    struct AlwaysDeny(&'static str);
    #[async_trait]
    impl ApprovalHandler for AlwaysDeny {
        async fn approve(&self, _: &str, _: &Value, _: ToolClass) -> ApprovalDecision {
            ApprovalDecision::Deny(self.0.to_string())
        }
    }

    /// Approval handler that takes 10s before answering — used to
    /// prove cancellation interrupts a hung approval.
    struct SlowApproval;
    #[async_trait]
    impl ApprovalHandler for SlowApproval {
        async fn approve(&self, _: &str, _: &Value, _: ToolClass) -> ApprovalDecision {
            tokio::time::sleep(Duration::from_secs(10)).await;
            ApprovalDecision::Allow
        }
    }

    #[tokio::test]
    async fn approval_deny_emits_error_tool_result_and_skips_execution() {
        let ran = Arc::new(AtomicBool::new(false));
        let ran_clone = Arc::clone(&ran);

        struct ObservingTool(Arc<AtomicBool>);
        #[async_trait]
        impl Tool for ObservingTool {
            fn name(&self) -> &str {
                "observe"
            }
            fn description(&self) -> &str {
                "observes whether it ran"
            }
            fn input_schema(&self) -> Value {
                json!({})
            }
            async fn execute(&self, _: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
                self.0.store(true, Ordering::SeqCst);
                Ok(ToolOutput::text("ran"))
            }
        }

        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(ObservingTool(ran_clone))]));
        let exec = ToolExecutor::with_approval(
            reg,
            Arc::new(AllowAll),
            Arc::new(AlwaysDeny("blocked by user")),
        );
        let res = exec.execute_one(call("observe", json!({})), &ctx()).await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };

        assert!(is_error, "denied call should yield is_error: true");
        assert!(
            content.contains("approval denied"),
            "content should mark approval denial, got: {content}"
        );
        assert!(
            content.contains("blocked by user"),
            "content should preserve the deny reason, got: {content}"
        );
        assert!(
            !ran.load(Ordering::SeqCst),
            "tool must NOT have executed after approval denial"
        );
    }

    #[tokio::test]
    async fn approval_cancel_during_approve_short_circuits() {
        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(Echo)]));
        let exec = ToolExecutor::with_approval(reg, Arc::new(AllowAll), Arc::new(SlowApproval));

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext {
            working_dir: PathBuf::from("/tmp"),
            cancel: cancel.clone(),
            depth: 0,
            max_depth: 1,
            executor: empty_executor(),
        };

        // Fire cancel after 50ms; SlowApproval would take 10s otherwise.
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });

        let started = std::time::Instant::now();
        let res = exec
            .execute_one(call("echo", json!({"msg": "x"})), &ctx)
            .await;
        let elapsed = started.elapsed();

        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(is_error, "cancel during approval should yield is_error");
        assert!(
            content.contains("cancelled"),
            "content should mention cancellation, got: {content}"
        );
        // Critical: the 10s SlowApproval future was racing the 50ms
        // cancel. With biased select! on cancel-first, we must beat
        // 10s by an order of magnitude. 1s is comfortable slack.
        assert!(
            elapsed < Duration::from_secs(1),
            "cancel should win the race against approve(); took {elapsed:?}"
        );
    }

    // --- Concurrency-model tests -------------------------------------------

    /// A probe tool that records the maximum concurrent invocation count.
    /// `class` lets a single struct serve as RO, default-Mutating, or
    /// promoted-Mutating depending on the test.
    struct ConcurrencyProbe {
        label: String,
        class: ToolClass,
        delay_ms: u64,
        active: Arc<AtomicUsize>,
        max_seen: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for ConcurrencyProbe {
        fn name(&self) -> &str {
            &self.label
        }
        fn description(&self) -> &str {
            "concurrency probe"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        fn class(&self) -> ToolClass {
            self.class
        }
        async fn execute(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            let cur = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            // Peek-max: bump max_seen if cur exceeds it.
            let mut prev = self.max_seen.load(Ordering::SeqCst);
            while cur > prev {
                match self
                    .max_seen
                    .compare_exchange(prev, cur, Ordering::SeqCst, Ordering::SeqCst)
                {
                    Ok(_) => break,
                    Err(actual) => prev = actual,
                }
            }
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolOutput::text(self.label.clone()))
        }
    }

    fn make_probe(
        label: &str,
        class: ToolClass,
        delay_ms: u64,
        active: &Arc<AtomicUsize>,
        max_seen: &Arc<AtomicUsize>,
    ) -> Arc<ConcurrencyProbe> {
        Arc::new(ConcurrencyProbe {
            label: label.into(),
            class,
            delay_ms,
            active: Arc::clone(active),
            max_seen: Arc::clone(max_seen),
        })
    }

    #[tokio::test]
    async fn default_mutator_serializes_via_width_one_semaphore() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let probe = make_probe("mut", ToolClass::Mutating, 30, &active, &max_seen);
        let reg = Arc::new(ToolRegistry::new(vec![probe]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));

        let calls: Vec<ToolCall> = (0..3)
            .map(|i| ToolCall {
                id: format!("{i}"),
                name: "mut".into(),
                input: json!({}),
            })
            .collect();

        let results = exec.execute_batch(calls, &ctx()).await;
        assert_eq!(results.len(), 3);
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            1,
            "default-Mutating routes to the width-1 serial pool"
        );
    }

    #[tokio::test]
    async fn promoted_mutator_runs_in_parallel_up_to_class_cap() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let probe = make_probe("pmut", ToolClass::Mutating, 60, &active, &max_seen);
        let reg = Arc::new(ToolRegistry::new(vec![probe]));
        // class cap = 3, no per-tool cap, promoted via on()
        let cfg = ConcurrencyConfig::new(20, 3, vec![("pmut".to_string(), ToolConcurrency::on())]);
        let exec = ToolExecutor::with_approval_and_concurrency(
            reg,
            Arc::new(AllowAll),
            Arc::new(AutoApprove),
            cfg,
        );

        let calls: Vec<ToolCall> = (0..5)
            .map(|i| ToolCall {
                id: format!("{i}"),
                name: "pmut".into(),
                input: json!({}),
            })
            .collect();

        let results = exec.execute_batch(calls, &ctx()).await;
        assert_eq!(results.len(), 5);
        let observed = max_seen.load(Ordering::SeqCst);
        assert_eq!(
            observed, 3,
            "promoted Mutating fills concurrent_mut up to class cap (got {observed})"
        );
    }

    #[tokio::test]
    async fn per_tool_cap_binds_below_class_cap() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let probe = make_probe("ptcap", ToolClass::Mutating, 60, &active, &max_seen);
        let reg = Arc::new(ToolRegistry::new(vec![probe]));
        // class cap = 10 (room), per-tool cap = 2 (binds first)
        let cfg = ConcurrencyConfig::new(
            20,
            10,
            vec![("ptcap".to_string(), ToolConcurrency::on().max(2))],
        );
        let exec = ToolExecutor::with_approval_and_concurrency(
            reg,
            Arc::new(AllowAll),
            Arc::new(AutoApprove),
            cfg,
        );

        let calls: Vec<ToolCall> = (0..5)
            .map(|i| ToolCall {
                id: format!("{i}"),
                name: "ptcap".into(),
                input: json!({}),
            })
            .collect();

        let results = exec.execute_batch(calls, &ctx()).await;
        assert_eq!(results.len(), 5);
        let observed = max_seen.load(Ordering::SeqCst);
        assert_eq!(
            observed, 2,
            "per-tool cap=2 binds below class cap=10 (got {observed})"
        );
    }

    #[tokio::test]
    async fn cancel_during_permit_acquire_short_circuits() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let probe = make_probe("slow", ToolClass::Mutating, 500, &active, &max_seen);
        let reg = Arc::new(ToolRegistry::new(vec![probe]));
        // Per-tool cap = 1 forces queueing.
        let cfg = ConcurrencyConfig::new(
            20,
            10,
            vec![("slow".to_string(), ToolConcurrency::on().max(1))],
        );
        let exec = ToolExecutor::with_approval_and_concurrency(
            reg,
            Arc::new(AllowAll),
            Arc::new(AutoApprove),
            cfg,
        );

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx_local = ToolContext {
            working_dir: PathBuf::from("/tmp"),
            cancel: cancel.clone(),
            depth: 0,
            max_depth: 1,
            executor: empty_executor(),
        };

        // Fire cancel ~50ms in — the first call has the per-tool permit
        // and is mid-sleep (500ms total), the second is parked on
        // acquire and must short-circuit.
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });

        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "slow".into(),
                input: json!({}),
            },
            ToolCall {
                id: "2".into(),
                name: "slow".into(),
                input: json!({}),
            },
        ];

        let results = exec.execute_batch(calls, &ctx_local).await;
        assert_eq!(results.len(), 2);

        // First: was already executing when cancel fired. The probe
        // doesn't honour cancel cooperatively, so it ran to completion.
        // Second: was waiting on the per-tool semaphore, must have
        // short-circuited with cancelled-before-execution.
        let r2 = &results[1];
        let Content::ToolResult {
            content, is_error, ..
        } = r2
        else {
            panic!("expected tool_result");
        };
        assert!(*is_error, "second call must be is_error after cancel");
        assert!(
            content.contains("cancelled before execution"),
            "second call must short-circuit, got: {content}"
        );

        // The probe's max_seen records max-concurrent-EXECUTED. Only the
        // first probe ever executed; second was cancelled at acquire.
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            1,
            "only the first probe should have entered execute()"
        );
    }

    #[tokio::test]
    async fn result_order_preserved_with_parallel_completion() {
        // Slow tool issued first, fast tool second. The fast tool finishes
        // first chronologically, but results must be returned in input
        // order — the indexed-slot collection in execute_batch is the
        // invariant being exercised.
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let slow = make_probe("slow", ToolClass::ReadOnly, 100, &active, &max_seen);
        let fast = make_probe("fast", ToolClass::ReadOnly, 0, &active, &max_seen);
        let reg = Arc::new(ToolRegistry::new(vec![slow, fast]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));

        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "slow".into(),
                input: json!({}),
            },
            ToolCall {
                id: "2".into(),
                name: "fast".into(),
                input: json!({}),
            },
        ];

        let results = exec.execute_batch(calls, &ctx()).await;
        assert_eq!(extract_text(&results[0]), "slow");
        assert_eq!(extract_text(&results[1]), "fast");
    }

    #[tokio::test]
    async fn mixed_class_batch_independent_pools() {
        // Three pools observed independently in one batch:
        //   - 5 RO calls cap 5 → max-observed = 5 (cap not hit)
        //   - 3 promoted-Mutating calls cap 2 → max-observed = 2
        //   - 2 default-Mutating calls cap 1 → max-observed = 1
        let ro_active = Arc::new(AtomicUsize::new(0));
        let ro_max = Arc::new(AtomicUsize::new(0));
        let pmut_active = Arc::new(AtomicUsize::new(0));
        let pmut_max = Arc::new(AtomicUsize::new(0));
        let smut_active = Arc::new(AtomicUsize::new(0));
        let smut_max = Arc::new(AtomicUsize::new(0));

        let ro = make_probe("ro", ToolClass::ReadOnly, 80, &ro_active, &ro_max);
        let pmut = make_probe("pmut", ToolClass::Mutating, 80, &pmut_active, &pmut_max);
        let smut = make_probe("smut", ToolClass::Mutating, 80, &smut_active, &smut_max);
        let reg = Arc::new(ToolRegistry::new(vec![ro, pmut, smut]));

        // RO cap 5, promoted-mutator class cap 2; pmut promoted, smut not.
        let cfg = ConcurrencyConfig::new(5, 2, vec![("pmut".to_string(), ToolConcurrency::on())]);
        let exec = ToolExecutor::with_approval_and_concurrency(
            reg,
            Arc::new(AllowAll),
            Arc::new(AutoApprove),
            cfg,
        );

        let mut calls: Vec<ToolCall> = Vec::new();
        for i in 0..5 {
            calls.push(ToolCall {
                id: format!("ro{i}"),
                name: "ro".into(),
                input: json!({}),
            });
        }
        for i in 0..3 {
            calls.push(ToolCall {
                id: format!("pmut{i}"),
                name: "pmut".into(),
                input: json!({}),
            });
        }
        for i in 0..2 {
            calls.push(ToolCall {
                id: format!("smut{i}"),
                name: "smut".into(),
                input: json!({}),
            });
        }

        let results = exec.execute_batch(calls, &ctx()).await;
        assert_eq!(results.len(), 10);

        assert_eq!(ro_max.load(Ordering::SeqCst), 5, "RO pool cap not hit");
        assert_eq!(
            pmut_max.load(Ordering::SeqCst),
            2,
            "promoted-mutator pool capped at 2"
        );
        assert_eq!(
            smut_max.load(Ordering::SeqCst),
            1,
            "default-mutator pool capped at 1"
        );
    }

    /// Mutator with side-effect: stores `true` into a shared flag after
    /// sleeping for `delay_ms`. Observable by a subsequent `FlagReader`
    /// in the same batch — but only if the executor sequences the Read
    /// after the Write.
    struct DelayedFlagSetter {
        flag: Arc<AtomicBool>,
        delay_ms: u64,
    }
    #[async_trait]
    impl Tool for DelayedFlagSetter {
        fn name(&self) -> &str {
            "set_flag"
        }
        fn description(&self) -> &str {
            "set"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        // class() defaults to Mutating.
        async fn execute(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.flag.store(true, Ordering::SeqCst);
            Ok(ToolOutput::text("set"))
        }
    }

    /// `ReadOnly` tool that observes a shared flag.
    struct FlagReader(Arc<AtomicBool>);
    #[async_trait]
    impl Tool for FlagReader {
        fn name(&self) -> &str {
            "read_flag"
        }
        fn description(&self) -> &str {
            "read"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        fn class(&self) -> ToolClass {
            ToolClass::ReadOnly
        }
        async fn execute(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            let v = self.0.load(Ordering::SeqCst);
            Ok(ToolOutput::text(if v { "true" } else { "false" }))
        }
    }

    #[tokio::test]
    async fn default_mutator_acts_as_barrier_against_subsequent_ro() {
        // Batch: [set_flag, read_flag]. The Mutator sleeps 100ms then
        // sets the flag. If the executor partitioned correctly, the
        // ReadOnly reads the flag *after* the Mutator finished and
        // observes "true". Without partitioning the Read would race
        // the Write and see "false" (the original P1 from Codex
        // review on PR #41).
        let flag = Arc::new(AtomicBool::new(false));
        let setter = Arc::new(DelayedFlagSetter {
            flag: Arc::clone(&flag),
            delay_ms: 100,
        });
        let reader = Arc::new(FlagReader(Arc::clone(&flag)));
        let reg = Arc::new(ToolRegistry::new(vec![setter, reader]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));

        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "set_flag".into(),
                input: json!({}),
            },
            ToolCall {
                id: "2".into(),
                name: "read_flag".into(),
                input: json!({}),
            },
        ];

        let results = exec.execute_batch(calls, &ctx()).await;
        assert_eq!(extract_text(&results[0]), "set");
        assert_eq!(
            extract_text(&results[1]),
            "true",
            "ReadOnly after default-Mutating must observe the mutation \
             — partition between class boundaries is the load-bearing invariant"
        );
    }

    /// Default-Mutating tool that records the wall-clock order in
    /// which `execute` was called. Used to assert that
    /// `execute_batch` admits adjacent `SerialMut` calls in their
    /// LLM-emitted input order, not in whatever order
    /// `FuturesUnordered` happens to poll them.
    struct OrderRecorder {
        label: &'static str,
        log: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }
    #[async_trait]
    impl Tool for OrderRecorder {
        fn name(&self) -> &str {
            self.label
        }
        fn description(&self) -> &str {
            "order recorder"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        // class() defaults to Mutating.
        async fn execute(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            // Sleep to widen the race window — without a delay, the
            // first future in `FuturesUnordered::next` always wins
            // trivially. With the sleep, a later future would
            // visibly race ahead if SerialMut runs were grouped
            // (the bug Codex flagged).
            tokio::time::sleep(Duration::from_millis(10)).await;
            self.log.lock().unwrap().push(self.label);
            Ok(ToolOutput::text(self.label))
        }
    }

    #[tokio::test]
    async fn adjacent_serial_mutators_execute_in_input_order() {
        // Codex P1: with adjacent default-Mutating calls in one
        // batch the executor must execute side effects in the LLM-
        // emitted order. The width-1 serial_mut semaphore alone
        // doesn't guarantee FIFO; `same_class_run_end` keeps
        // `SerialMut` runs length-1 to enforce strict input order.
        let log = Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
        let reg = Arc::new(ToolRegistry::new(vec![
            Arc::new(OrderRecorder {
                label: "first",
                log: Arc::clone(&log),
            }),
            Arc::new(OrderRecorder {
                label: "second",
                log: Arc::clone(&log),
            }),
            Arc::new(OrderRecorder {
                label: "third",
                log: Arc::clone(&log),
            }),
        ]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));

        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "first".into(),
                input: json!({}),
            },
            ToolCall {
                id: "2".into(),
                name: "second".into(),
                input: json!({}),
            },
            ToolCall {
                id: "3".into(),
                name: "third".into(),
                input: json!({}),
            },
        ];
        let _results = exec.execute_batch(calls, &ctx()).await;

        let observed = log.lock().unwrap().clone();
        assert_eq!(
            observed,
            vec!["first", "second", "third"],
            "adjacent SerialMut calls must execute side effects in input order"
        );
    }

    #[tokio::test]
    async fn promoted_mutator_acts_as_barrier_against_subsequent_ro() {
        // Same as above but the Mutator is promoted into the
        // concurrent-mutator pool. Promotion lifts the within-class cap
        // from 1 to N (default 10) — but does NOT relax the cross-class
        // ordering barrier. A subsequent ReadOnly still waits for the
        // promoted-Mutating run to complete.
        let flag = Arc::new(AtomicBool::new(false));
        let setter = Arc::new(DelayedFlagSetter {
            flag: Arc::clone(&flag),
            delay_ms: 100,
        });
        let reader = Arc::new(FlagReader(Arc::clone(&flag)));
        let reg = Arc::new(ToolRegistry::new(vec![setter, reader]));
        let cfg = ConcurrencyConfig::new(20, 10, vec![("set_flag".into(), ToolConcurrency::on())]);
        let exec = ToolExecutor::with_approval_and_concurrency(
            reg,
            Arc::new(AllowAll),
            Arc::new(AutoApprove),
            cfg,
        );

        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "set_flag".into(),
                input: json!({}),
            },
            ToolCall {
                id: "2".into(),
                name: "read_flag".into(),
                input: json!({}),
            },
        ];

        let results = exec.execute_batch(calls, &ctx()).await;
        assert_eq!(
            extract_text(&results[1]),
            "true",
            "ReadOnly after promoted-Mutating must observe the mutation"
        );
    }

    #[test]
    #[should_panic(expected = "max_concurrent_reads requires n > 0")]
    fn concurrency_config_panics_on_zero_read_cap() {
        let _ = ConcurrencyConfig::new(0, 10, std::iter::empty());
    }

    #[test]
    #[should_panic(expected = "max_concurrent_mutations requires n > 0")]
    fn concurrency_config_panics_on_zero_mutation_cap() {
        let _ = ConcurrencyConfig::new(20, 0, std::iter::empty());
    }

    #[test]
    #[should_panic(expected = "ToolConcurrency::max requires n > 0")]
    fn tool_concurrency_max_panics_on_zero() {
        let _ = ToolConcurrency::on().max(0);
    }
}
