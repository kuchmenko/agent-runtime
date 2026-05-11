use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, TraceHook};
use crate::approval::ApprovalHandler;
use crate::error::ToolError;
use crate::message::Message;
use crate::policy::{AllowList, IntersectPolicy};
use crate::provider::{LlmProvider, ThinkingConfig};
use crate::stream::StreamEvent;
use crate::tool::{Tool, ToolContext, ToolOutput};

const DEFAULT_NAME: &str = "agent";
const DEFAULT_DESCRIPTION: &str = "Spawn a sub-agent to handle a complex task autonomously. The sub-agent gets its own conversation context and inherits the parent agent's full tool set. Use this for tasks that require multi-step reasoning or focused exploration.";

/// Spawn a nested agent that reuses its parent's tool registry.
pub struct SubAgent {
    provider: Arc<dyn LlmProvider>,
    model: String,
    system: Option<String>,
    max_turns: usize,
    max_tokens: u32,
    temperature: Option<f32>,
    name: String,
    description: String,
    tools_allow: Option<HashSet<String>>,
    filter_tool_definitions: bool,
    approval_handler: Option<Arc<dyn ApprovalHandler>>,
    trace_hook: Option<TraceHook>,
    thinking: Option<ThinkingConfig>,
}

impl SubAgent {
    /// Construct a sub-agent tool named `agent`.
    ///
    /// Register multiple specialised sub-agents by giving each one a unique
    /// [`name`](Self::name). `AgentBuilder::build` rejects duplicate tool names.
    ///
    /// **Token cost considerations.** Each registered SubAgent costs
    /// system-prompt tokens on the parent. Three distinct costs:
    ///
    /// 1. **Per-Task-invocation context loading** — ~20k tokens per
    ///    child invocation (measured by Amit Kothari, 2025-10-11,
    ///    `https://amitkoth.com/claude-code-task-tool-vs-subagents/`).
    /// 2. **Flat tool-use system-prompt overhead** — 346 tokens for
    ///    `auto`/`none` tool choice on Opus 4.7
    ///    (`https://platform.claude.com/docs/en/build-with-claude/tool-use/overview`).
    /// 3. **Per-subagent-description registration** — ~30 tokens × N,
    ///    measured at ~4500 tokens for 142 subagent descriptions
    ///    (`https://github.com/anthropics/claude-code/issues/18245`).
    ///
    /// Budget the specialised SubAgent count with all three in mind.
    pub fn new(provider: Arc<dyn LlmProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            system: None,
            max_turns: 30,
            max_tokens: 4096,
            temperature: None,
            name: DEFAULT_NAME.into(),
            description: DEFAULT_DESCRIPTION.into(),
            tools_allow: None,
            filter_tool_definitions: false,
            approval_handler: None,
            trace_hook: None,
            thinking: None,
        }
    }

    /// Override the registered tool name. Default is `"agent"`. Two
    /// SubAgents with the same `name()` cause
    /// [`crate::BuildError::DuplicateToolName`] at
    /// [`crate::AgentBuilder::build`] — call this once per specialised
    /// child to give the LLM a routing handle (e.g. `"research"`,
    /// `"writer"`).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the description the parent LLM sees. Default is the
    /// generic SubAgent text. Match the description to the child's
    /// intended profile so the parent routes appropriately.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Default system prompt for the sub-agent. The LLM can override this
    /// per invocation via the `system` input field.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Limit child loop turns. Default: 30.
    ///
    /// Recommended values vary by profile shape: tight reasoning loops
    /// typically use ~5; research fan-out ~20; multi-step edit loops
    /// 50+. The default suits research-style children; raise it for
    /// mutating writers and lower it for autonomous reasoning.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Limit child output tokens per provider call. Default: 4096.
    ///
    /// Match the budget to the profile: short reasoning answers fit in
    /// ~1024; research summaries in ~4096; long mutating edit plans in
    /// 8192+. Larger budgets cost latency and money even when unused.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Restrict the child to this tool allow-list, intersected with the
    /// parent policy. Empty means no tools are allowed; unset means inherit all.
    pub fn tools_allow<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tools_allow = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Also hide disallowed tool definitions from the child LLM request.
    ///
    /// Default `false` preserves stable prompt-cache hashes; denied calls still
    /// surface as tool-result errors through policy.
    pub fn filter_tool_definitions(mut self, on: bool) -> Self {
        self.filter_tool_definitions = on;
        self
    }

    /// Per-child approval handler. Replaces the parent's handler for
    /// this SubAgent's child loop only — the parent and other siblings
    /// keep their inherited handler.
    ///
    /// Use [`crate::AutoApprove`] for an autonomous reasoning child
    /// that should bypass the parent's interactive approval prompts.
    /// **By design this narrows the safety guarantee for this child's
    /// subtree** — register it deliberately, not as a convenience.
    pub fn approval_handler(mut self, handler: Arc<dyn ApprovalHandler>) -> Self {
        self.approval_handler = Some(handler);
        self
    }

    /// Per-event trace hook on the child agent's stream.
    ///
    /// When set, [`SubAgent::execute`] routes through `Agent::stream`
    /// instead of `Agent::run` and forwards every [`StreamEvent`] —
    /// including `MessageDelta`, `Usage`, and `Done` — to `hook`. When
    /// unset, the child uses the buffered `Agent::run` fast path with
    /// zero per-event overhead.
    ///
    /// Closure must be `Send + Sync + 'static`: capture state via
    /// `Arc`, not `&`-references. Hook panics are caught with
    /// [`std::panic::catch_unwind`] and logged via
    /// [`tracing::error!`]; the agent loop continues — a crashing
    /// audit sink does not crash the agent.
    ///
    /// Cognition AI's "Share full agent traces" principle: mutating
    /// SubAgents (`tools_allow` containing `edit`/`write`/`bash`)
    /// should set this hook so the parent has per-turn visibility into
    /// the child's decisions. Read-only profiles ship safely without
    /// it.
    pub fn trace_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(&StreamEvent) + Send + Sync + 'static,
    {
        self.trace_hook = Some(Arc::new(hook));
        self
    }

    /// Per-call thinking override for the child's provider request.
    /// Overrides the provider instance default (e.g.
    /// [`crate::providers::Anthropic::with_thinking_budget`]).
    /// One `Anthropic` provider instance can therefore serve many
    /// SubAgents with different thinking budgets without cloning the
    /// HTTP client. See [`ThinkingConfig`] for provider-asymmetry
    /// notes (Budget is Anthropic-style; OpenAI providers ignore it).
    pub fn thinking(mut self, config: ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    async fn run_with_trace(
        &self,
        agent: &Agent,
        history: Vec<Message>,
        cancel: CancellationToken,
        hook: TraceHook,
    ) -> Result<ToolOutput, ToolError> {
        // Hook fires inside `run_streaming_loop` for every event,
        // including the absorbed `MessageDelta` / `Usage` / `Done` and
        // the agent-emitted `ToolCallPending`. We just need to drive
        // the stream to completion. Using `collect_result` discards the
        // public-channel events without forwarding them — the hook has
        // already seen them.
        let stream = agent.stream_with_trace_hook(history, cancel, hook);
        match stream.collect_result().await {
            Ok(result) => Ok(ToolOutput::text(result.text)),
            Err(e) => Ok(ToolOutput::error(format!("Sub-agent error: {e}"))),
        }
    }
}

#[async_trait]
impl Tool for SubAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the sub-agent to perform"
                },
                "system": {
                    "type": "string",
                    "description": "System prompt for the sub-agent (optional, overrides the default)"
                }
            },
            "required": ["prompt"]
        })
    }

    fn is_recursive(&self) -> bool {
        true
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        if ctx.depth >= ctx.max_depth {
            return Ok(ToolOutput::error(format!(
                "Max sub-agent depth ({}) reached. Cannot spawn further sub-agents.",
                ctx.max_depth
            )));
        }

        let prompt = input["prompt"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("prompt is required".into()))?;

        let system_override = input["system"].as_str().map(String::from);
        let system = system_override.or_else(|| self.system.clone());

        let parent_policy = ctx.executor.policy_arc_for_fork();
        let policy_override = self.tools_allow.as_ref().map(|allow| {
            Arc::new(IntersectPolicy {
                left: Arc::clone(&parent_policy),
                right: Arc::new(AllowList::new(allow.iter().cloned())),
            }) as Arc<dyn crate::executor::ToolPolicy>
        });

        let child_executor = ctx
            .executor
            .fork_for_subagent_with(policy_override, self.approval_handler.clone());

        let mut builder = Agent::builder()
            .provider_arc(Arc::clone(&self.provider))
            .model(&*self.model)
            .executor(child_executor)
            .max_turns(self.max_turns)
            .max_tokens(self.max_tokens)
            .working_dir(&ctx.working_dir)
            .max_depth(ctx.max_depth)
            .depth(ctx.depth + 1);

        if let Some(sys) = system {
            builder = builder.system(sys);
        }
        if let Some(temp) = self.temperature {
            builder = builder.temperature(temp);
        }
        if let Some(thinking) = self.thinking.clone() {
            builder = builder.thinking(thinking);
        }
        if self.filter_tool_definitions {
            let visible = ctx
                .executor
                .registry()
                .iter()
                .map(|tool| tool.name())
                .filter(|name| parent_policy.is_allowed(name))
                .filter(|name| {
                    self.tools_allow
                        .as_ref()
                        .is_none_or(|allow| allow.contains(*name))
                })
                .map(ToString::to_string)
                .collect();
            builder = builder.tool_definition_filter(visible);
        }

        let agent = match builder.build() {
            Ok(agent) => agent,
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "Sub-agent configuration error: {e}"
                )));
            }
        };
        let child_cancel = ctx.cancel.child_token();
        let history = vec![Message::user_text(prompt)];

        match &self.trace_hook {
            Some(hook) => {
                self.run_with_trace(&agent, history, child_cancel, Arc::clone(hook))
                    .await
            }
            None => match agent.run(history, child_cancel).await {
                Ok(result) => Ok(ToolOutput::text(result.text)),
                Err(e) => Ok(ToolOutput::error(format!("Sub-agent error: {e}"))),
            },
        }
    }
}
