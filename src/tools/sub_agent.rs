use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::Agent;
use crate::error::ToolError;
use crate::message::Message;
use crate::provider::LlmProvider;
use crate::tool::{Tool, ToolContext, ToolOutput};

/// Spawn a nested agent that reuses its parent's tool executor.
///
/// ## Design (Model 3)
///
/// `SubAgent` is **stateful**: it holds its own provider, model, and
/// per-run limits, captured at construction. The LLM can only set the
/// sub-agent's `prompt` and (optionally) override `system` per invocation
/// — `model`, `max_turns`, `max_tokens` etc. are fixed so the parent
/// agent can rely on predictable sub-agent behaviour.
///
/// At execute time the sub-agent builds a fresh [`Agent`] using the
/// **parent's [`ToolExecutor`] from [`ToolContext`]** rather than a
/// separately-configured toolset. This is the critical Model 3 property:
/// a child automatically inherits the parent's full registry — custom
/// tools, `WebFetch`, even another `SubAgent` instance — which enables
/// multi-level nesting up to `max_depth` without any explicit layering
/// or `Arc` cycles.
///
/// The child gets a cancellation token via [`tokio_util::sync::CancellationToken::child_token`],
/// so cancelling the parent cascades; and `ctx.depth + 1` is enforced
/// against `ctx.max_depth` to cap recursion.
pub struct SubAgent {
    provider: Arc<dyn LlmProvider>,
    model: String,
    system: Option<String>,
    max_turns: usize,
    max_tokens: u32,
    temperature: Option<f32>,
}

impl SubAgent {
    /// Construct a sub-agent tool. Provider and model are required; the
    /// rest are sensible defaults you can override with the builder-style
    /// setters.
    pub fn new(provider: Arc<dyn LlmProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            system: None,
            max_turns: 30,
            max_tokens: 4096,
            temperature: None,
        }
    }

    /// Default system prompt for the sub-agent. The LLM can override this
    /// per invocation via the `system` input field.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

#[async_trait]
impl Tool for SubAgent {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        "Spawn a sub-agent to handle a complex task autonomously. \
         The sub-agent gets its own conversation context and inherits \
         the parent agent's full tool set. Use this for tasks that \
         require multi-step reasoning or focused exploration."
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

    /// `SubAgent` is the canonical recursive tool — its `execute` body
    /// runs an entire nested `Agent::run` while the executor task that
    /// dispatched it stays parked. The executor uses this signal to
    /// route the call through the per-level-forked `concurrent_mut`
    /// pool even when the consumer hasn't explicitly promoted `agent`
    /// via `tool_concurrency`, avoiding the permanent stall that
    /// would otherwise arise from holding a shared `serial_mut`
    /// permit across the child's lifetime.
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

        // Fork the executor so the sub-agent runs with fresh
        // concurrency-permit accounting. Same registry, policy, and
        // approval — but independent semaphores. Without this fork,
        // a parent batch saturating `concurrent_mut` with promoted
        // `agent` calls would deadlock as soon as any child needed
        // its own promoted-mutator slot.
        let child_executor = ctx.executor.fork_for_subagent();

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

        let agent = builder.build();
        let child_cancel = ctx.cancel.child_token();
        let history = vec![Message::user_text(prompt)];

        match agent.run(history, child_cancel).await {
            Ok(result) => Ok(ToolOutput::text(result.text)),
            Err(e) => Ok(ToolOutput::error(format!("Sub-agent error: {e}"))),
        }
    }
}
