use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::agent::Agent;
use crate::approval::ApprovalHandler;
use crate::error::ToolError;
use crate::message::Message;
use crate::policy::{AllowList, IntersectPolicy};
use crate::provider::{LlmProvider, ThinkingConfig};
use crate::stream::StreamEvent;
use crate::tool::{Tool, ToolContext, ToolOutput};

const DEFAULT_NAME: &str = "agent";
const DEFAULT_DESCRIPTION: &str = "Spawn a sub-agent to handle a complex task autonomously. The sub-agent gets its own conversation context and inherits the parent agent's full tool set. Use this for tasks that require multi-step reasoning or focused exploration.";

type TraceHook = Arc<dyn Fn(&StreamEvent) + Send + Sync>;

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

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

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
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Limit child output tokens per provider call. Default: 4096.
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

    pub fn approval_handler(mut self, handler: Arc<dyn ApprovalHandler>) -> Self {
        self.approval_handler = Some(handler);
        self
    }

    pub fn trace_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(&StreamEvent) + Send + Sync + 'static,
    {
        self.trace_hook = Some(Arc::new(hook));
        self
    }

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
        let mut stream = agent.stream(history, cancel);
        while let Some(event) = stream.next().await {
            if let Ok(ev) = event {
                emit_trace_event(&hook, &ev);
            }
        }

        match stream.into_result().await {
            Ok(result) => Ok(ToolOutput::text(result.text)),
            Err(e) => Ok(ToolOutput::error(format!("Sub-agent error: {e}"))),
        }
    }
}

fn emit_trace_event(hook: &TraceHook, ev: &StreamEvent) {
    if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (hook)(ev))) {
        tracing::error!(?panic, "trace_hook closure panicked; suppressed");
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
            if let Some(allow) = &self.tools_allow {
                let visible = allow
                    .iter()
                    .filter(|name| parent_policy.is_allowed(name))
                    .cloned()
                    .collect();
                builder = builder.tool_definition_filter(visible);
            }
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
