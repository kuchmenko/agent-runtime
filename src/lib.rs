//! # tkach
//!
//! A provider-independent agent runtime for Rust with built-in tools.
//!
//! The agent is stateless — callers own the message history and pass
//! it in on every call.
//!
//! ## Quick Start
//!
//! ```ignore
//! use tkach::{Agent, CancellationToken, Message, providers::Anthropic, tools};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let agent = Agent::builder()
//!         .provider(Anthropic::from_env())
//!         .model("claude-sonnet-4-6")
//!         .system("You are a helpful coding assistant.")
//!         .tools(tools::defaults())
//!         .build();
//!
//!     let mut history = vec![Message::user_text(
//!         "What files are in the current directory?",
//!     )];
//!     let result = agent.run(history.clone(), CancellationToken::new()).await?;
//!     history.extend(result.new_messages);
//!     println!("{}", result.text);
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod approval;
pub mod error;
pub mod executor;
pub mod message;
pub mod provider;
pub mod providers;
pub mod stream;
pub mod tool;
pub mod tools;

// Re-export core types at the crate root for convenience.
pub use agent::{Agent, AgentBuilder, AgentResult, AgentStream};
pub use approval::{ApprovalDecision, ApprovalHandler, AutoApprove};
pub use error::{AgentError, ProviderError, ToolError};
pub use executor::{
    AllowAll, ConcurrencyConfig, ToolCall, ToolConcurrency, ToolExecutor, ToolPolicy, ToolRegistry,
};
pub use message::{
    CacheControl, CacheTtl, Content, Message, Role, StopReason, ThinkingMetadata, ThinkingProvider,
    Usage,
};
pub use provider::{LlmProvider, Request, Response, SystemBlock, ToolDefinition};
pub use stream::{ProviderEventStream, StreamEvent};
pub use tokio_util::sync::CancellationToken;
pub use tool::{Tool, ToolClass, ToolContext, ToolOutput};
