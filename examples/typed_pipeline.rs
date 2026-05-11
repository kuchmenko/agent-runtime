//! Typed model and thinking configuration without environment variables.
//!
//! This example is compile-time documentation: it shows the recommended typed
//! surface without making network calls.

use tkach::{
    Agent,
    model::{claude, gpt},
    providers::{OpenAIEffort, OpenAIResponses, OpenAISummary, anthropic::AnthropicEffort},
};

fn main() {
    let anthropic = tkach::providers::Anthropic::new("sk-ant-placeholder")
        .with_adaptive_thinking_effort(AnthropicEffort::High);

    let claude_agent = Agent::builder()
        .provider(anthropic)
        .model(claude::SONNET)
        .max_tokens(1024)
        .build()
        .unwrap();

    let openai = OpenAIResponses::new("sk-placeholder")
        .with_reasoning(OpenAIEffort::Medium, OpenAISummary::Detailed);

    let gpt_agent = Agent::builder()
        .provider(openai)
        .model(gpt::FIVE)
        .max_tokens(1024)
        .build()
        .unwrap();

    drop((claude_agent, gpt_agent));
}
