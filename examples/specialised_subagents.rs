use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use tkach::message::{Content, StopReason, Usage};
use tkach::provider::Response;
use tkach::providers::Mock;
use tkach::tools::SubAgent;
use tkach::{Agent, CancellationToken, Message, ThinkingConfig, ThinkingEffort};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let child = Arc::new(Mock::with_text("researched safely"));
    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);

    let parent = Mock::new(move |_req| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        match turn {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "research-1".into(),
                    name: "research".into(),
                    input: json!({"prompt": "Summarise the repository"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let research = SubAgent::new(child, "mock-child")
        .name("research")
        .description("Read-only repository research helper")
        .tools_allow(["read", "glob", "grep", "web_fetch"])
        .filter_tool_definitions(true)
        .thinking(ThinkingConfig::Effort(ThinkingEffort::High));

    let agent = Agent::builder()
        .provider(parent)
        .model("mock-parent")
        .tools(tkach::tools::defaults())
        .tool(research)
        .working_dir(std::env::current_dir()?)
        .build()?;

    let result = agent
        .run(
            vec![Message::user_text("delegate to research")],
            CancellationToken::new(),
        )
        .await?;

    assert_eq!(result.text, "done");
    assert_eq!(parent_turn.load(Ordering::SeqCst), 2);
    println!("specialised sub-agent completed: {}", result.text);
    Ok(())
}
