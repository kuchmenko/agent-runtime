use serde_json::{Value, json};
use tkach::{
    Agent, CancellationToken, Message, Tool, ToolContext, ToolError, ToolOutput, providers::Mock,
    tools,
};

/// Example: a custom tool that returns the current time.
struct CurrentTime;

#[async_trait::async_trait]
impl Tool for CurrentTime {
    fn name(&self) -> &str {
        "current_time"
    }

    fn description(&self) -> &str {
        "Returns the current UTC time"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        // In a real tool you'd use chrono or std::time
        Ok(ToolOutput::text("2026-04-04T12:00:00Z"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tkach::message::{Content, StopReason, Usage};
    use tkach::provider::Response;

    // Mock provider that first calls current_time, then responds with text
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "tool_1".into(),
                    name: "current_time".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("The current time is 2026-04-04T12:00:00Z")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("mock-model")
        .system("You are a helpful assistant.")
        .tool(CurrentTime)
        .tools(tools::defaults())
        .build()
        .unwrap();

    let history = vec![Message::user_text("What time is it?")];
    let result = agent.run(history, CancellationToken::new()).await?;
    println!("Agent response: {}", result.text);
    println!("Delta messages: {}", result.new_messages.len());

    Ok(())
}
