use tkach::{Agent, CancellationToken, Message, providers::Anthropic, tools};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let agent = Agent::builder()
        .provider(Anthropic::from_env())
        .model(tkach::model::claude::SONNET)
        .system("You are a helpful coding assistant. Be concise.")
        .tools(tools::defaults())
        .build();

    let history = vec![Message::user_text(
        "What files are in the current directory? List them.",
    )];

    let result = agent.run(history, CancellationToken::new()).await?;

    println!("{}", result.text);
    println!(
        "\n[tokens: {} in / {} out]",
        result.usage.input_tokens, result.usage.output_tokens
    );

    Ok(())
}
