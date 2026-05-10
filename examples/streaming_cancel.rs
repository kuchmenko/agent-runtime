//! Cancel a streaming run mid-flight.
//!
//! Asks the model to stream a long answer, lets ~1500ms of tokens
//! arrive, then fires `cancel.cancel()`. The cooperative cancellation
//! contract should:
//!
//! 1. The provider stream's `bytes_stream` future is dropped → reqwest
//!    aborts the underlying TCP connection.
//! 2. `Agent::run_streaming_loop` notices either the dropped stream
//!    or `cancel.is_cancelled()` on its next checkpoint and exits.
//! 3. The terminal `into_result()` returns `AgentError::Cancelled`
//!    with a partial holding whatever tokens did arrive.
//!
//! End-to-end timing should be **well under** the time it would take
//! the model to finish — we cancel after 1500ms; a full 200-token
//! Haiku response is several seconds.
//!
//! Run:  `cargo run --example streaming_cancel`

use std::io::Write;
use std::time::{Duration, Instant};

use futures::StreamExt;
use tkach::{Agent, AgentError, CancellationToken, Message, StreamEvent, providers::Anthropic};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let agent = Agent::builder()
        .provider(Anthropic::from_env())
        .model(tkach::model::claude::HAIKU_20251001)
        .system("You are verbose and detailed.")
        .max_turns(1)
        .max_tokens(2048)
        .build();

    let cancel = CancellationToken::new();
    let cancel_handle = cancel.clone();

    // Cancel after 1500ms — long enough for a few tokens to land, short
    // enough that the full response cannot have completed.
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        eprintln!("\n\n[firing cancel after 1500ms]");
        cancel_handle.cancel();
    });

    let started = Instant::now();
    let mut stream = agent.stream(
        vec![Message::user_text(
            "Write a detailed 500-word essay on the history of the Rust programming \
             language. Include sections on its origins at Mozilla, the design goals, \
             notable milestones in its release history, the role of the Rust Foundation, \
             and its current state. Be thorough.",
        )],
        cancel,
    );

    print!("> ");
    std::io::stdout().flush()?;

    let mut delta_count = 0usize;
    let mut text_seen = String::new();

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::ContentDelta(text)) => {
                delta_count += 1;
                print!("{text}");
                text_seen.push_str(&text);
                std::io::stdout().flush()?;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("\n[stream error: {e}]");
                break;
            }
        }
    }
    println!();

    let outcome = stream.into_result().await;
    let elapsed = started.elapsed();

    eprintln!();
    eprintln!("--- summary ---");
    eprintln!("delta count : {delta_count}");
    eprintln!("text seen   : {} chars", text_seen.len());
    eprintln!("wall time   : {elapsed:?}");
    eprintln!(
        "outcome     : {}",
        match &outcome {
            Ok(_) => "Ok (unexpected — should have been cancelled)".to_string(),
            Err(AgentError::Cancelled { partial }) => format!(
                "Cancelled with partial.text len={} chars",
                partial.text.len()
            ),
            Err(other) => format!("Err: {other:?}"),
        }
    );
    eprintln!();

    // --- assertions ---

    let err = outcome.expect_err("expected cancellation, got Ok");
    let AgentError::Cancelled { partial } = &err else {
        panic!("expected AgentError::Cancelled, got {err:?}");
    };

    assert!(
        delta_count >= 1,
        "expected to receive at least one ContentDelta before cancel fired"
    );
    assert_eq!(
        partial.stop_reason,
        tkach::StopReason::Cancelled,
        "partial should carry StopReason::Cancelled"
    );
    // Whole 500-word essay would take many seconds; we should have
    // wrapped up well before then. 5s is generous but firmly under
    // what a complete response would take.
    assert!(
        elapsed < Duration::from_secs(5),
        "expected prompt cancellation, took {elapsed:?}"
    );
    // Text actually streamed before cancel — proves the live path
    // is real, not just a fast-path skip.
    assert!(
        !text_seen.is_empty(),
        "expected non-empty text before cancel, got empty"
    );

    eprintln!("✓ all assertions passed");
    Ok(())
}
