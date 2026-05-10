//! Anthropic Batches — mixed-outcome handling.
//!
//! Submits 4 requests, one of which is intentionally malformed so the
//! server returns an `errored` outcome for it while the other 3
//! succeed. Demonstrates that per-row failures don't poison the batch
//! — they ride alongside successful outcomes in the result stream.
//!
//! Asserts: at least 1 `Errored`, at least 1 `Succeeded`, and the sum
//! of all outcome variants equals 4.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_batch_mixed

use std::time::Duration;

use tkach::providers::Anthropic;
use tkach::providers::anthropic::batch::{BatchOutcome, BatchRequest, BatchStatus};
use tkach::{Message, Request};

use futures::StreamExt;

const POLL_INTERVAL: Duration = Duration::from_secs(30);
const MAX_POLLS: u32 = 40;

fn ok_request(custom_id: &str, prompt: &str) -> BatchRequest {
    BatchRequest {
        custom_id: custom_id.into(),
        params: Request {
            model: tkach::model::claude::HAIKU_20251001.into(),
            system: None,
            messages: vec![Message::user_text(prompt)],
            tools: vec![],
            max_tokens: 64,
            temperature: Some(0.0),
        },
    }
}

/// Force a per-row `Errored` outcome by passing an invalid model name.
/// Anthropic rejects this row at run time without affecting siblings.
fn bad_request(custom_id: &str) -> BatchRequest {
    BatchRequest {
        custom_id: custom_id.into(),
        params: Request {
            model: "claude-does-not-exist-9999".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            max_tokens: 32,
            temperature: None,
        },
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let provider = Anthropic::from_env();

    let requests = vec![
        ok_request("req-1", "Say hello."),
        ok_request("req-2", "Name a colour."),
        bad_request("req-bad"),
        ok_request("req-3", "Say goodbye."),
    ];

    println!("[batch] submitting 4 requests (1 intentionally bad)");
    let handle = provider.create_batch(requests).await?;
    println!("[batch] submitted: {}", handle.id);

    let mut current = handle.clone();
    for poll in 0..MAX_POLLS {
        if current.is_terminal() {
            break;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
        current = provider.retrieve_batch(&handle.id).await?;
        println!(
            "[batch] poll #{}: status={:?}, succeeded={} errored={} processing={}",
            poll + 1,
            current.status,
            current.request_counts.succeeded,
            current.request_counts.errored,
            current.request_counts.processing,
        );
    }

    assert_eq!(
        current.status,
        BatchStatus::Ended,
        "batch did not reach Ended within polling window"
    );

    let mut stream = provider.batch_results(&handle.id).await?;
    let mut succeeded = 0u32;
    let mut errored = 0u32;
    let mut canceled = 0u32;
    let mut expired = 0u32;
    while let Some(item) = stream.next().await {
        let r = item?;
        match r.outcome {
            BatchOutcome::Succeeded(_) => {
                succeeded += 1;
                println!("[batch] result {}: succeeded", r.custom_id);
            }
            BatchOutcome::Errored {
                error_type,
                message,
            } => {
                errored += 1;
                println!(
                    "[batch] result {}: errored ({error_type}): {message}",
                    r.custom_id
                );
            }
            BatchOutcome::Canceled => canceled += 1,
            BatchOutcome::Expired => expired += 1,
        }
    }

    let total = succeeded + errored + canceled + expired;
    assert_eq!(total, 4, "expected 4 total outcomes, got {total}");
    assert!(
        succeeded >= 1,
        "expected at least 1 succeeded, got {succeeded}"
    );
    assert!(errored >= 1, "expected at least 1 errored, got {errored}");

    println!(
        "✅ {succeeded} succeeded, {errored} errored, {canceled} canceled, {expired} expired (per-row error isolation works)"
    );
    Ok(())
}
