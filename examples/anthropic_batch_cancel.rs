//! Anthropic Batches — cancel-then-fetch-partial.
//!
//! Submits 5 requests, calls `cancel_batch` after the first poll, then
//! waits for `status=Ended` and drains results. Cancel is best-effort:
//! rows the server already started running surface as `Succeeded`,
//! while still-queued rows surface as `Canceled`.
//!
//! Asserts: final status = Ended; ≥1 outcome is `Canceled`; total of
//! Succeeded + Canceled = 5.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_batch_cancel

use std::time::Duration;

use tkach::providers::Anthropic;
use tkach::providers::anthropic::batch::{BatchOutcome, BatchRequest, BatchStatus};
use tkach::{Message, Request};

use futures::StreamExt;

const POLL_INTERVAL: Duration = Duration::from_secs(15);
const MAX_POLLS: u32 = 40;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let provider = Anthropic::from_env();

    let requests: Vec<BatchRequest> = (1..=5)
        .map(|i| BatchRequest {
            custom_id: format!("req-{i}"),
            params: Request {
                model: tkach::model::claude::HAIKU_20251001.into(),
                system: None,
                messages: vec![Message::user_text(format!("Echo the number {i}."))],
                tools: vec![],
                max_tokens: 32,
                temperature: Some(0.0),
                thinking: None,
            },
        })
        .collect();

    println!("[batch] submitting 5 requests");
    let handle = provider.create_batch(requests).await?;
    println!("[batch] submitted: {}", handle.id);

    // Single poll for visibility, then cancel.
    tokio::time::sleep(Duration::from_secs(5)).await;
    let after_first = provider.retrieve_batch(&handle.id).await?;
    println!(
        "[batch] first poll: status={:?}, processing={}",
        after_first.status, after_first.request_counts.processing
    );

    println!("[batch] cancelling…");
    let cancelled = provider.cancel_batch(&handle.id).await?;
    println!("[batch] cancel ack: status={:?}", cancelled.status);

    // Wait for the cancellation to flush through to ended.
    let mut current = cancelled;
    for poll in 0..MAX_POLLS {
        if current.is_terminal() {
            break;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
        current = provider.retrieve_batch(&handle.id).await?;
        println!(
            "[batch] post-cancel poll #{}: status={:?}, succeeded={} canceled={}",
            poll + 1,
            current.status,
            current.request_counts.succeeded,
            current.request_counts.canceled,
        );
    }

    assert_eq!(
        current.status,
        BatchStatus::Ended,
        "batch did not reach Ended after cancel"
    );

    let mut stream = provider.batch_results(&handle.id).await?;
    let mut succeeded = 0u32;
    let mut canceled = 0u32;
    while let Some(item) = stream.next().await {
        let r = item?;
        match r.outcome {
            BatchOutcome::Succeeded(_) => {
                succeeded += 1;
                println!("[batch] result {}: succeeded", r.custom_id);
            }
            BatchOutcome::Canceled => {
                canceled += 1;
                println!("[batch] result {}: canceled", r.custom_id);
            }
            other => panic!("unexpected outcome for {}: {other:?}", r.custom_id),
        }
    }

    assert!(
        canceled >= 1,
        "expected at least 1 canceled, got {canceled}"
    );
    assert_eq!(
        succeeded + canceled,
        5,
        "expected 5 total outcomes, got {} succeeded + {} canceled",
        succeeded,
        canceled
    );
    println!(
        "✅ {} succeeded + {} canceled (cancellation honoured)",
        succeeded, canceled
    );
    Ok(())
}
