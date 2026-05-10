//! Anthropic Message Batches API — happy-path lifecycle.
//!
//! Submits 3 trivial requests, polls every 30 s until the batch ends,
//! then streams the results and asserts all 3 outcomes are `Succeeded`.
//!
//! **Latency.** Anthropic batches are async; even small batches
//! typically settle in 5-15 min. Run this example when you have time —
//! it will block on the polling loop until the server moves the batch
//! to `ended` (or hits the polling cap below).
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_batch

use std::time::Duration;

use tkach::providers::Anthropic;
use tkach::providers::anthropic::batch::{BatchOutcome, BatchRequest, BatchStatus};
use tkach::{Message, Request};

use futures::StreamExt;

const POLL_INTERVAL: Duration = Duration::from_secs(30);
const MAX_POLLS: u32 = 40; // 40 × 30 s = 20 min cap

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let provider = Anthropic::from_env();

    let prompts = ["Say hello.", "Count to three.", "Name one prime."];
    let requests: Vec<BatchRequest> = prompts
        .iter()
        .enumerate()
        .map(|(i, text)| BatchRequest {
            custom_id: format!("req-{}", i + 1),
            params: Request {
                model: tkach::model::claude::HAIKU_20251001.into(),
                system: None,
                messages: vec![Message::user_text(*text)],
                tools: vec![],
                max_tokens: 64,
                temperature: Some(0.0),
            },
        })
        .collect();

    println!("[batch] submitting {} requests", requests.len());
    let handle = provider.create_batch(requests).await?;
    println!(
        "[batch] submitted: {}, status={:?}",
        handle.id, handle.status
    );

    // Poll until ended.
    let mut current = handle.clone();
    for poll in 0..MAX_POLLS {
        if current.is_terminal() {
            break;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
        current = provider.retrieve_batch(&handle.id).await?;
        println!(
            "[batch] poll #{}: status={:?}, counts: processing={} succeeded={} errored={}",
            poll + 1,
            current.status,
            current.request_counts.processing,
            current.request_counts.succeeded,
            current.request_counts.errored,
        );
    }

    assert_eq!(
        current.status,
        BatchStatus::Ended,
        "batch did not reach Ended within {} polls ({:?} cap)",
        MAX_POLLS,
        POLL_INTERVAL * MAX_POLLS
    );

    // Drain results.
    let mut stream = provider.batch_results(&handle.id).await?;
    let mut succeeded = 0u32;
    while let Some(item) = stream.next().await {
        let r = item?;
        match r.outcome {
            BatchOutcome::Succeeded(resp) => {
                succeeded += 1;
                let text = resp
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        tkach::Content::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                println!(
                    "[batch] result {}: succeeded ({} in / {} out tokens) — {}",
                    r.custom_id, resp.usage.input_tokens, resp.usage.output_tokens, text
                );
                assert!(
                    resp.usage.input_tokens > 0,
                    "succeeded outcome must have input_tokens > 0"
                );
            }
            other => panic!(
                "expected all rows to succeed, got {other:?} for {}",
                r.custom_id
            ),
        }
    }

    assert_eq!(succeeded, 3, "expected 3 succeeded, got {succeeded}");
    println!("✅ all 3 requests succeeded");
    Ok(())
}
