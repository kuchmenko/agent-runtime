//! HTTP-level coverage of the Anthropic Batches API path.
//!
//! Uses `wiremock` to stand up a local HTTP server, redirects an
//! `Anthropic` provider at it via `with_base_url(...)`, and exercises
//! the lifecycle: submit, retrieve, stream results, cancel, list.
//!
//! Real-API smoke runs live in `examples/anthropic_batch*.rs` —
//! `cargo run --example anthropic_batch` etc. — and require a real
//! `ANTHROPIC_API_KEY`. The tests here run on every `cargo test`.

use futures::StreamExt;
use serde_json::json;
use tkach::ProviderError;
use tkach::message::{Content, Message};
use tkach::provider::{Request, ThinkingConfig};
use tkach::providers::Anthropic;
use tkach::providers::anthropic::batch::{BatchOutcome, BatchRequest, BatchStatus};
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn req(custom_id: &str) -> BatchRequest {
    BatchRequest {
        custom_id: custom_id.into(),
        params: Request {
            model: "claude-haiku-4-5".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            max_tokens: 64,
            temperature: None,
            thinking: None,
        },
    }
}

fn handle_json(id: &str, status: &str, counts: serde_json::Value) -> serde_json::Value {
    json!({
        "id": id,
        "type": "message_batch",
        "processing_status": status,
        "request_counts": counts,
        "created_at": "2026-04-29T12:00:00Z",
        "ended_at": null,
        "expires_at": "2026-05-28T12:00:00Z",
        "archived_at": null,
        "cancel_initiated_at": null,
        "results_url": null,
    })
}

fn anthropic(server: &MockServer) -> Anthropic {
    Anthropic::new("test-key").with_base_url(server.uri())
}

// ---------------------------------------------------------------------------
// Submit / retrieve / results — happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_batch_posts_requests_and_parses_handle() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(handle_json(
            "msgbatch_01ABC",
            "in_progress",
            json!({"processing": 2, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let handle = client
        .create_batch(vec![req("req-1"), req("req-2")])
        .await
        .expect("submit ok");
    assert_eq!(handle.id, "msgbatch_01ABC");
    assert_eq!(handle.status, BatchStatus::InProgress);
    assert_eq!(handle.request_counts.processing, 2);
    assert!(!handle.is_terminal());
}

#[tokio::test]
async fn create_batch_resolves_absolute_image_files() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .respond_with(ResponseTemplate::new(200).set_body_json(handle_json(
            "msgbatch_01ABC",
            "in_progress",
            json!({"processing": 1, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let image_path = std::env::temp_dir().join(format!(
        "tkach-batch-image-{}-{}.png",
        std::process::id(),
        server.address().port()
    ));
    tokio::fs::write(&image_path, b"png").await.unwrap();

    let mut request = req("req-1");
    request.params.messages = vec![Message::user(vec![
        Content::text("before"),
        Content::image_file("image/png", &image_path),
        Content::text("after"),
    ])];
    anthropic(&server)
        .create_batch(vec![request])
        .await
        .expect("submit ok");
    tokio::fs::remove_file(image_path).await.unwrap();

    let received = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
    let content = body["requests"][0]["params"]["messages"][0]["content"]
        .as_array()
        .unwrap();
    assert_eq!(content[0]["text"], "before");
    assert_eq!(
        content[1]["source"],
        json!({"type":"base64","media_type":"image/png","data":"cG5n"})
    );
    assert_eq!(content[2]["text"], "after");
}

#[tokio::test]
async fn create_batch_honors_per_request_thinking_override() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .respond_with(ResponseTemplate::new(200).set_body_json(handle_json(
            "msgbatch_01ABC",
            "in_progress",
            json!({"processing": 1, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let mut request = req("req-1");
    request.params.thinking = Some(ThinkingConfig::Budget(2048));

    let client = anthropic(&server).with_thinking_budget(1024);
    client.create_batch(vec![request]).await.expect("submit ok");

    let received = &server.received_requests().await.unwrap()[0];
    let body: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
    let params = &body["requests"][0]["params"];
    assert_eq!(params["thinking"]["type"], "enabled");
    assert_eq!(params["thinking"]["budget_tokens"], 2048);
}

#[tokio::test]
async fn create_batch_rejects_invalid_custom_id_before_http() {
    // Server set up but we expect zero requests to land — bad input
    // must short-circuit on the client side.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let bad = vec![BatchRequest {
        custom_id: "bad id with spaces".into(),
        params: req("dummy").params,
    }];
    let err = client.create_batch(bad).await.expect_err("rejected");
    assert!(matches!(err, ProviderError::Other(s) if s.contains("invalid custom_id")));
}

#[tokio::test]
async fn create_batch_rejects_duplicate_custom_id_before_http() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let dup = vec![req("same-id"), req("same-id")];
    let err = client.create_batch(dup).await.expect_err("dup rejected");
    assert!(matches!(err, ProviderError::Other(s) if s.contains("duplicate custom_id")));
}

#[tokio::test]
async fn retrieve_batch_returns_updated_counts() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/msgbatch_01ABC"))
        .respond_with(ResponseTemplate::new(200).set_body_json(handle_json(
            "msgbatch_01ABC",
            "ended",
            json!({"processing": 0, "succeeded": 3, "errored": 0, "canceled": 0, "expired": 0}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let h = client.retrieve_batch("msgbatch_01ABC").await.unwrap();
    assert_eq!(h.status, BatchStatus::Ended);
    assert_eq!(h.request_counts.succeeded, 3);
    assert!(h.is_terminal());
}

#[tokio::test]
async fn batch_results_streams_jsonl_lines_into_outcomes() {
    let server = MockServer::start().await;

    let body = concat!(
        r#"{"custom_id":"req-1","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"ok-1"}],"stop_reason":"end_turn","usage":{"input_tokens":11,"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}}"#,
        "\n",
        r#"{"custom_id":"req-2","result":{"type":"errored","error":{"type":"invalid_request_error","message":"max_tokens too high"}}}"#,
        "\n",
        r#"{"custom_id":"req-3","result":{"type":"canceled"}}"#,
        "\n",
        r#"{"custom_id":"req-4","result":{"type":"expired"}}"#,
        "\n",
    );

    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/b1/results"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "application/x-jsonl"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let stream = client.batch_results("b1").await.expect("stream ok");
    let collected: Vec<_> = stream.collect().await;
    assert_eq!(collected.len(), 4);

    let r1 = collected[0].as_ref().expect("ok");
    assert_eq!(r1.custom_id, "req-1");
    match &r1.outcome {
        BatchOutcome::Succeeded(resp) => {
            assert_eq!(resp.usage.input_tokens, 11);
            assert_eq!(resp.usage.output_tokens, 3);
        }
        other => panic!("expected Succeeded got {other:?}"),
    }

    let r2 = collected[1].as_ref().unwrap();
    assert!(
        matches!(&r2.outcome, BatchOutcome::Errored { error_type, .. } if error_type == "invalid_request_error")
    );

    assert!(matches!(
        collected[2].as_ref().unwrap().outcome,
        BatchOutcome::Canceled
    ));
    assert!(matches!(
        collected[3].as_ref().unwrap().outcome,
        BatchOutcome::Expired
    ));
}

// ---------------------------------------------------------------------------
// Premature batch_results — must surface BatchNotReady, not a vanilla 4xx
// ---------------------------------------------------------------------------

#[tokio::test]
async fn premature_batch_results_surfaces_batch_not_ready() {
    let server = MockServer::start().await;

    // First the /results call returns 4xx (Anthropic does this for
    // not-yet-ended batches).
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/b1/results"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "results not ready"}
        })))
        .expect(1)
        .mount(&server)
        .await;

    // Then our promote_not_ready() probe re-fetches the handle to
    // distinguish in_progress from a real 400.
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/b1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(handle_json(
            "b1",
            "in_progress",
            json!({"processing": 1, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    // batch_results returns Result<impl Stream, _>; the Ok side isn't
    // Debug, so we match instead of using expect_err().
    let result = client.batch_results("b1").await;
    match result {
        Ok(_) => panic!("expected BatchNotReady, got Ok stream"),
        Err(ProviderError::BatchNotReady { status }) => {
            assert_eq!(status, "in_progress");
        }
        Err(other) => panic!("expected BatchNotReady, got {other:?}"),
    }
}

#[tokio::test]
async fn batch_results_429_surfaces_rate_limit_not_batch_not_ready() {
    // Regression: previously every non-2xx /results response was probed
    // for the in_progress/canceling state and promoted to BatchNotReady,
    // which masked retryable transport errors (429, 5xx) and discarded
    // retry_after_ms. 429 must pass through unchanged so callers back
    // off using the server's hint instead of polling aggressively.
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/b1/results"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "8")
                .set_body_string(r#"{"error":{"message":"slow down"}}"#),
        )
        .expect(1)
        .mount(&server)
        .await;

    // Critically: do NOT mount /v1/messages/batches/b1 (the handle
    // probe) — if the bug regresses, the test fails on missing mock.
    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/b1"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let result = client.batch_results("b1").await;
    match result {
        Ok(_) => panic!("expected RateLimit, got Ok stream"),
        Err(ProviderError::RateLimit { retry_after_ms }) => {
            assert_eq!(retry_after_ms, Some(8_000));
        }
        Err(ProviderError::BatchNotReady { .. }) => {
            panic!("BUG: 429 incorrectly promoted to BatchNotReady");
        }
        Err(other) => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn batch_results_5xx_surfaces_retryable_api_error_not_batch_not_ready() {
    // Same regression class as the 429 case: 502 must pass through
    // with retryable=true so callers backoff-and-retry, not flip into
    // not-ready polling.
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/b1/results"))
        .respond_with(ResponseTemplate::new(502).set_body_string("upstream"))
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/b1"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let result = client.batch_results("b1").await;
    match result {
        Ok(_) => panic!("expected Api(502), got Ok stream"),
        Err(ProviderError::Api {
            status, retryable, ..
        }) => {
            assert_eq!(status, 502);
            assert!(retryable);
        }
        Err(ProviderError::BatchNotReady { .. }) => {
            panic!("BUG: 502 incorrectly promoted to BatchNotReady");
        }
        Err(other) => panic!("expected Api, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Error classification on batch endpoints
// ---------------------------------------------------------------------------

#[tokio::test]
async fn submit_429_maps_to_rate_limit_with_retry_after() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages/batches"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "12")
                .set_body_string(r#"{"error":{"message":"slow down"}}"#),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let err = client
        .create_batch(vec![req("req-1")])
        .await
        .expect_err("rate limited");
    match err {
        ProviderError::RateLimit { retry_after_ms } => {
            assert_eq!(retry_after_ms, Some(12_000));
        }
        other => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn retrieve_5xx_maps_to_retryable_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/b1"))
        .respond_with(ResponseTemplate::new(502).set_body_string("upstream"))
        .expect(1)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let err = client.retrieve_batch("b1").await.expect_err("5xx");
    match err {
        ProviderError::Api {
            status, retryable, ..
        } => {
            assert_eq!(status, 502);
            assert!(retryable);
        }
        other => panic!("expected Api, got {other:?}"),
    }
}

#[tokio::test]
async fn malformed_jsonl_line_surfaces_deserialization_error_on_stream_item() {
    let server = MockServer::start().await;

    let body = concat!(
        r#"{"custom_id":"req-1","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}}"#,
        "\n",
        r#"{"custom_id": malformed_json"#,
        "\n",
        r#"{"custom_id":"req-3","result":{"type":"canceled"}}"#,
        "\n",
    );

    Mock::given(method("GET"))
        .and(path("/v1/messages/batches/b1/results"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(1)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let stream = client.batch_results("b1").await.unwrap();
    let collected: Vec<_> = stream.collect().await;
    assert_eq!(collected.len(), 3);

    assert!(collected[0].is_ok());
    match &collected[1] {
        Err(ProviderError::Deserialization(_)) => {}
        other => panic!("expected Deserialization, got {other:?}"),
    }
    assert!(collected[2].is_ok()); // stream continues past the bad line
}

// ---------------------------------------------------------------------------
// Cancel + list
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cancel_batch_returns_canceling_handle() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages/batches/b1/cancel"))
        .respond_with(ResponseTemplate::new(200).set_body_json(handle_json(
            "b1",
            "canceling",
            json!({"processing": 3, "succeeded": 1, "errored": 0, "canceled": 0, "expired": 0}),
        )))
        .expect(1)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let h = client.cancel_batch("b1").await.unwrap();
    assert_eq!(h.status, BatchStatus::Canceling);
}

#[tokio::test]
async fn list_batches_threads_pagination_query_params() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/messages/batches"))
        .and(query_param("limit", "5"))
        .and(query_param("after_id", "msgbatch_xxx"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                handle_json(
                    "msgbatch_aaa",
                    "ended",
                    json!({"processing": 0, "succeeded": 3, "errored": 0, "canceled": 0, "expired": 0}),
                ),
            ],
            "has_more": false,
            "first_id": "msgbatch_aaa",
            "last_id": "msgbatch_aaa",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = anthropic(&server);
    let opts = tkach::providers::anthropic::batch::ListBatchesOpts {
        limit: Some(5),
        before_id: None,
        after_id: Some("msgbatch_xxx".into()),
    };
    let list = client.list_batches(opts).await.unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].id, "msgbatch_aaa");
}
