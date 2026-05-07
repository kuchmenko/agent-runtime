//! HTTP-level coverage of the `OpenAICodex` provider.
//!
//! Uses `wiremock` to assert request headers, request body shape, and
//! parser behavior over a real SSE response. Covers what the unit
//! tests in `src/providers/openai_codex.rs` cannot: that the provider
//! actually sends the headers and body it claims to, and that the
//! shared SSE parser is wired in.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use serde_json::{Value, json};
use tkach::ProviderError;
use tkach::message::{Message, StopReason};
use tkach::provider::{LlmProvider, Request, ToolDefinition};
use tkach::providers::{CodexCredentials, OpenAICodex};
use tkach::stream::StreamEvent;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request as MockRequest, ResponseTemplate};

fn build_request() -> Request {
    Request {
        model: "gpt-5-codex".into(),
        system: None,
        messages: vec![Message::user_text("ping")],
        tools: vec![ToolDefinition {
            name: "bash".into(),
            description: "shell".into(),
            input_schema: json!({"type":"object"}),
            cache_control: None,
        }],
        max_tokens: 256,
        temperature: None,
    }
}

fn ok_sse(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body)
}

#[tokio::test]
async fn stream_sends_codex_headers_and_body_shape() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/codex/responses"))
        .and(header("authorization", "Bearer secret-token"))
        .and(header("chatgpt-account-id", "acct_42"))
        .and(header("OpenAI-Beta", "responses=experimental"))
        .and(header("originator", "tkach"))
        .and(header("accept", "text/event-stream"))
        .and(header("content-type", "application/json"))
        .respond_with(ok_sse(
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":5}}}\n\n"
                .into(),
        ))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        OpenAICodex::with_static_credentials(CodexCredentials::new("secret-token", "acct_42"))
            .with_base_url(server.uri());

    let mut stream = provider.stream(build_request()).await.expect("stream");
    let mut saw_done = false;
    while let Some(event) = stream.next().await {
        if matches!(event.unwrap(), StreamEvent::Done) {
            saw_done = true;
        }
    }
    assert!(saw_done, "stream must end with Done");

    // Verify body shape on the captured request.
    let received = &server.received_requests().await.unwrap()[0];
    let body: Value = serde_json::from_slice(&received.body).unwrap();
    assert_eq!(body["model"], "gpt-5-codex");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["text"]["verbosity"], "low");
    assert_eq!(body["include"][0], "reasoning.encrypted_content");
    assert!(body.get("max_output_tokens").is_none());
    assert_eq!(body["tools"][0]["name"], "bash");
}

#[tokio::test]
async fn originator_header_can_be_overridden() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/codex/responses"))
        .and(header("originator", "sunny"))
        .respond_with(ok_sse(
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
                .into(),
        ))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("t", "acct_1"))
        .with_base_url(server.uri())
        .with_originator("sunny");

    let mut stream = provider.stream(build_request()).await.expect("stream");
    while stream.next().await.is_some() {}
}

#[tokio::test]
async fn credentials_provider_is_called_each_request() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/codex/responses"))
        .respond_with(|_req: &MockRequest| {
            ok_sse(
                "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
                    .into(),
            )
        })
        .mount(&server)
        .await;

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);

    let provider = OpenAICodex::new(move || {
        let counter = Arc::clone(&counter_clone);
        async move {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Ok(CodexCredentials::new(format!("token-{n}"), "acct"))
        }
    })
    .with_base_url(server.uri());

    for _ in 0..3 {
        let mut stream = provider.stream(build_request()).await.expect("stream");
        while stream.next().await.is_some() {}
    }
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn streaming_text_and_tool_calls_are_emitted() {
    let server = MockServer::start().await;

    let sse = String::new()
        + "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello \"}\n\n"
        + "data: {\"type\":\"response.output_text.delta\",\"delta\":\"world\"}\n\n"
        + "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"bash\",\"arguments\":\"\"}}\n\n"
        + "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n"
        + "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n"
        + "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}\n\n"
        + "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":4}}}\n\n";

    Mock::given(method("POST"))
        .and(path("/codex/responses"))
        .respond_with(ok_sse(sse))
        .mount(&server)
        .await;

    let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("t", "acct"))
        .with_base_url(server.uri());

    let mut stream = provider.stream(build_request()).await.expect("stream");
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut stop = StopReason::EndTurn;
    let mut input_tokens = 0;

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::ContentDelta(delta) => text.push_str(&delta),
            StreamEvent::ToolUse { id, name, input } => tool_calls.push((id, name, input)),
            StreamEvent::MessageDelta { stop_reason } => stop = stop_reason,
            StreamEvent::Usage(u) => input_tokens = u.input_tokens,
            StreamEvent::Done => break,
            _ => {}
        }
    }

    assert_eq!(text, "hello world");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].0, "call_1|fc_1");
    assert_eq!(tool_calls[0].1, "bash");
    assert_eq!(tool_calls[0].2["command"], "ls");
    assert_eq!(stop, StopReason::ToolUse);
    assert_eq!(input_tokens, 10);
}

#[tokio::test]
async fn complete_collects_stream_into_response() {
    let server = MockServer::start().await;

    let sse = String::new()
        + "data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n"
        + "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n\n";

    Mock::given(method("POST"))
        .and(path("/codex/responses"))
        .respond_with(ok_sse(sse))
        .mount(&server)
        .await;

    let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("t", "acct"))
        .with_base_url(server.uri());

    let response = provider.complete(build_request()).await.expect("complete");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.input_tokens, 2);
    assert_eq!(response.content.len(), 1);
    let text = match &response.content[0] {
        tkach::Content::Text { text, .. } => text.clone(),
        other => panic!("expected text content, got {other:?}"),
    };
    assert_eq!(text, "answer");
}

#[tokio::test]
async fn unauthorized_does_not_retry_internally() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/codex/responses"))
        .respond_with(ResponseTemplate::new(401).set_body_string("token expired"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("expired", "acct"))
        .with_base_url(server.uri());

    match provider.stream(build_request()).await {
        Ok(_) => panic!("must surface 401 before stream begins"),
        Err(ProviderError::Api {
            status, retryable, ..
        }) => {
            assert_eq!(status, 401);
            assert!(!retryable, "401 must be non-retryable");
        }
        Err(other) => panic!("expected ProviderError::Api, got {other:?}"),
    }
}

#[tokio::test]
async fn rate_limit_classification_uses_retry_after() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/codex/responses"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "7")
                .set_body_string("slow down"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("t", "acct"))
        .with_base_url(server.uri());

    match provider.stream(build_request()).await {
        Ok(_) => panic!("must 429 before stream begins"),
        Err(ProviderError::RateLimit { retry_after_ms }) => {
            assert_eq!(retry_after_ms, Some(7_000));
        }
        Err(other) => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn failed_response_event_surfaces_as_error() {
    let server = MockServer::start().await;

    let sse = String::new()
        + "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"model timeout\"}}}\n\n";

    Mock::given(method("POST"))
        .and(path("/codex/responses"))
        .respond_with(ok_sse(sse))
        .mount(&server)
        .await;

    let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("t", "acct"))
        .with_base_url(server.uri());

    let mut stream = provider.stream(build_request()).await.expect("stream");
    let mut saw_error = false;
    while let Some(event) = stream.next().await {
        if let Err(ProviderError::Api { message, .. }) = event {
            assert!(message.contains("model timeout"));
            saw_error = true;
        }
    }
    assert!(saw_error, "response.failed must surface as Err");
}

#[tokio::test]
async fn incomplete_max_tokens_maps_stop_reason() {
    let server = MockServer::start().await;

    let sse = String::new()
        + "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n"
        + "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n";

    Mock::given(method("POST"))
        .and(path("/codex/responses"))
        .respond_with(ok_sse(sse))
        .mount(&server)
        .await;

    let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("t", "acct"))
        .with_base_url(server.uri());

    let mut stream = provider.stream(build_request()).await.expect("stream");
    let mut stop = StopReason::EndTurn;
    while let Some(event) = stream.next().await {
        if let Ok(StreamEvent::MessageDelta { stop_reason }) = event {
            stop = stop_reason;
        }
    }
    assert_eq!(stop, StopReason::MaxTokens);
}
