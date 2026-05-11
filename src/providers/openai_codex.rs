//! OpenAI ChatGPT Codex subscription provider.
//!
//! Talks to the ChatGPT subscription Codex backend at
//! `https://chatgpt.com/backend-api/codex/responses`. The wire grammar
//! is the OpenAI Responses-style streaming protocol (same SSE event
//! shape as [`super::OpenAIResponses`]), so the SSE parser, input
//! mapping, and response decoding are shared via
//! [`super::openai_responses_proto`]. Only the HTTP envelope differs:
//! request URL, auth headers, request-body keys.
//!
//! ## Credentials are caller-owned
//!
//! `OpenAICodex` does **not** implement OAuth login, refresh-token
//! exchange, environment-variable lookup, keyring storage, or account
//! discovery. It only speaks HTTP + SSE.
//!
//! Pass a [`CodexCredentialsProvider`] (an async trait) and the
//! provider asks for fresh credentials before every request. If a
//! token is expired, your credential provider returns either a fresh
//! one or an error — `OpenAICodex` never tries to refresh internally.
//!
//! For short-lived callers with a static token, see
//! [`OpenAICodex::with_static_credentials`].
//!
//! ## Backend stability
//!
//! The Codex subscription backend is undocumented and unstable. Wire
//! shape and event names can change without notice. Production callers
//! should pin a `tkach` version they have validated end-to-end.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::{Value, json};

use super::openai_responses_proto::{self as proto, OpenAIEffort, OpenAISummary};
use crate::error::ProviderError;
use crate::message::{Content, StopReason, Usage};
use crate::provider::{LlmProvider, Request, Response, ThinkingConfig, ThinkingEffort};
use crate::stream::{ProviderEventStream, StreamEvent};

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
const DEFAULT_ORIGINATOR: &str = "tkach";
const DEFAULT_REASONING_SUMMARY: OpenAISummary = OpenAISummary::Auto;

/// Bearer credentials for the ChatGPT subscription Codex backend.
///
/// Both fields are caller-supplied. `OpenAICodex` does not store,
/// persist, or refresh them — they are passed through to a single
/// HTTP request and dropped.
///
/// `Debug` is manually implemented to redact `access_token`; the
/// `account_id` is shown because it is the public account identifier
/// already present in ChatGPT URLs.
#[derive(Clone)]
pub struct CodexCredentials {
    pub access_token: String,
    pub account_id: String,
}

impl CodexCredentials {
    pub fn new(access_token: impl Into<String>, account_id: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            account_id: account_id.into(),
        }
    }
}

impl fmt::Debug for CodexCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodexCredentials")
            .field("access_token", &"<redacted>")
            .field("account_id", &self.account_id)
            .finish()
    }
}

/// Caller-side credential source for [`OpenAICodex`].
///
/// Implementors return a fresh [`CodexCredentials`] before every
/// request. The provider does not cache; the trait method is the
/// single point where token freshness matters. A long-running consumer
/// typically wraps an OAuth client + cache here.
#[async_trait]
pub trait CodexCredentialsProvider: Send + Sync {
    async fn credentials(&self) -> Result<CodexCredentials, ProviderError>;
}

#[async_trait]
impl<F, Fut> CodexCredentialsProvider for F
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<CodexCredentials, ProviderError>> + Send,
{
    async fn credentials(&self) -> Result<CodexCredentials, ProviderError> {
        (self)().await
    }
}

struct StaticCredentials(CodexCredentials);

#[async_trait]
impl CodexCredentialsProvider for StaticCredentials {
    async fn credentials(&self) -> Result<CodexCredentials, ProviderError> {
        Ok(self.0.clone())
    }
}

/// Provider for the ChatGPT subscription Codex backend.
///
/// **Per-call thinking precedence:** same as
/// [`super::OpenAIResponses`]. [`crate::ThinkingConfig::Effort`]
/// overrides instance defaults set via [`Self::with_reasoning_effort`].
/// [`crate::ThinkingConfig::Disabled`] drops both effort and summary.
/// [`crate::ThinkingConfig::Budget`] is **Anthropic-style and silently
/// ignored** — the instance defaults apply.
pub struct OpenAICodex {
    credentials: Arc<dyn CodexCredentialsProvider>,
    client: reqwest::Client,
    base_url: String,
    originator: String,
    reasoning_effort: Option<OpenAIEffort>,
    reasoning_summary: Option<OpenAISummary>,
}

impl OpenAICodex {
    /// Build with a credential provider.
    pub fn new<P>(credentials: P) -> Self
    where
        P: CodexCredentialsProvider + 'static,
    {
        Self {
            credentials: Arc::new(credentials),
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            originator: DEFAULT_ORIGINATOR.to_string(),
            reasoning_effort: None,
            reasoning_summary: Some(DEFAULT_REASONING_SUMMARY),
        }
    }

    /// Build with a fixed credential pair. Use for tests, scripts, or
    /// short-lived callers where token refresh is not a concern.
    pub fn with_static_credentials(credentials: CodexCredentials) -> Self {
        Self::new(StaticCredentials(credentials))
    }

    /// Override the endpoint root. Default
    /// `https://chatgpt.com/backend-api`. The provider always appends
    /// `/codex/responses`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Override the `originator` request header. Default `tkach`.
    /// Apps embedding `tkach` may want to identify themselves
    /// (e.g. `sunny`).
    pub fn with_originator(mut self, originator: impl Into<String>) -> Self {
        self.originator = originator.into();
        self
    }

    /// Override the `reasoning.summary` field sent on every request.
    /// Default `"auto"`.
    ///
    /// The Codex backend does not emit `response.reasoning_summary_text.*`
    /// events unless `reasoning.summary` is set on the request.
    /// `include: ["reasoning.encrypted_content"]` alone gets you opaque
    /// replay state but no visible thinking text. Common values:
    /// `auto`, `concise`, `detailed`. OpenAI validates the exact set.
    pub fn with_reasoning_summary(mut self, summary: impl Into<OpenAISummary>) -> Self {
        self.reasoning_summary = Some(summary.into());
        self
    }

    /// Set `reasoning.effort` (Codex/Responses-style models). Typical
    /// values: `low`, `medium`, `high`. Off by default — the backend
    /// picks its own effort budget when this is unset.
    pub fn with_reasoning_effort(mut self, effort: impl Into<OpenAIEffort>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Stop sending `reasoning` on requests. The backend will not emit
    /// reasoning summary events; `include: ["reasoning.encrypted_content"]`
    /// still goes through, so opaque replay state continues to work.
    pub fn without_reasoning(mut self) -> Self {
        self.reasoning_effort = None;
        self.reasoning_summary = None;
        self
    }

    fn responses_url(&self) -> String {
        format!("{}/codex/responses", self.base_url.trim_end_matches('/'))
    }

    async fn send(&self, request: &Request) -> Result<reqwest::Response, ProviderError> {
        let body = build_request_body(
            request,
            effective_reasoning_effort(request, self.reasoning_effort.as_ref()).as_ref(),
            effective_reasoning_summary(request, self.reasoning_summary.as_ref()).as_ref(),
        );
        let credentials = self.credentials.credentials().await?;

        let response = self
            .client
            .post(self.responses_url())
            .bearer_auth(&credentials.access_token)
            .header("chatgpt-account-id", &credentials.account_id)
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", &self.originator)
            .header("accept", "text/event-stream")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();
        if status >= 400 {
            let retry_after_ms = proto::parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            return Err(proto::classify_error(status, text, retry_after_ms));
        }
        Ok(response)
    }
}

impl fmt::Debug for OpenAICodex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAICodex")
            .field("base_url", &self.base_url)
            .field("originator", &self.originator)
            .finish()
    }
}

#[async_trait]
impl LlmProvider for OpenAICodex {
    async fn stream(&self, request: Request) -> Result<ProviderEventStream, ProviderError> {
        let response = self.send(&request).await?;
        Ok(Box::pin(proto::responses_event_stream(
            response.bytes_stream().eventsource(),
        )))
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        // The Codex subscription backend has no documented non-streaming
        // endpoint, so collect from the stream. Accumulate text deltas,
        // pass through atomic ToolUse / ThinkingBlock events, and adopt
        // the terminal stop reason and usage.
        let mut stream = self.stream(request).await?;
        let mut content: Vec<Content> = Vec::new();
        let mut text_buf = String::new();
        let mut usage = Usage::default();
        let mut stop_reason = StopReason::EndTurn;

        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::ContentDelta(delta) => text_buf.push_str(&delta),
                StreamEvent::ThinkingDelta { .. } => {}
                StreamEvent::ThinkingBlock {
                    text,
                    provider,
                    metadata,
                } => {
                    flush_text(&mut text_buf, &mut content);
                    content.push(Content::Thinking {
                        text,
                        provider,
                        metadata,
                    });
                }
                StreamEvent::ToolUse { id, name, input } => {
                    flush_text(&mut text_buf, &mut content);
                    content.push(Content::ToolUse { id, name, input });
                }
                StreamEvent::ToolCallPending { .. } => {}
                StreamEvent::MessageDelta { stop_reason: sr } => stop_reason = sr,
                StreamEvent::Usage(u) => usage = u,
                StreamEvent::Done => break,
            }
        }
        flush_text(&mut text_buf, &mut content);

        Ok(Response {
            content,
            stop_reason,
            usage,
        })
    }
}

fn flush_text(buf: &mut String, content: &mut Vec<Content>) {
    if !buf.is_empty() {
        content.push(Content::text(std::mem::take(buf)));
    }
}

fn effective_reasoning_effort(
    request: &Request,
    instance: Option<&OpenAIEffort>,
) -> Option<OpenAIEffort> {
    match &request.thinking {
        Some(ThinkingConfig::Disabled) => None,
        Some(ThinkingConfig::Budget(_)) => instance.cloned(),
        Some(ThinkingConfig::Effort(effort)) => Some(map_thinking_effort(effort)),
        None => instance.cloned(),
    }
}

fn effective_reasoning_summary(
    request: &Request,
    instance: Option<&OpenAISummary>,
) -> Option<OpenAISummary> {
    match &request.thinking {
        Some(ThinkingConfig::Disabled) => None,
        Some(ThinkingConfig::Effort(_)) => Some(instance.cloned().unwrap_or(OpenAISummary::Auto)),
        Some(ThinkingConfig::Budget(_)) | None => instance.cloned(),
    }
}

fn map_thinking_effort(effort: &ThinkingEffort) -> OpenAIEffort {
    match effort {
        ThinkingEffort::Low => OpenAIEffort::Low,
        ThinkingEffort::Medium => OpenAIEffort::Medium,
        ThinkingEffort::High => OpenAIEffort::High,
        ThinkingEffort::Other(value) => OpenAIEffort::from(value.as_str()),
    }
}

fn build_request_body(
    request: &Request,
    reasoning_effort: Option<&OpenAIEffort>,
    reasoning_summary: Option<&OpenAISummary>,
) -> Value {
    let mut body = json!({
        "model": request.model,
        "store": false,
        "stream": true,
        "input": proto::build_input(&request.messages),
        "text": { "verbosity": "low" },
        "include": ["reasoning.encrypted_content"],
    });

    if let Some(instructions) = proto::instructions(request) {
        body["instructions"] = json!(instructions);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }

    if reasoning_effort.is_some() || reasoning_summary.is_some() {
        let mut reasoning = serde_json::Map::new();
        if let Some(effort) = reasoning_effort {
            reasoning.insert("effort".into(), json!(effort.as_wire()));
        }
        if let Some(summary) = reasoning_summary {
            reasoning.insert("summary".into(), json!(summary.as_wire()));
        }
        body["reasoning"] = Value::Object(reasoning);
    }

    let tools = proto::build_tools(&request.tools);
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
        body["tool_choice"] = json!("auto");
        body["parallel_tool_calls"] = json!(true);
    }

    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Content, Message};
    use crate::provider::{SystemBlock, ToolDefinition};

    fn sample_request() -> Request {
        Request {
            model: "gpt-5-codex".into(),
            system: Some(vec![SystemBlock::text("be brief")]),
            messages: vec![Message::user_text("hello")],
            tools: vec![ToolDefinition {
                name: "bash".into(),
                description: "shell".into(),
                input_schema: json!({"type":"object"}),
                cache_control: None,
            }],
            max_tokens: 256,
            temperature: Some(0.5),
            thinking: None,
        }
    }

    #[test]
    fn body_uses_codex_envelope() {
        let body = build_request_body(&sample_request(), None, Some(&OpenAISummary::Auto));

        assert_eq!(body["model"], "gpt-5-codex");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["text"]["verbosity"], "low");
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        assert!(
            body.get("max_output_tokens").is_none(),
            "Codex backend does not honor max_output_tokens; do not send it"
        );
        assert_eq!(body["instructions"], "be brief");
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "bash");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["input"][0]["role"], "user");
    }

    #[test]
    fn reasoning_summary_default_is_auto() {
        let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("t", "acct"));
        assert_eq!(
            provider.reasoning_summary.as_ref(),
            Some(&OpenAISummary::Auto)
        );
        assert!(provider.reasoning_effort.is_none());
    }

    #[test]
    fn body_emits_reasoning_summary_for_responses_api() {
        // Reproduces the bug Sunny found in real Codex traffic:
        // include=encrypted_content alone is not enough — the backend
        // only emits response.reasoning_summary_text.* when
        // reasoning.summary is set on the request.
        let body = build_request_body(&sample_request(), None, Some(&OpenAISummary::Auto));
        assert_eq!(body["reasoning"]["summary"], "auto");
        assert!(body["reasoning"].get("effort").is_none());
    }

    #[test]
    fn body_emits_both_effort_and_summary_when_set() {
        let body = build_request_body(
            &sample_request(),
            Some(&OpenAIEffort::Medium),
            Some(&OpenAISummary::Detailed),
        );
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert_eq!(body["reasoning"]["summary"], "detailed");
    }

    #[test]
    fn body_omits_reasoning_when_disabled() {
        let body = build_request_body(&sample_request(), None, None);
        assert!(body.get("reasoning").is_none());
        // include is independent of reasoning — encrypted replay state
        // still travels even when summary is disabled.
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
    }

    #[test]
    fn without_reasoning_clears_both_fields() {
        let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("t", "acct"))
            .with_reasoning_effort("high")
            .without_reasoning();
        assert!(provider.reasoning_effort.is_none());
        assert!(provider.reasoning_summary.is_none());
    }

    #[test]
    fn body_omits_tools_when_none() {
        let mut req = sample_request();
        req.tools.clear();
        let body = build_request_body(&req, None, Some(&OpenAISummary::Auto));
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn credentials_debug_redacts_access_token() {
        let creds = CodexCredentials::new("supersecret", "acct_123");
        let rendered = format!("{creds:?}");
        assert!(!rendered.contains("supersecret"));
        assert!(rendered.contains("<redacted>"));
        assert!(rendered.contains("acct_123"));
    }

    #[tokio::test]
    async fn static_credentials_returns_clone() {
        let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("token", "acct"));
        let creds = provider.credentials.credentials().await.unwrap();
        assert_eq!(creds.access_token, "token");
        assert_eq!(creds.account_id, "acct");
    }

    #[tokio::test]
    async fn closure_credential_provider_is_called_per_request() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_closure = Arc::clone(&calls);

        let provider = OpenAICodex::new(move || {
            let calls = Arc::clone(&calls_for_closure);
            async move {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                Ok(CodexCredentials::new(format!("token-{n}"), "acct"))
            }
        });

        let first = provider.credentials.credentials().await.unwrap();
        let second = provider.credentials.credentials().await.unwrap();
        assert_eq!(first.access_token, "token-0");
        assert_eq!(second.access_token, "token-1");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn debug_omits_credentials() {
        let provider =
            OpenAICodex::with_static_credentials(CodexCredentials::new("supersecret", "acct"));
        let rendered = format!("{provider:?}");
        assert!(!rendered.contains("supersecret"));
        assert!(rendered.contains("OpenAICodex"));
    }

    #[test]
    fn assistant_history_replays_function_call_id_split() {
        let req = Request {
            model: "gpt-5-codex".into(),
            system: None,
            messages: vec![
                Message::assistant(vec![Content::ToolUse {
                    id: "call_1|fc_1".into(),
                    name: "bash".into(),
                    input: json!({"command":"echo hi"}),
                }]),
                Message::user(vec![Content::tool_result("call_1|fc_1", "hi", false)]),
            ],
            tools: vec![],
            max_tokens: 128,
            temperature: None,
            thinking: None,
        };

        let body = build_request_body(&req, None, Some(&OpenAISummary::Auto));
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[0]["call_id"], "call_1");
        assert_eq!(input[0]["id"], "fc_1");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_1");

        // status is output-only on the Responses API; including it on
        // input items causes the backend to reject the request.
        assert!(input[0].get("status").is_none());
    }

    #[test]
    fn url_appends_codex_responses() {
        let provider = OpenAICodex::with_static_credentials(CodexCredentials::new("t", "a"))
            .with_base_url("http://localhost:1234/api/");
        assert_eq!(
            provider.responses_url(),
            "http://localhost:1234/api/codex/responses"
        );
    }
}

#[cfg(test)]
mod thinking_override_tests {
    //! Per-call `Request.thinking` precedence tests for the Codex
    //! provider. Mirrors the Anthropic/OpenAIResponses tests; covers
    //! issue #40 Phase 2 acceptance criteria.
    use super::*;
    use crate::message::Message;
    use crate::provider::{ThinkingConfig, ThinkingEffort};

    fn req_with(thinking: Option<ThinkingConfig>) -> Request {
        Request {
            model: "gpt-5-codex".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            max_tokens: 64,
            temperature: None,
            thinking,
        }
    }

    #[test]
    fn effort_high_emits_wire_reasoning_effort_high() {
        let r = req_with(Some(ThinkingConfig::Effort(ThinkingEffort::High)));
        let effort = effective_reasoning_effort(&r, None);
        let summary = effective_reasoning_summary(&r, None);
        let body = build_request_body(&r, effort.as_ref(), summary.as_ref());
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    #[test]
    fn per_call_effort_overrides_instance() {
        let r = req_with(Some(ThinkingConfig::Effort(ThinkingEffort::Low)));
        let effort = effective_reasoning_effort(&r, Some(&OpenAIEffort::High));
        let summary = effective_reasoning_summary(&r, Some(&OpenAISummary::Detailed));
        let body = build_request_body(&r, effort.as_ref(), summary.as_ref());
        assert_eq!(body["reasoning"]["effort"], "low");
        assert_eq!(body["reasoning"]["summary"], "detailed");
    }

    #[test]
    fn disabled_drops_reasoning_block() {
        let r = req_with(Some(ThinkingConfig::Disabled));
        let effort = effective_reasoning_effort(&r, Some(&OpenAIEffort::High));
        let summary = effective_reasoning_summary(&r, Some(&OpenAISummary::Auto));
        let body = build_request_body(&r, effort.as_ref(), summary.as_ref());
        assert!(
            body.get("reasoning").is_none(),
            "Disabled must drop reasoning entirely; got {body:?}"
        );
    }

    #[test]
    fn budget_falls_back_to_instance_silently() {
        // Same contract as OpenAIResponses: Budget is Anthropic-style;
        // Codex ignores and applies instance defaults.
        let r = req_with(Some(ThinkingConfig::Budget(8192)));
        let effort = effective_reasoning_effort(&r, Some(&OpenAIEffort::Medium));
        let summary = effective_reasoning_summary(&r, Some(&OpenAISummary::Auto));
        let body = build_request_body(&r, effort.as_ref(), summary.as_ref());
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert_eq!(body["reasoning"]["summary"], "auto");
    }

    #[test]
    fn other_effort_passes_through_verbatim() {
        let r = req_with(Some(ThinkingConfig::Effort(ThinkingEffort::Other(
            "xhigh".into(),
        ))));
        let effort = effective_reasoning_effort(&r, None);
        let summary = effective_reasoning_summary(&r, None);
        let body = build_request_body(&r, effort.as_ref(), summary.as_ref());
        assert_eq!(body["reasoning"]["effort"], "xhigh");
    }
}
