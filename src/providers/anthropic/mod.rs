use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ProviderError;
use crate::message::{
    CacheControl, Content, StopReason, ThinkingMetadata, ThinkingProvider, Usage,
};
use crate::provider::{
    LlmProvider, Request, Response, SystemBlock, ThinkingConfig, ThinkingEffort,
};
use crate::stream::{ProviderEventStream, StreamEvent};

pub mod batch;

/// Default Anthropic API base URL. Override via [`Anthropic::with_base_url`]
/// for testing against a mock server.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
pub(crate) const API_VERSION: &str = "2023-06-01";

/// Anthropic LLM provider (Claude).
pub struct Anthropic {
    api_key: String,
    client: reqwest::Client,
    base_url: String,
    thinking: Option<AnthropicThinkingConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
    Other(String),
}

impl AnthropicEffort {
    pub fn as_wire(&self) -> &str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
            Self::Other(value) => value.as_str(),
        }
    }
}

impl From<&str> for AnthropicEffort {
    fn from(value: &str) -> Self {
        if value.eq_ignore_ascii_case("low") {
            Self::Low
        } else if value.eq_ignore_ascii_case("medium") {
            Self::Medium
        } else if value.eq_ignore_ascii_case("high") {
            Self::High
        } else if value.eq_ignore_ascii_case("xhigh") {
            Self::XHigh
        } else if value.eq_ignore_ascii_case("max") {
            Self::Max
        } else {
            Self::Other(value.to_string())
        }
    }
}

impl From<String> for AnthropicEffort {
    fn from(value: String) -> Self {
        if value.eq_ignore_ascii_case("low") {
            Self::Low
        } else if value.eq_ignore_ascii_case("medium") {
            Self::Medium
        } else if value.eq_ignore_ascii_case("high") {
            Self::High
        } else if value.eq_ignore_ascii_case("xhigh") {
            Self::XHigh
        } else if value.eq_ignore_ascii_case("max") {
            Self::Max
        } else {
            Self::Other(value)
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum AnthropicThinkingConfig {
    Manual {
        budget_tokens: u32,
    },
    Adaptive {
        effort: Option<AnthropicEffort>,
        display: AnthropicThinkingDisplay,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AnthropicThinkingDisplay {
    Summarized,
}

impl AnthropicThinkingDisplay {
    fn as_wire_str(self) -> &'static str {
        match self {
            AnthropicThinkingDisplay::Summarized => "summarized",
        }
    }
}

impl Anthropic {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            thinking: None,
        }
    }

    /// Create from the `ANTHROPIC_API_KEY` environment variable.
    pub fn from_env() -> Self {
        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY env var is required");
        Self::new(api_key)
    }

    /// Override the API base URL. Pass the **scheme + host** (no
    /// trailing slash, no `/v1/...` path) — endpoints are appended
    /// internally. Primarily useful for routing tests through a local
    /// mock server (e.g. `wiremock`).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Enable Anthropic extended thinking with a manual token budget.
    ///
    /// Anthropic requires `budget_tokens >= 1024` and less than
    /// `max_tokens`; invalid combinations are rejected by the API. When
    /// enabled, tkach omits `temperature` because Anthropic marks it as
    /// incompatible with thinking.
    pub fn with_thinking_budget(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(AnthropicThinkingConfig::Manual { budget_tokens });
        self
    }

    /// Enable Anthropic adaptive thinking.
    ///
    /// Recommended by Anthropic for Claude Opus 4.7+, Opus 4.6, and
    /// Sonnet 4.6. Adaptive mode lets Claude decide whether and how much
    /// to think per request. tkach requests summarized display so
    /// streaming consumers can receive positive `ThinkingDelta` events
    /// on models whose API default is otherwise omitted thinking.
    pub fn with_adaptive_thinking(mut self) -> Self {
        self.thinking = Some(AnthropicThinkingConfig::Adaptive {
            effort: None,
            display: AnthropicThinkingDisplay::Summarized,
        });
        self
    }

    /// Enable adaptive thinking with explicit effort.
    ///
    /// Anthropic effort values are model-dependent; documented values are
    /// `low`, `medium`, `high`, `xhigh`, and `max`.
    pub fn with_adaptive_thinking_effort(mut self, effort: impl Into<AnthropicEffort>) -> Self {
        self.thinking = Some(AnthropicThinkingConfig::Adaptive {
            effort: Some(effort.into()),
            display: AnthropicThinkingDisplay::Summarized,
        });
        self
    }

    /// Endpoint URL for `complete` / `stream` (`{base}/v1/messages`).
    pub(crate) fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    /// Endpoint root for batch operations (`{base}/v1/messages/batches`).
    pub(crate) fn batches_url(&self) -> String {
        format!("{}/v1/messages/batches", self.base_url)
    }
}

#[async_trait]
impl LlmProvider for Anthropic {
    async fn stream(&self, request: Request) -> Result<ProviderEventStream, ProviderError> {
        let mut body = build_request_body_with_thinking(
            &request,
            effective_thinking(&request, self.thinking.as_ref()),
        );
        body.stream = true;

        let response = self
            .client
            .post(self.messages_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();

        if status >= 400 {
            // Pre-stream error: same classification as complete().
            let retry_after_ms = parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            return Err(classify_error(status, text, retry_after_ms));
        }

        // Stream OK; wrap the byte stream in an SSE parser then walk our
        // own state machine over Anthropic's event taxonomy.
        let event_stream = response.bytes_stream().eventsource();
        Ok(Box::pin(anthropic_event_stream(event_stream)))
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let body = build_request_body_with_thinking(
            &request,
            effective_thinking(&request, self.thinking.as_ref()),
        );

        let response = self
            .client
            .post(self.messages_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();

        if status >= 400 {
            let retry_after_ms = parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            return Err(classify_error(status, text, retry_after_ms));
        }

        let api_response: ApiResponse = response.json().await?;
        Ok(convert_response(api_response))
    }
}

/// Classify an Anthropic API error into a [`ProviderError`].
///
/// - 429 ⇒ `RateLimit`
/// - 529, 503 ⇒ `Overloaded` (529 is Anthropic-specific; 503 is generic
///   service unavailable — both are transient server-side pressure signals)
/// - 500, 502, 504 ⇒ retryable `Api`
/// - other 5xx ⇒ retryable `Api`, other 4xx ⇒ non-retryable `Api`
pub(crate) fn classify_error(
    status: u16,
    message: String,
    retry_after_ms: Option<u64>,
) -> ProviderError {
    match status {
        429 => ProviderError::RateLimit { retry_after_ms },
        529 | 503 => ProviderError::Overloaded { retry_after_ms },
        500 | 502 | 504 => ProviderError::Api {
            status,
            message,
            retryable: true,
        },
        s => ProviderError::Api {
            status: s,
            message,
            retryable: (500..600).contains(&s),
        },
    }
}

pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    // Spec allows either delay-seconds (integer) or HTTP-date. We only
    // parse the integer form — OpenAI/Anthropic both use seconds in practice.
    raw.trim().parse::<u64>().ok().map(|s| s * 1_000)
}

// --- Streaming SSE state machine ---
//
// Anthropic emits events in this lifecycle:
//
//   message_start            (carries initial input_tokens + cache fields
//                             in usage)
//   content_block_start[i]   (text or tool_use; tool_use has id, name)
//   content_block_delta[i] * (text_delta or input_json_delta fragments)
//   content_block_stop[i]    (close block; for tool_use we now emit
//                             one atomic StreamEvent::ToolUse with
//                             accumulated JSON parsed)
//   ... more blocks ...
//   message_delta            (final stop_reason + final output_tokens;
//                             cache fields are NOT re-sent here)
//   message_stop             (terminal)
//
// `ping` and unknown event types are ignored. `error` events surface
// as `Err(ProviderError::Other)` items in the stream and terminate it.
//
// Cache token merge: per Anthropic docs, cache_creation_input_tokens
// and cache_read_input_tokens arrive on `message_start` only. We
// remember them on the stream state and re-stamp every emitted Usage
// event with the running maximum, so a consumer that tracks "the
// latest Usage" never observes the cache fields collapse to 0 on
// message_delta. `merge_max` semantics are also exposed publicly via
// `Usage::merge_max`.

#[derive(Deserialize)]
#[serde(tag = "type")]
enum StreamingPayload {
    #[serde(rename = "message_start")]
    MessageStart {
        #[serde(default)]
        message: MessageStartPayload,
    },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockStart,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: BlockDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaPayload,
        #[serde(default)]
        usage: Option<ApiUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: ErrorPayload },
}

#[derive(Deserialize, Default)]
struct MessageStartPayload {
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlockStart {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum BlockDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Deserialize)]
struct MessageDeltaPayload {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ErrorPayload {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    message: String,
}

/// Per-block accumulation state. Text deltas are emitted live as
/// `ContentDelta`; tool_use blocks instead buffer `partial_json`
/// fragments and emit one atomic `ToolUse` event on `content_block_stop`.
enum BlockState {
    Text,
    Thinking {
        text_buf: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        json_buf: String,
    },
}

/// Streaming state carried across `unfold` polls. Holds the running
/// `Usage` so cache_creation/cache_read tokens observed on
/// `message_start` survive into `message_delta` and any subsequent
/// `Usage` emission.
struct StreamState<S> {
    sse: S,
    blocks: HashMap<usize, BlockState>,
    buffer: std::collections::VecDeque<Result<StreamEvent, ProviderError>>,
    /// Running merged `Usage` — re-stamped onto each emitted Usage event.
    usage: Usage,
}

fn anthropic_event_stream<S>(
    sse: S,
) -> impl futures::Stream<Item = Result<StreamEvent, ProviderError>>
where
    S: futures::Stream<
            Item = Result<
                eventsource_stream::Event,
                eventsource_stream::EventStreamError<reqwest::Error>,
            >,
        > + Send
        + Unpin
        + 'static,
{
    use std::collections::VecDeque;

    let initial = StreamState {
        sse,
        blocks: HashMap::new(),
        buffer: VecDeque::new(),
        usage: Usage::default(),
    };

    futures::stream::unfold(initial, |mut state| async move {
        // Drain any pending events first — one SSE event can produce
        // multiple StreamEvents (e.g. message_delta → MessageDelta + Usage).
        loop {
            if let Some(ev) = state.buffer.pop_front() {
                return Some((ev, state));
            }

            let next = state.sse.next().await?;
            let event = match next {
                Ok(ev) => ev,
                Err(e) => {
                    let err = ProviderError::Other(format!("SSE read error: {e}"));
                    return Some((Err(err), state));
                }
            };

            // Ignore unknown event types (ping etc. that we choose not to
            // type — eventsource-stream gives us them with .event set).
            // Real Anthropic events all have payload.type matching event.event.
            let payload: StreamingPayload = match serde_json::from_str(&event.data) {
                Ok(p) => p,
                Err(_) => continue, // unknown / malformed payload, skip
            };

            process_payload(
                payload,
                &mut state.blocks,
                &mut state.buffer,
                &mut state.usage,
            );
        }
    })
}

fn process_payload(
    payload: StreamingPayload,
    blocks: &mut HashMap<usize, BlockState>,
    buffer: &mut std::collections::VecDeque<Result<StreamEvent, ProviderError>>,
    running: &mut Usage,
) {
    match payload {
        StreamingPayload::MessageStart { message } => {
            if let Some(usage) = message.usage {
                running.merge_max(&usage_from_api(&usage));
                buffer.push_back(Ok(StreamEvent::Usage(running.clone())));
            }
        }
        StreamingPayload::ContentBlockStart {
            index,
            content_block,
        } => {
            let state = match content_block {
                ContentBlockStart::Text => BlockState::Text,
                ContentBlockStart::Thinking {
                    thinking,
                    signature,
                } => BlockState::Thinking {
                    text_buf: thinking,
                    signature,
                },
                ContentBlockStart::RedactedThinking { data } => {
                    BlockState::RedactedThinking { data }
                }
                ContentBlockStart::ToolUse { id, name, input } => BlockState::ToolUse {
                    id,
                    name,
                    // Some Anthropic responses ship a partial input on start;
                    // serialize it back as JSON to seed the buffer so later
                    // input_json_delta concatenation yields a valid object.
                    json_buf: if input.is_null() || input == Value::Object(Default::default()) {
                        String::new()
                    } else {
                        serde_json::to_string(&input).unwrap_or_default()
                    },
                },
            };
            blocks.insert(index, state);
        }
        StreamingPayload::ContentBlockDelta { index, delta } => match delta {
            BlockDelta::Text { text } => {
                buffer.push_back(Ok(StreamEvent::ContentDelta(text)));
            }
            BlockDelta::Thinking { thinking } => {
                if let Some(BlockState::Thinking { text_buf, .. }) = blocks.get_mut(&index) {
                    text_buf.push_str(&thinking);
                }
                buffer.push_back(Ok(StreamEvent::ThinkingDelta { text: thinking }));
            }
            BlockDelta::Signature { signature } => {
                if let Some(BlockState::Thinking { signature: sig, .. }) = blocks.get_mut(&index) {
                    sig.push_str(&signature);
                }
            }
            BlockDelta::InputJson { partial_json } => {
                if let Some(BlockState::ToolUse { json_buf, .. }) = blocks.get_mut(&index) {
                    json_buf.push_str(&partial_json);
                }
            }
        },
        StreamingPayload::ContentBlockStop { index } => {
            if let Some(block) = blocks.remove(&index) {
                match block {
                    BlockState::Text => {}
                    BlockState::Thinking {
                        text_buf,
                        signature,
                    } => {
                        buffer.push_back(Ok(StreamEvent::ThinkingBlock {
                            text: text_buf,
                            provider: ThinkingProvider::Anthropic,
                            metadata: ThinkingMetadata::Anthropic {
                                signature: (!signature.is_empty()).then_some(signature),
                            },
                        }));
                    }
                    BlockState::RedactedThinking { data } => {
                        buffer.push_back(Ok(StreamEvent::ThinkingBlock {
                            text: String::new(),
                            provider: ThinkingProvider::Anthropic,
                            metadata: ThinkingMetadata::AnthropicRedacted { data },
                        }));
                    }
                    BlockState::ToolUse { id, name, json_buf } => {
                        let input: Value = if json_buf.trim().is_empty() {
                            Value::Object(Default::default())
                        } else {
                            serde_json::from_str(&json_buf)
                                .unwrap_or_else(|_| Value::String(json_buf.clone()))
                        };
                        buffer.push_back(Ok(StreamEvent::ToolUse { id, name, input }));
                    }
                }
            }
        }
        StreamingPayload::MessageDelta { delta, usage } => {
            if let Some(stop) = delta.stop_reason {
                let stop_reason = map_stop_reason(&stop);
                buffer.push_back(Ok(StreamEvent::MessageDelta { stop_reason }));
            }
            if let Some(u) = usage {
                running.merge_max(&usage_from_api(&u));
                buffer.push_back(Ok(StreamEvent::Usage(running.clone())));
            }
        }
        StreamingPayload::MessageStop => {
            buffer.push_back(Ok(StreamEvent::Done));
        }
        StreamingPayload::Ping => {
            // Anthropic uses ping to keep the connection alive; nothing to emit.
        }
        StreamingPayload::Error { error } => {
            buffer.push_back(Err(ProviderError::Other(format!(
                "anthropic stream error ({}): {}",
                error.kind, error.message
            ))));
        }
    }
}

fn map_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        "pause_turn" => StopReason::PauseTurn,
        _ => StopReason::EndTurn,
    }
}

pub(crate) fn usage_from_api(api: &ApiUsage) -> Usage {
    Usage {
        input_tokens: api.input_tokens,
        output_tokens: api.output_tokens,
        cache_creation_input_tokens: api.cache_creation_input_tokens,
        cache_read_input_tokens: api.cache_read_input_tokens,
    }
}

// --- Anthropic API types ---

#[derive(Serialize)]
pub(crate) struct ApiRequest {
    pub(crate) model: String,
    pub(crate) max_tokens: u32,
    pub(crate) messages: Vec<ApiMessage>,
    /// Typed system blocks (Anthropic accepts either a free string or
    /// an array of typed blocks; we always emit the array form so
    /// `cache_control` works).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system: Option<Vec<ApiSystemBlock>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thinking: Option<ApiThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_config: Option<ApiOutputConfig>,
    /// `stream: true` switches the response to SSE; default false for
    /// `complete()`. Always false on the batch path.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) stream: bool,
}

#[derive(Serialize)]
pub(crate) struct ApiThinkingConfig {
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) budget_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) display: Option<&'static str>,
}

#[derive(Serialize)]
pub(crate) struct ApiOutputConfig {
    pub(crate) effort: String,
}

#[derive(Serialize)]
pub(crate) struct ApiMessage {
    pub(crate) role: String,
    pub(crate) content: Vec<ApiContent>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub(crate) enum ApiContent {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },

    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[derive(Serialize)]
pub(crate) struct ApiSystemBlock {
    /// Anthropic's typed system block has `type: "text"`.
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
pub(crate) struct ApiTool {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

#[derive(Deserialize)]
pub(crate) struct ApiResponse {
    pub(crate) content: Vec<ApiContent>,
    pub(crate) stop_reason: String,
    pub(crate) usage: ApiUsage,
}

#[derive(Deserialize)]
pub(crate) struct ApiUsage {
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    #[serde(default)]
    pub(crate) cache_creation_input_tokens: u32,
    #[serde(default)]
    pub(crate) cache_read_input_tokens: u32,
}

// --- Conversion ---

#[cfg(test)]
pub(crate) fn build_request_body(request: &Request) -> ApiRequest {
    build_request_body_with_thinking(request, None)
}

pub(crate) fn build_request_body_with_thinking(
    request: &Request,
    thinking: Option<AnthropicThinkingConfig>,
) -> ApiRequest {
    let messages = request.messages.iter().filter_map(message_to_api).collect();

    let tools = request
        .tools
        .iter()
        .map(|t| ApiTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
            cache_control: t.cache_control.clone(),
        })
        .collect();

    let api_thinking = thinking.as_ref().map(api_thinking_config);
    let output_config = thinking.as_ref().and_then(output_config);

    ApiRequest {
        model: request.model.clone(),
        max_tokens: request.max_tokens,
        messages,
        system: request.system.as_ref().map(|blocks| {
            blocks
                .iter()
                .map(|b: &SystemBlock| ApiSystemBlock {
                    kind: "text",
                    text: b.text.clone(),
                    cache_control: b.cache_control.clone(),
                })
                .collect()
        }),
        tools,
        temperature: thinking.is_none().then_some(request.temperature).flatten(),
        thinking: api_thinking,
        output_config,
        stream: false,
    }
}

fn effective_thinking(
    request: &Request,
    instance: Option<&AnthropicThinkingConfig>,
) -> Option<AnthropicThinkingConfig> {
    match &request.thinking {
        Some(ThinkingConfig::Disabled) => None,
        Some(ThinkingConfig::Budget(budget_tokens)) => Some(AnthropicThinkingConfig::Manual {
            budget_tokens: *budget_tokens,
        }),
        Some(ThinkingConfig::Effort(effort)) => Some(AnthropicThinkingConfig::Adaptive {
            effort: Some(map_thinking_effort(effort)),
            display: AnthropicThinkingDisplay::Summarized,
        }),
        None => instance.cloned(),
    }
}

fn map_thinking_effort(effort: &ThinkingEffort) -> AnthropicEffort {
    match effort {
        ThinkingEffort::Low => AnthropicEffort::Low,
        ThinkingEffort::Medium => AnthropicEffort::Medium,
        ThinkingEffort::High => AnthropicEffort::High,
        ThinkingEffort::Other(value) => AnthropicEffort::from(value.as_str()),
    }
}

fn api_thinking_config(config: &AnthropicThinkingConfig) -> ApiThinkingConfig {
    match config {
        AnthropicThinkingConfig::Manual { budget_tokens } => ApiThinkingConfig {
            kind: "enabled",
            budget_tokens: Some(*budget_tokens),
            display: None,
        },
        AnthropicThinkingConfig::Adaptive { display, .. } => ApiThinkingConfig {
            kind: "adaptive",
            budget_tokens: None,
            display: Some(display.as_wire_str()),
        },
    }
}

fn output_config(config: &AnthropicThinkingConfig) -> Option<ApiOutputConfig> {
    match config {
        AnthropicThinkingConfig::Adaptive {
            effort: Some(effort),
            ..
        } => Some(ApiOutputConfig {
            effort: effort.as_wire().to_string(),
        }),
        AnthropicThinkingConfig::Manual { .. }
        | AnthropicThinkingConfig::Adaptive { effort: None, .. } => None,
    }
}

fn message_to_api(msg: &crate::message::Message) -> Option<ApiMessage> {
    let content: Vec<ApiContent> = msg.content.iter().filter_map(content_to_api).collect();
    if content.is_empty() {
        return None;
    }

    Some(ApiMessage {
        role: match msg.role {
            crate::message::Role::User => "user".to_string(),
            crate::message::Role::Assistant => "assistant".to_string(),
        },
        content,
    })
}

fn content_to_api(content: &Content) -> Option<ApiContent> {
    match content {
        Content::Text {
            text,
            cache_control,
        } => Some(ApiContent::Text {
            text: text.clone(),
            cache_control: cache_control.clone(),
        }),
        Content::Thinking {
            text,
            provider: ThinkingProvider::Anthropic,
            metadata: ThinkingMetadata::Anthropic { signature },
        } => Some(ApiContent::Thinking {
            thinking: text.clone(),
            signature: signature.clone(),
        }),
        Content::Thinking {
            provider: ThinkingProvider::Anthropic,
            metadata: ThinkingMetadata::AnthropicRedacted { data },
            ..
        } => Some(ApiContent::RedactedThinking { data: data.clone() }),
        Content::Thinking { .. } => None,
        Content::ToolUse { id, name, input } => Some(ApiContent::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        }),
        Content::ToolResult {
            tool_use_id,
            content,
            is_error,
            cache_control,
        } => Some(ApiContent::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: *is_error,
            cache_control: cache_control.clone(),
        }),
    }
}

pub(crate) fn convert_response(api: ApiResponse) -> Response {
    let content = api
        .content
        .into_iter()
        .map(|c| match c {
            ApiContent::Text {
                text,
                cache_control,
            } => Content::Text {
                text,
                cache_control,
            },
            ApiContent::Thinking {
                thinking,
                signature,
            } => Content::Thinking {
                text: thinking,
                provider: ThinkingProvider::Anthropic,
                metadata: ThinkingMetadata::Anthropic { signature },
            },
            ApiContent::RedactedThinking { data } => Content::Thinking {
                text: String::new(),
                provider: ThinkingProvider::Anthropic,
                metadata: ThinkingMetadata::AnthropicRedacted { data },
            },
            ApiContent::ToolUse { id, name, input } => Content::ToolUse { id, name, input },
            ApiContent::ToolResult {
                tool_use_id,
                content,
                is_error,
                cache_control,
            } => Content::ToolResult {
                tool_use_id,
                content,
                is_error,
                cache_control,
            },
        })
        .collect();

    let stop_reason = map_stop_reason(&api.stop_reason);

    Response {
        content,
        stop_reason,
        usage: usage_from_api(&api.usage),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{CacheTtl, Message};
    use crate::provider::ToolDefinition;
    use serde_json::json;

    fn req_with_system(blocks: Vec<SystemBlock>) -> Request {
        Request {
            model: "claude-test".into(),
            system: Some(blocks),
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            max_tokens: 100,
            temperature: None,
            thinking: None,
        }
    }

    fn system_blocks_json(blocks: Vec<SystemBlock>) -> serde_json::Value {
        let req = req_with_system(blocks);
        let body = build_request_body(&req);
        serde_json::to_value(&body).unwrap()
    }

    #[test]
    fn anthropic_effort_variants_match_wire_names() {
        assert_eq!(AnthropicEffort::Low.as_wire(), "low");
        assert_eq!(AnthropicEffort::Medium.as_wire(), "medium");
        assert_eq!(AnthropicEffort::High.as_wire(), "high");
        assert_eq!(AnthropicEffort::XHigh.as_wire(), "xhigh");
        assert_eq!(AnthropicEffort::Max.as_wire(), "max");
        assert_eq!(AnthropicEffort::Other("ultra".into()).as_wire(), "ultra");
    }

    #[test]
    fn anthropic_effort_from_string_is_case_insensitive_and_preserves_other() {
        assert_eq!(AnthropicEffort::from("XHIGH"), AnthropicEffort::XHigh);
        assert_eq!(AnthropicEffort::from("MAX"), AnthropicEffort::Max);
        assert_eq!(
            AnthropicEffort::from("ultra"),
            AnthropicEffort::Other("ultra".into())
        );
        assert_eq!(
            AnthropicEffort::from(String::from("ULTRA")),
            AnthropicEffort::Other("ULTRA".into())
        );
    }

    #[test]
    fn system_blocks_serialize_as_typed_array() {
        let json = system_blocks_json(vec![
            SystemBlock::text("base instructions"),
            SystemBlock::cached("long stable context"),
        ]);
        let arr = json["system"].as_array().expect("system should be array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[1]["type"], "text");
    }

    #[test]
    fn system_blocks_serialize_text_payloads() {
        let json = system_blocks_json(vec![
            SystemBlock::text("base instructions"),
            SystemBlock::cached("long stable context"),
        ]);
        assert_eq!(json["system"][0]["text"], "base instructions");
        assert_eq!(json["system"][1]["text"], "long stable context");
    }

    #[test]
    fn system_blocks_serialize_cache_control_only_when_set() {
        let json = system_blocks_json(vec![
            SystemBlock::text("base instructions"),
            SystemBlock::cached("long stable context"),
        ]);
        assert!(json["system"][0].get("cache_control").is_none());
        assert_eq!(json["system"][1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn system_blocks_with_one_hour_ttl_serialize_inline() {
        let req = req_with_system(vec![SystemBlock::cached_1h("long-lived prefix")]);
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let cc = &json["system"][0]["cache_control"];
        assert_eq!(cc["type"], "ephemeral");
        assert_eq!(cc["ttl"], "1h");
    }

    #[test]
    fn tool_definition_cache_control_threads_to_api_tool() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::user_text("hi")],
            tools: vec![
                ToolDefinition {
                    name: "first".into(),
                    description: "first tool".into(),
                    input_schema: json!({"type":"object"}),
                    cache_control: None,
                },
                ToolDefinition {
                    name: "last".into(),
                    description: "last tool".into(),
                    input_schema: json!({"type":"object"}),
                    cache_control: Some(CacheControl::ephemeral()),
                },
            ],
            max_tokens: 10,
            temperature: None,
            thinking: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let tools = json["tools"].as_array().unwrap();
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn content_text_cache_control_threads_through() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::user(vec![Content::Text {
                text: "stable user prefix".into(),
                cache_control: Some(CacheControl::Ephemeral {
                    ttl: Some(CacheTtl::FiveMinutes),
                }),
            }])],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
            thinking: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let block = &json["messages"][0]["content"][0];
        assert_eq!(block["type"], "text");
        assert_eq!(block["cache_control"]["type"], "ephemeral");
        assert_eq!(block["cache_control"]["ttl"], "5m");
    }

    #[test]
    fn tool_result_cache_control_threads_through() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::user(vec![Content::ToolResult {
                tool_use_id: "t1".into(),
                content: "long output".into(),
                is_error: false,
                cache_control: Some(CacheControl::ephemeral()),
            }])],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
            thinking: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let block = &json["messages"][0]["content"][0];
        assert_eq!(block["type"], "tool_result");
        assert_eq!(block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn request_with_thinking_budget_serializes_top_level_thinking() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::user_text("solve")],
            tools: vec![],
            max_tokens: 2048,
            temperature: None,
            thinking: None,
        };
        let body = build_request_body_with_thinking(
            &req,
            Some(AnthropicThinkingConfig::Manual {
                budget_tokens: 1024,
            }),
        );
        let json = serde_json::to_value(&body).unwrap();

        assert_eq!(json["thinking"]["type"], "enabled");
        assert_eq!(json["thinking"]["budget_tokens"], 1024);
        assert!(json["thinking"].get("display").is_none());
    }

    #[test]
    fn request_with_adaptive_thinking_serializes_display_and_effort() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::user_text("solve")],
            tools: vec![],
            max_tokens: 2048,
            temperature: None,
            thinking: None,
        };
        let body = build_request_body_with_thinking(
            &req,
            Some(AnthropicThinkingConfig::Adaptive {
                effort: Some(AnthropicEffort::High),
                display: AnthropicThinkingDisplay::Summarized,
            }),
        );
        let json = serde_json::to_value(&body).unwrap();

        assert_eq!(json["thinking"]["type"], "adaptive");
        assert_eq!(json["thinking"]["display"], "summarized");
        assert!(json["thinking"].get("budget_tokens").is_none());
        assert_eq!(json["output_config"]["effort"], "high");
    }

    #[test]
    fn request_with_thinking_omits_temperature() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::user_text("solve")],
            tools: vec![],
            max_tokens: 2048,
            temperature: Some(0.2),
            thinking: None,
        };
        let body = build_request_body_with_thinking(
            &req,
            Some(AnthropicThinkingConfig::Manual {
                budget_tokens: 1024,
            }),
        );
        let json = serde_json::to_value(&body).unwrap();

        assert!(json.get("temperature").is_none());
    }

    #[test]
    fn anthropic_thinking_content_serializes_with_signature() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::assistant(vec![
                Content::thinking(
                    "reason",
                    ThinkingProvider::Anthropic,
                    ThinkingMetadata::anthropic(Some("sig".into())),
                ),
                Content::text("visible"),
            ])],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
            thinking: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let content = json["messages"][0]["content"].as_array().unwrap();

        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "reason");
        assert_eq!(content[0]["signature"], "sig");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], "visible");
    }

    #[test]
    fn anthropic_redacted_thinking_content_round_trips() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::assistant(vec![Content::thinking(
                "",
                ThinkingProvider::Anthropic,
                ThinkingMetadata::anthropic_redacted("opaque"),
            )])],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
            thinking: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let block = &json["messages"][0]["content"][0];
        assert_eq!(block["type"], "redacted_thinking");
        assert_eq!(block["data"], "opaque");

        let raw = json!({
            "content": [{"type":"redacted_thinking","data":"opaque"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let api: ApiResponse = serde_json::from_value(raw).unwrap();
        assert!(matches!(
            &convert_response(api).content[0],
            Content::Thinking {
                text,
                provider: ThinkingProvider::Anthropic,
                metadata: ThinkingMetadata::AnthropicRedacted { data },
            } if text.is_empty() && data == "opaque"
        ));
    }

    #[test]
    fn foreign_thinking_only_message_is_not_serialized_as_empty_message() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![
                Message::assistant(vec![Content::thinking(
                    "foreign",
                    ThinkingProvider::OpenAIResponses,
                    ThinkingMetadata::openai_responses(Some("rs_1".into()), None, 0, None),
                )]),
                Message::user_text("next"),
            ],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
            thinking: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let messages = json["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"][0]["text"], "next");
    }

    #[test]
    fn anthropic_provider_with_mismatched_metadata_is_dropped() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![
                Message::assistant(vec![Content::thinking(
                    "bad metadata",
                    ThinkingProvider::Anthropic,
                    ThinkingMetadata::None,
                )]),
                Message::user_text("next"),
            ],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
            thinking: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let messages = json["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"][0]["text"], "next");
    }

    #[test]
    fn response_with_cache_usage_parses_all_four_fields() {
        let raw = json!({
            "content": [{"type":"text","text":"ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 200
            }
        });
        let api: ApiResponse = serde_json::from_value(raw).unwrap();
        let resp = convert_response(api);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert_eq!(resp.usage.cache_creation_input_tokens, 100);
        assert_eq!(resp.usage.cache_read_input_tokens, 200);
    }

    #[test]
    fn response_without_cache_usage_defaults_to_zero() {
        let raw = json!({
            "content": [{"type":"text","text":"ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let api: ApiResponse = serde_json::from_value(raw).unwrap();
        let resp = convert_response(api);
        assert_eq!(resp.usage.cache_creation_input_tokens, 0);
        assert_eq!(resp.usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn streaming_thinking_delta_and_signature_emit_final_block() {
        use std::collections::VecDeque;
        let mut blocks: HashMap<usize, BlockState> = HashMap::new();
        let mut buffer: VecDeque<Result<StreamEvent, ProviderError>> = VecDeque::new();
        let mut running = Usage::default();

        process_payload(
            StreamingPayload::ContentBlockStart {
                index: 0,
                content_block: ContentBlockStart::Thinking {
                    thinking: String::new(),
                    signature: String::new(),
                },
            },
            &mut blocks,
            &mut buffer,
            &mut running,
        );
        process_payload(
            StreamingPayload::ContentBlockDelta {
                index: 0,
                delta: BlockDelta::Thinking {
                    thinking: "reason".into(),
                },
            },
            &mut blocks,
            &mut buffer,
            &mut running,
        );
        process_payload(
            StreamingPayload::ContentBlockDelta {
                index: 0,
                delta: BlockDelta::Signature {
                    signature: "sig".into(),
                },
            },
            &mut blocks,
            &mut buffer,
            &mut running,
        );
        process_payload(
            StreamingPayload::ContentBlockStop { index: 0 },
            &mut blocks,
            &mut buffer,
            &mut running,
        );

        assert!(matches!(
            buffer.pop_front().unwrap().unwrap(),
            StreamEvent::ThinkingDelta { text } if text == "reason"
        ));
        assert!(matches!(
            buffer.pop_front().unwrap().unwrap(),
            StreamEvent::ThinkingBlock {
                text,
                provider: ThinkingProvider::Anthropic,
                metadata: ThinkingMetadata::Anthropic {
                    signature: Some(signature),
                },
            } if text == "reason" && signature == "sig"
        ));
    }

    #[test]
    fn streaming_redacted_thinking_emits_final_block() {
        use std::collections::VecDeque;
        let mut blocks: HashMap<usize, BlockState> = HashMap::new();
        let mut buffer: VecDeque<Result<StreamEvent, ProviderError>> = VecDeque::new();
        let mut running = Usage::default();

        process_payload(
            StreamingPayload::ContentBlockStart {
                index: 0,
                content_block: ContentBlockStart::RedactedThinking {
                    data: "opaque".into(),
                },
            },
            &mut blocks,
            &mut buffer,
            &mut running,
        );
        process_payload(
            StreamingPayload::ContentBlockStop { index: 0 },
            &mut blocks,
            &mut buffer,
            &mut running,
        );

        assert!(matches!(
            buffer.pop_front().unwrap().unwrap(),
            StreamEvent::ThinkingBlock {
                text,
                provider: ThinkingProvider::Anthropic,
                metadata: ThinkingMetadata::AnthropicRedacted { data },
            } if text.is_empty() && data == "opaque"
        ));
        assert!(buffer.is_empty());
    }

    #[test]
    fn streaming_signature_without_thinking_delta_preserves_empty_block() {
        use std::collections::VecDeque;
        let mut blocks: HashMap<usize, BlockState> = HashMap::new();
        let mut buffer: VecDeque<Result<StreamEvent, ProviderError>> = VecDeque::new();
        let mut running = Usage::default();

        process_payload(
            StreamingPayload::ContentBlockStart {
                index: 0,
                content_block: ContentBlockStart::Thinking {
                    thinking: String::new(),
                    signature: String::new(),
                },
            },
            &mut blocks,
            &mut buffer,
            &mut running,
        );
        process_payload(
            StreamingPayload::ContentBlockDelta {
                index: 0,
                delta: BlockDelta::Signature {
                    signature: "sig-only".into(),
                },
            },
            &mut blocks,
            &mut buffer,
            &mut running,
        );
        process_payload(
            StreamingPayload::ContentBlockStop { index: 0 },
            &mut blocks,
            &mut buffer,
            &mut running,
        );

        assert!(matches!(
            buffer.pop_front().unwrap().unwrap(),
            StreamEvent::ThinkingBlock {
                text,
                provider: ThinkingProvider::Anthropic,
                metadata: ThinkingMetadata::Anthropic {
                    signature: Some(signature),
                },
            } if text.is_empty() && signature == "sig-only"
        ));
        assert!(buffer.is_empty());
    }

    #[test]
    fn streaming_usage_merges_cache_fields_across_message_start_and_delta() {
        // message_start carries cache fields; message_delta does not.
        // The running Usage must preserve cache fields through to the
        // final emitted Usage event.
        use std::collections::VecDeque;
        let mut blocks: HashMap<usize, BlockState> = HashMap::new();
        let mut buffer: VecDeque<Result<StreamEvent, ProviderError>> = VecDeque::new();
        let mut running = Usage::default();

        let start = StreamingPayload::MessageStart {
            message: MessageStartPayload {
                usage: Some(ApiUsage {
                    input_tokens: 50,
                    output_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 1000,
                }),
            },
        };
        process_payload(start, &mut blocks, &mut buffer, &mut running);

        let delta = StreamingPayload::MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("end_turn".into()),
            },
            usage: Some(ApiUsage {
                input_tokens: 50,
                output_tokens: 75,
                // Crucially: API does NOT re-send cache fields here.
                // serde(default) gives 0 — verifying merge_max keeps
                // the prior 1000.
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            }),
        };
        process_payload(delta, &mut blocks, &mut buffer, &mut running);

        // Drain Usage events; final emitted event should carry the
        // merged values.
        let usages: Vec<Usage> = buffer
            .into_iter()
            .filter_map(|r| match r.ok()? {
                StreamEvent::Usage(u) => Some(u),
                _ => None,
            })
            .collect();
        assert_eq!(usages.len(), 2);
        let last = usages.last().unwrap();
        assert_eq!(last.input_tokens, 50);
        assert_eq!(last.output_tokens, 75);
        assert_eq!(last.cache_read_input_tokens, 1000);
    }
}

#[cfg(test)]
mod thinking_override_tests {
    use super::*;
    use crate::provider::{ThinkingConfig, ThinkingEffort};

    fn request(thinking: Option<ThinkingConfig>) -> Request {
        Request {
            model: "claude-test".into(),
            system: None,
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: Some(0.2),
            thinking,
        }
    }

    #[test]
    fn request_budget_overrides_instance_default() {
        let req = request(Some(ThinkingConfig::Budget(2048)));
        let effective = effective_thinking(
            &req,
            Some(&AnthropicThinkingConfig::Adaptive {
                effort: Some(AnthropicEffort::Low),
                display: AnthropicThinkingDisplay::Summarized,
            }),
        );
        let body = build_request_body_with_thinking(&req, effective);
        let json = serde_json::to_value(body).unwrap();
        assert_eq!(json["thinking"]["type"], "enabled");
        assert_eq!(json["thinking"]["budget_tokens"], 2048);
        assert!(json.get("temperature").is_none());
    }

    #[test]
    fn request_effort_other_uses_anthropic_parse() {
        let req = request(Some(ThinkingConfig::Effort(ThinkingEffort::Other(
            "XHIGH".into(),
        ))));
        let body = build_request_body_with_thinking(&req, effective_thinking(&req, None));
        let json = serde_json::to_value(body).unwrap();
        assert_eq!(json["thinking"]["type"], "adaptive");
        assert_eq!(json["output_config"]["effort"], "xhigh");
    }

    #[test]
    fn request_disabled_omits_instance_default() {
        let req = request(Some(ThinkingConfig::Disabled));
        let body = build_request_body_with_thinking(
            &req,
            effective_thinking(
                &req,
                Some(&AnthropicThinkingConfig::Manual {
                    budget_tokens: 1024,
                }),
            ),
        );
        let json = serde_json::to_value(body).unwrap();
        assert!(json.get("thinking").is_none());
        assert!(
            json["temperature"]
                .as_f64()
                .is_some_and(|v| (v - 0.2).abs() < 1e-6)
        );
    }
}
