//! OpenAI Responses API provider.
//!
//! This is the provider to use for OpenAI reasoning/thinking streams.
//! `OpenAICompatible` targets Chat Completions (`/chat/completions`),
//! whose standard wire format has no reasoning-summary events. This
//! provider targets `/responses`, opts into reasoning summaries when
//! configured, and maps `response.reasoning_summary_text.*` events into
//! provider-neutral [`StreamEvent::ThinkingDelta`] /
//! [`StreamEvent::ThinkingBlock`] values.

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use serde_json::{Value, json};

use super::openai_responses_proto::{self as proto, OpenAIEffort, OpenAISummary};
use crate::error::ProviderError;
use crate::provider::{LlmProvider, Request, Response, ThinkingConfig, ThinkingEffort};
use crate::stream::ProviderEventStream;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Provider for OpenAI's `/responses` API.
///
/// Use this when you want first-class OpenAI reasoning summaries. For
/// Chat Completions-compatible endpoints, use [`super::OpenAICompatible`]
/// instead; that provider deliberately does not expose non-standard
/// `reasoning_content` fields as thinking.
///
/// **Per-call thinking precedence:**
/// [`crate::ThinkingConfig::Effort`] overrides the instance default
/// (set via [`Self::with_reasoning`]). [`crate::ThinkingConfig::Disabled`]
/// drops the entire reasoning block. [`crate::ThinkingConfig::Budget`]
/// is **Anthropic-style and silently ignored** here — the instance
/// default applies as if no per-call thinking were specified.
pub struct OpenAIResponses {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    reasoning: Option<ReasoningConfig>,
    include_encrypted_reasoning: bool,
}

#[derive(Debug, Clone)]
struct ReasoningConfig {
    effort: OpenAIEffort,
    summary: OpenAISummary,
}

impl OpenAIResponses {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
            reasoning: None,
            include_encrypted_reasoning: true,
        }
    }

    /// Read `OPENAI_API_KEY` from the environment.
    pub fn from_env() -> Self {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var is required");
        Self::new(api_key)
    }

    /// Override the endpoint root, without trailing `/responses`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Request reasoning effort + summaries from reasoning models.
    ///
    /// Typical values: effort `low|medium|high`, summary
    /// `auto|concise|detailed`. OpenAI validates the exact combinations
    /// per model.
    pub fn with_reasoning(
        mut self,
        effort: impl Into<OpenAIEffort>,
        summary: impl Into<OpenAISummary>,
    ) -> Self {
        self.reasoning = Some(ReasoningConfig {
            effort: effort.into(),
            summary: summary.into(),
        });
        self
    }

    /// Do not request encrypted reasoning replay blobs.
    ///
    /// Keeping this enabled is useful for stateless multi-turn replay:
    /// OpenAI can return opaque `reasoning.encrypted_content` that should
    /// be persisted but never displayed.
    pub fn without_encrypted_reasoning(mut self) -> Self {
        self.include_encrypted_reasoning = false;
        self
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl LlmProvider for OpenAIResponses {
    async fn stream(&self, request: Request) -> Result<ProviderEventStream, ProviderError> {
        let mut body = build_request_body(
            &request,
            effective_reasoning(&request, self.reasoning.as_ref()).as_ref(),
            self.include_encrypted_reasoning,
        );
        body["stream"] = json!(true);

        let response = self
            .client
            .post(self.responses_url())
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();
        if status >= 400 {
            let retry_after_ms = proto::parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            return Err(proto::classify_error(status, text, retry_after_ms));
        }

        Ok(Box::pin(proto::responses_event_stream(
            response.bytes_stream().eventsource(),
        )))
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let body = build_request_body(
            &request,
            effective_reasoning(&request, self.reasoning.as_ref()).as_ref(),
            self.include_encrypted_reasoning,
        );

        let response = self
            .client
            .post(self.responses_url())
            .bearer_auth(&self.api_key)
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

        let text = response.text().await?;
        let value = serde_json::from_str::<Value>(&text)?;
        proto::response_error(&value).map_or_else(|| proto::convert_response_value(&value), Err)
    }
}

fn build_request_body(
    request: &Request,
    reasoning: Option<&ReasoningConfig>,
    include_encrypted_reasoning: bool,
) -> Value {
    let mut body = json!({
        "model": request.model,
        "store": false,
        "stream": false,
        "input": proto::build_input(&request.messages),
        "max_output_tokens": request.max_tokens,
    });

    if let Some(instructions) = proto::instructions(request) {
        body["instructions"] = json!(instructions);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(reasoning) = reasoning {
        body["reasoning"] = json!({
            "effort": reasoning.effort.as_wire(),
            "summary": reasoning.summary.as_wire(),
        });
    }
    if include_encrypted_reasoning {
        body["include"] = json!(["reasoning.encrypted_content"]);
    }

    let tools = proto::build_tools(&request.tools);
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
        body["tool_choice"] = json!("auto");
        body["parallel_tool_calls"] = json!(true);
    }

    body
}

fn effective_reasoning(
    request: &Request,
    instance: Option<&ReasoningConfig>,
) -> Option<ReasoningConfig> {
    match &request.thinking {
        Some(ThinkingConfig::Disabled) => None,
        Some(ThinkingConfig::Budget(_)) => instance.cloned(),
        Some(ThinkingConfig::Effort(effort)) => Some(ReasoningConfig {
            effort: map_thinking_effort(effort),
            summary: instance
                .map(|r| r.summary.clone())
                .unwrap_or(OpenAISummary::Auto),
        }),
        None => instance.cloned(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Content, Message, StopReason, ThinkingMetadata, ThinkingProvider};
    use crate::provider::SystemBlock;
    use crate::stream::StreamEvent;
    use std::collections::VecDeque;

    #[test]
    fn request_includes_reasoning_and_encrypted_content() {
        let req = Request {
            model: "gpt-5".into(),
            system: Some(vec![SystemBlock::text("be brief")]),
            messages: vec![Message::user_text("solve")],
            tools: vec![],
            max_tokens: 128,
            temperature: None,
            thinking: None,
        };
        let reasoning = ReasoningConfig {
            effort: OpenAIEffort::Medium,
            summary: OpenAISummary::Auto,
        };

        let body = build_request_body(&req, Some(&reasoning), true);

        assert_eq!(body["model"], "gpt-5");
        assert_eq!(body["instructions"], "be brief");
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert_eq!(body["reasoning"]["summary"], "auto");
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        assert_eq!(body["input"][0]["role"], "user");
    }

    #[test]
    fn request_replays_openai_reasoning_and_tool_state() {
        let req = Request {
            model: "gpt-5".into(),
            system: None,
            messages: vec![
                Message::assistant(vec![
                    Content::thinking(
                        "summary",
                        ThinkingProvider::OpenAIResponses,
                        ThinkingMetadata::openai_responses(
                            Some("rs_1".into()),
                            Some(0),
                            0,
                            Some("enc".into()),
                        ),
                    ),
                    Content::ToolUse {
                        id: "call_1|fc_1".into(),
                        name: "bash".into(),
                        input: json!({"command":"echo hi"}),
                    },
                ]),
                Message::user(vec![Content::tool_result("call_1|fc_1", "hi", false)]),
            ],
            tools: vec![],
            max_tokens: 128,
            temperature: None,
            thinking: None,
        };

        let body = build_request_body(&req, None, true);
        let input = body["input"].as_array().unwrap();

        assert_eq!(input[0]["type"], "reasoning");
        assert_eq!(input[0]["id"], "rs_1");
        assert_eq!(input[0]["encrypted_content"], "enc");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["id"], "fc_1");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_1");

        // status is output-only on the Responses API; including it on
        // input items causes the backend to reject the request.
        assert!(input[0].get("status").is_none());
        assert!(input[1].get("status").is_none());
    }

    #[test]
    fn non_streaming_response_decodes_reasoning_text_and_tools() {
        let raw = json!({
            "status": "completed",
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type":"summary_text", "text":"checked constraints"}],
                    "encrypted_content": "opaque"
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "content": [{"type":"output_text", "text":"answer"}]
                },
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "{\"command\":\"echo hi\"}"
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 7}
        });

        let response = proto::convert_response_value(&raw).unwrap();

        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.usage.input_tokens, 10);
        assert!(matches!(
            &response.content[0],
            Content::Thinking {
                text,
                provider: ThinkingProvider::OpenAIResponses,
                metadata:
                    ThinkingMetadata::OpenAIResponses {
                        item_id: Some(item_id),
                        output_index: Some(0),
                        summary_index: 0,
                        encrypted_content: Some(encrypted),
                    },
            } if text == "checked constraints" && item_id == "rs_1" && encrypted == "opaque"
        ));
        assert!(matches!(&response.content[1], Content::Text { text, .. } if text == "answer"));
        assert!(matches!(
            &response.content[2],
            Content::ToolUse { id, name, input }
                if id == "call_1|fc_1" && name == "bash" && input["command"] == "echo hi"
        ));
    }

    #[test]
    fn streaming_reasoning_summary_emits_delta_and_final_block() {
        let mut parser = proto::ResponsesSseParser::default();
        let mut out = VecDeque::new();

        parser.process_value(
            json!({
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "delta": "checked"
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "text": "checked constraints"
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type":"summary_text", "text":"checked constraints"}],
                    "encrypted_content": "opaque"
                }
            }),
            &mut out,
        );

        assert!(matches!(
            out.pop_front().unwrap().unwrap(),
            StreamEvent::ThinkingDelta { text } if text == "checked"
        ));
        assert!(matches!(
            out.pop_front().unwrap().unwrap(),
            StreamEvent::ThinkingBlock {
                text,
                provider: ThinkingProvider::OpenAIResponses,
                metadata:
                    ThinkingMetadata::OpenAIResponses {
                        item_id: Some(item_id),
                        output_index: Some(0),
                        summary_index: 0,
                        encrypted_content: Some(encrypted),
                    },
            } if text == "checked constraints" && item_id == "rs_1" && encrypted == "opaque"
        ));
        assert!(out.is_empty());
    }

    #[test]
    fn streaming_ignores_raw_reasoning_text_events() {
        let mut parser = proto::ResponsesSseParser::default();
        let mut out = VecDeque::new();

        parser.process_value(
            json!({
                "type": "response.reasoning_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "raw chain of thought"
            }),
            &mut out,
        );

        assert!(out.is_empty());
    }

    #[test]
    fn streaming_tool_call_emits_atomic_tool_use() {
        let mut parser = proto::ResponsesSseParser::default();
        let mut out = VecDeque::new();

        parser.process_value(
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": ""
                }
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": "{\"command\":"
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "output_index": 0,
                "name": "bash",
                "arguments": "{\"command\":\"echo hi\"}"
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "{\"command\":\"echo hi\"}"
                }
            }),
            &mut out,
        );

        assert!(matches!(
            out.pop_front().unwrap().unwrap(),
            StreamEvent::ToolUse { id, name, input }
                if id == "call_1|fc_1" && name == "bash" && input["command"] == "echo hi"
        ));
        assert!(out.is_empty());
    }
}

#[cfg(test)]
mod thinking_override_tests {
    //! Per-call `Request.thinking` precedence tests. Mirror of the
    //! Anthropic provider's `thinking_override_tests`. Issue #40 Phase 2
    //! acceptance criteria require explicit coverage on every provider
    //! that supports thinking.
    use super::*;
    use crate::message::Message;
    use crate::provider::{SystemBlock, ThinkingConfig, ThinkingEffort};

    fn request(thinking: Option<ThinkingConfig>) -> Request {
        Request {
            model: "gpt-5".into(),
            system: Some(vec![SystemBlock::text("be brief")]),
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            max_tokens: 64,
            temperature: None,
            thinking,
        }
    }

    #[test]
    fn effort_high_emits_wire_reasoning_effort_high() {
        let req = request(Some(ThinkingConfig::Effort(ThinkingEffort::High)));
        let body = build_request_body(&req, effective_reasoning(&req, None).as_ref(), false);
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    #[test]
    fn per_call_effort_overrides_instance_default() {
        let instance = ReasoningConfig {
            effort: OpenAIEffort::Low,
            summary: OpenAISummary::Detailed,
        };
        let req = request(Some(ThinkingConfig::Effort(ThinkingEffort::High)));
        let body = build_request_body(
            &req,
            effective_reasoning(&req, Some(&instance)).as_ref(),
            false,
        );
        // Per-call hint wins over instance default.
        assert_eq!(body["reasoning"]["effort"], "high");
        // Instance summary preserved when not overridden.
        assert_eq!(body["reasoning"]["summary"], "detailed");
    }

    #[test]
    fn disabled_drops_reasoning_block_even_with_instance_default() {
        let instance = ReasoningConfig {
            effort: OpenAIEffort::High,
            summary: OpenAISummary::Auto,
        };
        let req = request(Some(ThinkingConfig::Disabled));
        let body = build_request_body(
            &req,
            effective_reasoning(&req, Some(&instance)).as_ref(),
            false,
        );
        assert!(
            body.get("reasoning").is_none(),
            "Disabled must drop reasoning entirely; got {body:?}"
        );
    }

    #[test]
    fn budget_falls_back_to_instance_silently() {
        // Budget is Anthropic-style; OpenAI providers ignore the value
        // and apply their instance defaults (no per-call override). This
        // test locks the documented contract; see ThinkingConfig::Budget
        // doc and the per-provider docstring.
        let instance = ReasoningConfig {
            effort: OpenAIEffort::Medium,
            summary: OpenAISummary::Auto,
        };
        let req = request(Some(ThinkingConfig::Budget(8192)));
        let body = build_request_body(
            &req,
            effective_reasoning(&req, Some(&instance)).as_ref(),
            false,
        );
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert_eq!(body["reasoning"]["summary"], "auto");
    }

    #[test]
    fn no_thinking_falls_through_to_instance() {
        let instance = ReasoningConfig {
            effort: OpenAIEffort::Low,
            summary: OpenAISummary::Auto,
        };
        let req = request(None);
        let body = build_request_body(
            &req,
            effective_reasoning(&req, Some(&instance)).as_ref(),
            false,
        );
        assert_eq!(body["reasoning"]["effort"], "low");
    }

    #[test]
    fn other_effort_passes_through_verbatim() {
        let req = request(Some(ThinkingConfig::Effort(ThinkingEffort::Other(
            "xhigh".into(),
        ))));
        let body = build_request_body(&req, effective_reasoning(&req, None).as_ref(), false);
        // `xhigh` resolves to OpenAIEffort::from("xhigh") which on the
        // wire emits "xhigh" verbatim (server-side validation handles
        // unknown tiers).
        assert_eq!(body["reasoning"]["effort"], "xhigh");
    }
}
