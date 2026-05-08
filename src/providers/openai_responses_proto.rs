//! Shared transport-agnostic helpers for OpenAI Responses-style backends.
//!
//! Two providers speak the same wire grammar:
//!
//! - [`super::OpenAIResponses`] — public OpenAI `/responses` endpoint.
//! - [`super::OpenAICodex`] — ChatGPT subscription Codex backend at
//!   `chatgpt.com/backend-api/codex/responses`.
//!
//! Both produce the same SSE event stream (`response.output_text.delta`,
//! `response.function_call_arguments.*`, `response.output_item.*`,
//! `response.reasoning_summary_text.*`, terminal `response.completed` /
//! `response.incomplete` / `response.failed`) and accept the same
//! `input` array shape (text messages, `function_call`,
//! `function_call_output`, `reasoning`).
//!
//! Differences between the two are confined to HTTP transport — base
//! URL, auth headers, request-body envelope keys (`max_output_tokens`,
//! `text.verbosity`, `reasoning`). Each provider keeps its own
//! `LlmProvider` impl and request-body builder; the shared logic lives
//! here so the parser cannot drift between them.

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{Value, json};

use crate::error::ProviderError;
use crate::message::{
    Content, Message, Role, StopReason, ThinkingMetadata, ThinkingProvider, Usage,
};
use crate::provider::{Request, Response, ToolDefinition};
use crate::stream::StreamEvent;

// --- Request body helpers -------------------------------------------------

/// Concatenate `Request.system` blocks with `\n\n`. `None` if absent or empty.
pub(super) fn instructions(request: &Request) -> Option<String> {
    let joined = request
        .system
        .as_ref()?
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!joined.is_empty()).then_some(joined)
}

/// Convert tools to the Responses `function` tool envelope.
pub(super) fn build_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": null,
            })
        })
        .collect()
}

/// Map message history into the Responses `input` array.
pub(super) fn build_input(messages: &[Message]) -> Vec<Value> {
    let mut input = Vec::new();
    for message in messages {
        match message.role {
            Role::User => push_user_message(&mut input, message),
            Role::Assistant => push_assistant_message(&mut input, message),
        }
    }
    input
}

fn push_user_message(input: &mut Vec<Value>, message: &Message) {
    let mut text = Vec::new();

    for content in &message.content {
        match content {
            Content::Text { text: chunk, .. } => text.push(chunk.as_str()),
            Content::ToolResult {
                tool_use_id,
                content,
                is_error,
                ..
            } => {
                push_text_message(input, "user", &text.join("\n"));
                text.clear();

                let output = if *is_error {
                    format!("[error] {content}")
                } else {
                    content.clone()
                };
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": tool_call_id_for_output(tool_use_id),
                    "output": output,
                }));
            }
            Content::Thinking { .. } | Content::ToolUse { .. } => {}
        }
    }

    push_text_message(input, "user", &text.join("\n"));
}

fn push_assistant_message(input: &mut Vec<Value>, message: &Message) {
    let mut text = Vec::new();

    for content in &message.content {
        match content {
            Content::Text { text: chunk, .. } => text.push(chunk.as_str()),
            Content::Thinking {
                text: thinking,
                provider: ThinkingProvider::OpenAIResponses,
                metadata:
                    ThinkingMetadata::OpenAIResponses {
                        item_id,
                        output_index: _,
                        summary_index: _,
                        encrypted_content,
                    },
            } => {
                push_text_message(input, "assistant", &text.join(""));
                text.clear();
                input.push(openai_reasoning_input(
                    item_id.as_deref(),
                    thinking,
                    encrypted_content.as_deref(),
                ));
            }
            Content::ToolUse {
                id,
                name,
                input: args,
            } => {
                push_text_message(input, "assistant", &text.join(""));
                text.clear();

                let (call_id, item_id) = split_tool_use_id(id);
                // `status` is output-only on the Responses API. Sending
                // it on input items returns an error from the backend.
                let mut item = json!({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": args.to_string(),
                });
                if let Some(item_id) = item_id {
                    item["id"] = json!(item_id);
                }
                input.push(item);
            }
            Content::Thinking { .. } | Content::ToolResult { .. } => {}
        }
    }

    push_text_message(input, "assistant", &text.join(""));
}

fn openai_reasoning_input(
    item_id: Option<&str>,
    text: &str,
    encrypted_content: Option<&str>,
) -> Value {
    // `status` is output-only on the Responses API. Sending it on
    // input reasoning items returns an error from the backend.
    let mut item = json!({
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
    });
    if let Some(item_id) = item_id {
        item["id"] = json!(item_id);
    }
    if let Some(encrypted_content) = encrypted_content {
        item["encrypted_content"] = json!(encrypted_content);
    }
    item
}

fn push_text_message(input: &mut Vec<Value>, role: &str, content: &str) {
    if !content.is_empty() {
        input.push(json!({ "role": role, "content": content }));
    }
}

fn split_tool_use_id(id: &str) -> (&str, Option<&str>) {
    id.split_once('|').map_or((id, None), |(call_id, item_id)| {
        (call_id, (!item_id.is_empty()).then_some(item_id))
    })
}

fn tool_call_id_for_output(id: &str) -> &str {
    split_tool_use_id(id).0
}

pub(super) fn combined_tool_use_id(call_id: &str, item_id: &str) -> String {
    if item_id.is_empty() {
        call_id.to_string()
    } else {
        format!("{call_id}|{item_id}")
    }
}

// --- Non-streaming response decoder --------------------------------------

pub(super) fn convert_response_value(value: &Value) -> Result<Response, ProviderError> {
    let mut content = Vec::new();
    if let Some(output) = value.get("output").and_then(Value::as_array) {
        for (output_index, item) in output.iter().enumerate() {
            convert_output_item(item, output_index, &mut content);
        }
    }

    if content.is_empty() {
        if let Some(text) = value.get("output_text").and_then(Value::as_str) {
            if !text.is_empty() {
                content.push(Content::text(text));
            }
        }
    }

    let has_tool_use = content.iter().any(|c| matches!(c, Content::ToolUse { .. }));
    Ok(Response {
        stop_reason: response_stop_reason(value, has_tool_use),
        usage: usage_from_response(value.get("usage")),
        content,
    })
}

fn convert_output_item(item: &Value, output_index: usize, content: &mut Vec<Content>) {
    match item.get("type").and_then(Value::as_str) {
        Some("message") => convert_message_item(item, content),
        Some("function_call") => {
            if let Some(tool) = tool_call_from_item(item) {
                content.push(tool_to_content(tool));
            }
        }
        Some("reasoning") => convert_reasoning_item(item, Some(output_index), content),
        _ => {}
    }
}

fn convert_message_item(item: &Value, content: &mut Vec<Content>) {
    let Some(parts) = item.get("content").and_then(Value::as_array) else {
        return;
    };
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("output_text") => {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        content.push(Content::text(text));
                    }
                }
            }
            Some("refusal") => {
                if let Some(text) = part.get("refusal").and_then(Value::as_str) {
                    if !text.is_empty() {
                        content.push(Content::text(text));
                    }
                }
            }
            Some("reasoning_text") => {}
            _ => {}
        }
    }
}

fn convert_reasoning_item(item: &Value, output_index: Option<usize>, content: &mut Vec<Content>) {
    let item_id = item.get("id").and_then(Value::as_str).map(str::to_string);
    let encrypted_content = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .map(str::to_string);

    let Some(summary) = item.get("summary").and_then(Value::as_array) else {
        return;
    };
    for (summary_index, part) in summary.iter().enumerate() {
        let Some(text) = part.get("text").and_then(Value::as_str) else {
            continue;
        };
        content.push(Content::thinking(
            text,
            ThinkingProvider::OpenAIResponses,
            ThinkingMetadata::openai_responses(
                item_id.clone(),
                output_index,
                summary_index,
                encrypted_content.clone(),
            ),
        ));
    }
}

fn response_stop_reason(value: &Value, has_tool_use: bool) -> StopReason {
    if has_tool_use {
        return StopReason::ToolUse;
    }

    match value.get("status").and_then(Value::as_str) {
        Some("incomplete") => match value
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
        {
            Some("max_output_tokens") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::EndTurn,
            _ => StopReason::EndTurn,
        },
        _ => StopReason::EndTurn,
    }
}

pub(super) fn usage_from_response(value: Option<&Value>) -> Usage {
    let Some(value) = value else {
        return Usage::default();
    };
    Usage {
        input_tokens: value
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        output_tokens: value
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: value
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
    }
}

// --- HTTP error classification -------------------------------------------

pub(super) fn classify_error(
    status: u16,
    message: String,
    retry_after_ms: Option<u64>,
) -> ProviderError {
    match status {
        429 => ProviderError::RateLimit { retry_after_ms },
        503 => ProviderError::Overloaded { retry_after_ms },
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

pub(super) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    raw.trim().parse::<u64>().ok().map(|s| s * 1_000)
}

pub(super) fn response_error(value: &Value) -> Option<ProviderError> {
    let error = value.get("error")?;
    if error.is_null() {
        return None;
    }
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("OpenAI Responses request failed")
        .to_string();
    Some(ProviderError::Api {
        status: 500,
        message,
        retryable: false,
    })
}

// --- SSE parser ----------------------------------------------------------

#[derive(Default)]
pub(super) struct ResponsesSseParser {
    usage: Usage,
    pending_tools: HashMap<String, PendingToolCall>,
    emitted_tool_items: HashSet<String>,
    pending_reasoning: HashMap<ReasoningKey, PendingReasoning>,
    saw_tool_use: bool,
    emitted_terminal: bool,
}

#[derive(Default)]
struct PendingToolCall {
    call_id: String,
    item_id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReasoningKey {
    item_id: String,
    summary_index: usize,
}

struct PendingReasoning {
    item_id: String,
    output_index: Option<usize>,
    summary_index: usize,
    text: String,
    encrypted_content: Option<String>,
    emitted: bool,
}

struct ResponsesStreamState<S> {
    sse: S,
    parser: ResponsesSseParser,
    outbox: VecDeque<Result<StreamEvent, ProviderError>>,
    done: bool,
}

pub(super) fn responses_event_stream<S>(
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
    futures::stream::unfold(
        ResponsesStreamState {
            sse,
            parser: ResponsesSseParser::default(),
            outbox: VecDeque::new(),
            done: false,
        },
        |mut state| async move {
            use futures::StreamExt;

            loop {
                if let Some(event) = state.outbox.pop_front() {
                    return Some((event, state));
                }
                if state.done {
                    return None;
                }

                let event = match state.sse.next().await {
                    Some(Ok(event)) => event,
                    Some(Err(err)) => {
                        return Some((
                            Err(ProviderError::Other(format!("SSE read error: {err}"))),
                            state,
                        ));
                    }
                    None => {
                        state.parser.finish(&mut state.outbox);
                        state.done = true;
                        continue;
                    }
                };

                let data = event.data.trim();
                if data == "[DONE]" {
                    state.parser.finish(&mut state.outbox);
                    state.done = true;
                    continue;
                }
                if data.is_empty() {
                    continue;
                }

                let Ok(value) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                if state.parser.process_value(value, &mut state.outbox) {
                    state.done = true;
                }
            }
        },
    )
}

impl ResponsesSseParser {
    pub(super) fn process_value(
        &mut self,
        value: Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) -> bool {
        match value.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => self.output_text_delta(&value, out),
            Some("response.reasoning_summary_text.delta") => {
                self.reasoning_summary_delta(&value, out);
            }
            Some("response.reasoning_summary_text.done")
            | Some("response.reasoning_summary_part.done") => self.reasoning_summary_done(&value),
            Some("response.output_item.added") => self.output_item_added(&value),
            Some("response.function_call_arguments.delta") => self.tool_arguments_delta(&value),
            Some("response.function_call_arguments.done") => self.tool_arguments_done(&value),
            Some("response.output_item.done") => self.output_item_done(&value, out),
            Some("response.completed") | Some("response.incomplete") => {
                self.terminal_response(&value, out);
                return true;
            }
            Some("response.failed") => {
                self.failed_response(&value, out);
                return true;
            }
            Some("error") | Some("response.error") => {
                out.push_back(Err(stream_error(&value)));
                return true;
            }
            Some("response.reasoning_text.delta") | Some("response.reasoning_text.done") => {}
            _ => {}
        }
        false
    }

    fn output_text_delta(
        &self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if let Some(delta) = value.get("delta").and_then(Value::as_str) {
            if !delta.is_empty() {
                out.push_back(Ok(StreamEvent::ContentDelta(delta.to_string())));
            }
        }
    }

    fn reasoning_summary_delta(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let Some((key, output_index)) = reasoning_key(value) else {
            return;
        };
        let Some(delta) = value.get("delta").and_then(Value::as_str) else {
            return;
        };

        self.pending_reasoning
            .entry(key.clone())
            .or_insert_with(|| PendingReasoning::new(key.clone(), output_index))
            .text
            .push_str(delta);

        if !delta.is_empty() {
            out.push_back(Ok(StreamEvent::ThinkingDelta {
                text: delta.to_string(),
            }));
        }
    }

    fn reasoning_summary_done(&mut self, value: &Value) {
        let Some((key, output_index)) = reasoning_key(value) else {
            return;
        };
        let text = value
            .get("text")
            .or_else(|| value.pointer("/part/text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        let pending = self
            .pending_reasoning
            .entry(key.clone())
            .or_insert_with(|| PendingReasoning::new(key, output_index));
        if !text.is_empty() {
            pending.text = text;
        }
    }

    fn output_item_added(&mut self, value: &Value) {
        if let Some(tool) = pending_tool_from_item(value.get("item")) {
            self.pending_tools.insert(tool.item_id.clone(), tool);
        }
    }

    fn tool_arguments_delta(&mut self, value: &Value) {
        let (Some(item_id), Some(delta)) = (
            value.get("item_id").and_then(Value::as_str),
            value.get("delta").and_then(Value::as_str),
        ) else {
            return;
        };
        self.pending_tools
            .entry(item_id.to_string())
            .or_insert_with(|| PendingToolCall {
                item_id: item_id.to_string(),
                ..PendingToolCall::default()
            })
            .arguments
            .push_str(delta);
    }

    fn tool_arguments_done(&mut self, value: &Value) {
        let Some(item_id) = value.get("item_id").and_then(Value::as_str) else {
            return;
        };
        let tool = self
            .pending_tools
            .entry(item_id.to_string())
            .or_insert_with(|| PendingToolCall {
                item_id: item_id.to_string(),
                ..PendingToolCall::default()
            });
        if let Some(arguments) = value.get("arguments").and_then(Value::as_str) {
            tool.arguments = arguments.to_string();
        }
        if let Some(name) = value.get("name").and_then(Value::as_str) {
            tool.name = name.to_string();
        }
    }

    fn output_item_done(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        match value.pointer("/item/type").and_then(Value::as_str) {
            Some("function_call") => {
                if let Some(tool) = self.completed_tool_call(value.get("item")) {
                    self.emit_tool_use(tool, out);
                }
            }
            Some("reasoning") => self.reasoning_item_done(value, out),
            _ => {}
        }
    }

    fn terminal_response(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if let Some(response) = value.get("response") {
            self.usage = usage_from_response(response.get("usage"));
            self.capture_reasoning_from_response(response, out);
            let status = response.get("status").and_then(Value::as_str);
            let incomplete_reason = response
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str);
            self.emit_terminal(status, incomplete_reason, out);
        } else {
            self.emit_terminal(None, None, out);
        }
    }

    fn failed_response(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if let Some(response) = value.get("response") {
            self.usage = usage_from_response(response.get("usage"));
        }
        out.push_back(Err(stream_error(value)));
        self.emitted_terminal = true;
    }

    fn capture_reasoning_from_response(
        &mut self,
        response: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let Some(output) = response.get("output").and_then(Value::as_array) else {
            return;
        };
        for (output_index, item) in output.iter().enumerate() {
            if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                self.capture_reasoning_item(item, Some(output_index), out);
            }
        }
    }

    fn reasoning_item_done(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let output_index = value
            .get("output_index")
            .and_then(Value::as_u64)
            .map(|i| i as usize);
        if let Some(item) = value.get("item") {
            self.capture_reasoning_item(item, output_index, out);
        }
    }

    fn capture_reasoning_item(
        &mut self,
        item: &Value,
        output_index: Option<usize>,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let item_id = item
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let encrypted_content = item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .map(str::to_string);

        if let Some(summary) = item.get("summary").and_then(Value::as_array) {
            for (summary_index, part) in summary.iter().enumerate() {
                let key = ReasoningKey {
                    item_id: item_id.clone(),
                    summary_index,
                };
                let pending = self
                    .pending_reasoning
                    .entry(key.clone())
                    .or_insert_with(|| PendingReasoning::new(key, output_index));
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    pending.text = text.to_string();
                }
                pending.encrypted_content = encrypted_content.clone();
                pending.output_index = pending.output_index.or(output_index);
            }
        }

        self.emit_reasoning_for_item(&item_id, out);
    }

    fn emit_reasoning_for_item(
        &mut self,
        item_id: &str,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let mut keys = self
            .pending_reasoning
            .keys()
            .filter(|key| key.item_id == item_id)
            .cloned()
            .collect::<Vec<_>>();
        keys.sort_by_key(|key| key.summary_index);

        for key in keys {
            self.emit_reasoning_key(&key, out);
        }
    }

    fn emit_all_reasoning(&mut self, out: &mut VecDeque<Result<StreamEvent, ProviderError>>) {
        let mut keys = self.pending_reasoning.keys().cloned().collect::<Vec<_>>();
        keys.sort_by(|a, b| {
            a.item_id
                .cmp(&b.item_id)
                .then(a.summary_index.cmp(&b.summary_index))
        });
        for key in keys {
            self.emit_reasoning_key(&key, out);
        }
    }

    fn emit_reasoning_key(
        &mut self,
        key: &ReasoningKey,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let Some(pending) = self.pending_reasoning.get_mut(key) else {
            return;
        };
        if pending.emitted {
            return;
        }
        pending.emitted = true;
        out.push_back(Ok(StreamEvent::ThinkingBlock {
            text: pending.text.clone(),
            provider: ThinkingProvider::OpenAIResponses,
            metadata: ThinkingMetadata::openai_responses(
                (!pending.item_id.is_empty()).then_some(pending.item_id.clone()),
                pending.output_index,
                pending.summary_index,
                pending.encrypted_content.clone(),
            ),
        }));
    }

    fn completed_tool_call(&mut self, item: Option<&Value>) -> Option<PendingToolCall> {
        let item = item?;
        let item_id = item
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if !item_id.is_empty() && self.emitted_tool_items.contains(&item_id) {
            return None;
        }

        let mut tool = if item_id.is_empty() {
            PendingToolCall::default()
        } else {
            self.pending_tools.remove(&item_id).unwrap_or_default()
        };
        if tool.item_id.is_empty() {
            tool.item_id = item_id;
        }
        if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
            tool.call_id = call_id.to_string();
        }
        if let Some(name) = item.get("name").and_then(Value::as_str) {
            tool.name = name.to_string();
        }
        if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
            tool.arguments = arguments.to_string();
        }
        (!tool.call_id.is_empty() || !tool.name.is_empty()).then_some(tool)
    }

    fn emit_tool_use(
        &mut self,
        tool: PendingToolCall,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if !tool.item_id.is_empty() {
            self.emitted_tool_items.insert(tool.item_id.clone());
        }
        self.saw_tool_use = true;
        out.push_back(Ok(StreamEvent::ToolUse {
            id: combined_tool_use_id(&tool.call_id, &tool.item_id),
            name: tool.name,
            input: parse_tool_arguments(&tool.arguments),
        }));
    }

    fn emit_pending_tools(&mut self, out: &mut VecDeque<Result<StreamEvent, ProviderError>>) {
        let tools = std::mem::take(&mut self.pending_tools);
        for (_, tool) in tools {
            if !tool.item_id.is_empty() && self.emitted_tool_items.contains(&tool.item_id) {
                continue;
            }
            self.emit_tool_use(tool, out);
        }
    }

    fn emit_terminal(
        &mut self,
        status: Option<&str>,
        incomplete_reason: Option<&str>,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if self.emitted_terminal {
            return;
        }
        self.emit_all_reasoning(out);
        self.emit_pending_tools(out);
        out.push_back(Ok(StreamEvent::Usage(self.usage.clone())));
        out.push_back(Ok(StreamEvent::MessageDelta {
            stop_reason: responses_stop_reason(status, incomplete_reason, self.saw_tool_use),
        }));
        out.push_back(Ok(StreamEvent::Done));
        self.emitted_terminal = true;
    }

    fn finish(&mut self, out: &mut VecDeque<Result<StreamEvent, ProviderError>>) {
        self.emit_terminal(None, None, out);
    }
}

impl PendingReasoning {
    fn new(key: ReasoningKey, output_index: Option<usize>) -> Self {
        Self {
            item_id: key.item_id,
            output_index,
            summary_index: key.summary_index,
            text: String::new(),
            encrypted_content: None,
            emitted: false,
        }
    }
}

fn reasoning_key(value: &Value) -> Option<(ReasoningKey, Option<usize>)> {
    let item_id = value.get("item_id")?.as_str()?.to_string();
    let summary_index = value.get("summary_index")?.as_u64()? as usize;
    let output_index = value
        .get("output_index")
        .and_then(Value::as_u64)
        .map(|i| i as usize);
    Some((
        ReasoningKey {
            item_id,
            summary_index,
        },
        output_index,
    ))
}

fn pending_tool_from_item(item: Option<&Value>) -> Option<PendingToolCall> {
    let item = item?;
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return None;
    }
    let item_id = item
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Some(PendingToolCall {
        call_id: item
            .get("call_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        item_id,
        name: item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        arguments: item
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    })
}

fn tool_call_from_item(item: &Value) -> Option<PendingToolCall> {
    pending_tool_from_item(Some(item))
}

fn tool_to_content(tool: PendingToolCall) -> Content {
    Content::ToolUse {
        id: combined_tool_use_id(&tool.call_id, &tool.item_id),
        name: tool.name,
        input: parse_tool_arguments(&tool.arguments),
    }
}

fn parse_tool_arguments(arguments: &str) -> Value {
    if arguments.trim().is_empty() {
        Value::Object(Default::default())
    } else {
        serde_json::from_str(arguments).unwrap_or(Value::Object(Default::default()))
    }
}

fn responses_stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
    saw_tool_use: bool,
) -> StopReason {
    if saw_tool_use {
        return StopReason::ToolUse;
    }
    match (status, incomplete_reason) {
        (Some("incomplete"), Some("max_output_tokens")) => StopReason::MaxTokens,
        (Some("incomplete"), Some("content_filter")) => StopReason::EndTurn,
        _ => StopReason::EndTurn,
    }
}

fn stream_error(value: &Value) -> ProviderError {
    let message = value
        .pointer("/response/error/message")
        .or_else(|| value.pointer("/error/message"))
        .or_else(|| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("OpenAI Responses stream failed")
        .to_string();
    ProviderError::Api {
        status: 500,
        message,
        retryable: false,
    }
}
