use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;

use crate::error::ProviderError;
use crate::message::{Content, StopReason, Usage};
use crate::provider::{LlmProvider, Request, Response};
use crate::stream::{ProviderEventStream, StreamEvent};

type ResponseFn = dyn Fn(&Request) -> Result<Response, ProviderError> + Send + Sync;

/// Mock LLM provider for testing.
///
/// # Example
///
/// ```ignore
/// use tkach::providers::Mock;
/// use tkach::message::{Content, StopReason, Usage};
/// use tkach::provider::Response;
///
/// let mock = Mock::new(|_req| {
///     Ok(Response {
///         content: vec![Content::text("Hello from mock!")],
///         stop_reason: StopReason::EndTurn,
///         usage: Usage::default(),
///     })
/// });
/// ```
pub struct Mock {
    handler: Arc<ResponseFn>,
    call_count: Arc<Mutex<usize>>,
    complete_count: Arc<Mutex<usize>>,
    stream_count: Arc<Mutex<usize>>,
}

impl Mock {
    /// Create a mock provider with a custom response function.
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(&Request) -> Result<Response, ProviderError> + Send + Sync + 'static,
    {
        Self {
            handler: Arc::new(handler),
            call_count: Arc::new(Mutex::new(0)),
            complete_count: Arc::new(Mutex::new(0)),
            stream_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a mock that always returns a text response.
    pub fn with_text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self::new(move |_| {
            Ok(Response {
                content: vec![Content::text(&text)],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        })
    }

    /// Create a mock from a sequence of responses.
    ///
    /// Each call to `complete()` returns the next response in order.
    /// After the sequence is exhausted, returns the last response repeatedly.
    pub fn with_responses(responses: Vec<Response>) -> Self {
        let responses = Arc::new(responses);
        let index = Arc::new(Mutex::new(0usize));
        Self::new(move |_| {
            let mut i = index.lock().unwrap();
            let resp_idx = (*i).min(responses.len() - 1);
            *i += 1;
            let r = &responses[resp_idx];
            Ok(Response {
                content: r.content.clone(),
                stop_reason: r.stop_reason,
                usage: r.usage.clone(),
            })
        })
    }

    /// Returns how many times `complete()` or `stream()` was called.
    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    /// Returns how many times `complete()` was called specifically.
    pub fn complete_count(&self) -> usize {
        *self.complete_count.lock().unwrap()
    }

    /// Returns how many times `stream()` was called specifically.
    /// Used by tests that need to assert which API surface a consumer
    /// took (e.g. SubAgent's trace_hook routing: stream when set, run
    /// when unset).
    pub fn stream_count(&self) -> usize {
        *self.stream_count.lock().unwrap()
    }
}

#[async_trait]
impl LlmProvider for Mock {
    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
        }
        {
            let mut count = self.complete_count.lock().unwrap();
            *count += 1;
        }
        (self.handler)(&request)
    }

    async fn stream(&self, request: Request) -> Result<ProviderEventStream, ProviderError> {
        // Mock streaming reuses the scripted Response: every Text block
        // becomes one ContentDelta, every ToolUse becomes one atomic
        // ToolUse event, then MessageDelta + Usage + Done. Tests can
        // exercise the streaming code path without real SSE.
        let response = {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            {
                let mut s = self.stream_count.lock().unwrap();
                *s += 1;
            }
            (self.handler)(&request)?
        };

        let events = response_to_events(response);
        Ok(Box::pin(stream::iter(events.into_iter().map(Ok))))
    }
}

fn response_to_events(response: Response) -> Vec<StreamEvent> {
    let mut events: Vec<StreamEvent> = Vec::new();
    for content in response.content {
        match content {
            Content::Text { text, .. } => events.push(StreamEvent::ContentDelta(text)),
            Content::Thinking {
                text,
                provider,
                metadata,
            } => {
                events.push(StreamEvent::ThinkingDelta { text: text.clone() });
                events.push(StreamEvent::ThinkingBlock {
                    text,
                    provider,
                    metadata,
                });
            }
            Content::ToolUse { id, name, input } => {
                events.push(StreamEvent::ToolUse { id, name, input })
            }
            Content::ToolResult { .. } => {
                // ToolResult doesn't appear in assistant output; skip.
            }
            Content::Image { .. } => {}
        }
    }
    events.push(StreamEvent::MessageDelta {
        stop_reason: response.stop_reason,
    });
    events.push(StreamEvent::Usage(response.usage));
    events.push(StreamEvent::Done);
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Request;
    use futures::StreamExt;
    use serde_json::json;

    fn empty_request() -> Request {
        Request {
            model: "test".into(),
            system: None,
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: None,
            thinking: None,
        }
    }

    #[tokio::test]
    async fn mock_stream_text_yields_one_delta_then_terminal_events() {
        let mock = Mock::with_text("hello");
        let mut s = mock.stream(empty_request()).await.unwrap();

        let mut events = Vec::new();
        while let Some(ev) = s.next().await {
            events.push(ev.unwrap());
        }

        assert!(matches!(events[0], StreamEvent::ContentDelta(ref t) if t == "hello"));
        assert!(matches!(
            events[1],
            StreamEvent::MessageDelta {
                stop_reason: StopReason::EndTurn
            }
        ));
        assert!(matches!(events[2], StreamEvent::Usage(_)));
        assert!(matches!(events[3], StreamEvent::Done));
    }

    #[tokio::test]
    async fn mock_stream_tool_use_emits_atomic_event() {
        let mock = Mock::new(|_| {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        });
        let mut s = mock.stream(empty_request()).await.unwrap();

        let first = s.next().await.unwrap().unwrap();
        match first {
            StreamEvent::ToolUse { id, name, input } => {
                assert_eq!(id, "t1");
                assert_eq!(name, "bash");
                assert_eq!(input["command"], "ls");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }
}
