use async_trait::async_trait;
use serde_json::Value;

use crate::error::ProviderError;
use crate::message::{CacheControl, Content, ImageSource, Message, Role, StopReason, Usage};
use crate::stream::ProviderEventStream;

/// Definition of a tool that gets sent to the LLM.
///
/// `cache_control` terminates a cached prefix segment at this tool
/// definition. The Anthropic API caches tool definitions in the order
/// they appear in the request — placing a breakpoint on the **last**
/// tool caches the entire toolset. Non-Anthropic providers ignore
/// this field.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub cache_control: Option<CacheControl>,
}

/// One block of the system prompt.
///
/// `Request.system` is a list because Anthropic's API accepts multiple
/// system blocks, each individually markable with [`CacheControl`] —
/// useful when part of the system prompt is stable (e.g. base
/// instructions) and part rotates per-call (e.g. user-specific context).
///
/// Non-Anthropic providers concatenate all blocks with `\n\n` into a
/// single system string and drop `cache_control`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemBlock {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    /// Plain system block, no cache breakpoint.
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            cache_control: None,
        }
    }

    /// System block marked as a cache breakpoint with the default 5m TTL.
    pub fn cached(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            cache_control: Some(CacheControl::ephemeral()),
        }
    }

    /// System block marked as a cache breakpoint with 1-hour TTL.
    pub fn cached_1h(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            cache_control: Some(CacheControl::ephemeral_1h()),
        }
    }
}

/// Request to the LLM provider.
#[derive(Debug, Clone)]
pub struct Request {
    pub model: String,
    pub system: Option<Vec<SystemBlock>>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    /// Optional per-call thinking/reasoning override. When set, providers
    /// that support thinking use it instead of their instance default.
    pub thinking: Option<ThinkingConfig>,
}

pub(crate) async fn resolve_request(mut request: Request) -> Result<Request, ProviderError> {
    for message in &mut request.messages {
        for content in &mut message.content {
            if let Content::Image { source } = content {
                if message.role != Role::User {
                    return Err(ProviderError::InvalidLocalInput {
                        message: "images are only valid in user messages".into(),
                    });
                }
                if let ImageSource::File { media_type, path } = source {
                    if !path.is_absolute() {
                        return Err(ProviderError::InvalidLocalInput {
                            message: format!(
                                "image file path must be absolute: {}",
                                path.display()
                            ),
                        });
                    }
                    let data = tokio::fs::read(&path).await.map_err(|source| {
                        ProviderError::ImageFileRead {
                            path: path.clone(),
                            source,
                        }
                    })?;
                    *source = ImageSource::Data {
                        media_type: std::mem::take(media_type),
                        data: data.into(),
                    };
                }
            }
        }
    }
    Ok(request)
}

/// Per-call thinking/reasoning configuration carried on [`Request`].
///
/// When set, this overrides the provider instance's construction-time
/// default (e.g. [`super::providers::Anthropic::with_thinking_budget`]).
/// Provider asymmetry: [`Self::Effort`] and [`Self::Disabled`] are
/// honored by every provider that supports thinking; [`Self::Budget`]
/// is **Anthropic-style** — OpenAI Responses and OpenAI Codex
/// providers ignore it and apply their instance defaults instead. See
/// the per-provider docstrings for the exact precedence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingConfig {
    /// Symbolic effort level. Each provider maps to its own native
    /// scale (Anthropic adaptive effort, OpenAI `reasoning.effort`).
    /// Portable across providers.
    Effort(ThinkingEffort),
    /// Explicit thinking budget in tokens. Anthropic-style only.
    /// OpenAI providers ignore this and fall back to their instance
    /// reasoning configuration.
    Budget(u32),
    /// Disable thinking for this call even when the provider instance
    /// has it enabled. Drops the entire `thinking`/`reasoning` block
    /// from the wire payload.
    Disabled,
}

/// Symbolic thinking effort tier. Per-call carrier — least common
/// denominator across providers. Vendor-specific tiers (Anthropic
/// `XHigh` / `Max`; future extensions) reach the wire via
/// [`Self::Other`] which forwards verbatim and routes through each
/// provider's case-insensitive `From<&str>` parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingEffort {
    /// Low / minimal reasoning effort.
    Low,
    /// Medium / balanced reasoning effort.
    Medium,
    /// High / maximum standard reasoning effort.
    High,
    /// Vendor-specific tier (e.g. `"xhigh"`, `"max"`). Forwarded
    /// verbatim. Each provider's `From<&str>` parse resolves
    /// recognised values to typed variants; unknown values are sent as
    /// raw strings and the server may 4xx — that's the caller's
    /// responsibility.
    Other(String),
}

/// Response from the LLM provider.
#[derive(Debug, Clone)]
pub struct Response {
    pub content: Vec<Content>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

/// Trait for LLM providers (Anthropic, OpenAI, etc.).
///
/// Implement this trait to add support for a new LLM provider.
///
/// Two API surfaces, two transports:
///
/// - [`complete`](LlmProvider::complete) — single-shot HTTP, fully
///   buffered response. Lowest overhead when the caller wants the
///   final answer in one go.
/// - [`stream`](LlmProvider::stream) — Server-Sent Events HTTP, emits
///   incremental [`StreamEvent`](crate::stream::StreamEvent)s as the
///   model produces them.
///
/// Implementations are **independent code paths**: streaming is not
/// derived from `complete()` and vice-versa. Errors that happen
/// before the stream begins (auth, malformed request, connection
/// refused) surface from the `stream(...)` async fn itself; errors
/// that surface mid-stream (parse failures, mid-body HTTP errors)
/// arrive as `Err` items inside the stream.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, request: Request) -> Result<Response, ProviderError>;

    async fn stream(&self, request: Request) -> Result<ProviderEventStream, ProviderError>;
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use std::path::PathBuf;

    fn request(content: Content, role: Role) -> Request {
        Request {
            model: "test".into(),
            system: None,
            messages: vec![Message {
                role,
                content: vec![content],
            }],
            tools: vec![],
            max_tokens: 1,
            temperature: None,
            thinking: None,
        }
    }

    #[tokio::test]
    async fn resolves_absolute_image_file_to_exact_bytes() {
        let path = std::env::temp_dir().join(format!("tkach-image-{}", std::process::id()));
        tokio::fs::write(&path, b"image bytes").await.unwrap();
        let resolved =
            resolve_request(request(Content::image_file("image/png", &path), Role::User))
                .await
                .unwrap();
        tokio::fs::remove_file(path).await.unwrap();

        assert!(matches!(
            &resolved.messages[0].content[0],
            Content::Image { source: ImageSource::Data { media_type, data } }
                if media_type == "image/png" && data.as_ref() == b"image bytes"
        ));
    }

    #[tokio::test]
    async fn rejects_relative_file_and_assistant_image_without_retry() {
        let relative = resolve_request(request(
            Content::image_file("image/png", PathBuf::from("image.png")),
            Role::User,
        ))
        .await
        .unwrap_err();
        assert!(matches!(relative, ProviderError::InvalidLocalInput { .. }));
        assert!(!relative.is_retryable());

        let assistant = resolve_request(request(
            Content::image_url("https://example.test/image.png"),
            Role::Assistant,
        ))
        .await
        .unwrap_err();
        assert!(matches!(assistant, ProviderError::InvalidLocalInput { .. }));
        assert!(!assistant.is_retryable());
    }

    #[tokio::test]
    async fn missing_image_file_error_carries_path() {
        let path = std::env::temp_dir().join("tkach-missing-image-does-not-exist");
        let error = resolve_request(request(Content::image_file("image/png", &path), Role::User))
            .await
            .unwrap_err();
        assert!(
            matches!(error, ProviderError::ImageFileRead { path: ref actual, .. } if actual == &path)
        );
        assert!(!error.is_retryable());
    }
}
