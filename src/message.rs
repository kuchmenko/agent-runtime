use std::path::PathBuf;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    #[serde(rename = "image")]
    Image { source: ImageSource },

    #[serde(rename = "thinking")]
    Thinking {
        text: String,
        provider: ThinkingProvider,
        metadata: ThinkingMetadata,
    },

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ImageSource {
    Data {
        media_type: String,
        #[serde(with = "base64_bytes")]
        data: Bytes,
    },
    File {
        media_type: String,
        path: PathBuf,
    },
    Url {
        url: String,
    },
}

mod base64_bytes {
    use base64::Engine;
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(data: &Bytes, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&base64::engine::general_purpose::STANDARD.encode(data))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Bytes, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map(Bytes::from)
            .map_err(serde::de::Error::custom)
    }
}

/// Provider that produced a persisted thinking/reasoning block.
///
/// Thinking blocks are protocol state, not just UI decoration: Anthropic
/// needs signatures, while OpenAI Responses/Codex can need reasoning item
/// IDs and encrypted replay state. Keeping provider identity typed avoids
/// accidentally replaying one provider's opaque state through another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingProvider {
    Anthropic,
    #[serde(rename = "openai_responses")]
    OpenAIResponses,
    #[serde(rename = "openai_compatible")]
    OpenAICompatible,
}

/// Provider-specific metadata for a completed thinking/reasoning block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingMetadata {
    None,
    Anthropic {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    AnthropicRedacted {
        data: String,
    },
    #[serde(rename = "openai_responses")]
    OpenAIResponses {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_index: Option<usize>,
        summary_index: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

/// Cache breakpoint marker for Anthropic prompt caching.
///
/// Place on a [`Content::Text`], [`Content::ToolResult`],
/// [`crate::provider::SystemBlock`], or [`crate::provider::ToolDefinition`]
/// to terminate a cached prefix segment at that point. The Anthropic API
/// caches everything from the start of the request up to and including
/// the marked block.
///
/// Anthropic accepts up to **4 cache breakpoints per request**; exceeding
/// this is rejected by the API.
///
/// Mutating any byte at or before a breakpoint invalidates the cached
/// prefix from that breakpoint forward — keep cached prefixes stable
/// across calls.
///
/// Pricing (relative to base input token cost):
/// - 5-minute cache writes: **1.25x**
/// - 1-hour cache writes: **2x**
/// - cache reads: **0.1x**
///
/// Net savings require a hit rate above ~25% for 5m TTL or ~50% for 1h.
///
/// Other providers ([`crate::providers::OpenAICompatible`],
/// [`crate::providers::Mock`]) silently ignore this field — they
/// pattern-match `Content` variants and rebuild their own wire types,
/// so the field is dropped at the conversion layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
    /// Ephemeral cache breakpoint. Default TTL 5 minutes; `Some(OneHour)`
    /// opts into the 1-hour TTL inline (no beta header required).
    Ephemeral {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ttl: Option<CacheTtl>,
    },
}

impl CacheControl {
    /// Convenience: `Ephemeral { ttl: None }`. Wire-equivalent to `ttl: "5m"`.
    pub fn ephemeral() -> Self {
        CacheControl::Ephemeral { ttl: None }
    }

    /// Convenience: `Ephemeral { ttl: Some(OneHour) }`.
    pub fn ephemeral_1h() -> Self {
        CacheControl::Ephemeral {
            ttl: Some(CacheTtl::OneHour),
        }
    }
}

/// Cache TTL on an [`CacheControl::Ephemeral`] breakpoint.
///
/// `None` on the parent enum and `Some(FiveMinutes)` are wire-equivalent
/// (both default to 5m at the API). `OneHour` is the only case that
/// changes observable behaviour.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

/// Reason the model stopped generating.
///
/// `EndTurn`, `ToolUse`, `MaxTokens`, `StopSequence`, `PauseTurn` are
/// reported by the provider. `Cancelled` is runtime-only: it appears in
/// [`crate::AgentResult::stop_reason`] on a partial result when the
/// caller's `CancellationToken` fires. No provider ever returns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    PauseTurn,
    Cancelled,
}

/// Token usage report from a provider call.
///
/// `cache_creation_input_tokens` and `cache_read_input_tokens` are
/// populated only by Anthropic responses when the request used
/// [`CacheControl`] breakpoints; non-Anthropic providers leave them at 0.
///
/// In streaming mode, the cache fields arrive on the `message_start`
/// event only; the Anthropic provider merges them across events so the
/// final emitted `Usage` carries all four numbers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

impl Usage {
    /// Merge another `Usage` into this one by taking the max of each
    /// field. Used by streaming consumers that receive multiple
    /// `Usage` events for the same turn (`message_start` carries
    /// `input_tokens` + cache fields, `message_delta` carries the
    /// final `output_tokens`).
    pub fn merge_max(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.max(other.input_tokens);
        self.output_tokens = self.output_tokens.max(other.output_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .max(other.cache_creation_input_tokens);
        self.cache_read_input_tokens = self
            .cache_read_input_tokens
            .max(other.cache_read_input_tokens);
    }

    /// Sum-aggregate another `Usage` into this one, field by field.
    /// Used to accumulate token counts across turns of an agent run —
    /// cache costs are per-turn, so summing them gives total cost.
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .saturating_add(other.cache_creation_input_tokens);
        self.cache_read_input_tokens = self
            .cache_read_input_tokens
            .saturating_add(other.cache_read_input_tokens);
    }
}

impl ThinkingMetadata {
    pub fn anthropic(signature: Option<String>) -> Self {
        ThinkingMetadata::Anthropic { signature }
    }

    pub fn anthropic_redacted(data: impl Into<String>) -> Self {
        ThinkingMetadata::AnthropicRedacted { data: data.into() }
    }

    pub fn openai_responses(
        item_id: Option<String>,
        output_index: Option<usize>,
        summary_index: usize,
        encrypted_content: Option<String>,
    ) -> Self {
        ThinkingMetadata::OpenAIResponses {
            item_id,
            output_index,
            summary_index,
            encrypted_content,
        }
    }
}

// --- Constructors ---

impl Message {
    pub fn user(content: Vec<Content>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::text(text)],
        }
    }

    pub fn assistant(content: Vec<Content>) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }

    /// Extract all text content joined together.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Extract all tool use blocks.
    pub fn tool_uses(&self) -> Vec<(&str, &str, &Value)> {
        self.content
            .iter()
            .filter_map(|c| match c {
                Content::ToolUse { id, name, input } => Some((id.as_str(), name.as_str(), input)),
                _ => None,
            })
            .collect()
    }
}

impl Content {
    pub fn text(text: impl Into<String>) -> Self {
        Content::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    pub fn image_bytes(media_type: impl Into<String>, data: impl Into<Bytes>) -> Self {
        Content::Image {
            source: ImageSource::Data {
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }

    pub fn image_file(media_type: impl Into<String>, absolute_path: impl Into<PathBuf>) -> Self {
        Content::Image {
            source: ImageSource::File {
                media_type: media_type.into(),
                path: absolute_path.into(),
            },
        }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Content::Image {
            source: ImageSource::Url { url: url.into() },
        }
    }

    pub fn thinking(
        text: impl Into<String>,
        provider: ThinkingProvider,
        metadata: ThinkingMetadata,
    ) -> Self {
        Content::Thinking {
            text: text.into(),
            provider,
            metadata,
        }
    }

    /// Text block marked as a cache breakpoint with the default 5m TTL.
    pub fn text_cached(text: impl Into<String>) -> Self {
        Content::Text {
            text: text.into(),
            cache_control: Some(CacheControl::ephemeral()),
        }
    }

    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Content::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error,
            cache_control: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_data_round_trips_as_base64() {
        let value = Content::image_bytes("image/png", Bytes::from_static(b"png"));
        let json = serde_json::to_value(&value).unwrap();
        let round_trip: Content = serde_json::from_value(json.clone()).unwrap();

        assert_eq!(json["source"]["data"], "cG5n");
        assert!(json["source"]["data"].is_string());
        assert!(matches!(
            round_trip,
            Content::Image {
                source: ImageSource::Data { media_type, data }
            } if media_type == "image/png" && data.as_ref() == b"png"
        ));
    }

    #[test]
    fn image_file_and_url_round_trip() {
        let file = Content::image_file("image/jpeg", PathBuf::from("/tmp/image.jpg"));
        let url = Content::image_url("https://example.test/image.webp");

        let file_json = serde_json::to_value(&file).unwrap();
        let url_json = serde_json::to_value(&url).unwrap();
        let _: Content = serde_json::from_value(file_json).unwrap();
        let _: Content = serde_json::from_value(url_json).unwrap();
    }

    #[test]
    fn image_data_rejects_invalid_base64() {
        let json = serde_json::json!({
            "type": "image",
            "source": {
                "type": "data",
                "media_type": "image/png",
                "data": "not base64"
            }
        });

        assert!(serde_json::from_value::<Content>(json).is_err());
    }

    #[test]
    fn thinking_content_serializes_with_provider_metadata() {
        let content = Content::thinking(
            "inspected repo",
            ThinkingProvider::OpenAIResponses,
            ThinkingMetadata::openai_responses(
                Some("rs_123".into()),
                None,
                0,
                Some("encrypted".into()),
            ),
        );

        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json["type"], "thinking");
        assert_eq!(json["provider"], "openai_responses");
        assert_eq!(json["metadata"]["type"], "openai_responses");
        assert_eq!(json["metadata"]["item_id"], "rs_123");

        assert_thinking_roundtrip(json);
    }

    fn assert_thinking_roundtrip(json: serde_json::Value) {
        let roundtrip: Content = serde_json::from_value(json).unwrap();
        let Content::Thinking {
            text,
            provider,
            metadata,
        } = roundtrip
        else {
            panic!("expected thinking content");
        };

        assert_eq!(text, "inspected repo");
        assert_eq!(provider, ThinkingProvider::OpenAIResponses);
        assert_eq!(
            metadata,
            ThinkingMetadata::OpenAIResponses {
                item_id: Some("rs_123".into()),
                output_index: None,
                summary_index: 0,
                encrypted_content: Some("encrypted".into()),
            }
        );
    }
}
