use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;

use crate::agent::AgentResult;

/// Error returned by [`crate::Agent::run`].
///
/// Every variant carries a `partial: Box<AgentResult>` holding whatever
/// delta was accumulated before the failure. The `Box` keeps `AgentError`
/// small enough to satisfy `clippy::result_large_err` — `AgentResult` is
/// ~72 bytes and would otherwise push the enum past the 128-byte
/// stack-Result threshold. Accessors auto-deref the box, so most callers
/// see `&AgentResult` and don't need to think about the indirection.
///
/// Callers that maintain their own history (the stateless-agent contract)
/// can persist progress on error:
///
/// ```ignore
/// let delta = match agent.run(history.clone(), cancel).await {
///     Ok(result) => result,
///     Err(e) => e.into_partial(),
/// };
/// history.extend(delta.new_messages);
/// save_session(&history);
/// ```
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("max turns ({turns}) reached without completion")]
    MaxTurnsReached {
        turns: usize,
        partial: Box<AgentResult>,
    },

    #[error("provider error: {source}")]
    Provider {
        #[source]
        source: ProviderError,
        partial: Box<AgentResult>,
    },

    #[error("cancelled")]
    Cancelled { partial: Box<AgentResult> },

    #[error("tool '{tool_name}' failed: {source}")]
    Tool {
        tool_name: String,
        #[source]
        source: ToolError,
        partial: Box<AgentResult>,
    },
}

impl AgentError {
    /// Borrow the partial result accumulated before the error.
    pub fn partial(&self) -> &AgentResult {
        match self {
            AgentError::MaxTurnsReached { partial, .. }
            | AgentError::Provider { partial, .. }
            | AgentError::Cancelled { partial }
            | AgentError::Tool { partial, .. } => partial,
        }
    }

    /// Consume the error and take ownership of the partial result.
    pub fn into_partial(self) -> AgentResult {
        match self {
            AgentError::MaxTurnsReached { partial, .. }
            | AgentError::Provider { partial, .. }
            | AgentError::Cancelled { partial }
            | AgentError::Tool { partial, .. } => *partial,
        }
    }
}

/// Errors returned by [`crate::LlmProvider`] implementations.
///
/// Retry policy lives with the caller — the provider only classifies.
/// Use [`ProviderError::is_retryable`] and [`ProviderError::retry_after`]
/// to drive retry/backoff logic.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("invalid local request content: {message}")]
    InvalidLocalInput { message: String },

    #[error("failed to read image file {path}: {source}")]
    ImageFileRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Transport-layer failure (connection refused, timeout, DNS, etc.).
    /// Retryable by default — the request likely never landed on the server.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// API-level error with an explicit HTTP status.
    /// `retryable` is set by the provider mapping (5xx transient ⇒ true).
    #[error("API error ({status}): {message}")]
    Api {
        status: u16,
        message: String,
        retryable: bool,
    },

    /// Server is temporarily overloaded (Anthropic 529, OpenAI 503 overloaded).
    /// Always retryable; `retry_after_ms` is parsed from the `Retry-After`
    /// header when present.
    #[error("overloaded (retry after: {retry_after_ms:?}ms)")]
    Overloaded { retry_after_ms: Option<u64> },

    /// Rate limit exceeded (HTTP 429). Always retryable after the indicated
    /// delay, when provided by the server.
    #[error("rate limited (retry after: {retry_after_ms:?}ms)")]
    RateLimit { retry_after_ms: Option<u64> },

    /// Response body was malformed — the server returned 2xx but the payload
    /// could not be parsed. Not retryable: re-sending the same request will
    /// produce the same garbage.
    #[error("deserialization error: {0}")]
    Deserialization(#[from] serde_json::Error),

    /// `Anthropic::batch_results` was called while the batch was still
    /// `in_progress` or `canceling`. Caller must keep polling
    /// `retrieve_batch` until the status reaches `ended`, then re-call.
    /// Carries the textual status (Anthropic's wire form) so callers can
    /// log without round-tripping through the typed enum.
    #[error("batch not ready (status: {status})")]
    BatchNotReady { status: String },

    /// Escape hatch for provider-specific errors that don't fit the above.
    /// Not retryable by default.
    #[error("{0}")]
    Other(String),
}

impl ProviderError {
    /// Is it safe to retry the same request after a backoff?
    ///
    /// `Http` errors are split: timeouts, connect failures, body-read
    /// glitches, and generic request failures are transient and
    /// retryable; decode failures (malformed bytes — same input means
    /// same parse failure), builder errors (caller bug — bad URL,
    /// invalid header), and redirect cycles are persistent and not.
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::Http(e) => is_transient_reqwest(e),
            ProviderError::Api { retryable, .. } => *retryable,
            ProviderError::Overloaded { .. } | ProviderError::RateLimit { .. } => true,
            ProviderError::Deserialization(_)
            | ProviderError::Other(_)
            | ProviderError::BatchNotReady { .. }
            | ProviderError::InvalidLocalInput { .. }
            | ProviderError::ImageFileRead { .. } => false,
        }
    }

    /// Suggested wait before retrying, if the server indicated one.
    /// Returns `None` when the caller should use its own backoff strategy.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            ProviderError::Overloaded { retry_after_ms }
            | ProviderError::RateLimit { retry_after_ms } => {
                retry_after_ms.map(Duration::from_millis)
            }
            _ => None,
        }
    }
}

/// Classify a `reqwest::Error` as transient (retryable) or permanent.
///
/// reqwest groups failures into orthogonal categories via `is_*` predicates.
/// We retry on transport-level glitches the server might recover from
/// (network blips, timeouts) and refuse to retry on caller-side bugs
/// (malformed URL, broken response decoding) — the same input would
/// produce the same failure on the next attempt.
fn is_transient_reqwest(e: &reqwest::Error) -> bool {
    if e.is_decode() || e.is_builder() || e.is_redirect() {
        return false;
    }
    // Timeout / connect / body-read / generic request → transient.
    e.is_timeout() || e.is_connect() || e.is_body() || e.is_request()
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The cooperative cancellation signal from [`crate::ToolContext::cancel`]
    /// fired before the tool completed. Long-running tools must return this
    /// rather than swallowing the signal or looping indefinitely.
    #[error("cancelled")]
    Cancelled,

    #[error("{0}")]
    Execution(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_retryable_flag_respected() {
        let retryable = ProviderError::Api {
            status: 500,
            message: "internal".into(),
            retryable: true,
        };
        assert!(retryable.is_retryable());
        assert!(retryable.retry_after().is_none());

        let fatal = ProviderError::Api {
            status: 400,
            message: "bad request".into(),
            retryable: false,
        };
        assert!(!fatal.is_retryable());
    }

    #[test]
    fn overloaded_always_retryable() {
        let with_hint = ProviderError::Overloaded {
            retry_after_ms: Some(5_000),
        };
        assert!(with_hint.is_retryable());
        assert_eq!(with_hint.retry_after(), Some(Duration::from_millis(5_000)));

        let without = ProviderError::Overloaded {
            retry_after_ms: None,
        };
        assert!(without.is_retryable());
        assert_eq!(without.retry_after(), None);
    }

    #[test]
    fn rate_limit_always_retryable() {
        let rl = ProviderError::RateLimit {
            retry_after_ms: Some(1_500),
        };
        assert!(rl.is_retryable());
        assert_eq!(rl.retry_after(), Some(Duration::from_millis(1_500)));
    }

    #[test]
    fn deserialization_never_retryable() {
        let err: serde_json::Error = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let e = ProviderError::Deserialization(err);
        assert!(!e.is_retryable());
        assert!(e.retry_after().is_none());
    }

    #[test]
    fn batch_not_ready_never_retryable() {
        let e = ProviderError::BatchNotReady {
            status: "in_progress".into(),
        };
        assert!(!e.is_retryable());
        assert!(e.retry_after().is_none());
    }

    #[test]
    fn other_never_retryable() {
        assert!(!ProviderError::Other("weird".into()).is_retryable());
    }
}
