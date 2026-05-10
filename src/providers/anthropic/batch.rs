//! Anthropic Message Batches API.
//!
//! Async submission of up to 100 k requests, run server-side over a
//! window of up to 24 h at **50 % off** input + output tokens. Stacks
//! with prompt caching for ≈85 % off when 1 h cache breakpoints are in
//! play.
//!
//! Lifecycle:
//! 1. [`Anthropic::create_batch`] — submit `Vec<BatchRequest>` (JSON
//!    body with a `requests` array on the wire), returns a
//!    [`BatchHandle`] with `status=InProgress`.
//! 2. [`Anthropic::retrieve_batch`] — poll the handle. Caller owns the
//!    cadence (5 min, 1 h, exp-backoff) — tkach exposes the primitive,
//!    not a blocking helper.
//! 3. [`Anthropic::batch_results`] — once `status=Ended`, stream
//!    [`BatchResult`]s line-by-line so a 100 k-row batch can be
//!    persisted without a 200 MB caller-side buffer.
//! 4. [`Anthropic::cancel_batch`] / [`Anthropic::list_batches`] — best-
//!    effort cancel and pagination over recent batches.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::OnceLock;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::{Stream, StreamExt};
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::{
    API_VERSION, Anthropic, ApiRequest, ApiResponse, build_request_body_with_thinking,
    classify_error, convert_response, parse_retry_after,
};
use crate::error::ProviderError;
use crate::provider::{Request, Response};

/// Pinned, boxed stream of batch results — ergonomic for callers that
/// drive it with `.next().await` without needing to `Box::pin` themselves.
pub type BatchResultStream = Pin<Box<dyn Stream<Item = Result<BatchResult, ProviderError>> + Send>>;

// --- Public types -----------------------------------------------------------

/// One row of a batch submission.
///
/// `custom_id` is caller-supplied and must:
/// - match `^[a-zA-Z0-9_-]{1,64}$`,
/// - be unique within the batch.
///
/// Both rules are enforced client-side by [`Anthropic::create_batch`]
/// before the HTTP call, so malformed input fails fast without paying
/// for a network round-trip.
///
/// `params` reuses the existing typed [`Request`] — model, system,
/// messages, tools, max_tokens, temperature. Once cache breakpoints are
/// attached via `SystemBlock::cached_1h(...)` they ride through the batch
/// path identically to the sync path.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub custom_id: String,
    pub params: Request,
}

/// Server-side state of a batch.
///
/// Returned by [`Anthropic::create_batch`], [`Anthropic::retrieve_batch`],
/// [`Anthropic::cancel_batch`] and [`Anthropic::list_batches`].
#[derive(Debug, Clone)]
pub struct BatchHandle {
    pub id: String,
    pub status: BatchStatus,
    pub request_counts: BatchCounts,
    pub created_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    /// 29 days after `created_at` per Anthropic policy.
    pub expires_at: DateTime<Utc>,
}

impl BatchHandle {
    /// True when `status == Ended`. Convenience for polling loops.
    pub fn is_terminal(&self) -> bool {
        self.status == BatchStatus::Ended
    }
}

/// Server-side processing state.
///
/// Maps to Anthropic's `processing_status` field on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchStatus {
    InProgress,
    Canceling,
    Ended,
}

impl BatchStatus {
    /// Anthropic's wire-form spelling (used in URLs, logs, and
    /// [`ProviderError::BatchNotReady`]).
    pub fn as_wire_str(self) -> &'static str {
        match self {
            BatchStatus::InProgress => "in_progress",
            BatchStatus::Canceling => "canceling",
            BatchStatus::Ended => "ended",
        }
    }
}

/// Per-outcome counts on a batch handle.
///
/// While `status=InProgress`, `processing` decreases as the server
/// resolves rows; the other counts increase as outcomes finalise.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchCounts {
    pub processing: u32,
    pub succeeded: u32,
    pub errored: u32,
    pub canceled: u32,
    pub expired: u32,
}

/// One row of batch results, identified by the caller's `custom_id`.
#[derive(Debug)]
pub struct BatchResult {
    pub custom_id: String,
    pub outcome: BatchOutcome,
}

/// Per-row terminal state.
#[derive(Debug)]
pub enum BatchOutcome {
    /// Row succeeded. Carries the same [`Response`] shape as a sync
    /// `complete()` call — content blocks, stop reason, and `Usage`
    /// (incl. `cache_creation_input_tokens` + `cache_read_input_tokens`
    /// when cache breakpoints were used).
    Succeeded(Response),
    /// Per-row server error. Distinct from a stream-level transport
    /// error, which surfaces as `Err(ProviderError)` on the stream
    /// item. Anthropic's per-row error envelope only carries
    /// `{type, message}` — no HTTP status — so we don't reuse
    /// [`ProviderError::Api`] here.
    Errored { error_type: String, message: String },
    /// Row was cancelled by [`Anthropic::cancel_batch`] before it ran.
    Canceled,
    /// Batch's 24 h processing window elapsed before the row ran.
    Expired,
}

/// Pagination options for [`Anthropic::list_batches`].
///
/// Anthropic uses cursor-style pagination — `before_id` / `after_id`
/// take the `id` of an item in a prior page.
#[derive(Debug, Default, Clone)]
pub struct ListBatchesOpts {
    /// 1..=100. Defaults to 20 server-side when `None`.
    pub limit: Option<u32>,
    pub before_id: Option<String>,
    pub after_id: Option<String>,
}

// --- Wire DTOs (private) ----------------------------------------------------

#[derive(Serialize)]
struct CreateBatchBody<'a> {
    requests: Vec<RequestEntry<'a>>,
}

#[derive(Serialize)]
struct RequestEntry<'a> {
    custom_id: &'a str,
    params: ApiRequest,
}

#[derive(Deserialize)]
struct ApiBatchHandle {
    id: String,
    processing_status: String,
    request_counts: ApiBatchCounts,
    created_at: DateTime<Utc>,
    #[serde(default)]
    ended_at: Option<DateTime<Utc>>,
    expires_at: DateTime<Utc>,
}

#[derive(Deserialize, Default)]
struct ApiBatchCounts {
    #[serde(default)]
    processing: u32,
    #[serde(default)]
    succeeded: u32,
    #[serde(default)]
    errored: u32,
    #[serde(default)]
    canceled: u32,
    #[serde(default)]
    expired: u32,
}

#[derive(Deserialize)]
struct ListBatchesResponse {
    data: Vec<ApiBatchHandle>,
}

#[derive(Deserialize)]
struct BatchResultLine {
    custom_id: String,
    result: ApiBatchOutcome,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiBatchOutcome {
    Succeeded { message: ApiResponse },
    Errored { error: ApiBatchError },
    Canceled,
    Expired,
}

#[derive(Deserialize)]
struct ApiBatchError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

// --- Conversions ------------------------------------------------------------

fn parse_status(raw: &str) -> Result<BatchStatus, ProviderError> {
    match raw {
        "in_progress" => Ok(BatchStatus::InProgress),
        "canceling" => Ok(BatchStatus::Canceling),
        "ended" => Ok(BatchStatus::Ended),
        other => Err(ProviderError::Other(format!(
            "unknown batch processing_status: {other:?}"
        ))),
    }
}

impl ApiBatchHandle {
    fn into_handle(self) -> Result<BatchHandle, ProviderError> {
        Ok(BatchHandle {
            status: parse_status(&self.processing_status)?,
            request_counts: BatchCounts {
                processing: self.request_counts.processing,
                succeeded: self.request_counts.succeeded,
                errored: self.request_counts.errored,
                canceled: self.request_counts.canceled,
                expired: self.request_counts.expired,
            },
            id: self.id,
            created_at: self.created_at,
            ended_at: self.ended_at,
            expires_at: self.expires_at,
        })
    }
}

impl From<BatchResultLine> for BatchResult {
    fn from(line: BatchResultLine) -> Self {
        let outcome = match line.result {
            ApiBatchOutcome::Succeeded { message } => {
                BatchOutcome::Succeeded(convert_response(ApiResponse {
                    content: message.content,
                    stop_reason: message.stop_reason,
                    usage: message.usage,
                }))
            }
            ApiBatchOutcome::Errored { error } => BatchOutcome::Errored {
                error_type: error.error_type,
                message: error.message,
            },
            ApiBatchOutcome::Canceled => BatchOutcome::Canceled,
            ApiBatchOutcome::Expired => BatchOutcome::Expired,
        };
        BatchResult {
            custom_id: line.custom_id,
            outcome,
        }
    }
}

// --- custom_id validation ---------------------------------------------------

/// Anthropic's documented constraint on `custom_id`.
fn custom_id_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[a-zA-Z0-9_-]{1,64}$").unwrap())
}

fn validate_unique_and_well_formed(requests: &[BatchRequest]) -> Result<(), ProviderError> {
    let re = custom_id_regex();
    let mut seen: HashSet<&str> = HashSet::with_capacity(requests.len());
    for req in requests {
        if !re.is_match(&req.custom_id) {
            return Err(ProviderError::Other(format!(
                "invalid custom_id {:?}: must match ^[a-zA-Z0-9_-]{{1,64}}$",
                req.custom_id
            )));
        }
        if !seen.insert(&req.custom_id) {
            return Err(ProviderError::Other(format!(
                "duplicate custom_id {:?} in batch (must be unique)",
                req.custom_id
            )));
        }
    }
    Ok(())
}

// --- Inherent methods on Anthropic -----------------------------------------

impl Anthropic {
    /// Submit a batch.
    ///
    /// Validates every `custom_id` (regex + dedup) before the HTTP call
    /// — bad input fails fast with [`ProviderError::Other`] instead of
    /// triggering an HTTP 400 round-trip.
    ///
    /// On success, returns a [`BatchHandle`] with `status=InProgress`.
    /// The actual processing happens server-side over up to 24 h.
    ///
    /// **Idempotency caveat.** Submission is *not* idempotent on the
    /// Anthropic side — re-submitting after a network glitch may create
    /// two batches. Callers that need at-most-once semantics should
    /// generate deterministic `custom_id`s and check
    /// [`Anthropic::list_batches`] for an existing batch before retrying.
    pub async fn create_batch(
        &self,
        requests: Vec<BatchRequest>,
    ) -> Result<BatchHandle, ProviderError> {
        validate_unique_and_well_formed(&requests)?;

        let entries: Vec<RequestEntry<'_>> = requests
            .iter()
            .map(|r| RequestEntry {
                custom_id: &r.custom_id,
                params: build_request_body_with_thinking(&r.params, self.thinking.clone()),
            })
            .collect();
        let body = CreateBatchBody { requests: entries };

        let response = self
            .post(&self.batches_url())
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();
        if status >= 400 {
            return Err(read_api_error(response).await);
        }

        let api: ApiBatchHandle = response.json().await?;
        api.into_handle()
    }

    /// Poll the server for the current status + counts.
    ///
    /// Cheap and idempotent — safe to call as fast as the caller wants
    /// to (within rate-limit reason).
    pub async fn retrieve_batch(&self, id: &str) -> Result<BatchHandle, ProviderError> {
        let url = format!("{}/{id}", self.batches_url());
        let response = self.get(&url).send().await?;

        let status = response.status().as_u16();
        if status >= 400 {
            return Err(read_api_error(response).await);
        }

        let api: ApiBatchHandle = response.json().await?;
        api.into_handle()
    }

    /// Stream `BatchResult`s for an ended batch.
    ///
    /// Returns `Err(ProviderError::BatchNotReady)` immediately if the
    /// batch is still `in_progress` or `canceling`. Once `ended`, the
    /// stream yields one [`BatchResult`] per JSONL line — caller can
    /// persist row-by-row without buffering the whole 200 MB body.
    ///
    /// Per-row server errors (model failed for this row) ride inside
    /// [`BatchOutcome::Errored`]. Stream-level errors (transport drop
    /// mid-body, malformed JSONL) ride as `Err(ProviderError)` on the
    /// stream item — caller decides whether to abort or skip.
    pub async fn batch_results(&self, id: &str) -> Result<BatchResultStream, ProviderError> {
        let url = format!("{}/{id}/results", self.batches_url());
        let response = self.get(&url).send().await?;

        let status = response.status().as_u16();
        if status >= 400 {
            let api_err = read_api_error(response).await;
            // Only plain "not ready" 4xx codes (400 / 404) warrant the
            // probe — those are what Anthropic returns when /results is
            // hit before the batch reaches `ended`. 429 and 5xx (incl.
            // Anthropic's 529) are retryable transport pressure
            // signals; masking them as `BatchNotReady` would flip
            // `is_retryable` to false and discard `retry_after_ms`,
            // pushing callers into aggressive polling against an
            // already-overloaded server.
            if is_premature_results_status(status) {
                return Err(promote_not_ready(self, id, api_err).await);
            }
            return Err(api_err);
        }

        let bytes = response.bytes_stream();
        Ok(Box::pin(jsonl_results(bytes)))
    }

    /// Best-effort cancel.
    ///
    /// Server flips to `canceling` and eventually to `ended`. Rows the
    /// server had already started processing complete normally
    /// ([`BatchOutcome::Succeeded`] / [`BatchOutcome::Errored`]); rows
    /// still queued surface as [`BatchOutcome::Canceled`] once results
    /// are read.
    pub async fn cancel_batch(&self, id: &str) -> Result<BatchHandle, ProviderError> {
        let url = format!("{}/{id}/cancel", self.batches_url());
        let response = self.post(&url).send().await?;

        let status = response.status().as_u16();
        if status >= 400 {
            return Err(read_api_error(response).await);
        }

        let api: ApiBatchHandle = response.json().await?;
        api.into_handle()
    }

    /// List recent batches.
    ///
    /// Cursor-style pagination via `before_id` / `after_id`. The default
    /// page size is 20 (Anthropic-side); pass `limit: Some(n)` for
    /// larger pages up to 100.
    pub async fn list_batches(
        &self,
        opts: ListBatchesOpts,
    ) -> Result<Vec<BatchHandle>, ProviderError> {
        let query = list_batches_query(opts);
        let mut req = self.get(&self.batches_url());
        if !query.is_empty() {
            req = req.query(&query);
        }

        let response = req.send().await?;

        let status = response.status().as_u16();
        if status >= 400 {
            return Err(read_api_error(response).await);
        }

        let api: ListBatchesResponse = response.json().await?;
        api.data
            .into_iter()
            .map(ApiBatchHandle::into_handle)
            .collect()
    }

    // --- HTTP helpers -------------------------------------------------------
    //
    // `batch` is a child module of `anthropic`, which means it has direct
    // access to `Anthropic`'s private fields. These helpers wrap the API
    // key + Anthropic-version header so each batch method body stays
    // focused on the verb + URL + body.

    fn post(&self, url: &str) -> reqwest::RequestBuilder {
        self.client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        self.client
            .get(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
    }
}

// --- HTTP error helpers -----------------------------------------------------

/// Project [`ListBatchesOpts`] into reqwest's query-pair shape, dropping
/// `None` fields. Pulled out of `list_batches` so the per-field
/// branching doesn't push the method past the cognitive-complexity
/// budget.
fn list_batches_query(opts: ListBatchesOpts) -> Vec<(&'static str, String)> {
    let mut query: Vec<(&'static str, String)> = Vec::new();
    if let Some(limit) = opts.limit {
        query.push(("limit", limit.to_string()));
    }
    if let Some(before) = opts.before_id {
        query.push(("before_id", before));
    }
    if let Some(after) = opts.after_id {
        query.push(("after_id", after));
    }
    query
}

async fn read_api_error(response: reqwest::Response) -> ProviderError {
    let status = response.status().as_u16();
    let retry_after_ms = parse_retry_after(response.headers());
    let text = response.text().await.unwrap_or_default();
    classify_error(status, text, retry_after_ms)
}

/// HTTP status codes Anthropic uses for the "results not ready yet"
/// signal on `/v1/messages/batches/{id}/results`. 429 and 5xx are
/// deliberately excluded — they're retryable transport pressure, not
/// not-ready, and would lose their retry-after hints if promoted.
fn is_premature_results_status(status: u16) -> bool {
    matches!(status, 400 | 404)
}

/// If the caller hit `batch_results` while the batch was still
/// `in_progress` or `canceling`, surface that as
/// [`ProviderError::BatchNotReady`] rather than a generic 4xx — caller
/// can drive a polling loop on that distinction.
async fn promote_not_ready(client: &Anthropic, id: &str, original: ProviderError) -> ProviderError {
    if let Ok(handle) = client.retrieve_batch(id).await {
        if handle.status != BatchStatus::Ended {
            return ProviderError::BatchNotReady {
                status: handle.status.as_wire_str().to_string(),
            };
        }
    }
    original
}

// --- JSONL line splitter ---------------------------------------------------

/// Split a stream of `Bytes` chunks into JSONL lines and parse each into
/// a [`BatchResult`]. Owns a small `Vec<u8>` buffer that holds the
/// trailing partial line between chunks — never buffers the whole
/// response.
fn jsonl_results<S>(bytes: S) -> impl Stream<Item = Result<BatchResult, ProviderError>>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send + 'static,
{
    futures::stream::unfold(
        (bytes, Vec::<u8>::new(), false),
        |(mut s, mut buf, finished)| async move {
            loop {
                if let Some(line) = drain_line(&mut buf) {
                    let parsed = parse_jsonl_line(&line);
                    return Some((parsed, (s, buf, finished)));
                }
                if finished {
                    return None;
                }
                match s.next().await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(e)) => {
                        return Some((Err(ProviderError::Http(e)), (s, buf, true)));
                    }
                    None => {
                        // Body fully drained. Drain any final un-newline-terminated
                        // line, then stop.
                        if buf.is_empty() {
                            return None;
                        }
                        let line = std::mem::take(&mut buf);
                        let parsed = parse_jsonl_line(&line);
                        return Some((parsed, (s, buf, true)));
                    }
                }
            }
        },
    )
}

fn drain_line(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    let nl = buf.iter().position(|&b| b == b'\n')?;
    let mut line: Vec<u8> = buf.drain(..=nl).collect();
    line.pop(); // strip trailing '\n'
    if line.last() == Some(&b'\r') {
        line.pop(); // strip trailing '\r' if CRLF
    }
    Some(line)
}

fn parse_jsonl_line(line: &[u8]) -> Result<BatchResult, ProviderError> {
    if line.iter().all(|b| b.is_ascii_whitespace()) {
        // Defensive: skip blank lines by surfacing nothing useful — we
        // shouldn't see these from Anthropic, but if we do, parsing
        // empty as JSON gives a confusing serde error.
        return Err(ProviderError::Other("empty JSONL line".into()));
    }
    let raw: BatchResultLine = serde_json::from_slice(line)?;
    Ok(BatchResult::from(raw))
}

// --- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

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

    #[test]
    fn validate_accepts_well_formed_unique_ids() {
        let v = vec![req("req-1"), req("req-2"), req("AB_cd-9")];
        validate_unique_and_well_formed(&v).expect("should accept");
    }

    #[test]
    fn validate_rejects_empty_custom_id() {
        let v = vec![req("")];
        let e = validate_unique_and_well_formed(&v).expect_err("empty rejected");
        assert!(matches!(e, ProviderError::Other(s) if s.contains("invalid custom_id")));
    }

    #[test]
    fn validate_rejects_too_long_custom_id() {
        let id = "x".repeat(65);
        let v = vec![req(&id)];
        let e = validate_unique_and_well_formed(&v).expect_err("too long rejected");
        assert!(matches!(e, ProviderError::Other(s) if s.contains("invalid custom_id")));
    }

    #[test]
    fn validate_rejects_special_chars() {
        let v = vec![req("req with space")];
        let e = validate_unique_and_well_formed(&v).expect_err("space rejected");
        assert!(matches!(e, ProviderError::Other(s) if s.contains("invalid custom_id")));
    }

    #[test]
    fn validate_rejects_duplicates() {
        let v = vec![req("req-1"), req("req-2"), req("req-1")];
        let e = validate_unique_and_well_formed(&v).expect_err("dup rejected");
        assert!(matches!(e, ProviderError::Other(s) if s.contains("duplicate custom_id")));
    }

    #[test]
    fn parse_status_maps_known_wire_forms() {
        assert_eq!(
            parse_status("in_progress").unwrap(),
            BatchStatus::InProgress
        );
        assert_eq!(parse_status("canceling").unwrap(), BatchStatus::Canceling);
        assert_eq!(parse_status("ended").unwrap(), BatchStatus::Ended);
    }

    #[test]
    fn parse_status_rejects_unknown() {
        assert!(parse_status("garbage").is_err());
    }

    #[test]
    fn as_wire_str_matches_anthropic_spelling() {
        assert_eq!(BatchStatus::InProgress.as_wire_str(), "in_progress");
        assert_eq!(BatchStatus::Canceling.as_wire_str(), "canceling");
        assert_eq!(BatchStatus::Ended.as_wire_str(), "ended");
    }

    #[test]
    fn handle_is_terminal_only_when_ended() {
        let mk = |s| BatchHandle {
            id: "x".into(),
            status: s,
            request_counts: BatchCounts::default(),
            created_at: Utc::now(),
            ended_at: None,
            expires_at: Utc::now(),
        };
        assert!(!mk(BatchStatus::InProgress).is_terminal());
        assert!(!mk(BatchStatus::Canceling).is_terminal());
        assert!(mk(BatchStatus::Ended).is_terminal());
    }

    #[test]
    fn drain_line_returns_complete_lines_only() {
        let mut buf = b"line one\nline two\nrema".to_vec();
        let l1 = drain_line(&mut buf).expect("line 1");
        assert_eq!(l1, b"line one");
        let l2 = drain_line(&mut buf).expect("line 2");
        assert_eq!(l2, b"line two");
        assert!(drain_line(&mut buf).is_none());
        assert_eq!(buf, b"rema");
    }

    #[test]
    fn drain_line_strips_crlf() {
        let mut buf = b"line\r\nrest".to_vec();
        let line = drain_line(&mut buf).expect("line");
        assert_eq!(line, b"line");
        assert_eq!(buf, b"rest");
    }

    #[test]
    fn jsonl_line_parses_succeeded_outcome() {
        let raw = br#"{"custom_id":"req-1","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}}"#;
        let r = parse_jsonl_line(raw).expect("ok");
        assert_eq!(r.custom_id, "req-1");
        match r.outcome {
            BatchOutcome::Succeeded(resp) => {
                assert_eq!(resp.usage.input_tokens, 10);
                assert_eq!(resp.usage.output_tokens, 5);
            }
            other => panic!("expected Succeeded, got {other:?}"),
        }
    }

    #[test]
    fn jsonl_line_parses_errored_outcome() {
        let raw = br#"{"custom_id":"req-2","result":{"type":"errored","error":{"type":"invalid_request_error","message":"max_tokens too high"}}}"#;
        let r = parse_jsonl_line(raw).expect("ok");
        match r.outcome {
            BatchOutcome::Errored {
                error_type,
                message,
            } => {
                assert_eq!(error_type, "invalid_request_error");
                assert!(message.contains("max_tokens"));
            }
            other => panic!("expected Errored, got {other:?}"),
        }
    }

    #[test]
    fn jsonl_line_parses_canceled_and_expired() {
        let canceled = br#"{"custom_id":"a","result":{"type":"canceled"}}"#;
        let expired = br#"{"custom_id":"b","result":{"type":"expired"}}"#;
        assert!(matches!(
            parse_jsonl_line(canceled).unwrap().outcome,
            BatchOutcome::Canceled
        ));
        assert!(matches!(
            parse_jsonl_line(expired).unwrap().outcome,
            BatchOutcome::Expired
        ));
    }
}
