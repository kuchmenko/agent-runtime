use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use futures::StreamExt;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::approval::{ApprovalHandler, AutoApprove};
use crate::error::{AgentError, ProviderError};
use crate::executor::{
    AllowAll, ConcurrencyConfig, ToolCall, ToolConcurrency, ToolExecutor, ToolPolicy, ToolRegistry,
};
use crate::message::{CacheControl, Content, Message, StopReason, Usage};
use crate::provider::{LlmProvider, Request, SystemBlock, ToolDefinition};
use crate::stream::StreamEvent;
use crate::tool::{Tool, ToolContext};

/// Fallback `stop_reason` for partial results returned before any turn
/// has completed (e.g. provider failure on turn 0). Documented as
/// "no successful turn yet" — callers can disambiguate using the
/// outer `AgentError` variant. `EndTurn` is the least-misleading
/// concrete StopReason for the empty-history case.
const FALLBACK_STOP_REASON: StopReason = StopReason::EndTurn;

/// Result of an agent run.
///
/// The agent is stateless: it does **not** retain conversation history
/// between calls. Callers pass in the full history each time and receive
/// back only the **delta** of new messages the agent appended during this
/// run (assistant responses + tool-result user messages).
///
/// Typical consumer pattern:
///
/// ```ignore
/// let mut history = load_session();
/// history.push(Message::user_text(input));
/// let result = agent.run(history.clone(), cancel).await?;
/// history.extend(result.new_messages);
/// save_session(&history);
/// ```
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// Messages appended by this run only — **not** the full history.
    pub new_messages: Vec<Message>,
    /// Final assistant text output.
    pub text: String,
    /// Aggregated token usage across all turns in this run.
    pub usage: Usage,
    /// Stop reason from the last provider response, or `Cancelled` if the
    /// caller's `CancellationToken` fired.
    pub stop_reason: StopReason,
}

/// The core agent runtime.
///
/// Runs an LLM-driven tool loop: sends messages to the LLM, executes any
/// requested tools via the [`ToolExecutor`], feeds results back, and
/// repeats until the LLM produces a final text response, max turns are
/// reached, or the caller cancels.
///
/// `Clone` is shallow (Arcs and small primitives) — cheap; used by
/// [`Agent::stream`] to move a copy into the background task.
#[derive(Clone)]
pub struct Agent {
    provider: Arc<dyn LlmProvider>,
    model: String,
    system: Option<Vec<SystemBlock>>,
    executor: Arc<ToolExecutor>,
    max_turns: usize,
    max_tokens: u32,
    temperature: Option<f32>,
    working_dir: PathBuf,
    max_depth: usize,
    depth: usize,
    /// When `Some`, the last `ToolDefinition` sent to the provider has
    /// its `cache_control` set, terminating a cached prefix segment
    /// over the entire toolset. Anthropic-only optimization;
    /// non-Anthropic providers ignore the field.
    cache_tools: Option<CacheControl>,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Borrow the tool executor this agent was built with. Exposed so that
    /// sub-agents can share the parent's registry + policy without having
    /// to reconstruct them.
    pub fn executor(&self) -> &Arc<ToolExecutor> {
        &self.executor
    }

    /// Tool definitions sent to the LLM, sorted by name for deterministic
    /// ordering (so prompt-cache hashes stay stable across turns).
    ///
    /// When `cache_tools` is set, `cache_control` is placed on the last
    /// (alphabetically-final) tool definition, caching the entire
    /// toolset as one segment.
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> = self
            .executor
            .registry()
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
                cache_control: None,
            })
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        if let Some(cc) = &self.cache_tools {
            if let Some(last) = defs.last_mut() {
                last.cache_control = Some(cc.clone());
            }
        }
        defs
    }

    fn make_context(&self, cancel: CancellationToken) -> ToolContext {
        ToolContext {
            working_dir: self.working_dir.clone(),
            cancel,
            depth: self.depth,
            max_depth: self.max_depth,
            executor: Arc::clone(&self.executor),
        }
    }

    /// Run the agent loop against the given message history.
    ///
    /// The agent is **stateless**: this method does not mutate `self` and
    /// the caller owns the conversation. `messages` is the full history
    /// (typically the prior session plus the new user message). The
    /// returned [`AgentResult::new_messages`] is the delta — the caller
    /// should extend their history with it to persist progress.
    ///
    /// `cancel` is a cooperative cancellation signal. The loop checks it
    /// between turns and after each tool batch, returning
    /// [`AgentError::Cancelled`] promptly. Tools receive the same token
    /// via [`ToolContext::cancel`] and are expected to honour it for any
    /// long-running work.
    ///
    /// On any error, the [`AgentError::partial`] accessor returns the
    /// progress accumulated up to the failure point, so the caller can
    /// still persist what succeeded.
    pub async fn run(
        &self,
        messages: Vec<Message>,
        cancel: CancellationToken,
    ) -> Result<AgentResult, AgentError> {
        let mut history = messages;
        let mut new_messages: Vec<Message> = Vec::new();
        let mut total_usage = Usage::default();
        // None until the first provider response lands. Using Option here
        // (rather than seeding with `EndTurn`) avoids a misleading
        // `partial.stop_reason: EndTurn` on first-turn provider failures.
        let mut last_stop: Option<StopReason> = None;

        let tool_defs = self.tool_definitions();
        let ctx = self.make_context(cancel.clone());

        for turn in 0..self.max_turns {
            info!(turn, "agent turn");

            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }

            let request = Request {
                model: self.model.clone(),
                system: self.system.clone(),
                messages: history.clone(),
                tools: tool_defs.clone(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
            };

            let response = match self.provider.complete(request).await {
                Ok(r) => r,
                Err(source) => {
                    return Err(AgentError::Provider {
                        source,
                        partial: build_partial(
                            &new_messages,
                            &total_usage,
                            last_stop.unwrap_or(FALLBACK_STOP_REASON),
                            "",
                        ),
                    });
                }
            };

            total_usage.add(&response.usage);
            last_stop = Some(response.stop_reason);

            let assistant_msg = Message::assistant(response.content.clone());
            history.push(assistant_msg.clone());
            new_messages.push(assistant_msg);

            let tool_calls: Vec<ToolCall> = response
                .content
                .iter()
                .filter_map(|c| match c {
                    Content::ToolUse { id, name, input } => Some(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    _ => None,
                })
                .collect();

            if tool_calls.is_empty() || response.stop_reason == StopReason::EndTurn {
                let text = extract_text(&response.content);
                info!(turn, "agent finished");
                // Safe to unwrap: we just assigned `Some` above when this
                // response was decoded.
                return Ok(AgentResult {
                    new_messages,
                    text,
                    usage: total_usage,
                    stop_reason: last_stop.unwrap_or(FALLBACK_STOP_REASON),
                });
            }

            // Cancellation can fire while `provider.complete` is in flight
            // (a multi-second LLM call). The pre-turn check at the top of
            // the loop misses this window. Bail out before invoking any
            // tools so a cancelled run never starts new mutating work.
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }

            debug!(count = tool_calls.len(), "executing tool batch");
            let results = self.executor.execute_batch(tool_calls, &ctx).await;

            let user_msg = Message::user(results);
            history.push(user_msg.clone());
            new_messages.push(user_msg);

            // If cancel fired while tools were running, skip the next
            // provider round-trip — cooperative tools have already returned
            // Cancelled error results, and another LLM call would only be
            // wasted tokens before the pre-turn check stops us anyway.
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }
        }

        Err(AgentError::MaxTurnsReached {
            turns: self.max_turns,
            partial: build_partial(
                &new_messages,
                &total_usage,
                last_stop.unwrap_or(FALLBACK_STOP_REASON),
                "",
            ),
        })
    }

    /// Run the agent loop in streaming mode.
    ///
    /// Returns immediately with an [`AgentStream`] handle. A background
    /// task is spawned (under the current Tokio runtime) that drives the
    /// loop, calling `provider.stream()` on each turn. As the model
    /// produces text, [`StreamEvent::ContentDelta`] arrives through the
    /// stream live; [`StreamEvent::ToolUse`] arrives atomically when the
    /// model finishes a tool block. Provider-internal events
    /// (`MessageDelta`, `Usage`, per-turn `Done`) are absorbed —
    /// consumers see only what they can render to the user.
    ///
    /// History accumulation happens silently: text deltas are joined
    /// into `Content::Text` blocks and finalized thinking blocks are
    /// preserved separately, in provider order, before being added to
    /// `new_messages`, so the next turn's request to the provider sees
    /// a clean conversation history.
    ///
    /// Call [`AgentStream::into_result`] (or `.collect_result()`) after
    /// the stream ends to receive the final [`AgentResult`] with
    /// `new_messages`, `text`, `usage`, and `stop_reason` — exactly the
    /// shape `run()` returns. Backpressure is provided by a bounded
    /// `mpsc(16)` channel: a slow consumer parks the producer task,
    /// which transitively parks the SSE reader, which lets the OS
    /// shrink the TCP receive window, all the way back to the LLM
    /// server.
    pub fn stream(&self, messages: Vec<Message>, cancel: CancellationToken) -> AgentStream {
        let agent = self.clone();
        let (events_tx, events_rx) = mpsc::channel::<Result<StreamEvent, ProviderError>>(16);
        let (result_tx, result_rx) = oneshot::channel();

        tokio::spawn(async move {
            let result = agent.run_streaming_loop(messages, cancel, events_tx).await;
            // If the consumer dropped before we finished, sending the
            // result is a no-op; that's fine.
            let _ = result_tx.send(result);
        });

        AgentStream {
            events_rx: ReceiverStream::new(events_rx),
            result_rx: Some(result_rx),
        }
    }

    /// Body of the streaming loop. Owns the task locally so the public
    /// API surface stays small. Mirrors `run()` structurally — same
    /// turn counter, same cancel checkpoints, same history shape — but
    /// the per-turn provider call yields a stream rather than a
    /// buffered Response.
    async fn run_streaming_loop(
        self,
        messages: Vec<Message>,
        cancel: CancellationToken,
        events_tx: mpsc::Sender<Result<StreamEvent, ProviderError>>,
    ) -> Result<AgentResult, AgentError> {
        let mut history = messages;
        let mut new_messages: Vec<Message> = Vec::new();
        let mut total_usage = Usage::default();
        let mut last_stop: Option<StopReason> = None;

        let tool_defs = self.tool_definitions();
        let ctx = self.make_context(cancel.clone());

        for turn in 0..self.max_turns {
            info!(turn, "agent stream turn");

            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }

            let request = Request {
                model: self.model.clone(),
                system: self.system.clone(),
                messages: history.clone(),
                tools: tool_defs.clone(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
            };

            let mut provider_stream = match self.provider.stream(request).await {
                Ok(s) => s,
                Err(source) => {
                    return Err(AgentError::Provider {
                        source,
                        partial: build_partial(
                            &new_messages,
                            &total_usage,
                            last_stop.unwrap_or(FALLBACK_STOP_REASON),
                            "",
                        ),
                    });
                }
            };

            // Per-turn accumulators. Visible text is tracked both as
            // `turn_text` for AgentResult.text and as ordered Text blocks
            // inside assistant_content. Thinking blocks are finalized by
            // provider events so signatures / replay metadata survive.
            let mut turn_text = String::new();
            let mut current_text_buf = String::new();
            let mut assistant_content: Vec<Content> = Vec::new();
            let mut tool_uses: Vec<ToolCall> = Vec::new();
            let mut turn_stop: Option<StopReason> = None;
            let mut turn_usage = Usage::default();

            // Inner SSE loop. We cannot use `while let Some(event) =
            // provider_stream.next().await` here because that blocks
            // until the next SSE chunk arrives and gives the cancel
            // token no chance to interrupt — a long generation would
            // run to completion ignoring `cancel.cancel()`. Instead
            // race each pull against the cancel future so external
            // cancellation aborts immediately, dropping
            // `provider_stream` (which closes the reqwest socket on
            // drop, terminating the SSE upstream).
            loop {
                let event = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        return Err(AgentError::Cancelled {
                            partial: build_partial(
                                &new_messages,
                                &total_usage,
                                StopReason::Cancelled,
                                &turn_text,
                            ),
                        });
                    }
                    next = provider_stream.next() => match next {
                        Some(e) => e,
                        None => break,
                    },
                };
                let ev = match event {
                    Ok(ev) => ev,
                    Err(source) => {
                        return Err(AgentError::Provider {
                            source,
                            partial: build_partial(
                                &new_messages,
                                &total_usage,
                                last_stop.unwrap_or(FALLBACK_STOP_REASON),
                                &turn_text,
                            ),
                        });
                    }
                };

                match ev {
                    StreamEvent::ContentDelta(delta) => {
                        turn_text.push_str(&delta);
                        current_text_buf.push_str(&delta);
                        if events_tx
                            .send(Ok(StreamEvent::ContentDelta(delta)))
                            .await
                            .is_err()
                        {
                            // Consumer hung up — abort like a cancel.
                            return Err(AgentError::Cancelled {
                                partial: build_partial(
                                    &new_messages,
                                    &total_usage,
                                    StopReason::Cancelled,
                                    &turn_text,
                                ),
                            });
                        }
                    }
                    StreamEvent::ThinkingDelta { text } => {
                        if events_tx
                            .send(Ok(StreamEvent::ThinkingDelta { text }))
                            .await
                            .is_err()
                        {
                            return Err(AgentError::Cancelled {
                                partial: build_partial(
                                    &new_messages,
                                    &total_usage,
                                    StopReason::Cancelled,
                                    &turn_text,
                                ),
                            });
                        }
                    }
                    StreamEvent::ThinkingBlock {
                        text,
                        provider,
                        metadata,
                    } => {
                        if !current_text_buf.is_empty() {
                            assistant_content
                                .push(Content::text(std::mem::take(&mut current_text_buf)));
                        }
                        assistant_content.push(Content::Thinking {
                            text: text.clone(),
                            provider,
                            metadata: metadata.clone(),
                        });
                        if events_tx
                            .send(Ok(StreamEvent::ThinkingBlock {
                                text,
                                provider,
                                metadata,
                            }))
                            .await
                            .is_err()
                        {
                            return Err(AgentError::Cancelled {
                                partial: build_partial(
                                    &new_messages,
                                    &total_usage,
                                    StopReason::Cancelled,
                                    &turn_text,
                                ),
                            });
                        }
                    }
                    StreamEvent::ToolUse { id, name, input } => {
                        if !current_text_buf.is_empty() {
                            assistant_content
                                .push(Content::text(std::mem::take(&mut current_text_buf)));
                        }
                        let call = ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        };
                        tool_uses.push(call);
                        assistant_content.push(Content::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });
                        if events_tx
                            .send(Ok(StreamEvent::ToolUse { id, name, input }))
                            .await
                            .is_err()
                        {
                            return Err(AgentError::Cancelled {
                                partial: build_partial(
                                    &new_messages,
                                    &total_usage,
                                    StopReason::Cancelled,
                                    &turn_text,
                                ),
                            });
                        }
                    }
                    StreamEvent::MessageDelta { stop_reason } => {
                        turn_stop = Some(stop_reason);
                    }
                    StreamEvent::Usage(u) => {
                        // Anthropic emits two Usage events (start with
                        // input_tokens + cache fields, end with
                        // output_tokens). The provider re-stamps cache
                        // fields onto every emission so merge_max here
                        // keeps the correct values across both events.
                        turn_usage.merge_max(&u);
                    }
                    StreamEvent::Done => break,
                    // ToolCallPending is an agent-emitted event and
                    // should never arrive from a provider's stream.
                    // Ignore defensively if a buggy provider injects it.
                    StreamEvent::ToolCallPending { .. } => {}
                }
            }
            if !current_text_buf.is_empty() {
                assistant_content.push(Content::text(current_text_buf));
            }
            // Drop provider_stream to free the underlying HTTP
            // connection before the next turn opens a new one.
            drop(provider_stream);

            total_usage.add(&turn_usage);
            let resolved_stop = turn_stop.unwrap_or(StopReason::EndTurn);
            last_stop = Some(resolved_stop);

            let assistant_msg = Message::assistant(assistant_content);
            history.push(assistant_msg.clone());
            new_messages.push(assistant_msg);

            if tool_uses.is_empty() || resolved_stop == StopReason::EndTurn {
                info!(turn, "agent stream finished");
                return Ok(AgentResult {
                    new_messages,
                    text: turn_text,
                    usage: total_usage,
                    stop_reason: resolved_stop,
                });
            }

            // Same cancel guard as run(): the provider stream may have
            // taken seconds to drain; bail before tool dispatch if the
            // caller cancelled meanwhile.
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }

            debug!(count = tool_uses.len(), "executing tool batch (stream)");
            let calls = tool_uses;

            // Emit one `ToolCallPending` per call before invoking the
            // executor. The consumer's UI uses this to render an
            // "approval pending" prompt while the executor's
            // `ApprovalHandler::approve` blocks on user input.
            // Class is resolved through the registry; a missing tool
            // (LLM hallucinated a name) falls back to Mutating —
            // safer default for unknown tools.
            for call in &calls {
                let class = self
                    .executor
                    .registry()
                    .get(&call.name)
                    .map(|t| t.class())
                    .unwrap_or(crate::tool::ToolClass::Mutating);
                let event = StreamEvent::ToolCallPending {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input: call.input.clone(),
                    class,
                };
                if events_tx.send(Ok(event)).await.is_err() {
                    return Err(AgentError::Cancelled {
                        partial: build_partial(
                            &new_messages,
                            &total_usage,
                            StopReason::Cancelled,
                            &turn_text,
                        ),
                    });
                }
            }

            let results = self.executor.execute_batch(calls, &ctx).await;
            let user_msg = Message::user(results);
            history.push(user_msg.clone());
            new_messages.push(user_msg);

            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }
        }

        warn!(turns = self.max_turns, "agent stream max turns");
        Err(AgentError::MaxTurnsReached {
            turns: self.max_turns,
            partial: build_partial(
                &new_messages,
                &total_usage,
                last_stop.unwrap_or(FALLBACK_STOP_REASON),
                "",
            ),
        })
    }
}

/// Live agent run.
///
/// `AgentStream` implements `Stream<Item = Result<StreamEvent,
/// ProviderError>>` for the live event channel and exposes
/// [`into_result`](Self::into_result) / [`collect_result`](Self::collect_result)
/// for the terminal [`AgentResult`].
pub struct AgentStream {
    events_rx: ReceiverStream<Result<StreamEvent, ProviderError>>,
    /// Optional so `into_result` can take it without dropping `self`.
    result_rx: Option<oneshot::Receiver<Result<AgentResult, AgentError>>>,
}

impl Stream for AgentStream {
    type Item = Result<StreamEvent, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().events_rx).poll_next(cx)
    }
}

impl AgentStream {
    /// Wait for the loop to finish and return the final result.
    ///
    /// Drops the live event channel — call this only after you've
    /// drained as many events as you care about (typically via the
    /// `Stream` impl in a `while let Some(ev) = stream.next().await`
    /// loop).
    pub async fn into_result(mut self) -> Result<AgentResult, AgentError> {
        let rx = self
            .result_rx
            .take()
            .expect("into_result called twice on AgentStream");
        // Drop the events receiver; if the producer is still running
        // it'll see the channel close and short-circuit.
        drop(self.events_rx);
        match rx.await {
            Ok(result) => result,
            Err(_) => Err(AgentError::Cancelled {
                partial: Box::new(AgentResult {
                    new_messages: Vec::new(),
                    text: String::new(),
                    usage: Usage::default(),
                    stop_reason: StopReason::Cancelled,
                }),
            }),
        }
    }

    /// Drain remaining events (discarding them) and return the result.
    /// Convenient when the caller only cares about the final outcome —
    /// equivalent to `while let Some(_) = stream.next().await {}` then
    /// `into_result()`.
    pub async fn collect_result(mut self) -> Result<AgentResult, AgentError> {
        while self.events_rx.next().await.is_some() {}
        let rx = self
            .result_rx
            .take()
            .expect("collect_result called after into_result");
        rx.await.unwrap_or_else(|_| {
            Err(AgentError::Cancelled {
                partial: Box::new(AgentResult {
                    new_messages: Vec::new(),
                    text: String::new(),
                    usage: Usage::default(),
                    stop_reason: StopReason::Cancelled,
                }),
            })
        })
    }
}

fn build_partial(
    new_messages: &[Message],
    usage: &Usage,
    stop_reason: StopReason,
    text: &str,
) -> Box<AgentResult> {
    Box::new(AgentResult {
        new_messages: new_messages.to_vec(),
        text: text.to_string(),
        usage: usage.clone(),
        stop_reason,
    })
}

fn extract_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

// --- Builder ---

pub struct AgentBuilder {
    provider: Option<Arc<dyn LlmProvider>>,
    model: Option<String>,
    system: Option<Vec<SystemBlock>>,
    tools: Vec<Arc<dyn Tool>>,
    policy: Option<Arc<dyn ToolPolicy>>,
    approval: Option<Arc<dyn ApprovalHandler>>,
    executor_override: Option<Arc<ToolExecutor>>,
    max_turns: usize,
    max_tokens: u32,
    temperature: Option<f32>,
    working_dir: Option<PathBuf>,
    max_depth: usize,
    depth: usize,
    cache_tools: Option<CacheControl>,
    read_cap: usize,
    mut_cap: usize,
    tool_concurrencies: Vec<(String, ToolConcurrency)>,
}

impl AgentBuilder {
    fn new() -> Self {
        Self {
            provider: None,
            model: None,
            system: None,
            tools: Vec::new(),
            policy: None,
            approval: None,
            executor_override: None,
            max_turns: 50,
            max_tokens: 16384,
            temperature: None,
            working_dir: None,
            max_depth: 3,
            depth: 0,
            cache_tools: None,
            read_cap: 20,
            mut_cap: 10,
            tool_concurrencies: Vec::new(),
        }
    }

    pub fn provider(mut self, provider: impl LlmProvider + 'static) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    /// Use a shared provider (typically for sub-agent spawning).
    pub fn provider_arc(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the system prompt as a single uncached block.
    ///
    /// Use [`Self::system_blocks`] for a typed multi-block system
    /// prompt with [`crate::CacheControl`] breakpoints (Anthropic
    /// prompt caching).
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(vec![SystemBlock::text(system)]);
        self
    }

    /// Set the system prompt as a list of typed blocks. Each block
    /// can carry a [`crate::CacheControl`] cache breakpoint —
    /// useful when part of the prompt is stable across calls (good
    /// to cache) and part rotates per-call.
    ///
    /// Non-Anthropic providers concatenate the blocks with `\n\n`
    /// and drop cache_control.
    pub fn system_blocks(mut self, blocks: Vec<SystemBlock>) -> Self {
        self.system = Some(blocks);
        self
    }

    /// Cache the entire toolset as a single prefix segment with the
    /// given TTL. The agent places `cache_control` on the last
    /// (alphabetically-final) tool definition each turn, which makes
    /// Anthropic cache every preceding tool too.
    ///
    /// For long stable toolsets (the default 8 built-in tools clear
    /// several KB of JSON Schema), this is the highest-leverage
    /// breakpoint in tkach. Non-Anthropic providers ignore it.
    pub fn cache_tools(mut self, cache_control: CacheControl) -> Self {
        self.cache_tools = Some(cache_control);
        self
    }

    /// Register a tool by value — convenient for concrete built-in tools.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Register tools as shared trait objects. Matches the shape of
    /// [`crate::tools::defaults`] and allows one tool instance to live in
    /// multiple registries.
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Install a tool-invocation policy. Without this, [`AllowAll`] is used.
    pub fn policy(mut self, policy: impl ToolPolicy + 'static) -> Self {
        self.policy = Some(Arc::new(policy));
        self
    }

    /// Install a per-call approval handler. Without this,
    /// [`AutoApprove`](crate::AutoApprove) is used (every call allowed).
    /// The handler runs after [`ToolPolicy::is_allowed`] succeeds and
    /// before the tool actually executes; denials surface to the model
    /// as `is_error: true` tool_results so the LLM can adapt.
    ///
    /// Sub-agents inherit this handler automatically through their
    /// shared [`ToolExecutor`] (Model 3): one handler gates the whole
    /// agent tree.
    pub fn approval(mut self, approval: impl ApprovalHandler + 'static) -> Self {
        self.approval = Some(Arc::new(approval));
        self
    }

    /// Re-use an existing [`ToolExecutor`] instead of building one from
    /// the `tools` + `policy` accumulated in the builder. Intended for
    /// sub-agent spawning, where the child inherits the parent's full
    /// registry automatically.
    ///
    /// When set, `.tool()`, `.tools()`, and `.policy()` are ignored.
    pub fn executor(mut self, executor: Arc<ToolExecutor>) -> Self {
        self.executor_override = Some(executor);
        self
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Maximum nesting depth for sub-agent recursion. Default: 3.
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub(crate) fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Maximum concurrent `ReadOnly`-class tool calls in a single batch.
    /// Default: 20.
    ///
    /// Ignored when [`Self::executor`] was called — the inherited
    /// `ToolExecutor` already carries its own [`ConcurrencyConfig`].
    ///
    /// # Panics
    /// Panics if `n == 0`.
    #[must_use]
    pub fn max_concurrent_reads(mut self, n: usize) -> Self {
        assert!(n > 0, "max_concurrent_reads requires n > 0");
        self.read_cap = n;
        self
    }

    /// Maximum concurrent `Mutating`-class tool calls that have been
    /// promoted via [`Self::tool_concurrency`] with
    /// [`ToolConcurrency::on`]. Tools without an explicit promotion
    /// continue to serialise globally regardless of this value (they
    /// share a separate width-1 semaphore). Default: 10.
    ///
    /// Sub-agents share the parent's `ToolExecutor` and therefore the
    /// parent's mutator pool — the cap applies across the entire agent
    /// tree. With a 3-level deep tree fanning out 5 sub-agents per
    /// level and `max_concurrent_mutations(10)`, leaf calls can sit
    /// waiting for permits held elsewhere in the tree. This is
    /// intended behaviour — global admission control.
    ///
    /// Ignored when [`Self::executor`] was called.
    ///
    /// # Panics
    /// Panics if `n == 0`.
    #[must_use]
    pub fn max_concurrent_mutations(mut self, n: usize) -> Self {
        assert!(n > 0, "max_concurrent_mutations requires n > 0");
        self.mut_cap = n;
        self
    }

    /// Configure concurrency for a single tool by name.
    ///
    /// The most common use is [`ToolConcurrency::on`] to promote a
    /// default-`Mutating` tool into the concurrent-mutator pool. Chain
    /// `.max(N)` for a per-tool cap that bounds parallelism for *this
    /// specific tool* below the class cap. The effective concurrent
    /// count is `min(per_tool_cap, class_cap_remaining)`.
    ///
    /// Promotion is a *consumer's responsibility* contract: the
    /// framework can no longer prevent racing calls to the same tool
    /// with conflicting inputs (e.g. two writes to the same path).
    /// Promote only when the LLM-emitted batch shape — and the tool's
    /// own resource semantics — make racing safe.
    ///
    /// Later calls for the same tool name override earlier ones.
    /// Ignored when [`Self::executor`] was called.
    #[must_use]
    pub fn tool_concurrency(mut self, name: impl Into<String>, cfg: ToolConcurrency) -> Self {
        self.tool_concurrencies.push((name.into(), cfg));
        self
    }

    /// Bulk variant of [`Self::tool_concurrency`] for configuring
    /// multiple tools at once.
    #[must_use]
    pub fn tool_concurrencies<I>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (String, ToolConcurrency)>,
    {
        self.tool_concurrencies.extend(entries);
        self
    }

    pub fn build(self) -> Agent {
        let executor = self.executor_override.unwrap_or_else(|| {
            let registry = Arc::new(ToolRegistry::new(self.tools));
            let policy: Arc<dyn ToolPolicy> = self.policy.unwrap_or_else(|| Arc::new(AllowAll));
            let approval: Arc<dyn ApprovalHandler> =
                self.approval.unwrap_or_else(|| Arc::new(AutoApprove));
            let concurrency =
                ConcurrencyConfig::new(self.read_cap, self.mut_cap, self.tool_concurrencies);
            Arc::new(ToolExecutor::with_approval_and_concurrency(
                registry,
                policy,
                approval,
                concurrency,
            ))
        });

        Agent {
            provider: self.provider.expect("provider is required"),
            model: self.model.expect("model is required"),
            system: self.system,
            executor,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            working_dir: self
                .working_dir
                .unwrap_or_else(|| std::env::current_dir().expect("failed to get current dir")),
            max_depth: self.max_depth,
            depth: self.depth,
            cache_tools: self.cache_tools,
        }
    }
}
