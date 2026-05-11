use std::collections::{HashMap, HashSet};
use std::future::Future;
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
use crate::provider::{LlmProvider, Request, SystemBlock, ThinkingConfig, ToolDefinition};
use crate::steering::{self, AgentControl, SteerCommand, TurnId};
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
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error(
        "duplicate tool name '{name}' (registered {count} times); each tool must have a unique name"
    )]
    DuplicateToolName { name: String, count: usize },
}

/// Type alias for the per-event trace hook used by `stream_with_trace_hook`
/// and the SubAgent execute path.
pub(crate) type TraceHook = Arc<dyn Fn(&StreamEvent) + Send + Sync>;

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
    thinking: Option<ThinkingConfig>,
    tool_definition_filter: Option<HashSet<String>>,
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
            .filter(|t| {
                self.tool_definition_filter
                    .as_ref()
                    .is_none_or(|allowed| allowed.contains(t.name()))
            })
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
        let (future, _handle) = self.run_with_handle(messages, cancel);
        future.await
    }

    pub fn run_with_handle(
        &self,
        messages: Vec<Message>,
        cancel: CancellationToken,
    ) -> (
        impl Future<Output = Result<AgentResult, AgentError>> + Send + 'static,
        crate::AgentHandle,
    ) {
        let agent = self.clone();
        let (control, handle) = steering::control_pair(cancel.clone());
        let future = async move { agent.run_loop(messages, cancel, control).await };
        (future, handle)
    }

    async fn run_loop(
        self,
        messages: Vec<Message>,
        cancel: CancellationToken,
        mut control: AgentControl,
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
            let turn_id = begin_turn(&control, turn);
            info!(turn, %turn_id, "agent turn");

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
                thinking: self.thinking.clone(),
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
                clear_active_turn(&control);
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
            let results = self
                .executor
                .execute_batch_with_tracker(
                    tool_calls,
                    &ctx,
                    Some(control.handle_inner.tool_runs.clone()),
                )
                .await;

            let user_msg = Message::user(results);
            history.push(user_msg.clone());
            new_messages.push(user_msg);
            drain_queued_user_messages(&mut control, &mut history, &mut new_messages);

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

        clear_active_turn(&control);
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
        let (stream, _handle) = self.stream_with_handle(messages, cancel);
        stream
    }

    pub fn stream_with_handle(
        &self,
        messages: Vec<Message>,
        cancel: CancellationToken,
    ) -> (AgentStream, crate::AgentHandle) {
        self.stream_internal(messages, cancel, None)
    }

    /// Streaming variant that also fires `trace_hook` for **every**
    /// `StreamEvent` observed by the loop — including
    /// `MessageDelta`, `Usage`, `Done`, and the agent-emitted
    /// `ToolCallPending`. The public stream channel keeps its
    /// existing contract (Done is the terminal marker on that channel
    /// and is NOT forwarded). Used by [`crate::tools::SubAgent`] when
    /// `trace_hook` is set so per-turn observability is lossless
    /// without changing public stream semantics.
    pub(crate) fn stream_with_trace_hook(
        &self,
        messages: Vec<Message>,
        cancel: CancellationToken,
        hook: TraceHook,
    ) -> AgentStream {
        let (stream, _handle) = self.stream_internal(messages, cancel, Some(hook));
        stream
    }

    fn stream_internal(
        &self,
        messages: Vec<Message>,
        cancel: CancellationToken,
        trace_hook: Option<TraceHook>,
    ) -> (AgentStream, crate::AgentHandle) {
        let agent = self.clone();
        let (control, handle) = steering::control_pair(cancel.clone());
        let (events_tx, events_rx) = mpsc::channel::<Result<StreamEvent, ProviderError>>(16);
        let (result_tx, result_rx) = oneshot::channel();

        tokio::spawn(async move {
            let result = agent
                .run_streaming_loop(messages, cancel, control, events_tx, trace_hook)
                .await;
            // If the consumer dropped before we finished, sending the
            // result is a no-op; that's fine.
            let _ = result_tx.send(result);
        });

        (
            AgentStream {
                events_rx: ReceiverStream::new(events_rx),
                result_rx: Some(result_rx),
            },
            handle,
        )
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
        mut control: AgentControl,
        events_tx: mpsc::Sender<Result<StreamEvent, ProviderError>>,
        trace_hook: Option<TraceHook>,
    ) -> Result<AgentResult, AgentError> {
        let mut history = messages;
        let mut new_messages: Vec<Message> = Vec::new();
        let mut total_usage = Usage::default();
        let mut last_stop: Option<StopReason> = None;

        let tool_defs = self.tool_definitions();
        let ctx = self.make_context(cancel.clone());

        for turn in 0..self.max_turns {
            let turn_id = begin_turn(&control, turn);
            info!(turn, %turn_id, "agent stream turn");
            let turn_event = StreamEvent::TurnStarted {
                turn_id: turn_id.clone(),
            };
            if let Some(hook) = &trace_hook {
                emit_trace(hook, &turn_event);
            }
            if events_tx.send(Ok(turn_event)).await.is_err() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }

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
                thinking: self.thinking.clone(),
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
                        let ev = StreamEvent::ContentDelta(delta);
                        if let Some(hook) = &trace_hook {
                            emit_trace(hook, &ev);
                        }
                        if events_tx.send(Ok(ev)).await.is_err() {
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
                        let ev = StreamEvent::ThinkingDelta { text };
                        if let Some(hook) = &trace_hook {
                            emit_trace(hook, &ev);
                        }
                        if events_tx.send(Ok(ev)).await.is_err() {
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
                        let ev = StreamEvent::ThinkingBlock {
                            text,
                            provider,
                            metadata,
                        };
                        if let Some(hook) = &trace_hook {
                            emit_trace(hook, &ev);
                        }
                        if events_tx.send(Ok(ev)).await.is_err() {
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
                        let ev = StreamEvent::ToolUse { id, name, input };
                        if let Some(hook) = &trace_hook {
                            emit_trace(hook, &ev);
                        }
                        if events_tx.send(Ok(ev)).await.is_err() {
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
                        // Forwarded only to the optional `trace_hook`
                        // below; intentionally NOT forwarded on the
                        // public `Agent::stream` channel because Done
                        // (the event right after) is documented as the
                        // terminal marker and consumers break on it.
                        if let Some(hook) = &trace_hook {
                            emit_trace(hook, &StreamEvent::MessageDelta { stop_reason });
                        }
                        turn_stop = Some(stop_reason);
                    }
                    StreamEvent::Usage(u) => {
                        // Anthropic emits two Usage events (start with
                        // input_tokens + cache fields, end with
                        // output_tokens). The provider re-stamps cache
                        // fields onto every emission so merge_max here
                        // keeps the correct values across both events.
                        if let Some(hook) = &trace_hook {
                            emit_trace(hook, &StreamEvent::Usage(u.clone()));
                        }
                        turn_usage.merge_max(&u);
                    }
                    StreamEvent::Done => {
                        // Per-turn provider boundary, fired into the
                        // trace_hook for full child observability per
                        // issue #40 P6, but absorbed by the public
                        // stream so `Done` keeps its documented
                        // "stream terminated" terminal contract.
                        if let Some(hook) = &trace_hook {
                            emit_trace(hook, &StreamEvent::Done);
                        }
                        break;
                    }
                    // Agent-emitted events should never arrive from a
                    // provider's stream. Ignore defensively if a buggy
                    // provider injects one.
                    StreamEvent::TurnStarted { .. } | StreamEvent::ToolCallPending { .. } => {}
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
                clear_active_turn(&control);
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
                if let Some(hook) = &trace_hook {
                    emit_trace(hook, &event);
                }
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

            let results = self
                .executor
                .execute_batch_with_tracker(
                    calls,
                    &ctx,
                    Some(control.handle_inner.tool_runs.clone()),
                )
                .await;
            let user_msg = Message::user(results);
            history.push(user_msg.clone());
            new_messages.push(user_msg);
            drain_queued_user_messages(&mut control, &mut history, &mut new_messages);

            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }
        }

        clear_active_turn(&control);
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

fn begin_turn(control: &AgentControl, turn: usize) -> TurnId {
    let turn_id = TurnId::new();
    *control
        .handle_inner
        .active_turn
        .write()
        .expect("agent handle turn lock poisoned") = Some(turn_id.clone());
    debug!(turn, %turn_id, "active turn set");
    turn_id
}

fn clear_active_turn(control: &AgentControl) {
    *control
        .handle_inner
        .active_turn
        .write()
        .expect("agent handle turn lock poisoned") = None;
}

fn drain_queued_user_messages(
    control: &mut AgentControl,
    history: &mut Vec<Message>,
    new_messages: &mut Vec<Message>,
) {
    let mut content = Vec::new();
    while let Ok(command) = control.steer_rx.try_recv() {
        match command {
            SteerCommand::Append {
                content: mut queued,
            } => content.append(&mut queued),
        }
    }
    if !content.is_empty() {
        let message = Message::user(content);
        history.push(message.clone());
        new_messages.push(message);
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

/// Fire a trace hook closure with panic isolation. A panicking hook is
/// caught and logged, but the agent loop continues — an audit sink
/// failure must not crash the agent.
fn emit_trace(hook: &TraceHook, ev: &StreamEvent) {
    if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (hook)(ev))) {
        tracing::error!(?panic, "trace_hook closure panicked; suppressed");
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
    thinking: Option<ThinkingConfig>,
    tool_definition_filter: Option<HashSet<String>>,
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
            thinking: None,
            tool_definition_filter: None,
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
    /// When set, the override carries its own registry, policy, approval
    /// handler, AND [`ConcurrencyConfig`] — so the following builder
    /// methods are silently ignored at [`Self::build`]:
    ///
    /// - [`Self::tool`], [`Self::tools`]
    /// - [`Self::policy`]
    /// - [`Self::approval`]
    /// - [`Self::max_concurrent_reads`], [`Self::max_concurrent_mutations`]
    /// - [`Self::tool_concurrency`], [`Self::tool_concurrencies`]
    ///
    /// To customise concurrency for a sub-agent, build a forked executor
    /// via [`ToolExecutor::fork_for_subagent`] (or construct one directly
    /// with [`ToolExecutor::with_approval_and_concurrency`]) and pass it
    /// here.
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

    /// Default per-call thinking config threaded into every
    /// [`Request`] this agent issues. Provider precedence: an explicit
    /// per-call `Request.thinking` (set by, for example, a SubAgent's
    /// own [`crate::tools::SubAgent::thinking`]) overrides this; this
    /// in turn overrides the provider instance's construction-time
    /// default. See [`ThinkingConfig`] for provider-asymmetry notes.
    pub fn thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub(crate) fn tool_definition_filter(mut self, allowed: HashSet<String>) -> Self {
        self.tool_definition_filter = Some(allowed);
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

    /// Maximum concurrent `Mutating`-class tool calls that share the
    /// `concurrent_mut` pool. Default: 10.
    ///
    /// Two paths admit a call into this pool:
    ///
    /// 1. The consumer promotes a default-`Mutating` tool via
    ///    [`Self::tool_concurrency`] with [`ToolConcurrency::on`].
    /// 2. The tool sets [`crate::Tool::is_recursive`] to `true` (the
    ///    canonical example is `SubAgent`). Recursive tools are routed
    ///    through `concurrent_mut` regardless of explicit promotion to
    ///    avoid the permit-held-during-nested-execute deadlock that
    ///    would otherwise arise on the shared `serial_mut` pool.
    ///
    /// **Scope of the cap is per nesting level**, not tree-wide.
    /// `SubAgent::execute` calls [`ToolExecutor::fork_for_subagent`],
    /// which forks a fresh `concurrent_mut` semaphore (same numeric
    /// cap, independent permit accounting) so a parent saturating its
    /// pool cannot deadlock children needing their own
    /// promoted-mutator slots. Total in-flight promoted calls in a
    /// deep tree therefore scale as `O(fanout^depth × cap)` — capacity-
    /// plan accordingly when promoting tools that hit shared external
    /// resources (rate-limited APIs, finite connection pools).
    ///
    /// `serial_mut` (cap 1, fixed) and `read` (default cap 20) stay
    /// **shared** across the agent tree:
    ///
    /// - Default-`Mutating` tools without an explicit promotion still
    ///   serialise globally — two sibling sub-agents writing the same
    ///   file via the built-in `write` tool will not race even when
    ///   the consumer opted into none of the concurrency knobs.
    /// - The read pool acts as a global throughput throttle.
    ///
    /// See [`crate::ConcurrencyConfig::fork`] for the exact share-vs-
    /// fork policy.
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
    /// **Per-tool caps fork per nesting level**: a sub-agent inherits
    /// the configured cap as a fresh semaphore via
    /// [`ToolExecutor::fork_for_subagent`]. A `.max(2)` cap therefore
    /// admits 2 concurrent calls *per nesting level*, not 2 tree-wide.
    /// Plan caps with this in mind for tools that hit shared external
    /// resources (rate-limited APIs, finite connection pools); see
    /// [`crate::ConcurrencyConfig::fork`].
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

    pub fn build(self) -> Result<Agent, BuildError> {
        self.validate_tool_names()?;
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

        Ok(Agent {
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
            thinking: self.thinking,
            tool_definition_filter: self.tool_definition_filter,
        })
    }

    fn validate_tool_names(&self) -> Result<(), BuildError> {
        if self.executor_override.is_some() {
            return Ok(());
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for tool in &self.tools {
            *counts.entry(tool.name().to_string()).or_insert(0) += 1;
        }
        if let Some((name, count)) = counts.into_iter().find(|(_, count)| *count > 1) {
            return Err(BuildError::DuplicateToolName { name, count });
        }
        Ok(())
    }
}
