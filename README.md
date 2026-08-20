# tkach

A provider-independent agent runtime for Rust. Stateless agent loop, pluggable LLM providers (Anthropic, OpenAI Responses, ChatGPT Codex, OpenAI-compatible), built-in file/shell tools, real SSE streaming with reasoning summaries, cooperative cancellation, and per-call approval gating.

[![CI](https://github.com/kuchmenko/tkach/actions/workflows/ci.yml/badge.svg)](https://github.com/kuchmenko/tkach/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Status:** pre-1.0 (`0.6.0`). Breaking changes are signalled via `feat!:` conventional commits and recorded in [`CHANGELOG.md`](./CHANGELOG.md). Before 1.0, breaking changes increment the minor version.

## Features

- **Stateless `Agent::run`** — caller owns the message history; the agent returns the **delta** of new messages it appended. Resume, multi-turn chat, fork & retry all become composable.
- **Atomic events with one cancel surface** — `ToolUse` events are emitted whole, never as partial JSON; a single `CancellationToken` shuts down the loop, the SSE pull, the in-flight HTTP body, and any `bash` child process via `kill_on_drop`.
- **Provider parity, including reasoning** — Anthropic (adaptive and manual extended thinking), OpenAI Responses (reasoning summary), ChatGPT Codex (subscription endpoint), and any OpenAI-compatible Chat Completions endpoint share one `StreamEvent` API surface — not three SDKs.
- **Inherited sub-agent controls** — sub-agents fork the parent executor while retaining its tool registry, policy, approval handler, and concurrency settings.

## Quick start

```toml
[dependencies]
tkach = { git = "https://github.com/kuchmenko/tkach" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use tkach::{Agent, CancellationToken, Message, providers::Anthropic, tools};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::builder()
        .provider(Anthropic::from_env())
        .model(tkach::model::claude::HAIKU_20251001)
        .system("You are a concise assistant.")
        .tools(tools::defaults())
        .build()?;

    let mut history = vec![Message::user_text(
        "List the .rs files in this directory and summarise each.",
    )];

    let result = agent.run(history.clone(), CancellationToken::new()).await?;

    history.extend(result.new_messages);   // caller owns history
    println!("{}", result.text);
    println!("[{} in / {} out tokens]", result.usage.input_tokens, result.usage.output_tokens);
    Ok(())
}
```

> **New to tkach?** Run `cargo run --example basic` for a ~30-line working agent against Anthropic, or `cargo run --example streaming` for the streaming variant. Full list under [Examples](#examples) below.

### Image input

User messages can mix text and images in exact block order. Images are valid only in user-role messages and may come from in-memory bytes, an absolute local file path, or a URL:

```rust
use tkach::{Content, Message};

let first_capture = std::path::PathBuf::from("/absolute/path/first.png");
let second_capture = std::path::PathBuf::from("/absolute/path/second.png");

history.push(Message::user(vec![
    Content::text("First region:"),
    Content::image_file("image/png", first_capture),
    Content::text("Second region:"),
    Content::image_file("image/png", second_capture),
    Content::text("Compare these regions."),
]));

let result = agent.run(history.clone(), CancellationToken::new()).await?;
history.extend(result.new_messages);
history.push(Message::user_text("Which region contains the setting?"));
```

Local image files are read asynchronously before each provider request and must use absolute paths. The caller owns those files and their lifecycle. Stateless replay resends and rereads every image retained in history on later turns. `Content::image_bytes` creates a self-contained history entry whose JSON uses base64; `Content::image_url` relies on the URL remaining available. Image input through the ChatGPT Codex subscription provider is experimental because that endpoint is undocumented.

## Architecture at a glance

```
┌───────────┐  messages + cancel    ┌─────────────────────────────┐
│  caller   │──────────────────────▶│         Agent::run          │
└───────────┘   new_messages,        │     (or ::stream)           │
              text, usage,           │                             │
              stop_reason            └────┬───────────────────────┘
                                          │
                       ┌──────────────────┴────────────┐
                       ▼                               ▼
                ┌────────────────────┐          ┌───────────────────┐
                │   LlmProvider      │          │   ToolExecutor    │
                │                    │          │ ┌───────────────┐ │
                │ Anthropic          │          │ │  ToolPolicy   │ │
                │ OpenAIResponses    │          │ ├───────────────┤ │
                │ OpenAICodex        │          │ │ApprovalHandler│ │
                │ OpenAICompatible   │          │ ├───────────────┤ │
                │ Mock               │          │ │ ToolRegistry  │ │
                └────────────────────┘          │ └───────────────┘ │
                                                └─────────┬─────────┘
                                                          │
                                              ┌───────────┴────────────┐
                                              │    Tool scheduling     │
                                              │ ReadOnly: concurrent   │
                                              │ Recursive: concurrent  │
                                              │ Mutating: serial by    │
                                              │ default; opt-in        │
                                              │ concurrency available  │
                                              └────────────────────────┘
```

The diagram glosses over a few invariants worth knowing up front:

- **Stateless turn loop.** `Agent::run` walks turn-by-turn until the model
  stops emitting `ToolUse` or `max_turns` is reached. Each turn is one
  `LlmProvider` call and one `ToolExecutor::execute_batch` against the
  resulting `ToolUse` block. The caller appends `result.new_messages`
  to its own history for the next call — the runtime keeps no per-call
  state, so resume / fork / retry are just calls with different histories.
- **Atomic streaming.** `Agent::stream` accumulates `input_json_delta`
  chunks before emitting `ToolUse`, so consumers never see partial JSON.
  `ToolCallPending` fires after `ToolUse` and before the approval gate,
  so UIs can render an "approval pending" state independently of the
  tool actually running.
- **One cancellation surface.** A single `CancellationToken` reaches the
  agent loop, the SSE pull, the in-flight HTTP body, every
  `ApprovalHandler::approve` call, every `Bash` child via `kill_on_drop`,
  and every `WebFetch`. Cancellation is cooperative for custom tools,
  which must observe `ToolContext::cancel` themselves.
- **Sub-agents fork the executor.** A child retains the parent's registry,
  policy, approval behavior, and numeric concurrency settings. The read
  and serial-mutator pools remain shared across the tree; the concurrent-
  mutator and per-tool pools are forked per nesting level to prevent
  parent/child permit deadlocks.
- **Ordered concurrency.** `execute_batch` partitions calls into contiguous
  runs of the same routing class and executes calls concurrently within
  each run using `FuturesUnordered`. It waits between runs and returns
  results in model-issued order. Read-only calls default to a shared cap
  of 20. Recursive tools and explicitly promoted mutators use a cap of 10
  per nesting level. Other mutators remain globally serial.

Concurrency can be tuned from the builder. Promoting a mutating tool is a
caller-owned safety decision: only promote tools whose calls may overlap.

```rust
use tkach::{Agent, ToolConcurrency};

let agent = Agent::builder()
    // provider, model, and tools omitted
    .max_concurrent_reads(8)
    .max_concurrent_mutations(4)
    .tool_concurrency("independent_write", ToolConcurrency::on().max(2))
    .build()?;
```

## Built-in tools

Read-only (`ToolClass::ReadOnly` — concurrent within an ordered run):

- `Read` — read file contents (numbered lines, offset/limit).
- `Glob` — find files matching a glob (sorted by mtime).
- `Grep` — regex search in files (with context, ignore patterns).
- `WebFetch` — HTTP GET a URL, returns body text.

Mutating (`ToolClass::Mutating` — serial unless explicitly promoted):

- `Write` — write a file (creates parents).
- `Edit` — replace an exact string in a file.
- `Bash` — run a shell command (cancel-aware via `kill_on_drop`).

Recursive:

- `SubAgent` — spawn a nested agent with a forked executor. Recursive tools use
  the concurrent-mutator pool so multiple sub-agents in one run may overlap.

`tools::defaults()` returns `Read + Write + Edit + Glob + Grep + Bash`. Add `WebFetch` and `SubAgent::new(provider, model)` explicitly when you want them.

## Providers

```rust
use tkach::providers::{Anthropic, OpenAIEffort, OpenAICompatible, OpenAIResponses, OpenAISummary};

// Anthropic Messages API.
let p = Anthropic::from_env();   // ANTHROPIC_API_KEY

// OpenAI Chat Completions-compatible API: text + tool calls, no standard thinking.
let p = OpenAICompatible::from_env();   // OPENAI_API_KEY

// OpenAI Responses API — required for reasoning-summary streams.
let p = OpenAIResponses::from_env()
    .with_reasoning(OpenAIEffort::Medium, OpenAISummary::Detailed);

// Any OpenAI-compatible Chat Completions endpoint:
//   OpenRouter
let p = OpenAICompatible::new(key)
    .with_base_url("https://openrouter.ai/api/v1");
//   Local Ollama
let p = OpenAICompatible::new("ignored")
    .with_base_url("http://localhost:11434/v1");
//   Moonshot, DeepSeek, Together, Groq — same shape
```

Implementing your own provider: implement `LlmProvider` (one `complete` and one `stream` method).

### Typed configuration

Prefer typed constants/enums for autocomplete and typo resistance:

```rust
use tkach::{
    Agent,
    model::{claude, gpt, openrouter},
    providers::{OpenAIEffort, OpenAISummary, anthropic::AnthropicEffort},
};

Agent::builder().model(claude::SONNET);
Agent::builder().model(gpt::FIVE);
Agent::builder().model(openrouter::OPENAI_GPT_5_5);

let effort = AnthropicEffort::High;
let reasoning = (OpenAIEffort::Medium, OpenAISummary::Detailed);
```

Raw strings still work for new vendor values; they route through `Other(String)`.

### Anthropic extended thinking

`Anthropic::with_adaptive_thinking_effort` (recommended on Claude Sonnet/Opus 4.6+) lets the model decide when to think. `with_thinking_budget` is the older fixed-token mode.

```rust
// Adaptive thinking — recommended.
use tkach::providers::anthropic::AnthropicEffort;

let p = Anthropic::from_env()
    .with_adaptive_thinking_effort(AnthropicEffort::High);

// Manual budget — fixed-token mode.
let p = Anthropic::from_env()
    .with_thinking_budget(1024);
```

Both paths emit the same `StreamEvent::ThinkingDelta` and `StreamEvent::ThinkingBlock` events; downstream code does not branch on which mode produced them.

### Anthropic prompt caching

`SystemBlock::cached`, `Content::text_cached`, and `AgentBuilder::cache_tools` mark cache breakpoints; `Usage` reports `cache_creation_input_tokens` / `cache_read_input_tokens` so callers can measure hit rate. Default TTL is 5 min, with 1 h available via `CacheControl::ephemeral_1h()`. Cache reads bill at 0.1× base input; writes at 1.25× (5 min) or 2× (1 h). See `examples/anthropic_caching.rs` and `examples/anthropic_caching_streaming.rs`.

### Anthropic Message Batches

Anthropic's [Message Batches API](https://docs.anthropic.com/en/api/messages-batches) takes the same `Request` body and processes it asynchronously. Anthropic documents a processing window of up to 24 hours and a 50% discount on input and output tokens. It is intended for workloads that do not require immediate results, such as backfills and evaluations. Prompt caching can also apply when a request has a stable prefix; check Anthropic's current pricing before estimating cost.

```rust
use futures::StreamExt;
use tkach::providers::Anthropic;
use tkach::providers::anthropic::batch::{BatchOutcome, BatchRequest};
use tkach::{Message, Request};

let provider = Anthropic::from_env();

let requests = vec![BatchRequest {
    custom_id: "req-1".into(),               // ^[a-zA-Z0-9_-]{1,64}$, unique within batch
    params: Request {
        model: tkach::model::claude::HAIKU_20251001.into(),
        system: None,
        messages: vec![Message::user_text("Say hello.")],
        tools: vec![],
        max_tokens: 64,
        temperature: None,
        thinking: None,
    },
}];

let handle = provider.create_batch(requests).await?;          // status=InProgress
loop {
    let h = provider.retrieve_batch(&handle.id).await?;
    if h.is_terminal() { break }                              // status=Ended
    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
}

let mut stream = provider.batch_results(&handle.id).await?;   // JSONL line-by-line
while let Some(item) = stream.next().await {
    match item?.outcome {
        BatchOutcome::Succeeded(resp) => { /* same Response shape as complete() */ }
        BatchOutcome::Errored { error_type, message } => { /* per-row error */ }
        BatchOutcome::Canceled | BatchOutcome::Expired => {}
    }
}
```

`custom_id`s are validated client-side (regex + dedup) before the HTTP call. Caller owns the polling cadence — there is no `await_batch` helper because the right interval (every 5 min vs every 1 h vs exp-backoff) is workload-dependent. See `examples/anthropic_batch.rs`, `examples/anthropic_batch_cancel.rs`, `examples/anthropic_batch_mixed.rs`.

### OpenAI ChatGPT Codex subscription

`OpenAICodex` targets the ChatGPT subscription Codex backend at `https://chatgpt.com/backend-api/codex/responses`. Wire grammar matches `OpenAIResponses` (same SSE events: `response.output_text.delta`, atomic `function_call`, `response.reasoning_summary_text.*`), so text, tool calls, and reasoning summaries flow through the standard `StreamEvent` API.

Credentials are caller-owned. The provider does **not** implement OAuth login, refresh-token exchange, environment-variable lookup, keyring storage, or account discovery — it asks a `CodexCredentialsProvider` for fresh credentials before every request and surfaces `401` to the caller without internal retry.

```rust
use async_trait::async_trait;
use tkach::ProviderError;
use tkach::providers::{
    CodexCredentials, CodexCredentialsProvider, OpenAICodex, OpenAIEffort, OpenAISummary,
};

struct MyTokenCache { /* OAuth client, refresh logic, keyring ... */ }

#[async_trait]
impl CodexCredentialsProvider for MyTokenCache {
    async fn credentials(&self) -> Result<CodexCredentials, ProviderError> {
        // Call your token cache here. Refresh on expiry; surface errors otherwise.
        Ok(CodexCredentials::new("access-token", "account-id"))
    }
}

let provider = OpenAICodex::new(MyTokenCache { /* ... */ })
    .with_originator("my-app")                  // optional, defaults to "tkach"
    .with_reasoning_summary(OpenAISummary::Auto)     // optional, default "auto"
    .with_reasoning_effort(OpenAIEffort::Medium);     // optional, off by default

// Static credentials are useful for tests and short-lived scripts:
let provider = OpenAICodex::with_static_credentials(
    CodexCredentials::new("token", "acct"),
);
```

Reasoning summary is on by default (`reasoning: { summary: "auto" }`). The Codex backend does not emit `response.reasoning_summary_text.*` events unless this is set — `include: ["reasoning.encrypted_content"]` alone gets you opaque replay state but no visible thinking text. Call `.without_reasoning()` to drop the field; the encrypted-replay include is independent and still travels.

The Codex subscription backend is undocumented and unstable. Wire shape and event names can change without notice — pin a `tkach` version you have validated end-to-end if you ship this in production. See `examples/streaming_openai_codex.rs`.

## Streaming

```rust
use tkach::{Agent, CancellationToken, Message, StreamEvent};
use futures::StreamExt;

let mut stream = agent.stream(history, CancellationToken::new());

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TurnStarted { turn_id } => {
            eprintln!("[turn: {turn_id}]");          // correlate steering calls
        }
        StreamEvent::ContentDelta(text) => {
            print!("{text}");                    // visible answer tokens
        }
        StreamEvent::ThinkingDelta { text } => {
            eprint!("[thinking] {text}");         // provider-returned summary, not final text
        }
        StreamEvent::ThinkingBlock { .. } => {
            // Finalized thinking/reasoning block with replay metadata.
            // Persisted in AgentResult.new_messages, excluded from AgentResult.text.
        }
        StreamEvent::ToolUse { id, name, input } => {
            // Atomic: parser accumulated all `input_json_delta` chunks
            // before emitting; you never see partial JSON.
            eprintln!("[tool: {name}({input})]");
        }
        StreamEvent::ToolCallPending { id, name, input, class } => {
            // Agent-emitted: render an "approval pending" prompt in the UI.
            // Fires after ToolUse, before the executor's approval gate runs.
        }
        // MessageDelta, Usage, Done are absorbed by the agent loop and
        // not forwarded on the public stream. The loop ends naturally
        // when the channel closes; collect the final result below.
        _ => {}
    }
}

let result = stream.into_result().await?;        // final AgentResult
```

The agent stream can also emit steering and policy lifecycle events:
`ModeChanged`, `ModeChangeRequested`, `ContinuationInjected`, `GuardAborted`,
`PolicyInstalled`, `PolicyRemoved`, and `PolicyApplied`. Provider-level
`MessageDelta`, `Usage`, and `Done` events are consumed by the agent loop.
Downstream exhaustive matches should include a fallback arm when they do not
need every lifecycle event.

Provider boundary: Anthropic thinking requires `Anthropic::with_adaptive_thinking*` or `with_thinking_budget(...)`; OpenAI thinking requires `OpenAIResponses` (`/responses` with `reasoning.summary`). `OpenAICompatible` is Chat Completions and intentionally asserts the no-thinking contract because that wire format has no standard reasoning-summary event.

`AgentStream` uses a bounded 16-event channel, so a slow consumer pauses the
producer after the buffer fills. Cancellation is raced against each provider
pull and against approval and execution admission. Actual shutdown latency
still depends on the provider, network, and whether custom tools cooperate.

Steering-aware callers can use `run_with_handle` / `stream_with_handle` to get an `AgentHandle` for the active run:

```rust
let (mut stream, handle) = agent.stream_with_handle(history, CancellationToken::new());

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TurnStarted { turn_id } => {
            handle.queue_user_message("Also include post-2023 sources", Some(turn_id))?;
        }
        StreamEvent::ToolCallPending { id, .. } if should_interrupt(&id) => {
            handle.interrupt(InterruptTarget::Tool { tool_call_id: id })?;
        }
        StreamEvent::ContentDelta(text) => print!("{text}"),
        _ => {}
    }
}

let result = stream.into_result().await?;
```

Queued user messages are appended at the next provider-call boundary, never mid-tool. Tool interrupts cancel only that tool's child token; the agent feeds the cancellation result back to the model and continues the turn. The same handle also exposes mode gates (`PlanMode`, `AcceptEditsMode`, custom `AgentMode`), root-thread `ask_user(...)` via a caller-provided `UserInputBridge`, synchronous `ContinuationGuard` predicates for keep-working loops, and runtime prompt policies.

Prompt policies append traceable system-prompt blocks at provider-call boundaries without changing tool-dispatch authority:

```rust
handle.install_prompt_policy(tkach::PromptPolicy {
    name: "diagnose-first".into(),
    scope: tkach::PolicyScope::NextTurn,
    content: "Prefer diagnosis before code.".into(),
    precedence: 10,
    trigger: tkach::PolicyTrigger::Always,
})?;
```

`PolicyScope::NextTurn` removes itself after the next matching provider request. `EveryTurnUntilRemoved` and `Persistent` stay installed for this handle until removed; `Persistent` does not cross process or handle lifetime boundaries. Streaming callers can observe `PolicyInstalled`, `PolicyRemoved`, and `PolicyApplied` events.

See [`examples/streaming_cancel.rs`](./examples/streaming_cancel.rs) for live cancel timing and [`examples/steering_edge_cases.rs`](./examples/steering_edge_cases.rs) for deterministic steering boundary checks.

## Approval flow

```rust
use tkach::{ApprovalDecision, ApprovalHandler, ToolClass};
use async_trait::async_trait;
use serde_json::Value;

struct MyApproval;

#[async_trait]
impl ApprovalHandler for MyApproval {
    async fn approve(&self, name: &str, input: &Value, class: ToolClass) -> ApprovalDecision {
        if class == ToolClass::ReadOnly {
            return ApprovalDecision::Allow;             // blanket-allow reads
        }
        // Hand off to UI; wait for user click.
        match prompt_user(name, input).await {
            true  => ApprovalDecision::Allow,
            false => ApprovalDecision::Deny("user declined".into()),
        }
    }
}

let agent = Agent::builder()
    .provider(Anthropic::from_env())
    .model(tkach::model::claude::HAIKU_20251001)
    .tools(tools::defaults())
    .approval(MyApproval)
    .build()?;
```

`Deny(reason)` flows back to the model as `is_error: true` tool_result so the LLM can adapt — it is **not** an `AgentError`. The runtime races `approve()` against `cancel.cancelled()`, so an outer cancel always wins over a hung UI handler.

## Specialised SubAgents

`SubAgent::new(provider, model)` still registers the default `agent` tool. For multiple children, give each one a unique tool name; `AgentBuilder::build()` returns `BuildError::DuplicateToolName` instead of silently shadowing a prior registration.

```rust
use std::sync::Arc;
use tkach::{Agent, ThinkingConfig, ThinkingEffort, providers::Anthropic, tools::SubAgent};

let haiku = Arc::new(Anthropic::from_env());
let research = SubAgent::new(haiku, tkach::model::claude::HAIKU_20251001)
    .name("research")
    .description("Read-only research helper")
    .tools_allow(["read", "glob", "grep", "web_fetch"])
    .filter_tool_definitions(true)
    .thinking(ThinkingConfig::Effort(ThinkingEffort::High));

let agent = Agent::builder()
    .provider(Anthropic::from_env())
    .model(tkach::model::claude::SONNET)
    .tools(tkach::tools::defaults())
    .tool(research)
    .build()?;
```

Tool scoping is an intersection with the parent policy: a child allow-list can remove capability, never re-add a tool denied by the parent. Leave `filter_tool_definitions(false)` to keep prompt-cache hashes stable and let denied calls return tool-result errors; enable it when the child should not even see disallowed tools.

`trace_hook` is optional. Configure it when the parent needs every child
`StreamEvent`, for example to observe a mutating child's intermediate tool
calls. Without a hook, the child uses the buffered `Agent::run` path and the
parent receives its final summary. The hook affects observability, not tool
permissions; use `tools_allow`, `ToolPolicy`, and `ApprovalHandler` to control
capabilities.

```rust
let writer = SubAgent::new(provider, tkach::model::claude::SONNET)
    .name("writer")
    .description("Mutating writer with full trace observability")
    .tools_allow(["read", "edit", "write", "bash"])
    .trace_hook(move |ev| audit_sink.record(ev));
```

See [`examples/specialised_subagents.rs`](./examples/specialised_subagents.rs) for three configurations registered side-by-side: read-only research, autonomous reasoning with `approval_handler(AutoApprove)`, and a mutating writer with `trace_hook`.

## Custom tools

```rust
use tkach::{Tool, ToolClass, ToolContext, ToolError, ToolOutput};
use serde_json::{Value, json};

struct CurrentTime;

#[async_trait::async_trait]
impl Tool for CurrentTime {
    fn name(&self) -> &str { "current_time" }
    fn description(&self) -> &str { "Returns the current UTC time as ISO 8601." }
    fn class(&self) -> ToolClass { ToolClass::ReadOnly }
    fn input_schema(&self) -> Value { json!({ "type": "object", "properties": {} }) }

    async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(chrono::Utc::now().to_rfc3339()))
    }
}

let agent = Agent::builder()
    .provider(...)
    .tool(CurrentTime)
    .build()?;
```

Long-running tools should `tokio::select!` on `ctx.cancel.cancelled()` and return `ToolError::Cancelled` promptly — the loop trusts the contract and does not race tools at the outer level.

## Examples

All examples are compiled in CI. Deterministic examples assert their key
behavior; `typed_pipeline.rs` is compile-time configuration documentation.

- [`basic.rs`](./examples/basic.rs) — Minimal `agent.run`.
- [`streaming.rs`](./examples/streaming.rs) — Anthropic streaming with visible/thinking event handling.
- [`streaming_anthropic_thinking.rs`](./examples/streaming_anthropic_thinking.rs) — Anthropic manual extended-thinking stream; asserts positive thinking blocks.
- [`streaming_anthropic_adaptive_thinking.rs`](./examples/streaming_anthropic_adaptive_thinking.rs) — Anthropic adaptive-thinking stream; asserts positive thinking blocks.
- [`streaming_multi_tool.rs`](./examples/streaming_multi_tool.rs) — Multi-turn write→edit→read chain via `Agent::stream`.
- [`streaming_subagent.rs`](./examples/streaming_subagent.rs) — Sonnet streams, delegates to a Haiku sub-agent.
- [`specialised_subagents.rs`](./examples/specialised_subagents.rs) — Named child tool with an allow-list and per-call thinking override.
- [`streaming_openai_tools.rs`](./examples/streaming_openai_tools.rs) — OpenAI-compatible tool call + no-thinking contract through Chat Completions.
- [`streaming_openai_responses_thinking.rs`](./examples/streaming_openai_responses_thinking.rs) — OpenAI Responses reasoning-summary stream; asserts positive thinking blocks.
- [`streaming_openai_codex.rs`](./examples/streaming_openai_codex.rs) — ChatGPT Codex subscription stream; reasoning summary + atomic tool calls.
- [`streaming_cancel.rs`](./examples/streaming_cancel.rs) — Cancel mid-generation, partial text preserved.
- [`streaming_resilience.rs`](./examples/streaming_resilience.rs) — Tool failure + cancel-during-tool + multi-block turns.
- [`approval_flow.rs`](./examples/approval_flow.rs) — Live denial flow with custom `ApprovalHandler`.
- [`parallel_tools.rs`](./examples/parallel_tools.rs) — Ordered read-only and mutating runs.
- [`parallel_writes.rs`](./examples/parallel_writes.rs) — Explicitly promote independent writes for concurrency.
- [`parallel_subagents.rs`](./examples/parallel_subagents.rs) — Concurrent sub-agent fan-out with bounded admission.
- [`steering_edge_cases.rs`](./examples/steering_edge_cases.rs) — Deterministic queue, interrupt, mode, and prompt-policy checks.
- [`typed_pipeline.rs`](./examples/typed_pipeline.rs) — Compile-time typed model and thinking configuration.
- [`custom_tool.rs`](./examples/custom_tool.rs) — Defining your own tool.
- [`anthropic_caching.rs`](./examples/anthropic_caching.rs) — Prompt caching: cache_creation vs cache_read on the second call.
- [`anthropic_caching_streaming.rs`](./examples/anthropic_caching_streaming.rs) — Same shape, but through the streaming API.
- [`anthropic_batch.rs`](./examples/anthropic_batch.rs) — Batch API happy path: submit → poll → stream results (50 % off, 24 h async).
- [`anthropic_batch_cancel.rs`](./examples/anthropic_batch_cancel.rs) — Batch cancel-then-fetch-partial; mixed `Succeeded` and `Canceled` outcomes.
- [`anthropic_batch_mixed.rs`](./examples/anthropic_batch_mixed.rs) — Per-row error isolation; bad request rides alongside successes as `Errored`.

Examples that talk to live APIs read `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and optional OpenAI override vars from `.env` — see [`.env.example`](./.env.example).

## Testing

```sh
cargo test                       # unit + mock-based integration (no network)
cargo test -- --ignored          # adds real-API smoke tests (needs ANTHROPIC_API_KEY)
cargo test --test integration image_input -- --ignored --test-threads=1 --nocapture
cargo run --example streaming    # any of the runnable examples
```

The image-input smoke tests use a generated two-color PNG and verify real multimodal interpretation across each configured provider. Anthropic additionally verifies an absolute file source followed by a text-only turn that replays the same image history. OpenAI-compatible, Responses, and Codex tests skip when their provider-specific credentials are absent; see [`.env.example`](./.env.example) for model and endpoint overrides.

CI runs formatting, clippy, all-target tests with coverage, RustSec audit,
`cargo deny`, and an MSRV 1.86 check. Clippy warnings other than
`cognitive_complexity` and `too_many_lines` fail the build; those two are
reported through SARIF without gating CI. Real-API tests run only through
`Actions → Integration Tests → Run workflow → tier=smoke|full`.

## Versioning & releases

Conventional commits + [release-please](https://github.com/googleapis/release-please) drive the version bump and changelog. See [`RELEASING.md`](./RELEASING.md). `feat!:` commits cut a breaking-change release; pre-1.0 those bump the minor version.

## License

[MIT](./LICENSE).
