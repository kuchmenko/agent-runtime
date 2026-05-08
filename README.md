# tkach

A provider-independent agent runtime for Rust. Stateless agent loop, pluggable LLM providers (Anthropic, OpenAI Responses, ChatGPT Codex, OpenAI-compatible), built-in file/shell tools, real SSE streaming with reasoning summaries, cooperative cancellation, and per-call approval gating.

[![Crates.io](https://img.shields.io/crates/v/tkach.svg)](https://crates.io/crates/tkach)
[![Docs.rs](https://img.shields.io/docsrs/tkach)](https://docs.rs/tkach)
[![CI](https://github.com/kuchmenko/tkach/actions/workflows/ci.yml/badge.svg)](https://github.com/kuchmenko/tkach/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Status:** pre-1.0 (`0.4.0`). Breaking changes are signalled via `feat!:` conventional commits and recorded in [`CHANGELOG.md`](./CHANGELOG.md). The core API has stabilised across foundation, streaming, approval, and reasoning milestones — and is settling — but expect motion.

## Features

- **Stateless `Agent::run`** — caller owns the message history; the agent returns the **delta** of new messages it appended. Resume, multi-turn chat, fork & retry all become composable.
- **Atomic events with one cancel surface** — `ToolUse` events are emitted whole, never as partial JSON; a single `CancellationToken` shuts down the loop, the SSE pull, the in-flight HTTP body, and any `bash` child process via `kill_on_drop`.
- **Provider parity, including reasoning** — Anthropic (adaptive and manual extended thinking), OpenAI Responses (reasoning summary), ChatGPT Codex (subscription endpoint), and any OpenAI-compatible Chat Completions endpoint share one `StreamEvent` API surface — not three SDKs.
- **Sub-agents inherit the parent's executor** — one `ApprovalHandler`, one `ToolPolicy`, one tool registry gate the whole agent tree without explicit re-plumbing (Model 3).

## Quick start

```toml
[dependencies]
tkach = "0.4"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use tkach::{Agent, CancellationToken, Message, providers::Anthropic, tools};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder()
        .provider(Anthropic::from_env())
        .model("claude-haiku-4-5-20251001")
        .system("You are a concise assistant.")
        .tools(tools::defaults())
        .build();

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
                ┌────────────┐                 ┌───────────────────┐
                │  Provider  │                 │   ToolExecutor    │
                │            │                 │ ┌───────────────┐ │
                │ Anthropic  │                 │ │  ToolPolicy   │ │
                │ OpenAI     │                 │ ├───────────────┤ │
                │ Responses/ │                 │ │ApprovalHandler│ │
                │ Codex/     │                 │ ├───────────────┤ │
                │ compatible │                 │ │ ToolRegistry  │ │
                │ Mock       │                 │ └───────────────┘ │
                └────────────┘                 └─────────┬─────────┘
                                                         │
                                              read-only batches in
                                              parallel via join_all,
                                              mutating sequentially
```

## Built-in tools

Read-only (`ToolClass::ReadOnly` — batched in parallel):

- `Read` — read file contents (numbered lines, offset/limit).
- `Glob` — find files matching a glob (sorted by mtime).
- `Grep` — regex search in files (with context, ignore patterns).
- `WebFetch` — HTTP GET a URL, returns body text.

Mutating (`ToolClass::Mutating` — executed sequentially):

- `Write` — write a file (creates parents).
- `Edit` — replace an exact string in a file.
- `Bash` — run a shell command (cancel-aware via `kill_on_drop`).
- `SubAgent` — spawn a nested agent that inherits the parent's tools and policies.

`tools::defaults()` returns `Read + Write + Edit + Glob + Grep + Bash`. Add `WebFetch` and `SubAgent::new(provider, model)` explicitly when you want them.

## Providers

```rust
use tkach::providers::{Anthropic, OpenAICompatible, OpenAIResponses};

// Anthropic Messages API.
let p = Anthropic::from_env();   // ANTHROPIC_API_KEY

// OpenAI Chat Completions-compatible API: text + tool calls, no standard thinking.
let p = OpenAICompatible::from_env();   // OPENAI_API_KEY

// OpenAI Responses API — required for reasoning-summary streams.
let p = OpenAIResponses::from_env()
    .with_reasoning("medium", "detailed");

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

### Anthropic extended thinking

`Anthropic::with_adaptive_thinking_effort` (recommended on Claude Sonnet/Opus 4.6+) lets the model decide when to think. `with_thinking_budget` is the older fixed-token mode.

```rust
// Adaptive thinking — recommended.
let p = Anthropic::from_env()
    .with_adaptive_thinking_effort("high");

// Manual budget — fixed-token mode.
let p = Anthropic::from_env()
    .with_thinking_budget(1024);
```

Both paths emit the same `StreamEvent::ThinkingDelta` and `StreamEvent::ThinkingBlock` events; downstream code does not branch on which mode produced them.

### Anthropic prompt caching

`SystemBlock::cached`, `Content::text_cached`, and `AgentBuilder::cache_tools` mark cache breakpoints; `Usage` reports `cache_creation_input_tokens` / `cache_read_input_tokens` so callers can measure hit rate. Default TTL is 5 min, with 1 h available via `CacheControl::ephemeral_1h()`. Cache reads bill at 0.1× base input; writes at 1.25× (5 min) or 2× (1 h). See `examples/anthropic_caching.rs` and `examples/anthropic_caching_streaming.rs`.

### Anthropic Message Batches (50 % async)

Anthropic's [Message Batches API](https://docs.anthropic.com/en/api/messages-batches) takes the same `Request` body, runs it asynchronously over up to 24 h, and bills **50 % off** input + output tokens. Stack with `SystemBlock::cached_1h(...)` for ≈85 % off when prefixes are stable across batches. Right call for overnight backfills, scheduled recompute jobs, evals, or any workload that doesn't care about p99.

```rust
use futures::StreamExt;
use tkach::providers::Anthropic;
use tkach::providers::anthropic::batch::{BatchOutcome, BatchRequest};
use tkach::{Message, Request};

let provider = Anthropic::from_env();

let requests = vec![BatchRequest {
    custom_id: "req-1".into(),               // ^[a-zA-Z0-9_-]{1,64}$, unique within batch
    params: Request {
        model: "claude-haiku-4-5-20251001".into(),
        system: None,
        messages: vec![Message::user_text("Say hello.")],
        tools: vec![],
        max_tokens: 64,
        temperature: None,
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
use tkach::providers::{CodexCredentials, CodexCredentialsProvider, OpenAICodex};

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
    .with_reasoning_summary("auto")             // optional, default "auto"
    .with_reasoning_effort("medium");           // optional, off by default

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
        StreamEvent::Done => break,
        _ => {}                                  // MessageDelta, Usage
    }
}

let result = stream.into_result().await?;        // final AgentResult
```

`ThinkingDelta` and `ThinkingBlock` are public `StreamEvent` variants. Downstream exhaustive matches must add arms for them when upgrading.

Provider boundary: Anthropic thinking requires `Anthropic::with_adaptive_thinking*` or `with_thinking_budget(...)`; OpenAI thinking requires `OpenAIResponses` (`/responses` with `reasoning.summary`). `OpenAICompatible` is Chat Completions and intentionally asserts the no-thinking contract because that wire format has no standard reasoning-summary event.

Backpressure is real: a slow consumer parks the producer task, which closes the SSE read side, which lets the OS shrink the TCP receive window — all the way back to the LLM server. Cancellation works mid-stream too: `cancel.cancel()` aborts the current SSE pull within milliseconds via `tokio::select!`.

See [`examples/streaming_cancel.rs`](./examples/streaming_cancel.rs) for live cancel timing.

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
    .model("claude-haiku-4-5-20251001")
    .tools(tools::defaults())
    .approval(MyApproval)
    .build();
```

`Deny(reason)` flows back to the model as `is_error: true` tool_result so the LLM can adapt — it is **not** an `AgentError`. The runtime races `approve()` against `cancel.cancelled()`, so an outer cancel always wins over a hung UI handler.

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
    .build();
```

Long-running tools should `tokio::select!` on `ctx.cancel.cancelled()` and return `ToolError::Cancelled` promptly — the loop trusts the contract and does not race tools at the outer level.

## Examples

Each runnable demo also asserts its invariants — `cargo run --example NAME` either prints the demo and exits 0, or panics with a clear message.

- [`basic.rs`](./examples/basic.rs) — Minimal `agent.run`.
- [`streaming.rs`](./examples/streaming.rs) — Anthropic streaming with visible/thinking event handling.
- [`streaming_anthropic_thinking.rs`](./examples/streaming_anthropic_thinking.rs) — Anthropic manual extended-thinking stream; asserts positive thinking blocks.
- [`streaming_anthropic_adaptive_thinking.rs`](./examples/streaming_anthropic_adaptive_thinking.rs) — Anthropic adaptive-thinking stream; asserts positive thinking blocks.
- [`streaming_multi_tool.rs`](./examples/streaming_multi_tool.rs) — Multi-turn write→edit→read chain via `Agent::stream`.
- [`streaming_subagent.rs`](./examples/streaming_subagent.rs) — Sonnet streams, delegates to a Haiku sub-agent.
- [`streaming_openai_tools.rs`](./examples/streaming_openai_tools.rs) — OpenAI-compatible tool call + no-thinking contract through Chat Completions.
- [`streaming_openai_responses_thinking.rs`](./examples/streaming_openai_responses_thinking.rs) — OpenAI Responses reasoning-summary stream; asserts positive thinking blocks.
- [`streaming_openai_codex.rs`](./examples/streaming_openai_codex.rs) — ChatGPT Codex subscription stream; reasoning summary + atomic tool calls.
- [`streaming_cancel.rs`](./examples/streaming_cancel.rs) — Cancel mid-generation, partial text preserved.
- [`streaming_resilience.rs`](./examples/streaming_resilience.rs) — Tool failure + cancel-during-tool + multi-block turns.
- [`approval_flow.rs`](./examples/approval_flow.rs) — Live denial flow with custom `ApprovalHandler`.
- [`parallel_tools.rs`](./examples/parallel_tools.rs) — Read-only tools running in parallel.
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
cargo run --example streaming    # any of the runnable examples
```

CI runs fmt, clippy (with cognitive-complexity gates), MSRV (1.86), and `cargo deny` on every PR. Real-API smoke runs are gated behind `Actions → Integration Tests → Run workflow → tier=smoke|full`.

## Versioning & releases

Conventional commits + [release-please](https://github.com/googleapis/release-please) drive the version bump and changelog. See [`RELEASING.md`](./RELEASING.md). `feat!:` commits cut a breaking-change release; pre-1.0 those bump the minor version.

## License

[MIT](./LICENSE).
