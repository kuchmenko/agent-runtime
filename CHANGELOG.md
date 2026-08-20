# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0](https://github.com/kuchmenko/tkach/compare/tkach-v0.5.0...tkach-v0.6.0) (2026-08-20)


### ⚠ BREAKING CHANGES

* add native image input ([#51](https://github.com/kuchmenko/tkach/issues/51))

### Features

* add native image input ([#51](https://github.com/kuchmenko/tkach/issues/51)) ([351ddc0](https://github.com/kuchmenko/tkach/commit/351ddc0fc9dcfb3e16228f069e1be7d03b0492f9))


### Bug Fixes

* **deps:** update h2 for RUSTSEC-2026-0258 ([bde1d16](https://github.com/kuchmenko/tkach/commit/bde1d169fc6f6c13d7130f0ce32d68003510e2f3))

## [0.5.0](https://github.com/kuchmenko/tkach/compare/tkach-v0.4.0...tkach-v0.5.0) (2026-05-12)

### Added

* **Steering handle:** `Agent::run_with_handle` and `Agent::stream_with_handle` return an `AgentHandle` so callers can steer an active run without rebuilding the agent loop.
* **Queued user input:** `AgentHandle::queue_user_message` appends operator context at the next safe provider-call boundary.
* **Interrupts:** `AgentHandle::interrupt` can cancel the session, active turn, or one in-flight tool while preserving partial results.
* **Modes:** `AgentMode`, `PlanMode`, and `AcceptEditsMode` gate tool dispatch independently from approval handlers.
* **User input bridge:** root agents can request structured operator input through `AgentHandle::ask_user`.
* **Continuation guards:** callers can install bounded predicates that inject follow-up prompts or abort runaway loops.
* **Runtime prompt policies:** callers can install scoped system-prompt addenda with `NextTurn`, `EveryTurnUntilRemoved`, or handle-lifetime `Persistent` scope.
* **Steering events:** streams now expose turn, mode, guard, and prompt-policy lifecycle events for UI and tracing integrations.
* **Sub-agent profiles:** `SubAgent` supports named specialised child-agent profiles and forwards full child stream events to `trace_hook`.
* **Tool concurrency controls:** `AgentBuilder` exposes read/mutator class caps and per-tool concurrency opt-ins.

### Fixed

* Hardened steering boundaries around turn ids, queue draining, tool admission, mode changes, continuation guards, and prompt-policy matching.
* Prevented mode and prompt-policy lock-ordering hazards, including reentrant prompt-policy matchers.
* Cleared stale tool-call context before later text-only prompt-policy matches.
* Preserved mode and policy events before stream completion.
* Applied parent tool policy correctly when filtering child-agent tool definitions.

## [0.4.0](https://github.com/kuchmenko/tkach/compare/tkach-v0.3.0...tkach-v0.4.0) (2026-05-08)


### ⚠ BREAKING CHANGES

* add thinking provider examples and replay guards

### Features

* add Anthropic adaptive thinking ([2b5dd21](https://github.com/kuchmenko/tkach/commit/2b5dd21d4cf16a51d18a81f8f5712e49f6ec0475))
* add OpenAI Codex subscription provider ([51efee3](https://github.com/kuchmenko/tkach/commit/51efee3bcf60482e55b72abab8ab00a7bd51ab31))
* add OpenAI Codex subscription provider ([0152052](https://github.com/kuchmenko/tkach/commit/01520525833f54d6256971de5b5e573b5f8f2d3a)), closes [#31](https://github.com/kuchmenko/tkach/issues/31)
* add OpenAI Responses thinking provider ([002c19b](https://github.com/kuchmenko/tkach/commit/002c19b650d1d3b61179d181c92564dd36d23482))
* add provider thinking events ([5db0a6e](https://github.com/kuchmenko/tkach/commit/5db0a6e500ce59268655f773ea9350aebcd235c9))
* add thinking provider examples and replay guards ([b67f2b2](https://github.com/kuchmenko/tkach/commit/b67f2b293f4136a9db8fe1800e8d39ac196a7c8f))
* enable Anthropic thinking smoke example ([17417c5](https://github.com/kuchmenko/tkach/commit/17417c5e09b4d649428ba98f143118d887b47bd4))


### Bug Fixes

* address Codex/CodeRabbit review — terminal handling ([79984d6](https://github.com/kuchmenko/tkach/commit/79984d6c324ce6f203a903d6e37a2a39d9bdd1be))
* address thinking review feedback ([da2d45f](https://github.com/kuchmenko/tkach/commit/da2d45f95ac4808412d1b6b25061f6e86be8c32b))
* drop output-only status field from Responses input items ([b6bf17d](https://github.com/kuchmenko/tkach/commit/b6bf17dfd2c48ddbb3a5f28956e88a9b13acb146))
* preserve Anthropic redacted thinking ([2ba19d7](https://github.com/kuchmenko/tkach/commit/2ba19d7e1a7edc61fcd0c88786c71495b4668b91))
* send reasoning.summary so Codex emits summary events ([29dcc6b](https://github.com/kuchmenko/tkach/commit/29dcc6b89def53e1711f7426779d1eaf47df30da))

## [0.3.0](https://github.com/kuchmenko/tkach/compare/tkach-v0.2.0...tkach-v0.3.0) (2026-04-29)


### ⚠ BREAKING CHANGES

* anthropic prompt caching (cache_control, SystemBlock, cache token usage) ([#25](https://github.com/kuchmenko/tkach/issues/25))

### Features

* Anthropic Message Batches API (50% off async) ([#27](https://github.com/kuchmenko/tkach/issues/27)) ([f46f447](https://github.com/kuchmenko/tkach/commit/f46f447e12872e8e99a7091da8ce74f51792c2ea))
* anthropic prompt caching (cache_control, SystemBlock, cache token usage) ([#25](https://github.com/kuchmenko/tkach/issues/25)) ([3c7d871](https://github.com/kuchmenko/tkach/commit/3c7d8714e48a5b59de3baee03cfa4b7b84e06e93))

## [0.2.0](https://github.com/kuchmenko/tkach/compare/tkach-v0.1.0...tkach-v0.2.0) (2026-04-26)


### ⚠ BREAKING CHANGES

* rename crate fellowship-rs → tkach
* `ToolContext` loses provider/model/max_turns/ max_tokens/temperature/agent_depth/max_agent_depth; gains `depth`, `max_depth`, `executor`. `SubAgent` is no longer a unit struct — construct via `SubAgent::new(provider, model)` and configure with `.system()`, `.max_turns()`, etc. `AgentBuilder::max_agent_depth` renamed to `max_depth`.
* `ToolContext` gains the required `cancel` field; direct constructions must supply it. `ToolError::Cancelled` variant added.
* none at the API level (execute_one still exists), but observable concurrency semantics change — callers that rely on sequential execution of consecutive read-only tools must reclassify them to `Mutating`.
* `tools::defaults()` returns `Vec<Arc<dyn Tool>>` (was `Vec<Box<dyn Tool>>`); `tools::all()` removed; `AgentBuilder::tools` signature changed to match.
* `Tool` trait gains a `class()` method (defaulted, so existing implementations compile; override if the tool is read-only).
* `Agent::run(&str)` replaced by `Agent::run(Vec<Message>, CancellationToken)`; `AgentResult::messages` renamed to `new_messages` and now holds only the delta; `AgentError::MaxTurnsReached(usize)` and
* `StopReason` gains three variants — exhaustive `match` patterns on it must be updated.
* `ProviderError::Api` gains a `retryable: bool` field; two new variants added.

### Features

* add integration tests with real Anthropic API ([35fd5d3](https://github.com/kuchmenko/tkach/commit/35fd5d35721616856f27be591cd4cb9c3a40f70d))
* add OpenAI-compatible provider (non-streaming, API-key) ([37fdfef](https://github.com/kuchmenko/tkach/commit/37fdfef3e15ada9e788e7a93c96467124132db19))
* add Tool::class() to classify side effects ([6ae797f](https://github.com/kuchmenko/tkach/commit/6ae797fac1d8864612f5938cab5a269f74962f10))
* Agent::stream + AgentStream handle ([668b6cb](https://github.com/kuchmenko/tkach/commit/668b6cb5304ae0de7bccc22482ff5b9f2a675e47))
* Agent::stream emits ToolCallPending before approval gate ([4adb3bc](https://github.com/kuchmenko/tkach/commit/4adb3bcc13d023b5874247a6dd8f22764acceb7a))
* AgentBuilder::approval + StreamEvent::ToolCallPending variant ([198aef9](https://github.com/kuchmenko/tkach/commit/198aef9eb459683c09472dcd301c0aaee7c0cd04))
* approval flow — ApprovalHandler, ToolCallPending events, cancel-aware gate ([e67ccec](https://github.com/kuchmenko/tkach/commit/e67ccec0b6760a1e307f8671c7a26da7289f59fe))
* ApprovalHandler trait + AutoApprove default ([f0322bc](https://github.com/kuchmenko/tkach/commit/f0322bc44e8f4db1a56057d6cdcd631226ccdfb1))
* classify ProviderError for retry decisions ([013e335](https://github.com/kuchmenko/tkach/commit/013e33532fd1fa12c1a2a997f4b76dabc7222daf))
* implement Anthropic SSE streaming ([2dbf9be](https://github.com/kuchmenko/tkach/commit/2dbf9be436cbbc3511110c66e311d3adb52aa630))
* implement OpenAI-compatible SSE streaming ([1c8a917](https://github.com/kuchmenko/tkach/commit/1c8a91714605c18b195e83a7a6c62ba05e065a21))
* integrate release-please for automated releases ([4099f2e](https://github.com/kuchmenko/tkach/commit/4099f2e93fe3efc80d32394b0e5b4126638ff0ba))
* introduce StreamEvent and LlmProvider::stream ([5249fc0](https://github.com/kuchmenko/tkach/commit/5249fc0f86fe9f811934b3cf9d26d1c7cd2d475d))
* introduce ToolRegistry, ToolExecutor, ToolPolicy ([6ddf009](https://github.com/kuchmenko/tkach/commit/6ddf009a846a70f63907daa3540d7b476249b2e5))
* make Agent::run stateless with CancellationToken ([bd3d83d](https://github.com/kuchmenko/tkach/commit/bd3d83d3993c75c1e9cea5a50320d86a3f79fc68))
* parallelise read-only tool batches in execute_batch ([b8a22d1](https://github.com/kuchmenko/tkach/commit/b8a22d1658c032579957be167065951bd10ec8b6))
* propagate cancellation through ctx into Bash/WebFetch ([adde3b6](https://github.com/kuchmenko/tkach/commit/adde3b645fa645a96bc4324b278b30133ba203a9))
* rework SubAgent to Model 3 (stateful, executor via ctx) ([abeb778](https://github.com/kuchmenko/tkach/commit/abeb778c8df15bdcf6d44a9eab2cf5923d186ae3))
* streaming — StreamEvent, Agent::stream, real SSE for Anthropic and OpenAI-compat ([9b11cff](https://github.com/kuchmenko/tkach/commit/9b11cffa8d4d841519541d27cfe5f3a1abbb74c1))
* ToolExecutor approval gate with cancel-aware select! ([a7748d0](https://github.com/kuchmenko/tkach/commit/a7748d03424695571f0480157c0613c6f4e3381c))
* widen StopReason with StopSequence, PauseTurn, Cancelled ([1611ece](https://github.com/kuchmenko/tkach/commit/1611ece32bb9e45ec5f3b3258b156ab9b90ac8e2))


### Bug Fixes

* cancel mid-stream and four end-to-end verification examples ([ae96069](https://github.com/kuchmenko/tkach/commit/ae960698f401d773b9ad34b7a4faffeb44925124))
* integration test exhaustive match for StreamEvent::ToolCallPending ([872f4a6](https://github.com/kuchmenko/tkach/commit/872f4a627650869b69dcabe9f4f5d6ed51824d7e))
* OpenAI-compat wire correctness + narrow Http retryability (P2) ([7b73de4](https://github.com/kuchmenko/tkach/commit/7b73de4d0c45278b7a3a56632ec8c8245e271950))
* stop dispatching tools after cancellation fires (P1) ([e71ee2b](https://github.com/kuchmenko/tkach/commit/e71ee2b6cb2a0a68f63239b7d21ee46b19f8b5f0))


### Miscellaneous Chores

* rename crate fellowship-rs → tkach ([3fcef20](https://github.com/kuchmenko/tkach/commit/3fcef2013717bed20d912afcc13ca5f7a4ffe211))

## [0.1.0] - 2026-04-26

Initial public release of `tkach`.

A provider-independent agent runtime for Rust with a stateless agent
loop, pluggable LLM providers (Anthropic, OpenAI-compatible), built-in
file/shell tools, real SSE streaming, cooperative cancellation, and
per-call approval gating.

### Core API

- `Agent::run` — stateless agent loop; caller owns message history
- `Agent::stream` + `AgentStream` — live token streaming with atomic `ToolUse` events
- `LlmProvider` trait — providers: Anthropic, OpenAICompatible (OpenAI / OpenRouter / Ollama / Moonshot / DeepSeek / Together / Groq), Mock
- `Tool` trait + `ToolClass::{ReadOnly, Mutating}` — read-only batches run in parallel; mutating runs sequentially
- `ToolExecutor` + `ToolPolicy` + `ApprovalHandler` — two-gate execution model with cancel-aware approval
- `CancellationToken` — propagates through the loop, SSE pull, HTTP body, and `Bash` child processes via `kill_on_drop`
- `SubAgent` (Model 3) — nested agents inherit parent's executor; one approval handler / policy / registry gates the whole tree

### Built-in tools

`Read`, `Glob`, `Grep`, `WebFetch` (read-only) · `Write`, `Edit`, `Bash`, `SubAgent` (mutating)
