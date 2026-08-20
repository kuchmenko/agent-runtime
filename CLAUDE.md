# tkach

Provider-independent single-agent runtime for Rust with built-in tools.
Crate name and library import path are both `tkach`.

The repository is `kuchmenko/tkach` on GitHub. It was previously named
`kuchmenko/agent-runtime` (until 2026-04-26) and `kuchmenko/fellowship-rs`
(briefly, 2026-04-26). GitHub keeps automatic redirects from those old
URLs, so links in commit messages, prior issues, and `CHANGELOG.md`
continue to resolve.

## Architecture

Single crate with modules:
- `agent` — Agent struct, builder, agent loop, AgentStream
- `approval` — ApprovalHandler trait, ApprovalDecision, AutoApprove
- `tool` — Tool trait, ToolContext, ToolOutput, ToolClass
- `executor` — ToolExecutor, ToolRegistry, ToolPolicy, AllowAll
- `provider` — LlmProvider trait, Request/Response
- `stream` — StreamEvent, ProviderEventStream
- `message` — Message, Content, Role, StopReason, Usage
- `error` — AgentError, ProviderError, ToolError
- `tools/` — Built-in tools: Read, Write, Edit, Glob, Grep, Bash, SubAgent, WebFetch
- `providers/` — Anthropic, OpenAICompatible, OpenAIResponses, OpenAICodex, Mock

## Commands

- `cargo test` — run all tests
- `cargo clippy --all-targets -- -D warnings` — lint
- `cargo fmt --check` — format check

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — new feature (bumps **patch** pre-1.0; minor post-1.0)
- `fix:` — bug fix (bumps patch)
- `feat!:` or `fix!:` — breaking change (bumps minor pre-1.0; major post-1.0)
- `chore:`, `docs:`, `refactor:`, `test:`, `ci:` — no release

Pre-1.0 bumping is governed by `release-please-config.json` flags
`bump-minor-pre-major: true` and `bump-patch-for-minor-pre-major: true`.
The result: only breaking commits cut a new minor; everything else rides patches.

## Release process

Automated via [release-please](https://github.com/googleapis/release-please).
See `RELEASING.md` for details.

**Flow:** conventional commits on `main` → release-please creates Release PR → merge PR → GitHub Release + tag. All automatic.

**Do NOT** manually edit `CHANGELOG.md` or bump the version — release-please handles both.

**When to suggest merging the Release PR:**
- When meaningful features or fixes have accumulated
- After a breaking API change
- When the user asks about versioning or shipping
