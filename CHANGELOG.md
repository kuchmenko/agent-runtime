# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1](https://github.com/kuchmenko/agent-runtime/compare/agent-runtime-v0.1.0...agent-runtime-v0.1.1) (2026-04-05)


### Features

* integrate release-please for automated releases ([4099f2e](https://github.com/kuchmenko/agent-runtime/commit/4099f2e93fe3efc80d32394b0e5b4126638ff0ba))

## [Unreleased]

## [0.1.0] - 2026-04-04

### Added

- Core agent loop with sequential tool execution
- `LlmProvider` trait for provider-independent LLM integration
- `Tool` trait for custom tool implementation
- `Agent` builder with configurable model, system prompt, tools, and limits
- Built-in tools: Read, Write, Edit, Glob, Grep, Bash, SubAgent, WebFetch
- Anthropic Messages API provider
- Mock provider for testing
- Sub-agent spawning with depth limiting
- CI pipeline: fmt, clippy, test, audit, deny, MSRV
- Release workflow with GitHub Releases on tag push

[Unreleased]: https://github.com/kuchmenko/agent-runtime/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kuchmenko/agent-runtime/releases/tag/v0.1.0
