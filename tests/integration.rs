// Integration tests that call real LLM APIs.
// All tests are #[ignore] so the default `cargo test` doesn't hit the network.
//
// Local: `cargo test -- --ignored` (loads .env via dotenvy — see .env.example).
// CI:    Actions → "Integration Tests" → Run workflow → tier=smoke|full.

use std::path::Path;
use std::sync::{Arc, Once};

use futures::StreamExt;
use tkach::message::{Content, Message, ThinkingProvider};
use tkach::provider::{Request, SystemBlock};
use tkach::providers::{Anthropic, OpenAICompatible, OpenAIResponses};
use tkach::tools::SubAgent;
use tkach::{Agent, AgentResult, CancellationToken, LlmProvider, StreamEvent};

/// Load `.env` once per test process. `cargo test` runs every `#[test]` on
/// the same process by default, so the `Once` ensures a single load.
/// Failure is silently ignored — env vars from the shell still take
/// precedence and CI sets them directly.
fn load_env() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = dotenvy::dotenv();
    });
}

fn prompt(text: &str) -> Vec<Message> {
    vec![Message::user_text(text)]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn require_api_key() -> Anthropic {
    load_env();
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        panic!("ANTHROPIC_API_KEY is required for integration tests (set it in .env)");
    }
    Anthropic::from_env()
}

fn haiku_agent(working_dir: &Path) -> Agent {
    Agent::builder()
        .provider(require_api_key())
        .model(tkach::model::claude::HAIKU_20251001)
        .system("You are a concise assistant. Use tools when needed. Be brief.")
        .tools(tkach::tools::defaults())
        .max_turns(10)
        .max_tokens(1024)
        .working_dir(working_dir)
        .build()
        .unwrap()
}

fn sonnet_agent(working_dir: &Path) -> Agent {
    let provider: Arc<dyn LlmProvider> = Arc::new(require_api_key());
    let sub_agent = SubAgent::new(Arc::clone(&provider), tkach::model::claude::HAIKU_20251001)
        .max_turns(10)
        .max_tokens(2048);

    Agent::builder()
        .provider_arc(provider)
        .model(tkach::model::claude::SONNET)
        .system("You are a concise coding assistant. Use tools when needed. Be brief.")
        .tools(tkach::tools::defaults())
        .tool(tkach::tools::WebFetch)
        .tool(sub_agent)
        .max_turns(15)
        .max_tokens(4096)
        .working_dir(working_dir)
        .build()
        .unwrap()
}

fn assert_tool_called(result: &AgentResult, tool_name: &str) {
    let called = result.new_messages.iter().any(|msg| {
        msg.content
            .iter()
            .any(|c| matches!(c, Content::ToolUse { name, .. } if name == tool_name))
    });
    assert!(
        called,
        "Expected tool '{tool_name}' to be called. Tools called: {:?}",
        collect_tool_calls(result)
    );
}

fn assert_no_tool_errors(result: &AgentResult) {
    for msg in &result.new_messages {
        for content in &msg.content {
            if let Content::ToolResult {
                is_error: true,
                content,
                ..
            } = content
            {
                panic!("Unexpected tool error in conversation: {content}");
            }
        }
    }
}

fn collect_tool_calls(result: &AgentResult) -> Vec<String> {
    result
        .new_messages
        .iter()
        .flat_map(|msg| msg.content.iter())
        .filter_map(|c| match c {
            Content::ToolUse { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn assert_file_contains(path: &Path, expected: &str) {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    assert!(
        content.contains(expected),
        "File {} does not contain '{expected}'. Content:\n{content}",
        path.display()
    );
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("tkach_integration").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// Tier 1: Smoke tests (haiku) — provider conversion + basic loop
// ---------------------------------------------------------------------------

/// Raw provider roundtrip: send message, get text back (no tools).
#[tokio::test]
#[ignore]
async fn smoke_provider_roundtrip() {
    let agent = Agent::builder()
        .provider(require_api_key())
        .model(tkach::model::claude::HAIKU_20251001)
        .system("Reply with exactly: PONG")
        .max_turns(1)
        .max_tokens(32)
        .build()
        .unwrap();

    let result = agent
        .run(prompt("PING"), CancellationToken::new())
        .await
        .unwrap();

    assert!(!result.text.is_empty(), "Response should not be empty");
    assert!(
        result.text.contains("PONG"),
        "Expected 'PONG' in response, got: {}",
        result.text
    );
    assert!(result.usage.input_tokens > 0, "Should have input tokens");
    assert!(result.usage.output_tokens > 0, "Should have output tokens");
}

/// Agent reads a known file using the read tool.
#[tokio::test]
#[ignore]
async fn smoke_agent_reads_file() {
    let dir = temp_dir("smoke_read");
    std::fs::write(dir.join("hello.txt"), "The secret code is 42.").unwrap();

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("Read the file hello.txt and tell me the secret code."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "read");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains("42"),
        "Agent should mention the secret code. Got: {}",
        result.text
    );
}

/// Agent runs a bash command.
#[tokio::test]
#[ignore]
async fn smoke_agent_runs_bash() {
    let dir = temp_dir("smoke_bash");

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("Run `echo 'hello_from_bash'` and tell me what it printed."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "bash");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains("hello_from_bash"),
        "Agent should report the command output. Got: {}",
        result.text
    );
}

/// Agent creates a new file.
#[tokio::test]
#[ignore]
async fn smoke_agent_writes_file() {
    let dir = temp_dir("smoke_write");

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("Create a file called output.txt with the content 'agent was here'."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "write");
    assert_no_tool_errors(&result);

    let output_file = dir.join("output.txt");
    assert!(output_file.exists(), "File should have been created");
    assert_file_contains(&output_file, "agent was here");
}

/// Agent uses glob to find files.
#[tokio::test]
#[ignore]
async fn smoke_agent_finds_files() {
    let dir = temp_dir("smoke_glob");
    std::fs::write(dir.join("foo.rs"), "fn foo() {}").unwrap();
    std::fs::write(dir.join("bar.rs"), "fn bar() {}").unwrap();
    std::fs::write(dir.join("readme.md"), "# Hello").unwrap();

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("How many .rs files are in this directory? Use glob to find them."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "glob");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains('2') || result.text.to_lowercase().contains("two"),
        "Agent should find 2 .rs files. Got: {}",
        result.text
    );
}

/// Agent uses grep to search file contents.
#[tokio::test]
#[ignore]
async fn smoke_agent_greps() {
    let dir = temp_dir("smoke_grep");
    std::fs::write(
        dir.join("code.rs"),
        "fn main() {\n    let x = TODO_FIX;\n}\n",
    )
    .unwrap();
    std::fs::write(dir.join("lib.rs"), "pub fn helper() {}\n").unwrap();

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("Search for 'TODO_FIX' in the files here. Which file contains it?"),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "grep");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains("code.rs"),
        "Agent should identify code.rs. Got: {}",
        result.text
    );
}

// ---------------------------------------------------------------------------
// Tier 2: Full tests (sonnet) — complex multi-tool scenarios
// ---------------------------------------------------------------------------

/// Agent reads a file, edits it, then the edit is verified.
#[tokio::test]
#[ignore]
async fn full_agent_edit_chain() {
    let dir = temp_dir("full_edit");
    std::fs::write(
        dir.join("config.toml"),
        "[server]\nhost = \"localhost\"\nport = 8080\n",
    )
    .unwrap();

    let agent = sonnet_agent(&dir);
    let result = agent
        .run(
            prompt("Read config.toml, then change the port from 8080 to 9090."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "read");
    assert_tool_called(&result, "edit");
    assert_no_tool_errors(&result);
    assert_file_contains(&dir.join("config.toml"), "9090");
}

/// Agent combines glob + grep to find a pattern across files.
#[tokio::test]
#[ignore]
async fn full_agent_multi_tool_search() {
    let dir = temp_dir("full_search");
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(
        dir.join("src/main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/lib.rs"),
        "pub async fn process() {\n    // async work\n}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/utils.rs"),
        "pub fn helper() {\n    // sync helper\n}\n",
    )
    .unwrap();

    let agent = sonnet_agent(&dir);
    let result = agent
        .run(
            prompt(
                "Use the grep tool to search for the pattern 'async' in the src/ directory. \
                 Tell me which files contain it.",
            ),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "grep");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains("lib.rs"),
        "Agent should find lib.rs. Got: {}",
        result.text
    );
    let tools_used = collect_tool_calls(&result);
    assert!(
        !tools_used.is_empty(),
        "Agent should have used at least one tool"
    );
}

/// Agent delegates to a sub-agent.
#[tokio::test]
#[ignore]
async fn full_agent_sub_agent() {
    let dir = temp_dir("full_subagent");
    std::fs::write(dir.join("data.txt"), "The answer is 7.").unwrap();

    let agent = sonnet_agent(&dir);
    let result = agent
        .run(
            prompt(
                "Use a sub-agent to read data.txt and report what it says. \
                 Pass this prompt to the agent tool: 'Read data.txt and return its contents.'",
            ),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "agent");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains('7') || result.text.contains("seven"),
        "Agent should relay the sub-agent's finding. Got: {}",
        result.text
    );
}

/// Full scenario: create a project structure and make changes.
#[tokio::test]
#[ignore]
async fn full_agent_create_and_modify() {
    let dir = temp_dir("full_create_modify");

    let agent = sonnet_agent(&dir);
    let result = agent
        .run(
            prompt(
                "Create a file called hello.py with a function greet(name) that \
                 prints 'Hello, {name}!'. Then read it back to verify it's correct.",
            ),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "write");
    assert_tool_called(&result, "read");
    assert_no_tool_errors(&result);

    let py_file = dir.join("hello.py");
    assert!(py_file.exists(), "hello.py should have been created");
    assert_file_contains(&py_file, "greet");
    assert_file_contains(&py_file, "Hello");
}

// ---------------------------------------------------------------------------
// OpenAI-compatible provider smoke
// ---------------------------------------------------------------------------

/// Round-trip against any OpenAI-compatible endpoint to validate our
/// `OpenAICompatible` provider end-to-end (request shape, tool_calls /
/// finish_reason mapping, retry classification on real responses).
///
/// **Status: opportunistic.** This is a fully functional smoke — when
/// `OPENAI_API_KEY` is set in the environment (or `.env`), it performs
/// a real round-trip; when unset / empty / still the placeholder from
/// `.env.example`, it skips with a printed reason. Drop a key into
/// `.env` (or point at a local Ollama with
/// `OPENAI_BASE_URL=http://localhost:11434/v1`, or OpenRouter via
/// `OPENAI_BASE_URL=https://openrouter.ai/api/v1`) to enable.
///
/// TODO(openai): once a key/endpoint is provisioned in CI, add it to
/// `.github/workflows/integration.yml` so `tier=smoke` exercises both
/// providers, not just Anthropic.
#[tokio::test]
#[ignore]
async fn smoke_openai_compatible_roundtrip() {
    load_env();
    // Skip if the var is unset, empty, or still the .env.example placeholder.
    // Without these checks the provider would round-trip a bogus key into
    // a 401 from the upstream and look like a real failure.
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() && !k.starts_with("sk-...") => k,
        _ => {
            eprintln!(
                "skipping smoke_openai_compatible_roundtrip: \
                 OPENAI_API_KEY missing, empty, or still the placeholder"
            );
            return;
        }
    };

    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
    let model = std::env::var("OPENAI_SMOKE_MODEL")
        .unwrap_or_else(|_| tkach::model::openrouter::OPENAI_GPT_5_5.to_string());

    let provider = OpenAICompatible::new(api_key).with_base_url(base_url);

    // Hit the provider directly (no agent loop) for the tightest possible
    // assertion on the wire bridge — system prompt routing, response
    // decoding, finish_reason mapping. 256 tokens of headroom because
    // some models (Kimi, reasoning Llamas) are chatty even when told to
    // be brief; we don't want the assertion to fight model verbosity.
    let request = Request {
        model: model.clone(),
        system: Some(vec![SystemBlock::text(
            "Reply with exactly the single word: PONG",
        )]),
        messages: vec![Message::user_text("PING")],
        tools: vec![],
        max_tokens: 256,
        temperature: Some(0.0),
        thinking: None,
    };

    let response = provider
        .complete(request)
        .await
        .expect("provider round-trip should succeed");

    let text: String = response
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();

    eprintln!(
        "[smoke openai-compat | model={model}] stop={:?} \
         in={} out={} text={text:?}",
        response.stop_reason, response.usage.input_tokens, response.usage.output_tokens
    );

    assert!(response.usage.input_tokens > 0, "should have prompt tokens");
    assert!(
        response.usage.output_tokens > 0,
        "should have completion tokens"
    );
    // Don't pin StopReason — different models legitimately stop on
    // EndTurn, MaxTokens, or StopSequence. The semantic check is that
    // the model said PONG somewhere in its output.
    assert!(
        text.to_uppercase().contains("PONG"),
        "expected PONG in response, got: {text:?}"
    );
}

// ---------------------------------------------------------------------------
// Anthropic streaming smoke (real SSE round-trip)
// ---------------------------------------------------------------------------

/// Streams a PING/PONG response through the Anthropic SSE pipeline.
/// Validates: bearer auth via `x-api-key`, `stream: true` switching the
/// transport, our SSE state machine progressing through `message_start`
/// → `content_block_*` → `message_delta` → `message_stop`, and the
/// final assembled text matching what `complete()` would have returned.
#[tokio::test]
#[ignore]
async fn smoke_anthropic_stream_roundtrip() {
    load_env();
    let provider = require_api_key();

    let request = Request {
        model: tkach::model::claude::HAIKU_20251001.into(),
        system: Some(vec![SystemBlock::text("Reply with exactly: PONG")]),
        messages: vec![Message::user_text("PING")],
        tools: vec![],
        max_tokens: 32,
        temperature: Some(0.0),
        thinking: None,
    };

    let mut stream = provider.stream(request).await.expect("open stream");
    let mut text = String::new();
    let mut delta_count = 0usize;
    let mut got_message_delta = false;
    let mut got_done = false;
    let mut input_tokens = 0u32;
    let mut output_tokens = 0u32;

    while let Some(event) = stream.next().await {
        let ev = event.expect("event ok");
        match ev {
            StreamEvent::ContentDelta(t) => {
                delta_count += 1;
                text.push_str(&t);
            }
            StreamEvent::ToolUse { .. } => panic!("no tools in this prompt"),
            StreamEvent::MessageDelta { .. } => got_message_delta = true,
            StreamEvent::Usage(u) => {
                if u.input_tokens > 0 {
                    input_tokens = u.input_tokens;
                }
                if u.output_tokens > 0 {
                    output_tokens = u.output_tokens;
                }
            }
            StreamEvent::Done => got_done = true,
            StreamEvent::TurnStarted { .. }
            | StreamEvent::ModeChanged { .. }
            | StreamEvent::ModeChangeRequested { .. }
            | StreamEvent::ContinuationInjected { .. }
            | StreamEvent::GuardAborted { .. }
            | StreamEvent::PolicyInstalled { .. }
            | StreamEvent::PolicyRemoved { .. }
            | StreamEvent::PolicyApplied { .. }
            | StreamEvent::ThinkingDelta { .. }
            | StreamEvent::ThinkingBlock { .. }
            | StreamEvent::ToolCallPending { .. } => {}
        }
    }

    eprintln!(
        "[smoke anthropic stream] deltas={delta_count} \
         in={input_tokens} out={output_tokens} text={text:?}"
    );

    assert!(delta_count >= 1, "should have received at least one delta");
    assert!(got_message_delta, "should have received MessageDelta");
    assert!(got_done, "should have received Done terminal");
    assert!(input_tokens > 0, "should have prompt tokens");
    assert!(output_tokens > 0, "should have completion tokens");
    assert!(
        text.to_uppercase().contains("PONG"),
        "expected PONG in assembled text, got: {text:?}"
    );
}

/// Streams a PING/PONG response through any OpenAI-compatible endpoint.
/// Same opportunistic-skip semantics as the non-streaming OpenAI smoke.
#[tokio::test]
#[ignore]
async fn smoke_openai_compatible_stream_roundtrip() {
    load_env();
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() && !k.starts_with("sk-...") => k,
        _ => {
            eprintln!(
                "skipping smoke_openai_compatible_stream_roundtrip: \
                 OPENAI_API_KEY missing, empty, or still the placeholder"
            );
            return;
        }
    };

    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
    let model = std::env::var("OPENAI_SMOKE_MODEL")
        .unwrap_or_else(|_| tkach::model::openrouter::OPENAI_GPT_5_5.to_string());

    let provider = OpenAICompatible::new(api_key).with_base_url(base_url);

    let request = Request {
        model: model.clone(),
        system: Some(vec![SystemBlock::text(
            "Reply with exactly the single word: PONG",
        )]),
        messages: vec![Message::user_text("PING")],
        tools: vec![],
        max_tokens: 256,
        temperature: Some(0.0),
        thinking: None,
    };

    let mut stream = provider.stream(request).await.expect("open stream");
    let mut text = String::new();
    let mut delta_count = 0usize;
    let mut got_done = false;

    while let Some(event) = stream.next().await {
        match event.expect("event ok") {
            StreamEvent::ContentDelta(t) => {
                delta_count += 1;
                text.push_str(&t);
            }
            StreamEvent::ToolUse { .. } => panic!("no tools in this prompt"),
            StreamEvent::Done => got_done = true,
            _ => {}
        }
    }

    eprintln!("[smoke openai-compat stream | model={model}] deltas={delta_count} text={text:?}");

    assert!(got_done, "should have received Done terminal");
    assert!(delta_count >= 1, "should have received at least one delta");
    assert!(
        text.to_uppercase().contains("PONG"),
        "expected PONG in assembled text, got: {text:?}"
    );
}

/// Streams through OpenAI `/responses` and requires positive reasoning-summary events.
///
/// This smoke is opt-in via `OPENAI_RESPONSES_API_KEY` so a generic
/// OpenRouter `OPENAI_API_KEY` does not accidentally make `cargo test -- --ignored`
/// fail against a Chat Completions-only endpoint.
#[tokio::test]
#[ignore]
async fn smoke_openai_responses_thinking_stream() {
    load_env();
    let api_key = match std::env::var("OPENAI_RESPONSES_API_KEY") {
        Ok(k) if !k.is_empty() && !k.starts_with("sk-...") => k,
        _ => {
            eprintln!(
                "skipping smoke_openai_responses_thinking_stream: \
                 OPENAI_RESPONSES_API_KEY missing, empty, or still the placeholder"
            );
            return;
        }
    };

    let base_url = std::env::var("OPENAI_RESPONSES_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let model = std::env::var("OPENAI_RESPONSES_MODEL")
        .unwrap_or_else(|_| tkach::model::gpt::FIVE.to_string());
    let effort =
        std::env::var("OPENAI_RESPONSES_REASONING_EFFORT").unwrap_or_else(|_| "medium".into());
    let summary =
        std::env::var("OPENAI_RESPONSES_REASONING_SUMMARY").unwrap_or_else(|_| "detailed".into());

    let provider = OpenAIResponses::new(api_key)
        .with_base_url(base_url)
        .with_reasoning(effort, summary);

    let request = Request {
        model: model.clone(),
        system: Some(vec![SystemBlock::text(
            "Answer in one sentence. Do not include reasoning in the final answer.",
        )]),
        messages: vec![Message::user_text(
            "Solve carefully: a box has 3 red balls and 2 blue balls. \
             Without replacement, what is P(two red draws)?",
        )],
        tools: vec![],
        max_tokens: 1024,
        temperature: None,
        thinking: None,
    };

    let mut stream = provider
        .stream(request)
        .await
        .expect("open responses stream");
    let mut text = String::new();
    let mut thinking_delta_chars = 0usize;
    let mut thinking_blocks = 0usize;
    let mut got_done = false;

    while let Some(event) = stream.next().await {
        match event.expect("event ok") {
            StreamEvent::ContentDelta(t) => text.push_str(&t),
            StreamEvent::ThinkingDelta { text } => thinking_delta_chars += text.chars().count(),
            StreamEvent::ThinkingBlock { provider, .. } => {
                assert_eq!(provider, ThinkingProvider::OpenAIResponses);
                thinking_blocks += 1;
            }
            StreamEvent::Done => got_done = true,
            StreamEvent::ToolUse { .. } => panic!("no tools in this prompt"),
            _ => {}
        }
    }

    eprintln!(
        "[smoke openai responses thinking | model={model}] \
         thinking={thinking_delta_chars} chars / {thinking_blocks} blocks text={text:?}"
    );

    assert!(got_done, "should have received Done terminal");
    assert!(thinking_blocks > 0, "should emit finalized thinking blocks");
    assert!(!text.trim().is_empty(), "should emit final visible text");
}

// ---------------------------------------------------------------------------
// Agent::stream end-to-end smoke (real Anthropic, full loop)
// ---------------------------------------------------------------------------

/// Drives a complete agent run through `Agent::stream` against the
/// Anthropic API. Validates: live `ContentDelta` events, atomic
/// `ToolUse` events, the agent loop assembling deltas into a single
/// `Content::Text` for history, and `into_result` returning a normal
/// `AgentResult` with non-zero usage.
#[tokio::test]
#[ignore]
async fn smoke_agent_stream_end_to_end() {
    load_env();
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        panic!("ANTHROPIC_API_KEY required");
    }

    let dir = temp_dir("smoke_agent_stream");
    std::fs::write(dir.join("note.txt"), "The codeword is BANANA.").unwrap();

    let agent = haiku_agent(&dir);

    let mut stream = agent.stream(
        prompt("Read the file note.txt and tell me the codeword. Be brief."),
        CancellationToken::new(),
    );

    let mut delta_count = 0usize;
    let mut tool_uses = Vec::new();
    while let Some(ev) = stream.next().await {
        match ev.expect("stream event") {
            StreamEvent::ContentDelta(_) => delta_count += 1,
            StreamEvent::ToolUse { name, .. } => tool_uses.push(name),
            _ => {}
        }
    }
    let result = stream.into_result().await.expect("agent stream result");

    eprintln!(
        "[smoke agent stream] deltas={delta_count} tools={tool_uses:?} \
         in={} out={} text={:?}",
        result.usage.input_tokens, result.usage.output_tokens, result.text
    );

    assert!(delta_count >= 1, "should have streamed at least one delta");
    assert!(
        tool_uses.iter().any(|n| n == "read"),
        "agent should have called the read tool"
    );
    assert!(
        result.text.contains("BANANA"),
        "final text should contain the codeword: {:?}",
        result.text
    );
    assert!(
        result.usage.input_tokens > 0 && result.usage.output_tokens > 0,
        "usage should have non-zero counts"
    );
}
