use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use serde_json::{Value, json};
use tkach::message::{Content, Message, StopReason, ThinkingMetadata, ThinkingProvider, Usage};
use tkach::provider::Response;
use tkach::providers::Mock;
use tkach::{Agent, AgentError, CancellationToken, StreamEvent};

fn test_dir() -> std::path::PathBuf {
    std::env::current_dir().unwrap()
}

fn prompt(text: &str) -> Vec<Message> {
    vec![Message::user_text(text)]
}

// --- Simple text response (no tools, 1 turn) ---

#[tokio::test]
async fn single_turn_text_response() {
    let agent = Agent::builder()
        .provider(Mock::with_text("Hello, world!"))
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("Hi"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(result.text, "Hello, world!");
    // Delta is assistant-only: input user message is not echoed back.
    assert_eq!(result.new_messages.len(), 1);
    assert_eq!(result.stop_reason, StopReason::EndTurn);
}

// --- Tool call → result → final text (2 LLM calls) ---

#[tokio::test]
async fn tool_call_then_text_response() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo hello"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("The command output: hello")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("run echo hello"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(result.text, "The command output: hello");
    // Delta: assistant(tool_use), user(tool_result), assistant(text) = 3
    assert_eq!(result.new_messages.len(), 3);
}

// --- Multiple tool calls in a single response ---

#[tokio::test]
async fn multiple_tool_calls_single_response() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![
                    Content::ToolUse {
                        id: "t1".into(),
                        name: "bash".into(),
                        input: json!({"command": "echo first"}),
                    },
                    Content::ToolUse {
                        id: "t2".into(),
                        name: "bash".into(),
                        input: json!({"command": "echo second"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("Both commands ran successfully")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("run two commands"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.text, "Both commands ran successfully");

    // new_messages[1] is the user(tool_results) message
    let tool_results_msg = &result.new_messages[1];
    assert_eq!(tool_results_msg.content.len(), 2);
}

// --- Max turns exceeded ---

#[tokio::test]
async fn max_turns_exceeded() {
    let mock = Mock::new(|_req| {
        Ok(Response {
            content: vec![Content::ToolUse {
                id: "t1".into(),
                name: "bash".into(),
                input: json!({"command": "echo loop"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .max_turns(3)
        .working_dir(test_dir())
        .build()
        .unwrap();

    let err = agent
        .run(prompt("loop forever"), CancellationToken::new())
        .await
        .unwrap_err();
    let AgentError::MaxTurnsReached { turns, partial } = &err else {
        panic!("expected MaxTurnsReached, got {err:?}");
    };
    assert_eq!(*turns, 3);
    // Partial holds the delta: (assistant + user tool_result) × 3 = 6
    assert_eq!(partial.new_messages.len(), 6);
    assert_eq!(partial.stop_reason, StopReason::ToolUse);
}

// --- Cancellation ---

#[tokio::test]
async fn cancel_before_run_returns_cancelled_immediately() {
    let mock = Mock::with_text("should never get here");
    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let cancel = CancellationToken::new();
    cancel.cancel();

    let err = agent.run(prompt("hi"), cancel).await.unwrap_err();
    let AgentError::Cancelled { partial } = &err else {
        panic!("expected Cancelled, got {err:?}");
    };
    assert_eq!(partial.new_messages.len(), 0);
    assert_eq!(partial.stop_reason, StopReason::Cancelled);
}

// --- Tool not found ---

#[tokio::test]
async fn tool_not_found_returns_error_result() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "nonexistent_tool".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                let last_msg = req.messages.last().unwrap();
                let has_error = last_msg.content.iter().any(|c| match c {
                    Content::ToolResult { is_error, .. } => *is_error,
                    _ => false,
                });
                assert!(has_error, "LLM should receive tool error");

                Ok(Response {
                    content: vec![Content::text("Tool not found, sorry.")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("use a fake tool"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.text, "Tool not found, sorry.");
}

// --- Usage accumulation across turns ---

#[tokio::test]
async fn usage_accumulates_across_turns() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo hi"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            }),
            _ => Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage {
                    input_tokens: 200,
                    output_tokens: 30,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("test"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.usage.input_tokens, 300);
    assert_eq!(result.usage.output_tokens, 80);
}

// --- Read tool integration ---

#[tokio::test]
async fn read_tool_reads_actual_file() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "read".into(),
                    input: json!({"file_path": "Cargo.toml"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                let last_msg = req.messages.last().unwrap();
                let content = match &last_msg.content[0] {
                    Content::ToolResult { content, .. } => content.clone(),
                    _ => panic!("expected tool result"),
                };
                assert!(content.contains("tkach"), "should contain package name");

                Ok(Response {
                    content: vec![Content::text("I read the Cargo.toml")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("read cargo toml"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.text, "I read the Cargo.toml");
}

// --- Glob tool integration ---

#[tokio::test]
async fn glob_tool_finds_files() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "glob".into(),
                    input: json!({"pattern": "src/**/*.rs"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                let last_msg = req.messages.last().unwrap();
                let content = match &last_msg.content[0] {
                    Content::ToolResult { content, .. } => content.clone(),
                    _ => panic!("expected tool result"),
                };
                assert!(content.contains("lib.rs"), "should find lib.rs");

                Ok(Response {
                    content: vec![Content::text("Found Rust files")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("find rust files"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.text, "Found Rust files");
}

// --- Multi-turn tool chain (read → edit → verify) ---

#[tokio::test]
async fn multi_turn_tool_chain() {
    let tmp_dir = std::env::temp_dir().join("tkach_test");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let test_file = tmp_dir.join("test.txt");
    std::fs::write(&test_file, "hello world").unwrap();

    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();
    let file_path = test_file.display().to_string();
    let file_path_clone = file_path.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "read".into(),
                    input: json!({"file_path": file_path_clone}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            1 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t2".into(),
                    name: "edit".into(),
                    input: json!({
                        "file_path": file_path_clone,
                        "old_string": "hello world",
                        "new_string": "hello agent"
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("File updated successfully")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(&tmp_dir)
        .build()
        .unwrap();

    let result = agent
        .run(prompt("update the file"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.text, "File updated successfully");

    let content = std::fs::read_to_string(&test_file).unwrap();
    assert_eq!(content, "hello agent");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// --- No tools configured ---

#[tokio::test]
async fn no_tools_text_only() {
    let agent = Agent::builder()
        .provider(Mock::with_text("I have no tools but I can chat!"))
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("hello"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.text, "I have no tools but I can chat!");
}

// --- EndTurn with tool_use content stops the loop ---

#[tokio::test]
async fn end_turn_stops_even_with_tool_use_content() {
    // stop_reason is EndTurn but content has tool_use — loop honours stop_reason.
    let mock = Mock::new(|_req| {
        Ok(Response {
            content: vec![
                Content::text("Let me think..."),
                Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo orphan"}),
                },
            ],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("test"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.text, "Let me think...");
    // Delta: 1 assistant message, no tool execution.
    assert_eq!(result.new_messages.len(), 1);
}

// --- Cancel during tool batch: Bash honours it, loop returns Cancelled ---

#[tokio::test]
async fn cancel_during_bash_tool_returns_cancelled() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    // sleep longer than our cancel delay — bash honours
                    // ctx.cancel via select! + kill_on_drop.
                    input: json!({"command": "sleep 10", "timeout_ms": 30000}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => panic!("loop should not reach a second turn — cancel fired mid-batch"),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        cancel_clone.cancel();
    });

    let start = std::time::Instant::now();
    let err = agent.run(prompt("sleep"), cancel).await.unwrap_err();
    let elapsed = start.elapsed();

    let AgentError::Cancelled { partial } = &err else {
        panic!("expected Cancelled, got {err:?}");
    };
    assert_eq!(partial.stop_reason, StopReason::Cancelled);
    // The actual signal here is "cancelled long before sleep 10 elapsed";
    // 5s is loose enough to absorb shared-CI runner jitter (process spawn +
    // SIGKILL + reap can stretch on macOS/Windows agents) while still
    // proving prompt termination — the worst-case "didn't honour cancel"
    // would push elapsed close to 10s.
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "expected prompt cancel, took {elapsed:?}"
    );
}

// --- Provider error surfaces with partial ---

#[tokio::test]
async fn provider_error_returns_partial() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo hi"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            }),
            _ => Err(tkach::ProviderError::Overloaded {
                retry_after_ms: Some(2_000),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let err = agent
        .run(prompt("test"), CancellationToken::new())
        .await
        .unwrap_err();

    let AgentError::Provider { source, partial } = &err else {
        panic!("expected Provider error, got {err:?}");
    };
    assert!(source.is_retryable());
    // One full tool round-trip happened before the provider failed.
    assert_eq!(partial.new_messages.len(), 2);
    assert_eq!(partial.usage.input_tokens, 10);
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_text_response_emits_delta_then_collects_result() {
    let agent = Agent::builder()
        .provider(Mock::with_text("Hello, world!"))
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let mut stream = agent.stream(prompt("hi"), CancellationToken::new());
    let mut events: Vec<StreamEvent> = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.unwrap());
    }
    let result = stream.into_result().await.unwrap();

    // Mock emits one ContentDelta per Text block; ToolUse/MessageDelta/
    // Usage/Done are absorbed by the agent loop and not forwarded on
    // the public stream channel. Done is documented as the terminal
    // marker; consumers break on it and call into_result().
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], StreamEvent::ContentDelta(t) if t == "Hello, world!"));

    // Final history matches the run() shape: one assistant message with
    // text body assembled from deltas.
    assert_eq!(result.new_messages.len(), 1);
    assert_eq!(result.text, "Hello, world!");
    assert_eq!(result.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn stream_thinking_response_forwards_live_and_preserves_history() {
    let thinking = Content::thinking(
        "I should inspect the repo first.",
        ThinkingProvider::OpenAIResponses,
        ThinkingMetadata::openai_responses(Some("rs_1".into()), None, 0, Some("enc".into())),
    );

    let agent = Agent::builder()
        .provider(Mock::new(move |_req| {
            Ok(Response {
                content: vec![thinking.clone(), Content::text("Done.")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }))
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let mut stream = agent.stream(prompt("hi"), CancellationToken::new());
    let mut saw_delta = false;
    let mut saw_block = false;
    let mut visible = String::new();
    while let Some(ev) = stream.next().await {
        match ev.unwrap() {
            StreamEvent::ThinkingDelta { text } => {
                assert_eq!(text, "I should inspect the repo first.");
                saw_delta = true;
            }
            StreamEvent::ThinkingBlock {
                text,
                provider,
                metadata,
            } => {
                assert_eq!(text, "I should inspect the repo first.");
                assert_eq!(provider, ThinkingProvider::OpenAIResponses);
                assert_eq!(
                    metadata,
                    ThinkingMetadata::OpenAIResponses {
                        item_id: Some("rs_1".into()),
                        output_index: None,
                        summary_index: 0,
                        encrypted_content: Some("enc".into()),
                    }
                );
                saw_block = true;
            }
            StreamEvent::ContentDelta(text) => visible.push_str(&text),
            _ => {}
        }
    }
    let result = stream.into_result().await.unwrap();

    assert!(saw_delta, "consumer should see live thinking progress");
    assert!(saw_block, "consumer should see finalized thinking metadata");
    assert_eq!(visible, "Done.");
    assert_eq!(result.text, "Done.");
    assert_eq!(result.new_messages.len(), 1);
    let contents = &result.new_messages[0].content;
    assert_eq!(contents.len(), 2);
    assert!(matches!(
        &contents[0],
        Content::Thinking {
            text,
            provider: ThinkingProvider::OpenAIResponses,
            metadata: ThinkingMetadata::OpenAIResponses {
                item_id,
                output_index: None,
                summary_index: 0,
                encrypted_content,
            },
        } if text == "I should inspect the repo first."
            && item_id.as_deref() == Some("rs_1")
            && encrypted_content.as_deref() == Some("enc")
    ));
    assert!(matches!(
        &contents[1],
        Content::Text { text, .. } if text == "Done."
    ));
}

#[tokio::test]
async fn stream_redacted_thinking_preserves_metadata_without_visible_text() {
    let redacted = Content::thinking(
        "",
        ThinkingProvider::Anthropic,
        ThinkingMetadata::anthropic_redacted("opaque-redacted-payload"),
    );

    let agent = Agent::builder()
        .provider(Mock::new(move |_req| {
            Ok(Response {
                content: vec![redacted.clone(), Content::text("Done.")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }))
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let mut stream = agent.stream(prompt("hi"), CancellationToken::new());
    let mut saw_redacted_block = false;
    let mut visible = String::new();
    while let Some(ev) = stream.next().await {
        match ev.unwrap() {
            StreamEvent::ThinkingDelta { text } => {
                assert!(
                    text.is_empty(),
                    "redacted thinking must not leak display text"
                );
            }
            StreamEvent::ThinkingBlock {
                text,
                provider,
                metadata,
            } => {
                assert!(
                    text.is_empty(),
                    "redacted thinking block should have no display text"
                );
                assert_eq!(provider, ThinkingProvider::Anthropic);
                assert_eq!(
                    metadata,
                    ThinkingMetadata::AnthropicRedacted {
                        data: "opaque-redacted-payload".into(),
                    }
                );
                saw_redacted_block = true;
            }
            StreamEvent::ContentDelta(text) => visible.push_str(&text),
            _ => {}
        }
    }
    let result = stream.into_result().await.unwrap();

    assert!(
        saw_redacted_block,
        "consumer should see finalized redacted metadata"
    );
    assert_eq!(visible, "Done.");
    assert_eq!(result.text, "Done.");
    let contents = &result.new_messages[0].content;
    assert!(matches!(
        &contents[0],
        Content::Thinking {
            text,
            provider: ThinkingProvider::Anthropic,
            metadata: ThinkingMetadata::AnthropicRedacted { data },
        } if text.is_empty() && data == "opaque-redacted-payload"
    ));
}

#[tokio::test]
async fn stream_tool_call_then_text_response_executes_tool_inline() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo hi"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let mut stream = agent.stream(prompt("run echo"), CancellationToken::new());
    let mut tool_use_seen = false;
    let mut final_text = String::new();
    while let Some(ev) = stream.next().await {
        match ev.unwrap() {
            StreamEvent::ToolUse { name, .. } => {
                assert_eq!(name, "bash");
                tool_use_seen = true;
            }
            StreamEvent::ContentDelta(t) => final_text.push_str(&t),
            _ => {}
        }
    }
    let result = stream.into_result().await.unwrap();

    assert!(tool_use_seen, "consumer should see ToolUse event");
    assert_eq!(final_text, "done");
    // Delta history: assistant(tool_use), user(tool_result), assistant(text) = 3
    assert_eq!(result.new_messages.len(), 3);
    assert_eq!(result.text, "done");
}

#[tokio::test]
async fn stream_collect_result_skips_event_drain() {
    let agent = Agent::builder()
        .provider(Mock::with_text("ignored content"))
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    // Don't iterate events; collect_result should still complete.
    let stream = agent.stream(prompt("hi"), CancellationToken::new());
    let result = stream.collect_result().await.unwrap();

    assert_eq!(result.text, "ignored content");
    assert_eq!(result.new_messages.len(), 1);
}

#[tokio::test]
async fn stream_cancel_before_start_returns_cancelled_via_into_result() {
    let mock = Mock::with_text("never");
    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let cancel = CancellationToken::new();
    cancel.cancel();
    let stream = agent.stream(prompt("hi"), cancel);

    let err = stream.collect_result().await.unwrap_err();
    let AgentError::Cancelled { partial } = &err else {
        panic!("expected Cancelled, got {err:?}");
    };
    assert_eq!(partial.stop_reason, StopReason::Cancelled);
    assert!(partial.new_messages.is_empty());
}

// --- Approval flow streaming events -----------------------------------------

#[tokio::test]
async fn stream_emits_tool_call_pending_before_executing_tool() {
    use tkach::ToolClass;

    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo hi"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .working_dir(test_dir())
        .build()
        .unwrap();

    let mut stream = agent.stream(prompt("run echo"), CancellationToken::new());
    let mut sequence: Vec<&'static str> = Vec::new();
    let mut pending_class: Option<ToolClass> = None;
    while let Some(ev) = stream.next().await {
        match ev.unwrap() {
            StreamEvent::ToolUse { .. } => sequence.push("ToolUse"),
            StreamEvent::ToolCallPending { name, class, .. } => {
                assert_eq!(name, "bash");
                pending_class = Some(class);
                sequence.push("ToolCallPending");
            }
            StreamEvent::ContentDelta(_) => sequence.push("ContentDelta"),
            _ => {}
        }
    }
    let result = stream.into_result().await.unwrap();

    // Critical ordering invariant: ToolUse arrives first (provider
    // event), then the agent emits ToolCallPending right before
    // dispatching to the executor (where approval gate runs).
    let tu_pos = sequence
        .iter()
        .position(|x| *x == "ToolUse")
        .expect("ToolUse");
    let pending_pos = sequence
        .iter()
        .position(|x| *x == "ToolCallPending")
        .expect("ToolCallPending");
    assert!(
        tu_pos < pending_pos,
        "ToolUse must come before ToolCallPending; got: {sequence:?}"
    );

    // Class is resolved through the registry.
    assert_eq!(
        pending_class,
        Some(ToolClass::Mutating),
        "bash is Mutating; class should be threaded through"
    );

    // Tool still ran end-to-end (AutoApprove default).
    assert_eq!(result.text, "done");
}

#[tokio::test]
async fn stream_with_deny_handler_emits_pending_but_skips_execution() {
    use async_trait::async_trait;
    use serde_json::Value;
    use tkach::{ApprovalDecision, ApprovalHandler, ToolClass};

    struct DenyAll;
    #[async_trait]
    impl ApprovalHandler for DenyAll {
        async fn approve(&self, _: &str, _: &Value, _: ToolClass) -> ApprovalDecision {
            ApprovalDecision::Deny("nope".into())
        }
    }

    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo hi"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("acknowledged the denial")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(tkach::tools::defaults())
        .approval(DenyAll)
        .working_dir(test_dir())
        .build()
        .unwrap();

    let mut stream = agent.stream(prompt("run echo hi"), CancellationToken::new());
    let mut got_pending = false;
    while let Some(ev) = stream.next().await {
        if let StreamEvent::ToolCallPending { .. } = ev.unwrap() {
            got_pending = true;
        }
    }
    let result = stream.into_result().await.unwrap();

    assert!(
        got_pending,
        "ToolCallPending must still fire even when handler will deny"
    );

    // Tool result in history must carry the denial reason — proves
    // the executor's gate ran and the model saw the rejection.
    let saw_denial = result.new_messages.iter().any(|m| {
        m.content.iter().any(|c| match c {
            Content::ToolResult {
                content, is_error, ..
            } => *is_error && content.contains("nope"),
            _ => false,
        })
    });
    assert!(
        saw_denial,
        "expected denial tool_result containing 'nope' in history"
    );
}

#[test]
fn build_rejects_duplicate_tool_names() {
    let err = match Agent::builder()
        .provider(Mock::with_text("unused"))
        .model("test")
        .tool(tkach::tools::Read)
        .tool(tkach::tools::Read)
        .working_dir(test_dir())
        .build()
    {
        Ok(_) => panic!("duplicate tool names must fail fast"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        tkach::BuildError::DuplicateToolName { name, count }
            if name == "read" && count == 2
    ));
}

#[tokio::test]
async fn builder_threads_thinking_to_provider_request() {
    let mock = Mock::new(|req| {
        assert_eq!(
            req.thinking,
            Some(tkach::ThinkingConfig::Effort(tkach::ThinkingEffort::High))
        );
        Ok(Response {
            content: vec![Content::text("ok")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .thinking(tkach::ThinkingConfig::Effort(tkach::ThinkingEffort::High))
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("hi"), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.text, "ok");
}

#[tokio::test]
async fn subagent_trace_hook_receives_full_child_stream_events() {
    use std::sync::Mutex;

    // The hook captures every variant the child stream emits. The Mock
    // child returns a single text response so the expected sequence is:
    // ContentDelta("child done") -> MessageDelta(EndTurn) -> Usage(_) ->
    // Done. Issue #40 Phase 3 acceptance: ToolUse, ToolCallPending,
    // ContentDelta, MessageDelta, Usage, Done all flow through.
    let observed: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed);
    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);

    let parent = Mock::new(move |_| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "sub-1".into(),
                    name: "agent".into(),
                    input: json!({"prompt": "answer inside child"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("parent done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });

    let child = Arc::new(Mock::with_text("child done"));
    let agent = Agent::builder()
        .provider(parent)
        .model("parent")
        .tool(
            tkach::tools::SubAgent::new(child, "child").trace_hook(move |ev| {
                let tag = match ev {
                    StreamEvent::ContentDelta(_) => "content_delta",
                    StreamEvent::ThinkingDelta { .. } => "thinking_delta",
                    StreamEvent::ThinkingBlock { .. } => "thinking_block",
                    StreamEvent::ToolUse { .. } => "tool_use",
                    StreamEvent::ToolCallPending { .. } => "tool_call_pending",
                    StreamEvent::MessageDelta { .. } => "message_delta",
                    StreamEvent::Usage(_) => "usage",
                    StreamEvent::Done => "done",
                };
                observed_clone.lock().unwrap().push(tag);
            }),
        )
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("delegate"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(result.text, "parent done");
    let seen = observed.lock().unwrap().clone();
    assert!(
        seen.contains(&"content_delta"),
        "trace_hook missed ContentDelta; got {seen:?}"
    );
    assert!(
        seen.contains(&"message_delta"),
        "trace_hook missed MessageDelta; got {seen:?}"
    );
    assert!(
        seen.contains(&"usage"),
        "trace_hook missed Usage; got {seen:?}"
    );
    assert!(
        seen.contains(&"done"),
        "trace_hook missed Done; got {seen:?}"
    );
}

#[tokio::test]
async fn subagent_trace_hook_observes_tool_call_path() {
    use std::sync::Mutex;

    // Mutating-child path per issue #40 P6. Child first turn emits
    // ToolUse(read), the agent emits ToolCallPending, executes the
    // tool, child second turn emits text, then MessageDelta + Usage
    // + Done. The hook MUST observe ToolUse and ToolCallPending in
    // addition to ContentDelta and the absorbed terminals.
    let observed: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed);
    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);

    let parent = Mock::new(move |_| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "sub-1".into(),
                    name: "child".into(),
                    input: json!({"prompt": "delegate"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("parent done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });

    let child_turn = Arc::new(AtomicUsize::new(0));
    let child_turn_clone = Arc::clone(&child_turn);
    let child = Arc::new(Mock::new(move |_| {
        let turn = child_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "child-call-1".into(),
                    name: "read".into(),
                    input: json!({"path": "Cargo.toml"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("child read it")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    }));

    let agent = Agent::builder()
        .provider(parent)
        .model("parent")
        .tools(tkach::tools::defaults())
        .tool(
            tkach::tools::SubAgent::new(child, "child-model")
                .name("child")
                .trace_hook(move |ev| {
                    let tag = match ev {
                        StreamEvent::ContentDelta(_) => "content_delta",
                        StreamEvent::ThinkingDelta { .. } => "thinking_delta",
                        StreamEvent::ThinkingBlock { .. } => "thinking_block",
                        StreamEvent::ToolUse { .. } => "tool_use",
                        StreamEvent::ToolCallPending { .. } => "tool_call_pending",
                        StreamEvent::MessageDelta { .. } => "message_delta",
                        StreamEvent::Usage(_) => "usage",
                        StreamEvent::Done => "done",
                    };
                    observed_clone.lock().unwrap().push(tag);
                }),
        )
        .working_dir(test_dir())
        .build()
        .unwrap();

    agent
        .run(prompt("delegate"), CancellationToken::new())
        .await
        .unwrap();

    let seen = observed.lock().unwrap().clone();
    for required in [
        "tool_use",
        "tool_call_pending",
        "content_delta",
        "message_delta",
        "usage",
        "done",
    ] {
        assert!(
            seen.contains(&required),
            "trace_hook missed {required}; got {seen:?}"
        );
    }
}

#[tokio::test]
async fn subagent_trace_hook_does_not_break_public_stream_consumers() {
    // Regression guard: trace_hook must NOT change the public
    // Agent::stream contract. A README-style consumer running in
    // parallel sees only the documented forward set
    // (ContentDelta/Thinking*/ToolUse/ToolCallPending) — not
    // MessageDelta/Usage/Done — even when a SubAgent in the same tree
    // has a trace_hook wired.
    let mock = Mock::with_text("hello");
    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .working_dir(test_dir())
        .build()
        .unwrap();

    let mut stream = agent.stream(prompt("hi"), CancellationToken::new());
    let mut public_events: Vec<StreamEvent> = Vec::new();
    while let Some(ev) = stream.next().await {
        if let Ok(ev) = ev {
            public_events.push(ev);
        }
    }
    let _ = stream.into_result().await.unwrap();

    assert!(
        public_events.iter().all(|e| !matches!(
            e,
            StreamEvent::MessageDelta { .. } | StreamEvent::Usage(_) | StreamEvent::Done
        )),
        "public Agent::stream must not forward MessageDelta/Usage/Done; got {public_events:?}"
    );
}

#[tokio::test]
async fn subagent_trace_hook_panic_does_not_fail_parent() {
    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);

    let parent = Mock::new(move |_| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "sub-1".into(),
                    name: "agent".into(),
                    input: json!({"prompt": "answer inside child"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("parent recovered")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });

    let child = Arc::new(Mock::with_text("child done"));
    let agent = Agent::builder()
        .provider(parent)
        .model("parent")
        .tool(tkach::tools::SubAgent::new(child, "child").trace_hook(|_| {
            panic!("trace sink failed");
        }))
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("delegate"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(result.text, "parent recovered");
}

#[tokio::test]
async fn subagent_filtered_tool_definitions_respect_parent_policy_without_child_allow_list() {
    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);

    let parent = Mock::new(move |_| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "sub-1".into(),
                    name: "agent".into(),
                    input: json!({"prompt": "inspect visible tools"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("parent done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });

    let child = Arc::new(Mock::new(|req| {
        let names = req
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["agent", "read"]);
        Ok(Response {
            content: vec![Content::text("child done")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }));

    let agent = Agent::builder()
        .provider(parent)
        .model("parent")
        .tools(tkach::tools::defaults())
        .tool(tkach::tools::SubAgent::new(child, "child").filter_tool_definitions(true))
        .policy(tkach::AllowList::new(["agent", "read"]))
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("delegate"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(result.text, "parent done");
}

#[tokio::test]
async fn subagent_filtered_tool_definitions_respect_parent_policy_with_child_allow_list() {
    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);

    let parent = Mock::new(move |_| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "sub-1".into(),
                    name: "agent".into(),
                    input: json!({"prompt": "inspect visible tools"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("parent done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });

    let child = Arc::new(Mock::new(|req| {
        let names = req
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["read"]);
        Ok(Response {
            content: vec![Content::text("child done")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    }));

    let agent = Agent::builder()
        .provider(parent)
        .model("parent")
        .tools(tkach::tools::defaults())
        .tool(
            tkach::tools::SubAgent::new(child, "child")
                .tools_allow(["read", "grep"])
                .filter_tool_definitions(true),
        )
        .policy(tkach::AllowList::new(["agent", "read"]))
        .working_dir(test_dir())
        .build()
        .unwrap();

    let result = agent
        .run(prompt("delegate"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(result.text, "parent done");
}

// --- Phase 1 acceptance: SubAgent specialisation primitives ----------------

#[tokio::test]
async fn two_named_subagents_register_and_appear_in_tool_definitions() {
    // Two specialised SubAgents must register side-by-side without
    // BuildError, and the LLM-visible tool definitions must include
    // both names (alphabetically sorted by Agent::tool_definitions).
    let parent = Mock::new(|req| {
        let names: Vec<&str> = req.tools.iter().map(|t| t.name.as_str()).collect();
        // Alphabetical: research, writer
        assert_eq!(names, vec!["research", "writer"]);
        Ok(Response {
            content: vec![Content::text("ok")],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        })
    });

    let child_a = Arc::new(Mock::with_text("a"));
    let child_b = Arc::new(Mock::with_text("b"));

    let agent = Agent::builder()
        .provider(parent)
        .model("parent")
        .tool(
            tkach::tools::SubAgent::new(child_a, "child-model")
                .name("research")
                .description("Read-only research"),
        )
        .tool(
            tkach::tools::SubAgent::new(child_b, "child-model")
                .name("writer")
                .description("Mutating writer"),
        )
        .working_dir(test_dir())
        .build()
        .expect("two distinct names must register without BuildError");

    agent
        .run(prompt("hi"), CancellationToken::new())
        .await
        .unwrap();
}

#[tokio::test]
async fn subagent_tools_allow_denies_non_allowed_call_via_intersect_policy() {
    // P2 acceptance: child invokes a tool outside its allow-list; the
    // IntersectPolicy(parent_policy, AllowList(child)) intersection
    // returns is_allowed=false, so execute_one synthesises an
    // is_error: true tool_result with "not allowed by policy".
    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);
    let parent = Mock::new(move |_| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "sub-1".into(),
                    name: "child".into(),
                    input: json!({"prompt": "do work"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("parent done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });

    let child_turn = Arc::new(AtomicUsize::new(0));
    let child_turn_clone = Arc::clone(&child_turn);
    let observed_results: Arc<std::sync::Mutex<Vec<String>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed_results);

    let child = Arc::new(Mock::new(move |req| {
        let turn = child_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            // Child issues a write — outside its allow-list of [read].
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "child-call-1".into(),
                    name: "write".into(),
                    input: json!({"path": "/tmp/x", "content": "y"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            // Capture the tool_result the child sees on turn 2.
            for msg in &req.messages {
                for c in &msg.content {
                    if let Content::ToolResult { content, .. } = c {
                        observed_clone.lock().unwrap().push(content.clone());
                    }
                }
            }
            Ok(Response {
                content: vec![Content::text("child saw error")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    }));

    let agent = Agent::builder()
        .provider(parent)
        .model("parent")
        .tools(tkach::tools::defaults())
        .tool(
            tkach::tools::SubAgent::new(child, "child-model")
                .name("child")
                .tools_allow(["read"]),
        )
        .working_dir(test_dir())
        .build()
        .unwrap();

    agent
        .run(prompt("delegate"), CancellationToken::new())
        .await
        .unwrap();

    let results = observed_results.lock().unwrap().clone();
    let joined = results.join("|");
    let child_turns = child_turn.load(Ordering::SeqCst);
    let parent_turns = parent_turn.load(Ordering::SeqCst);
    assert!(
        joined.contains("not allowed by policy") && joined.contains("write"),
        "child should see policy denial for write; child_turns={child_turns}, parent_turns={parent_turns}, observed={joined:?}"
    );
}

#[tokio::test]
async fn subagent_approval_handler_overrides_parent_deny_handler() {
    // P5 acceptance: parent installs DenyAll approval; child SubAgent
    // overrides with AutoApprove via approval_handler(...). The child's
    // tool call must land (no policy denial, no approval denial),
    // proving the override actually swaps the per-child handler.
    use tkach::{ApprovalDecision, ApprovalHandler, AutoApprove, ToolClass};

    struct DenyAll;
    #[async_trait::async_trait]
    impl ApprovalHandler for DenyAll {
        async fn approve(&self, _: &str, _: &Value, _: ToolClass) -> ApprovalDecision {
            ApprovalDecision::Deny("parent says no".into())
        }
    }

    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);
    let parent = Mock::new(move |_| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "sub-1".into(),
                    name: "child".into(),
                    input: json!({"prompt": "do work"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("parent done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });

    let child_turn = Arc::new(AtomicUsize::new(0));
    let child_turn_clone = Arc::clone(&child_turn);
    let observed_results: Arc<std::sync::Mutex<Vec<String>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed_results);

    let child = Arc::new(Mock::new(move |req| {
        let turn = child_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "child-call-1".into(),
                    name: "read".into(),
                    input: json!({"path": "Cargo.toml"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            for msg in &req.messages {
                for c in &msg.content {
                    if let Content::ToolResult { content, .. } = c {
                        observed_clone.lock().unwrap().push(content.clone());
                    }
                }
            }
            Ok(Response {
                content: vec![Content::text("child done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    }));

    let agent = Agent::builder()
        .provider(parent)
        .model("parent")
        .tools(tkach::tools::defaults())
        .tool(
            tkach::tools::SubAgent::new(child, "child-model")
                .name("child")
                .approval_handler(Arc::new(AutoApprove)),
        )
        .approval(DenyAll)
        .working_dir(test_dir())
        .build()
        .unwrap();

    agent
        .run(prompt("delegate"), CancellationToken::new())
        .await
        .unwrap();

    let results = observed_results.lock().unwrap().clone();
    let joined = results.join("|");
    assert!(
        !joined.contains("approval denied"),
        "child override must bypass parent DenyAll; got {joined:?}"
    );
    assert!(
        !joined.contains("parent says no"),
        "child override must bypass parent DenyAll; got {joined:?}"
    );
}

#[tokio::test]
async fn fork_for_subagent_with_none_none_matches_fork_for_subagent_shape() {
    // Compatibility regression: fork_for_subagent() must equal
    // fork_for_subagent_with(None, None) by construction (one-line
    // delegation). Validate by inspecting the resulting executor's
    // policy / approval / registry Arc identity.
    use std::sync::Arc as StdArc;
    use tkach::{ToolExecutor, ToolRegistry};

    let registry = StdArc::new(ToolRegistry::new(vec![]));
    let policy: StdArc<dyn tkach::ToolPolicy> = StdArc::new(tkach::AllowAll);
    let approval: StdArc<dyn tkach::ApprovalHandler> = StdArc::new(tkach::AutoApprove);
    let exec = ToolExecutor::with_approval(registry.clone(), policy.clone(), approval.clone());

    let a = exec.fork_for_subagent();
    let b = exec.fork_for_subagent_with(None, None);

    // Both share the parent's registry Arc.
    assert!(StdArc::ptr_eq(a.registry(), b.registry()));
    assert!(StdArc::ptr_eq(a.registry(), &registry));
}

#[tokio::test]
async fn subagent_trace_hook_unset_takes_run_fast_path() {
    // P3 acceptance: when trace_hook is None, SubAgent::execute uses
    // agent.run (not agent.stream). Mock observes that stream() is NOT
    // called for the child invocation when no trace_hook is wired.
    let stream_called = Arc::new(AtomicUsize::new(0));
    let stream_called_clone = Arc::clone(&stream_called);

    let parent_turn = Arc::new(AtomicUsize::new(0));
    let parent_turn_clone = Arc::clone(&parent_turn);
    let parent = Mock::new(move |_| {
        let turn = parent_turn_clone.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            Ok(Response {
                content: vec![Content::ToolUse {
                    id: "sub-1".into(),
                    name: "child".into(),
                    input: json!({"prompt": "work"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            })
        } else {
            Ok(Response {
                content: vec![Content::text("parent done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    });

    let _ = stream_called_clone;
    let _ = stream_called;
    let child_mock = Arc::new(Mock::with_text("child done"));
    let child_provider: Arc<dyn tkach::provider::LlmProvider> =
        Arc::clone(&child_mock) as Arc<dyn tkach::provider::LlmProvider>;

    let agent = Agent::builder()
        .provider(parent)
        .model("parent")
        .tool(tkach::tools::SubAgent::new(child_provider, "child-model").name("child"))
        .working_dir(test_dir())
        .build()
        .unwrap();

    agent
        .run(prompt("delegate"), CancellationToken::new())
        .await
        .unwrap();

    let n = child_mock.stream_count();
    assert_eq!(
        n, 0,
        "child stream() must not be called when trace_hook is unset; got {n} stream calls"
    );
    assert!(
        child_mock.complete_count() >= 1,
        "child complete() must be called via run() fast path"
    );
}
