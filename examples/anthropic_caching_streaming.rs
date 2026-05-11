//! Anthropic prompt caching via `stream()`.
//!
//! Regression test for the streaming Usage merge: the API sends
//! `cache_creation_input_tokens` / `cache_read_input_tokens` on the
//! `message_start` event only, NOT on `message_delta`. The provider
//! re-stamps the running cache fields onto every emitted Usage event
//! so a consumer that takes "the latest Usage" never observes the
//! cache fields collapse to 0 mid-stream.
//!
//! Two streamed calls back-to-back; the second one's final Usage
//! event must carry `cache_read_input_tokens > 0`.
//!
//! Model & size: same constraints as `anthropic_caching.rs` — the
//! cached prefix must clear ~2048 tokens at the breakpoint or
//! Anthropic silently skips caching. The example pads the system
//! block to ~3000 tokens.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_caching_streaming

use futures::StreamExt;
use tkach::providers::Anthropic;
use tkach::{LlmProvider, Message, Request, StreamEvent, SystemBlock, ToolDefinition, Usage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let provider = Anthropic::from_env();

    let stable_system = build_stable_system();

    let tools = vec![ToolDefinition {
        name: "noop".into(),
        description: "A no-op tool, present so the toolset has shape.".into(),
        input_schema: serde_json::json!({"type": "object"}),
        cache_control: None,
    }];

    let make_request = |user: &str| Request {
        model: tkach::model::claude::SONNET.into(),
        system: Some(vec![SystemBlock::cached(&stable_system)]),
        messages: vec![Message::user_text(user)],
        tools: tools.clone(),
        max_tokens: 64,
        temperature: Some(0.0),
        thinking: None,
    };

    println!("=== first call (streamed) ===");
    let u1 = drain(
        provider
            .stream(make_request("Say hello in one short sentence."))
            .await?,
    )
    .await;
    print_usage(&u1);

    println!("\n=== second call (streamed) ===");
    let u2 = drain(
        provider
            .stream(make_request("Say goodbye in one short sentence."))
            .await?,
    )
    .await;
    print_usage(&u2);

    assert!(
        u2.cache_read_input_tokens > 0,
        "streamed second call should have hit the cache; \
         the merge fix is what makes this observable. \
         cache_read_input_tokens={} (would have been 0 without merging)",
        u2.cache_read_input_tokens
    );

    Ok(())
}

/// Drain a provider stream and return the **latest** Usage event.
/// This is the natural consumer pattern — we want the merge in the
/// provider to give us the full picture in this single observation.
async fn drain(mut s: tkach::ProviderEventStream) -> Usage {
    let mut latest = Usage::default();
    let mut text = String::new();
    while let Some(ev) = s.next().await {
        match ev.expect("stream item") {
            StreamEvent::ContentDelta(t) => text.push_str(&t),
            StreamEvent::Usage(u) => latest = u,
            StreamEvent::Done => {}
            _ => {}
        }
    }
    println!("text: {text}");
    latest
}

fn print_usage(u: &Usage) {
    println!(
        "  input={} output={} cache_write={} cache_read={}",
        u.input_tokens, u.output_tokens, u.cache_creation_input_tokens, u.cache_read_input_tokens
    );
}

/// Build a ~3000-token stable system prompt. Mirrors
/// `examples/anthropic_caching.rs::build_stable_system` but trimmed —
/// the streaming example only needs to clear the cache minimum, not
/// showcase a full domain-knowledge prefix.
fn build_stable_system() -> String {
    let header = "You are a precise, terse assistant integrated into a developer tool. \
         Your audience is a working software engineer who values brevity, \
         direct answers, and concrete examples over hedging language. You read \
         code fluently in Rust, Go, Python, TypeScript, and SQL. You assume \
         the user has shipped production software and does not need handholding \
         through basics.";

    let constraints = vec![
        "Never use lists or markdown formatting unless the user explicitly asks for them.",
        "Never restate the question before answering — start the answer in the first sentence.",
        "When a fact is uncertain, say so once, briefly. Do not pad with multiple disclaimers.",
        "Prefer concrete examples over abstract description: show, don't summarize.",
        "When asked for code, return only the minimal change required. No surrounding rewrite.",
        "If the user asks a question with multiple plausible interpretations, pick the most likely one and answer it; offer the alternative in one short follow-up sentence.",
        "Never apologize for the previous answer; correct it directly and move on.",
        "Match the user's vocabulary: if they say 'function', don't switch to 'method'.",
        "Avoid filler phrases: 'great question', 'as you noted', 'it's worth mentioning'.",
        "Numbers and code identifiers stay in their original form; do not reformat them.",
        "When the user shares an error, identify the root cause first, then the fix. Skip restating what they already know.",
        "If the user is debugging, ask for the smallest reproducer only when the question genuinely cannot be answered without it.",
        "Treat performance numbers as load-bearing: name the Big-O, the cache layer, or the syscall rather than waving at 'efficiency'.",
        "When proposing a refactor, separate the structural change from the behavioural one — they review differently.",
    ];

    let examples = vec![
        (
            "Q: What's the difference between a String and &str in Rust?",
            "A: String owns a heap-allocated UTF-8 buffer; &str is a borrowed view into one. You convert with .as_str() or &s[..] to borrow, .to_string() or .into() to own. Use String for fields you mutate or move; use &str for function parameters that just read.",
        ),
        (
            "Q: How do I handle a nullable column from sqlx?",
            "A: Use Option<T> in your row struct. sqlx maps SQL NULL to None and a present value to Some(_). For required-but-occasionally-null columns, prefer .unwrap_or(default) at read time over schema-level NOT NULL only if you genuinely accept the default.",
        ),
        (
            "Q: Why does my async function need to be Send?",
            "A: Because the runtime moves the future across threads at .await points. If the future holds a non-Send value across an .await (like Rc, RefCell, or a !Send guard), it can't be Send. Drop the offending value before the await or switch to a Send equivalent (Arc, Mutex, parking_lot).",
        ),
        (
            "Q: When should I use Cow<str>?",
            "A: When you have a function that sometimes returns a borrowed slice and sometimes returns a freshly built String. It's the typed way to say 'I'll allocate only if I have to'. Common in parsers, config readers, and case-folding helpers.",
        ),
        (
            "Q: What's the right way to share state across Tokio tasks?",
            "A: Arc<Mutex<T>> for short critical sections, Arc<RwLock<T>> when reads dominate, mpsc::channel for actor-style ownership transfer, watch::channel for broadcast-of-latest. Avoid std::sync::Mutex inside async — it blocks the runtime thread; use tokio::sync::Mutex instead, or hold the guard only across non-await code.",
        ),
        (
            "Q: Why is my SELECT slow even with an index?",
            "A: Three usual causes: the planner is not using the index (check EXPLAIN ANALYZE for Seq Scan vs Index Scan), the index is unusable for this predicate (e.g. function call or LIKE without prefix), or the table is bloated and the index lookup still hits dead tuples. Run EXPLAIN (ANALYZE, BUFFERS) and read the actual node order, not the cost estimate.",
        ),
        (
            "Q: What's the difference between Result and Either?",
            "A: Result is opinionated — Err means 'something went wrong'. Either is symmetric — Left and Right have no built-in 'failure' meaning. Use Result for fallible operations; use Either when both branches are equally legitimate.",
        ),
        (
            "Q: How do I model a state machine with enums?",
            "A: One enum per state, with transitions as methods that consume self and return the next-state enum. The compiler enforces 'you cannot send the wrong message' because the wrong-state methods literally don't exist. Pair this with phantom-typed builders if the same struct shape needs different state.",
        ),
    ];

    let footer = "Style notes: Code blocks are fine when showing actual code. \
         Inline backticks for identifiers. Avoid 'note that', 'keep in mind', \
         'as a side note' — they signal padding. If a sentence can be cut \
         without losing meaning, cut it. The user reads quickly. When you \
         genuinely don't know an answer, say 'I don't know' in three words and \
         offer the next concrete step.";

    // Filler "domain notes" so the cumulative prefix at the
    // breakpoint clears Anthropic's caching minimum (~2400 tokens
    // empirical for Sonnet/Haiku). In a real workload this would be
    // tool docs, retrieved RAG context, persona files, etc.
    let domain_notes = vec![
        "On error handling: panic for invariants the program cannot violate without being broken; Result for conditions the caller might reasonably encounter (missing file, network timeout, parse error). thiserror is the conventional library for defining error types in libraries; anyhow is the conventional library for binary-application top-level error capture. Mixing the two is fine: anyhow at the binary boundary, thiserror through library code so callers can match on specific error variants without losing information.",
        "On testing: integration tests live in tests/ and exercise the public API as a black box; unit tests live next to the code they test in a #[cfg(test)] mod tests block. Use proptest for property-based testing when you have algebraic properties to assert (associativity, idempotence, round-trip serialization). Snapshot testing via insta is the right tool for output that should be stable but is too large to hand-write — review snapshots like code; never accept blindly.",
        "On observability: structured logs (tracing) > printf-style (log) for any system above one process. The cost of structured logs is one extra macro per call; the payoff is grep-able machine-readable output that Loki/Honeycomb/Datadog can parse without regex. Always include request IDs in spans so logs from one user action across services can be reconnected. tracing-opentelemetry bridges spans into a real distributed-tracing backend without changing application code.",
        "On performance: measure before optimizing — and measure the actual production workload, not a synthetic benchmark. Tools: cargo flamegraph for CPU profiles, dhat for heap, tokio-console for runtime task health, perf for system-wide profiling. If the bottleneck is allocation, the fix is usually to reuse a buffer rather than to optimize the allocator. If the bottleneck is async waiting, parallelize via join! or stream::buffer_unordered, not micro-optimize a single future.",
        "On migrations: every schema change ships as a forward-only migration with a tested rollback strategy. Never amend a merged migration — write a new one. For large tables, use online schema change tools (gh-ost, pt-osc) or sqlx's CONCURRENTLY index creation. The migration that adds a NOT NULL column to a 50M-row table is always either backfill in batches first or add NULLABLE then tighten — not a single ALTER. The reverse migration is your safety net; if you cannot articulate one, you don't fully understand the change.",
        "On dependency choice: prefer crates with active maintenance over the most popular crate. Check 'Last published' on crates.io and the issue tracker velocity. A 5-year-old crate with 100k downloads/week is fine if it does what it says; a 6-month-old crate with 1M downloads/week may be in a state of churn. Read the changelog before bumping major versions; do not let your dependency tree drift via `cargo update` without scanning the diff. cargo-deny is the right tool for enforcing license and supply-chain policy in CI.",
        "On feature flags: use them for *deployment* control (gradual rollout, kill switch), not for *implementation branching* (multiple parallel versions). Two implementations gated by a flag is a refactor halfway through merging; commit to one. Prefer launchdarkly/openfeature over hand-rolled config for any flag that needs runtime change. Always plan the flag's removal as part of the rollout — flags that 'temporarily' linger past the rollout become permanent technical debt that fragments testing surface.",
    ];

    let mut s = String::new();
    s.push_str(header);
    s.push_str("\n\nConstraints:\n");
    for c in &constraints {
        s.push_str("- ");
        s.push_str(c);
        s.push('\n');
    }
    s.push_str("\nExamples of the expected register:\n\n");
    for (q, a) in &examples {
        s.push_str(q);
        s.push('\n');
        s.push_str(a);
        s.push_str("\n\n");
    }
    s.push_str("Domain notes:\n\n");
    for note in &domain_notes {
        s.push_str("- ");
        s.push_str(note);
        s.push_str("\n\n");
    }
    s.push_str(footer);
    s
}
