//! Mid-task steering primitives.
//!
//! The public handle API is append/interrupt oriented. Callers send small
//! control messages into a running agent; the loop applies them only at safe
//! boundaries so provider streams and tool executions are not mutated in place.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::message::Content;

static TURN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Opaque identifier for one agent turn.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnId(String);

impl TurnId {
    #[must_use]
    pub fn new() -> Self {
        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default();
        let suffix = TURN_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(format!("turn_{unix_ms}_{suffix}"))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TurnId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for TurnId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<TurnId> for String {
    fn from(value: TurnId) -> Self {
        value.0
    }
}

/// Converts ergonomic queue inputs into message content blocks.
pub trait IntoQueueContent {
    fn into_queue_content(self) -> Vec<Content>;
}

impl IntoQueueContent for &str {
    fn into_queue_content(self) -> Vec<Content> {
        vec![Content::text(self)]
    }
}

impl IntoQueueContent for String {
    fn into_queue_content(self) -> Vec<Content> {
        vec![Content::text(self)]
    }
}

impl IntoQueueContent for Vec<Content> {
    fn into_queue_content(self) -> Vec<Content> {
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SteerError {
    #[error("queued user message content is empty")]
    EmptyContent,
    #[error("no active turn")]
    NoActiveTurn,
    #[error("expected active turn {expected}, got {actual}")]
    ExpectedTurnMismatch { expected: TurnId, actual: TurnId },
    #[error("agent steering channel is closed")]
    ChannelClosed,
}

#[derive(Debug, Clone)]
pub enum SteerCommand {
    Append { content: Vec<Content> },
}

#[derive(Debug, Clone)]
pub enum InterruptTarget {
    Tool {
        tool_call_id: String,
    },
    Turn {
        turn_id: TurnId,
        reason: InterruptReason,
    },
    Session,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptOutcome {
    Cancelled,
    NotInFlight,
    AlreadyDone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptReason {
    UserAbort,
    Replaced,
    PolicyViolation,
}

#[derive(Debug, thiserror::Error)]
pub enum InterruptError {
    #[error("unknown turn {0}")]
    UnknownTurn(TurnId),
    #[error("agent steering channel is closed")]
    ChannelClosed,
}

#[derive(Debug, Clone)]
pub(crate) struct ToolRunTracker {
    inner: Arc<Mutex<ToolRunState>>,
}

#[derive(Debug, Default)]
struct ToolRunState {
    in_flight: HashMap<String, CancellationToken>,
    completed: HashSet<String>,
}

impl ToolRunTracker {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ToolRunState::default())),
        }
    }

    pub(crate) fn register(&self, tool_call_id: &str, cancel: CancellationToken) {
        let mut state = self.inner.lock().expect("tool tracker lock poisoned");
        state.completed.remove(tool_call_id);
        state.in_flight.insert(tool_call_id.to_string(), cancel);
    }

    pub(crate) fn mark_done(&self, tool_call_id: &str) {
        let mut state = self.inner.lock().expect("tool tracker lock poisoned");
        state.in_flight.remove(tool_call_id);
        state.completed.insert(tool_call_id.to_string());
    }

    pub(crate) fn interrupt(&self, tool_call_id: &str) -> InterruptOutcome {
        let state = self.inner.lock().expect("tool tracker lock poisoned");
        if let Some(cancel) = state.in_flight.get(tool_call_id) {
            cancel.cancel();
            InterruptOutcome::Cancelled
        } else if state.completed.contains(tool_call_id) {
            InterruptOutcome::AlreadyDone
        } else {
            InterruptOutcome::NotInFlight
        }
    }
}

#[derive(Debug)]
pub(crate) struct AgentHandleInner {
    pub(crate) active_turn: RwLock<Option<TurnId>>,
    pub(crate) steer_tx: mpsc::UnboundedSender<SteerCommand>,
    pub(crate) cancel: CancellationToken,
    pub(crate) tool_runs: ToolRunTracker,
}

pub(crate) struct AgentControl {
    pub(crate) handle_inner: Arc<AgentHandleInner>,
    pub(crate) steer_rx: mpsc::UnboundedReceiver<SteerCommand>,
}

pub(crate) fn control_pair(
    cancel: CancellationToken,
) -> (AgentControl, crate::handle::AgentHandle) {
    let (steer_tx, steer_rx) = mpsc::unbounded_channel();
    let inner = Arc::new(AgentHandleInner {
        active_turn: RwLock::new(None),
        steer_tx,
        cancel,
        tool_runs: ToolRunTracker::new(),
    });
    let control = AgentControl {
        handle_inner: Arc::clone(&inner),
        steer_rx,
    };
    let handle = crate::handle::AgentHandle { inner };
    (control, handle)
}
