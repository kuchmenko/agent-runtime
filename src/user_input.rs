//! Operator input bridge for root-agent elicitation.

use std::time::Duration;

use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionSet {
    pub questions: Vec<Question>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    pub header: String,
    pub question: String,
    pub options: Vec<QuestionOption>,
    pub multi_select: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionOption {
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub question_index: usize,
    pub selected_options: Vec<usize>,
    pub text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserInputResponse {
    Answered(Vec<Answer>),
    Cancelled,
    TimedOut,
}

#[derive(Debug, thiserror::Error)]
pub enum AskUserError {
    #[error("ask_user is only available from the root agent thread")]
    NotRootThread,
    #[error("no user input bridge is registered")]
    NoUserInputBridge,
    #[error(transparent)]
    BridgeError(#[from] BridgeError),
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BridgeError {
    #[error("bridge transport error: {0}")]
    Transport(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("bridge protocol error: {0}")]
    Protocol(String),
    #[error("bridge error: {0}")]
    Custom(#[source] Box<dyn std::error::Error + Send + Sync>),
}

#[async_trait]
pub trait UserInputBridge: Send + Sync {
    async fn collect(&self, questions: &QuestionSet) -> Result<UserInputResponse, BridgeError>;
}

pub(crate) async fn collect_with_timeout(
    bridge: &dyn UserInputBridge,
    questions: &QuestionSet,
    timeout: Duration,
) -> Result<UserInputResponse, BridgeError> {
    match tokio::time::timeout(timeout, bridge.collect(questions)).await {
        Ok(result) => result,
        Err(_) => Ok(UserInputResponse::TimedOut),
    }
}
