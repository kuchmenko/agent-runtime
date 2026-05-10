//! Canonical model name constants.
//!
//! These constants are plain `&'static str` values, so they work with existing
//! APIs that accept `impl Into<String>` while giving callers autocomplete and a
//! single source of truth for common model identifiers.

/// Anthropic Claude model names.
pub mod claude {
    /// Latest Opus tier alias.
    pub const OPUS: &str = "claude-opus-4-7";
    /// Latest Sonnet tier alias.
    pub const SONNET: &str = "claude-sonnet-4-6";
    /// Latest Haiku tier alias.
    pub const HAIKU: &str = "claude-haiku-4-5";

    /// Claude Opus 4.6 generation alias.
    pub const OPUS_4_6: &str = "claude-opus-4-6";
    /// Claude Haiku 4.5 pinned snapshot.
    pub const HAIKU_20251001: &str = "claude-haiku-4-5-20251001";
}

/// OpenAI model names.
pub mod gpt {
    /// GPT-5.
    pub const FIVE: &str = "gpt-5";
    /// GPT-5 Codex.
    pub const FIVE_CODEX: &str = "gpt-5-codex";
    /// GPT-5.5.
    pub const FIVE_FIVE: &str = "gpt-5.5";
}

/// OpenRouter model names.
pub mod openrouter {
    /// Anthropic Claude Haiku through OpenRouter.
    pub const ANTHROPIC_HAIKU: &str = "anthropic/claude-haiku-4-5";
    /// OpenAI GPT-5.5 through OpenRouter.
    pub const OPENAI_GPT_5_5: &str = "openai/gpt-5.5";
}

#[cfg(test)]
mod tests {
    use super::{claude, gpt, openrouter};

    #[test]
    fn claude_constants_match_wire_names() {
        assert_eq!(claude::OPUS, "claude-opus-4-7");
        assert_eq!(claude::SONNET, "claude-sonnet-4-6");
        assert_eq!(claude::HAIKU, "claude-haiku-4-5");
        assert_eq!(claude::OPUS_4_6, "claude-opus-4-6");
        assert_eq!(claude::HAIKU_20251001, "claude-haiku-4-5-20251001");
    }

    #[test]
    fn gpt_constants_match_wire_names() {
        assert_eq!(gpt::FIVE, "gpt-5");
        assert_eq!(gpt::FIVE_CODEX, "gpt-5-codex");
        assert_eq!(gpt::FIVE_FIVE, "gpt-5.5");
    }

    #[test]
    fn openrouter_constants_match_wire_names() {
        assert_eq!(openrouter::ANTHROPIC_HAIKU, "anthropic/claude-haiku-4-5");
        assert_eq!(openrouter::OPENAI_GPT_5_5, "openai/gpt-5.5");
    }
}
