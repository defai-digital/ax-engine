//! Grammar-constrained generation.
//!
//! Restricts token sampling to outputs that conform to a specified grammar.
//! Constraints are applied by masking disallowed token logits to `-inf`
//! before softmax, so masked tokens receive zero probability.
//!
//! Supported constraint types:
//! - [`ChoiceConstraint`]: output must be one of a fixed set of strings
//! - [`PatternConstraint`]: output must match a positional character-class
//!   pattern (digits, letters, literals, etc.)
//!
//! # Pipeline integration
//!
//! Grammar masking should be applied after repetition penalty and temperature
//! scaling, but before top-k / top-p filtering:
//!
//! ```text
//! 1. Repetition penalty
//! 2. Temperature scaling
//! 3. >>> Grammar masking <<<   (apply_grammar_mask)
//! 4. Top-k filtering
//! 5. Top-p filtering
//! 6. Softmax
//! 7. Categorical sampling
//! ```

/// Trait for stateful grammar constraints on token generation.
///
/// Implementations track what text has been generated so far and
/// determine which tokens are allowed next.
pub trait GrammarConstraint {
    /// Write `true` into `mask[i]` for each token `i` that is allowed next.
    ///
    /// `mask` is pre-filled with `false`. `vocab_tokens[i]` is the text
    /// representation of token ID `i`.
    fn compute_mask(&self, mask: &mut [bool], vocab_tokens: &[String]);

    /// Advance the constraint state after accepting a token.
    fn advance(&mut self, token_text: &str);

    /// Whether the constraint is in an accepting state (generation can stop).
    fn is_complete(&self) -> bool;

    /// Reset to the initial state for a new generation.
    fn reset(&mut self);
}

/// Apply a grammar constraint mask to logits.
///
/// Sets logits of disallowed tokens to [`f32::NEG_INFINITY`] so that softmax
/// assigns them zero probability. If the constraint would disallow *all*
/// tokens (unsatisfiable state), no masking is applied as a safety fallback.
pub fn apply_grammar_mask(
    logits: &mut [f32],
    constraint: &dyn GrammarConstraint,
    vocab_tokens: &[String],
) {
    assert_eq!(
        logits.len(),
        vocab_tokens.len(),
        "logits length ({}) must match vocab size ({})",
        logits.len(),
        vocab_tokens.len()
    );

    let mut mask = vec![false; logits.len()];
    constraint.compute_mask(&mut mask, vocab_tokens);

    // Safety: if no tokens are allowed, skip masking entirely to avoid
    // producing an all-NEG_INFINITY distribution that would NaN after softmax.
    if !mask.iter().any(|&m| m) {
        return;
    }

    for (logit, &allowed) in logits.iter_mut().zip(mask.iter()) {
        if !allowed {
            *logit = f32::NEG_INFINITY;
        }
    }
}

// ── ChoiceConstraint ────────────────────────────────────────────────────────

/// Constrain generation to exactly one of a fixed set of strings.
///
/// At each step, only tokens whose text keeps the generated output
/// on track to match one of the choices are allowed.
///
/// # Example
///
/// ```ignore
/// let mut c = ChoiceConstraint::new(&["yes", "no"]);
/// // Initially, tokens "y", "ye", "yes", "n", "no" are allowed
/// c.advance("y");
/// // Now only tokens "e", "es" are allowed (and "yes" if it's a single token)
/// c.advance("es");
/// assert!(c.is_complete()); // "yes" matches
/// ```
pub struct ChoiceConstraint {
    /// Original choices (preserved for reset).
    all_choices: Vec<String>,
    /// Choices that still match the generated prefix.
    remaining: Vec<String>,
    /// Text generated so far.
    generated: String,
}

impl ChoiceConstraint {
    /// Create a constraint that allows exactly one of `choices`.
    ///
    /// Panics if `choices` is empty.
    pub fn new(choices: &[&str]) -> Self {
        assert!(!choices.is_empty(), "must provide at least one choice");
        let all: Vec<String> = choices.iter().map(|s| s.to_string()).collect();
        Self {
            remaining: all.clone(),
            all_choices: all,
            generated: String::new(),
        }
    }

    /// The text generated so far.
    pub fn generated(&self) -> &str {
        &self.generated
    }

    /// Choices that still match the generated prefix.
    pub fn remaining_choices(&self) -> &[String] {
        &self.remaining
    }
}

impl GrammarConstraint for ChoiceConstraint {
    fn compute_mask(&self, mask: &mut [bool], vocab_tokens: &[String]) {
        for (i, token_text) in vocab_tokens.iter().enumerate() {
            if token_text.is_empty() {
                continue;
            }

            let candidate = format!("{}{}", self.generated, token_text);

            // Token is allowed if the candidate is a prefix of any remaining
            // choice, OR if any remaining choice is a prefix of the candidate
            // (the candidate completes or extends through the choice).
            // However, we don't allow overshooting: candidate must not be
            // longer than the choice unless it exactly matches.
            mask[i] = self.remaining.iter().any(|choice| {
                // candidate is still building toward this choice
                choice.starts_with(&candidate)
            });
        }
    }

    fn advance(&mut self, token_text: &str) {
        self.generated.push_str(token_text);
        self.remaining.retain(|c| c.starts_with(&self.generated));
    }

    fn is_complete(&self) -> bool {
        self.remaining.contains(&self.generated)
    }

    fn reset(&mut self) {
        self.generated.clear();
        self.remaining = self.all_choices.clone();
    }
}

// ── PatternConstraint ───────────────────────────────────────────────────────

/// Character class for pattern matching.
#[derive(Debug, Clone)]
pub enum CharClass {
    /// Any digit `[0-9]`.
    Digit,
    /// Any ASCII letter `[a-zA-Z]`.
    Alpha,
    /// Any alphanumeric `[a-zA-Z0-9]`.
    AlphaNum,
    /// A specific literal character.
    Literal(char),
    /// Any of the given characters.
    OneOf(Vec<char>),
    /// Any character (wildcard).
    Any,
}

impl CharClass {
    /// Check if a character matches this class.
    pub fn matches(&self, c: char) -> bool {
        match self {
            CharClass::Digit => c.is_ascii_digit(),
            CharClass::Alpha => c.is_ascii_alphabetic(),
            CharClass::AlphaNum => c.is_ascii_alphanumeric(),
            CharClass::Literal(expected) => c == *expected,
            CharClass::OneOf(chars) => chars.contains(&c),
            CharClass::Any => true,
        }
    }
}

/// Constrain generation to match a positional character-class pattern.
///
/// Each element in the pattern specifies which characters are allowed at
/// that position. The output must have exactly `pattern.len()` characters,
/// each matching its corresponding class.
///
/// # Example
///
/// ```ignore
/// use ax_core::sampling::grammar::{PatternConstraint, CharClass};
///
/// // Date pattern: YYYY-MM-DD
/// let pattern = PatternConstraint::new(vec![
///     CharClass::Digit, CharClass::Digit, CharClass::Digit, CharClass::Digit,
///     CharClass::Literal('-'),
///     CharClass::Digit, CharClass::Digit,
///     CharClass::Literal('-'),
///     CharClass::Digit, CharClass::Digit,
/// ]);
/// ```
pub struct PatternConstraint {
    pattern: Vec<CharClass>,
    /// Current position in the pattern (number of characters generated).
    position: usize,
}

impl PatternConstraint {
    /// Create a pattern constraint from a sequence of character classes.
    pub fn new(pattern: Vec<CharClass>) -> Self {
        Self {
            pattern,
            position: 0,
        }
    }

    /// Build a pattern from a simple format string.
    ///
    /// Format characters:
    /// - `N` → digit
    /// - `A` → letter
    /// - `W` → alphanumeric
    /// - `?` → any character
    /// - anything else → literal
    ///
    /// Example: `"NNNN-NN-NN"` → date pattern (4 digits, dash, 2 digits, dash, 2 digits)
    pub fn from_format(fmt: &str) -> Self {
        let pattern = fmt
            .chars()
            .map(|c| match c {
                'N' => CharClass::Digit,
                'A' => CharClass::Alpha,
                'W' => CharClass::AlphaNum,
                '?' => CharClass::Any,
                other => CharClass::Literal(other),
            })
            .collect();
        Self::new(pattern)
    }

    /// Current position in the pattern.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Total length of the pattern.
    pub fn pattern_len(&self) -> usize {
        self.pattern.len()
    }

    /// Check if all characters in `text` match the pattern starting at `pos`.
    /// Returns the number of matched characters (0 if any fails).
    fn matches_from(&self, text: &str, pos: usize) -> Option<usize> {
        let mut offset = 0;
        for c in text.chars() {
            let idx = pos + offset;
            if idx >= self.pattern.len() {
                // Token extends beyond pattern — not allowed
                return None;
            }
            if !self.pattern[idx].matches(c) {
                return None;
            }
            offset += 1;
        }
        Some(offset)
    }
}

impl GrammarConstraint for PatternConstraint {
    fn compute_mask(&self, mask: &mut [bool], vocab_tokens: &[String]) {
        if self.position >= self.pattern.len() {
            // Pattern fully matched — no more tokens allowed
            return;
        }

        for (i, token_text) in vocab_tokens.iter().enumerate() {
            if token_text.is_empty() {
                continue;
            }
            // Check if every character in the token matches the pattern
            // at the current position onward
            if self.matches_from(token_text, self.position).is_some() {
                mask[i] = true;
            }
        }
    }

    fn advance(&mut self, token_text: &str) {
        // Count characters (not bytes) to advance position
        self.position += token_text.chars().count();
    }

    fn is_complete(&self) -> bool {
        self.position >= self.pattern.len()
    }

    fn reset(&mut self) {
        self.position = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test vocabulary ─────────────────────────────────────────────

    fn test_vocab() -> Vec<String> {
        vec![
            "<unk>".into(), // 0
            "<s>".into(),   // 1
            "</s>".into(),  // 2
            "y".into(),     // 3
            "n".into(),     // 4
            "e".into(),     // 5
            "s".into(),     // 6
            "o".into(),     // 7
            "ye".into(),    // 8
            "yes".into(),   // 9
            "no".into(),    // 10
            "maybe".into(), // 11
            "1".into(),     // 12
            "2".into(),     // 13
            "3".into(),     // 14
            "0".into(),     // 15
            "-".into(),     // 16
            "12".into(),    // 17
            "99".into(),    // 18
            "abc".into(),   // 19
            "a".into(),     // 20
            " ".into(),     // 21
            "".into(),      // 22
        ]
    }

    // ── apply_grammar_mask ──────────────────────────────────────────

    #[test]
    fn test_apply_grammar_mask_basic() {
        let vocab = test_vocab();
        let c = ChoiceConstraint::new(&["yes", "no"]);
        let mut logits = vec![1.0f32; vocab.len()];

        apply_grammar_mask(&mut logits, &c, &vocab);

        // Allowed: "y" (3), "n" (4), "ye" (8), "yes" (9), "no" (10)
        assert!(logits[3].is_finite(), "y should be allowed");
        assert!(logits[4].is_finite(), "n should be allowed");
        assert!(logits[8].is_finite(), "ye should be allowed");
        assert!(logits[9].is_finite(), "yes should be allowed");
        assert!(logits[10].is_finite(), "no should be allowed");

        // Disallowed
        assert_eq!(logits[5], f32::NEG_INFINITY, "e should be masked");
        assert_eq!(logits[6], f32::NEG_INFINITY, "s should be masked");
        assert_eq!(logits[7], f32::NEG_INFINITY, "o should be masked");
        assert_eq!(logits[11], f32::NEG_INFINITY, "maybe should be masked");
        assert_eq!(logits[12], f32::NEG_INFINITY, "1 should be masked");
    }

    #[test]
    fn test_apply_grammar_mask_unsatisfiable() {
        // If no tokens match, masking is skipped (safety)
        let vocab = test_vocab();
        let mut c = ChoiceConstraint::new(&["xyz"]);
        // No token starts with "x", so nothing will match
        // But ChoiceConstraint checks prefix match, and "xyz" starts with none of our tokens
        // Actually "xyz" — no token is a prefix of it... wait
        // Actually we check choice.starts_with(&candidate), so candidate = "" + token_text
        // "xyz".starts_with("y") = false, "xyz".starts_with("n") = false, etc.
        // Hmm, but does "xyz" start with any token? No token in our vocab starts "x".
        // So no token is allowed → safety fallback applies
        c.advance("impossible_prefix");
        let mut logits = vec![1.0f32; vocab.len()];
        apply_grammar_mask(&mut logits, &c, &vocab);

        // All logits should remain unchanged (fallback)
        assert!(logits.iter().all(|&v| v == 1.0));
    }

    // ── ChoiceConstraint ────────────────────────────────────────────

    #[test]
    fn test_choice_constraint_initial_mask() {
        let vocab = test_vocab();
        let c = ChoiceConstraint::new(&["yes", "no"]);
        let mut mask = vec![false; vocab.len()];
        c.compute_mask(&mut mask, &vocab);

        assert!(mask[3], "y is prefix of yes");
        assert!(mask[4], "n is prefix of no");
        assert!(mask[8], "ye is prefix of yes");
        assert!(mask[9], "yes matches exactly");
        assert!(mask[10], "no matches exactly");
        assert!(!mask[7], "o is not a prefix of yes or no");
        assert!(!mask[11], "maybe is not a prefix of yes or no");
    }

    #[test]
    fn test_choice_constraint_advance() {
        let vocab = test_vocab();
        let mut c = ChoiceConstraint::new(&["yes", "no"]);
        assert!(!c.is_complete());

        c.advance("y");
        assert!(!c.is_complete());
        assert_eq!(c.generated(), "y");
        assert_eq!(c.remaining_choices(), &["yes"]);

        let mut mask = vec![false; vocab.len()];
        c.compute_mask(&mut mask, &vocab);
        // After "y", only "e", "es" should match (building toward "yes")
        assert!(mask[5], "e extends y→ye (prefix of yes)");
        assert!(!mask[3], "y would make yy — not prefix of yes");
        assert!(!mask[4], "n would make yn — not prefix");
    }

    #[test]
    fn test_choice_constraint_complete() {
        let mut c = ChoiceConstraint::new(&["yes", "no"]);
        c.advance("ye");
        assert!(!c.is_complete());
        c.advance("s");
        assert!(c.is_complete());
        assert_eq!(c.generated(), "yes");
    }

    #[test]
    fn test_choice_constraint_no_overshoot() {
        let vocab = test_vocab();
        let c = ChoiceConstraint::new(&["no"]);
        let mut mask = vec![false; vocab.len()];
        c.compute_mask(&mut mask, &vocab);

        assert!(mask[4], "n is prefix of no");
        assert!(mask[10], "no matches exactly");
        // "no" is the choice but "maybe" doesn't start with "no" prefix
        assert!(!mask[11], "maybe should not match");
    }

    #[test]
    fn test_choice_constraint_reset() {
        let mut c = ChoiceConstraint::new(&["yes", "no"]);
        c.advance("ye");
        assert_eq!(c.remaining_choices().len(), 1);

        c.reset();
        assert_eq!(c.generated(), "");
        assert_eq!(c.remaining_choices().len(), 2);
        assert!(!c.is_complete());
    }

    #[test]
    fn test_choice_constraint_single_token_match() {
        let vocab = test_vocab();
        let c = ChoiceConstraint::new(&["no"]);
        let mut mask = vec![false; vocab.len()];
        c.compute_mask(&mut mask, &vocab);

        assert!(mask[10], "no should match as single token");
        assert!(mask[4], "n should match as prefix of no");
    }

    #[test]
    #[should_panic(expected = "must provide at least one choice")]
    fn test_choice_constraint_empty_panics() {
        let _c = ChoiceConstraint::new(&[]);
    }

    // ── CharClass ───────────────────────────────────────────────────

    #[test]
    fn test_char_class_digit() {
        let cc = CharClass::Digit;
        assert!(cc.matches('0'));
        assert!(cc.matches('9'));
        assert!(!cc.matches('a'));
        assert!(!cc.matches('-'));
    }

    #[test]
    fn test_char_class_alpha() {
        let cc = CharClass::Alpha;
        assert!(cc.matches('a'));
        assert!(cc.matches('Z'));
        assert!(!cc.matches('0'));
        assert!(!cc.matches(' '));
    }

    #[test]
    fn test_char_class_alphanumeric() {
        let cc = CharClass::AlphaNum;
        assert!(cc.matches('a'));
        assert!(cc.matches('Z'));
        assert!(cc.matches('5'));
        assert!(!cc.matches('-'));
    }

    #[test]
    fn test_char_class_literal() {
        let cc = CharClass::Literal('-');
        assert!(cc.matches('-'));
        assert!(!cc.matches('a'));
    }

    #[test]
    fn test_char_class_one_of() {
        let cc = CharClass::OneOf(vec!['x', 'y', 'z']);
        assert!(cc.matches('x'));
        assert!(cc.matches('z'));
        assert!(!cc.matches('a'));
    }

    #[test]
    fn test_char_class_any() {
        let cc = CharClass::Any;
        assert!(cc.matches('a'));
        assert!(cc.matches('0'));
        assert!(cc.matches(' '));
        assert!(cc.matches('!'));
    }

    // ── PatternConstraint ───────────────────────────────────────────

    #[test]
    fn test_pattern_from_format() {
        let p = PatternConstraint::from_format("NN-NN");
        assert_eq!(p.pattern_len(), 5);
        assert_eq!(p.position(), 0);
        assert!(!p.is_complete());
    }

    #[test]
    fn test_pattern_digit_mask() {
        let vocab = test_vocab();
        let p = PatternConstraint::from_format("NNN");
        let mut mask = vec![false; vocab.len()];
        p.compute_mask(&mut mask, &vocab);

        // Allowed: single digits "1"(12), "2"(13), "3"(14), "0"(15)
        assert!(mask[12], "1 should be allowed at digit position");
        assert!(mask[13], "2 should be allowed");
        assert!(mask[14], "3 should be allowed");
        assert!(mask[15], "0 should be allowed");
        // Multi-digit tokens: "12"(17), "99"(18) — both are 2 digits
        assert!(mask[17], "12 should be allowed (2 digits fit)");
        assert!(mask[18], "99 should be allowed");
        // Not allowed: letters, symbols
        assert!(!mask[3], "y is not a digit");
        assert!(!mask[16], "- is not a digit");
        assert!(!mask[19], "abc is not digits");
    }

    #[test]
    fn test_pattern_advance_and_complete() {
        let mut p = PatternConstraint::from_format("NN");
        assert!(!p.is_complete());

        p.advance("1");
        assert_eq!(p.position(), 1);
        assert!(!p.is_complete());

        p.advance("2");
        assert_eq!(p.position(), 2);
        assert!(p.is_complete());
    }

    #[test]
    fn test_pattern_advance_multi_char_token() {
        let mut p = PatternConstraint::from_format("NNNN");
        p.advance("12"); // 2 characters
        assert_eq!(p.position(), 2);
        p.advance("34"); // 2 more
        assert_eq!(p.position(), 4);
        assert!(p.is_complete());
    }

    #[test]
    fn test_pattern_mixed_classes() {
        let vocab = test_vocab();
        // Pattern: digit, dash, digit
        let p = PatternConstraint::from_format("N-N");
        let mut mask = vec![false; vocab.len()];
        p.compute_mask(&mut mask, &vocab);

        // At position 0, only digits are allowed
        assert!(mask[12], "1 matches digit at pos 0");
        assert!(!mask[16], "- does not match digit at pos 0");
    }

    #[test]
    fn test_pattern_after_advance() {
        let vocab = test_vocab();
        let mut p = PatternConstraint::from_format("N-N");

        // After generating "1", position is 1 → next must be "-"
        p.advance("1");
        let mut mask = vec![false; vocab.len()];
        p.compute_mask(&mut mask, &vocab);

        assert!(mask[16], "- should match literal at pos 1");
        assert!(!mask[12], "1 should not match - at pos 1");
        assert!(!mask[3], "y should not match - at pos 1");
    }

    #[test]
    fn test_pattern_no_overshoot() {
        let vocab = test_vocab();
        let mut p = PatternConstraint::from_format("NN");
        p.advance("1"); // position = 1

        let mut mask = vec![false; vocab.len()];
        p.compute_mask(&mut mask, &vocab);

        // "12" (token 17) would put us at position 3, past pattern length 2
        // matches_from("12", 1) → position 1+0=1 → '1' is digit ✓, 1+1=2 → '2' at position 2 ≥ pattern.len() → None
        assert!(!mask[17], "12 overshoots pattern of length 2 at pos 1");
        // Single digit is fine
        assert!(mask[12], "1 fits at pos 1");
    }

    #[test]
    fn test_pattern_complete_blocks_all() {
        let vocab = test_vocab();
        let mut p = PatternConstraint::from_format("N");
        p.advance("5");
        assert!(p.is_complete());

        let mut mask = vec![false; vocab.len()];
        p.compute_mask(&mut mask, &vocab);
        // All false — pattern is complete, no more tokens allowed
        assert!(mask.iter().all(|&m| !m));
    }

    #[test]
    fn test_pattern_reset() {
        let mut p = PatternConstraint::from_format("NNN");
        p.advance("12");
        assert_eq!(p.position(), 2);
        p.reset();
        assert_eq!(p.position(), 0);
        assert!(!p.is_complete());
    }

    #[test]
    fn test_pattern_date_format() {
        // YYYY-MM-DD
        let p = PatternConstraint::from_format("NNNN-NN-NN");
        assert_eq!(p.pattern_len(), 10);
        assert!(!p.is_complete());
    }

    #[test]
    fn test_pattern_alpha_mask() {
        let vocab = test_vocab();
        let p = PatternConstraint::from_format("AAA");
        let mut mask = vec![false; vocab.len()];
        p.compute_mask(&mut mask, &vocab);

        assert!(mask[19], "abc matches 3 letters");
        assert!(mask[3], "y matches letter at pos 0");
        assert!(mask[20], "a matches letter at pos 0");
        assert!(!mask[12], "1 does not match letter");
        assert!(!mask[16], "- does not match letter");
    }

    #[test]
    fn test_pattern_format_literal() {
        // "(NN)"
        let p = PatternConstraint::from_format("(NN)");
        assert_eq!(p.pattern_len(), 4);
        // First char must be literal '('
        // No token in our test vocab is '(' so nothing matches
        let vocab = test_vocab();
        let mut mask = vec![false; vocab.len()];
        p.compute_mask(&mut mask, &vocab);
        assert!(mask.iter().all(|&m| !m), "no token matches ( in test vocab");
    }

    // ── Integration ─────────────────────────────────────────────────

    #[test]
    fn test_choice_full_generation_flow() {
        let vocab = test_vocab();
        let mut c = ChoiceConstraint::new(&["yes", "no"]);

        // Step 1: mask logits
        let mut logits = vec![1.0f32; vocab.len()];
        apply_grammar_mask(&mut logits, &c, &vocab);

        // Simulate selecting "no" (token 10)
        assert!(logits[10].is_finite());
        c.advance("no");
        assert!(c.is_complete());
    }

    #[test]
    fn test_choice_multi_step_generation() {
        let vocab = test_vocab();
        let mut c = ChoiceConstraint::new(&["yes", "no"]);

        // Step 1: generate "y"
        let mut logits = vec![1.0f32; vocab.len()];
        apply_grammar_mask(&mut logits, &c, &vocab);
        assert!(logits[3].is_finite()); // "y"
        c.advance("y");
        assert!(!c.is_complete());

        // Step 2: generate "e"
        let mut logits = vec![1.0f32; vocab.len()];
        apply_grammar_mask(&mut logits, &c, &vocab);
        assert!(logits[5].is_finite()); // "e"
        assert!(!logits[4].is_finite()); // "n" should be masked now
        c.advance("e");
        assert!(!c.is_complete());

        // Step 3: generate "s"
        let mut logits = vec![1.0f32; vocab.len()];
        apply_grammar_mask(&mut logits, &c, &vocab);
        assert!(logits[6].is_finite()); // "s"
        c.advance("s");
        assert!(c.is_complete());
    }

    #[test]
    fn test_pattern_multi_step_generation() {
        let vocab = test_vocab();
        let mut p = PatternConstraint::from_format("NN-N");

        // Step 1: generate "1" at pos 0
        let mut logits = vec![1.0f32; vocab.len()];
        apply_grammar_mask(&mut logits, &p, &vocab);
        assert!(logits[12].is_finite()); // "1"
        assert!(!logits[16].is_finite()); // "-" not allowed at pos 0
        p.advance("1");

        // Step 2: generate "2" at pos 1
        let mut logits = vec![1.0f32; vocab.len()];
        apply_grammar_mask(&mut logits, &p, &vocab);
        assert!(logits[13].is_finite()); // "2"
        p.advance("2");

        // Step 3: generate "-" at pos 2
        let mut logits = vec![1.0f32; vocab.len()];
        apply_grammar_mask(&mut logits, &p, &vocab);
        assert!(logits[16].is_finite()); // "-"
        assert!(!logits[12].is_finite()); // "1" not allowed at literal pos
        p.advance("-");

        // Step 4: generate "3" at pos 3
        let mut logits = vec![1.0f32; vocab.len()];
        apply_grammar_mask(&mut logits, &p, &vocab);
        assert!(logits[14].is_finite()); // "3"
        p.advance("3");
        assert!(p.is_complete());
    }
}
