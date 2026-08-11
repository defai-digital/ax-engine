//! DeepSeek DSML tool-call parsing.
//!
//! DeepSeek models emit tool calls as DSML stanzas delimited by dedicated
//! fullwidth-bar control markers (ds4 reference parity). Markers are built
//! from `U+FF5C` escapes so the source stays byte-exact without embedding
//! fullwidth literals.
//!
//! Tag matching is lenient in the ds4 sense: stray ASCII whitespace and
//! duplicated fullwidth bars inside the tag delimiters are ignored (for
//! example `</｜DSML｜invoke >` or `<｜DSML｜｜tool_calls>`). Parameter
//! values between tags are preserved byte-exact; only the structural
//! delimiters are normalized. Malformed constructs still fail closed.

use serde_json::{Map, Value};

use super::schema::OpenAiFunctionCall;
use super::tool_names;

const DSML_WORD: &str = "DSML";
const TAG_TOOL_CALLS: &str = "tool_calls";
const TAG_INVOKE: &str = "invoke";
const TAG_PARAMETER: &str = "parameter";

/// Filler tolerated inside DSML tag delimiters (ds4 parity): ASCII
/// whitespace plus stray fullwidth bars the model sometimes duplicates.
fn is_dsml_filler(c: char) -> bool {
    c.is_ascii_whitespace() || c == '\u{FF5C}'
}

fn skip_dsml_filler(text: &str, mut index: usize) -> usize {
    while let Some(c) = text[index..].chars().next() {
        if !is_dsml_filler(c) {
            break;
        }
        index += c.len_utf8();
    }
    index
}

/// Leniently match one complete DSML tag (`<...>`) at the start of `s`.
/// Returns the byte length of the tag, ending just past the closing `>`.
fn match_dsml_complete_tag(s: &str, closing: bool, kind: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'<') {
        return None;
    }
    let mut index = skip_dsml_filler(s, 1);
    if closing {
        if bytes.get(index) != Some(&b'/') {
            return None;
        }
        index = skip_dsml_filler(s, index + 1);
    }
    if !s[index..].starts_with(DSML_WORD) {
        return None;
    }
    index = skip_dsml_filler(s, index + DSML_WORD.len());
    if !s[index..].starts_with(kind) {
        return None;
    }
    index = skip_dsml_filler(s, index + kind.len());
    if bytes.get(index) != Some(&b'>') {
        return None;
    }
    Some(index + 1)
}

/// Leniently match the head of an attribute-bearing DSML open tag
/// (`<｜DSML｜invoke` / `<｜DSML｜parameter`) at the start of `s`. Returns
/// the byte offset of the attribute region (just past the kind word's
/// filler run); the caller owns scanning to the terminating `>`.
fn match_dsml_open_head(s: &str, kind: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'<') {
        return None;
    }
    let mut index = skip_dsml_filler(s, 1);
    // Closing tags carry no attributes; they go through the complete matcher.
    if bytes.get(index) == Some(&b'/') {
        return None;
    }
    if !s[index..].starts_with(DSML_WORD) {
        return None;
    }
    index = skip_dsml_filler(s, index + DSML_WORD.len());
    if !s[index..].starts_with(kind) {
        return None;
    }
    index += kind.len();
    // Kind-word boundary: anything but filler or `>` starts another word
    // (for example `invokes`), which is not a DSML tag.
    match s[index..].chars().next() {
        None => Some(index),
        Some(c) if is_dsml_filler(c) || c == '>' => Some(skip_dsml_filler(s, index)),
        Some(_) => None,
    }
}

/// Find the next complete DSML tag of `kind` at or after `from`. Returns
/// `(tag_start, end)` with `end` just past the closing `>`.
fn find_dsml_tag(text: &str, from: usize, closing: bool, kind: &str) -> Option<(usize, usize)> {
    let mut search = from;
    while let Some(rel) = text[search..].find('<') {
        let at = search + rel;
        if let Some(len) = match_dsml_complete_tag(&text[at..], closing, kind) {
            return Some((at, at + len));
        }
        search = at + 1;
    }
    None
}

/// Find the next attribute-bearing DSML open tag of `kind` at or after
/// `from`. Returns `(tag_start, gt)` with `gt` at the terminating `>`.
fn find_dsml_open_tag(text: &str, from: usize, kind: &str) -> Option<(usize, usize)> {
    let mut search = from;
    while let Some(rel) = text[search..].find('<') {
        let at = search + rel;
        if let Some(attrs_from) = match_dsml_open_head(&text[at..], kind) {
            let gt_rel = text[at + attrs_from..].find('>')?;
            return Some((at, at + attrs_from + gt_rel));
        }
        search = at + 1;
    }
    None
}

/// Lenient presence check for a DSML tool-call stanza (ds4 parity): stray
/// whitespace or duplicated bars inside the open marker still count.
pub(crate) fn contains_dsml_tool_calls(text: &str) -> bool {
    find_dsml_tag(text, 0, false, TAG_TOOL_CALLS).is_some()
}

/// Extract DSML tool calls from model output.
///
/// Returns the parsed calls plus the leftover assistant content (text
/// outside the stanzas, ds4 keeps it). Any malformed construct fails closed
/// with `None`, leaving the caller to surface the raw text unchanged.
pub(crate) fn parse_dsml_tool_calls(text: &str) -> Option<(Vec<OpenAiFunctionCall>, String)> {
    let (first_start, _) = find_dsml_tag(text, 0, false, TAG_TOOL_CALLS)?;
    let mut calls = Vec::new();
    let mut leftover = text[..first_start].trim().to_string();
    let mut cursor = first_start;
    loop {
        let (_, body_start) = find_dsml_tag(text, cursor, false, TAG_TOOL_CALLS)?;
        let (close_start, after) = find_dsml_tag(text, body_start, true, TAG_TOOL_CALLS)?;
        let before = calls.len();
        parse_dsml_stanza(&text[body_start..close_start], &mut calls)?;
        if calls.len() == before {
            return None;
        }
        match find_dsml_tag(text, after, false, TAG_TOOL_CALLS) {
            Some((next_start, _)) => {
                let between = text[after..next_start].trim();
                if !between.is_empty() {
                    if !leftover.is_empty() {
                        leftover.push('\n');
                    }
                    leftover.push_str(between);
                }
                cursor = next_start;
            }
            None => {
                let tail = text[after..].trim();
                if !tail.is_empty() {
                    if !leftover.is_empty() {
                        leftover.push('\n');
                    }
                    leftover.push_str(tail);
                }
                break;
            }
        }
    }
    if calls.is_empty() {
        return None;
    }
    Some((calls, leftover))
}

fn parse_dsml_stanza(body: &str, calls: &mut Vec<OpenAiFunctionCall>) -> Option<()> {
    let mut cursor = 0;
    while let Some((open_start, gt)) = find_dsml_open_tag(body, cursor, TAG_INVOKE) {
        let open_tag = &body[open_start..=gt];
        let name = dsml_attr(open_tag, "name")?;
        if !tool_names::is_valid(&name) {
            return None;
        }
        let (close_start, close_end) = find_dsml_tag(body, gt + 1, true, TAG_INVOKE)?;
        let inner = &body[gt + 1..close_start];
        let arguments = parse_dsml_parameters(inner)?;
        calls.push(OpenAiFunctionCall { name, arguments });
        cursor = close_end;
    }
    Some(())
}

fn parse_dsml_parameters(inner: &str) -> Option<String> {
    let mut args = Map::new();
    let mut cursor = 0;
    while let Some((open_start, gt)) = find_dsml_open_tag(inner, cursor, TAG_PARAMETER) {
        let open_tag = &inner[open_start..=gt];
        let name = dsml_attr(open_tag, "name")?;
        let is_string = dsml_attr(open_tag, "string").is_some_and(|value| value == "true");
        let value_start = gt + 1;
        let (close_start, close_end) = find_dsml_tag(inner, value_start, true, TAG_PARAMETER)?;
        // Parameter values ride byte-exact (ds4 parity): only the structural
        // delimiters tolerate filler, never the value text itself.
        let raw = &inner[value_start..close_start];
        let value = if is_string {
            Value::String(raw.to_string())
        } else {
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        args.insert(name, value);
        cursor = close_end;
    }
    Some(Value::Object(args).to_string())
}

fn dsml_attr(tag: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = tag.find(&needle)? + needle.len();
    let rest = &tag[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BAR: &str = "\u{FF5C}";

    fn stanza_open() -> String {
        format!("<{BAR}DSML{BAR}tool_calls>")
    }

    fn stanza_close() -> String {
        format!("</{BAR}DSML{BAR}tool_calls>")
    }

    fn invoke(name: &str, params: &str) -> String {
        format!("<{BAR}DSML{BAR}invoke name=\"{name}\">{params}</{BAR}DSML{BAR}invoke>")
    }

    fn param(name: &str, string: bool, value: &str) -> String {
        format!(
            "<{BAR}DSML{BAR}parameter name=\"{name}\" string=\"{string}\">{value}</{BAR}DSML{BAR}parameter>",
            string = if string { "true" } else { "false" }
        )
    }

    fn stanza(body: &str) -> String {
        format!("{}{}{}", stanza_open(), body, stanza_close())
    }

    #[test]
    fn markers_use_fullwidth_bar() {
        assert!(stanza_open().chars().any(|c| c == '\u{FF5C}'));
        assert!(!stanza_open().contains('|'));
    }

    #[test]
    fn parses_single_invoke_with_mixed_params() {
        let text = format!(
            "Let me check.\n{}",
            stanza(&invoke(
                "bash",
                &format!(
                    "{}{}",
                    param("command", true, "ls -la /tmp"),
                    param("timeout", false, "30")
                )
            ))
        );
        let (calls, leftover) = parse_dsml_tool_calls(&text).expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["command"], Value::String("ls -la /tmp".to_string()));
        assert_eq!(args["timeout"], Value::from(30));
        assert_eq!(leftover, "Let me check.");
    }

    #[test]
    fn parses_multiple_invokes_and_stanzas() {
        let text = format!(
            "{}\nbetween\n{}",
            stanza(&invoke("a", &param("x", true, "1"))),
            stanza(&invoke("b", &param("y", false, "true")))
        );
        let (calls, leftover) = parse_dsml_tool_calls(&text).expect("parse");
        assert_eq!(
            calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(leftover, "between");
    }

    #[test]
    fn non_string_param_falls_back_to_raw_string() {
        let text = stanza(&invoke("a", &param("blob", false, "not json")));
        let (calls, _) = parse_dsml_tool_calls(&text).expect("parse");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["blob"], Value::String("not json".to_string()));
    }

    #[test]
    fn tolerates_whitespace_inside_closing_tags() {
        // ds4 parity: closing tags may carry stray whitespace before `>`.
        let text = format!(
            "{}<{BAR}DSML{BAR}invoke name=\"a\">{}< /{BAR}DSML{BAR}invoke >{} ",
            stanza_open(),
            param("x", true, "1").replace(
                &format!("</{BAR}DSML{BAR}parameter>"),
                &format!("</{BAR}DSML{BAR}parameter  >")
            ),
            stanza_close(),
        );
        let (calls, _) = parse_dsml_tool_calls(&text).expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["x"], Value::String("1".to_string()));
    }

    #[test]
    fn tolerates_extra_fullwidth_bars_in_markers() {
        // Duplicated bars in the stanza and invoke markers.
        let text = format!(
            "<{BAR}DSML{BAR}{BAR}tool_calls><{BAR}DSML{BAR}invoke name=\"a\">{}</{BAR}{BAR}DSML{BAR}invoke></{BAR}DSML{BAR}{BAR}tool_calls>",
            param("x", true, "1")
        );
        let (calls, _) = parse_dsml_tool_calls(&text).expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
        assert!(contains_dsml_tool_calls(&text));
    }

    #[test]
    fn tolerates_filler_around_attributes() {
        let text = format!(
            "{}<{BAR}DSML{BAR}invoke {BAR} name=\"a\" >{}< /{BAR} DSML {BAR}invoke>{}",
            stanza_open(),
            param("x", false, "2"),
            stanza_close(),
        );
        let (calls, _) = parse_dsml_tool_calls(&text).expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["x"], Value::from(2));
    }

    #[test]
    fn lenient_delimiters_preserve_values_byte_exact() {
        // Filler tolerance applies to structural tags only; the string value
        // keeps its surrounding whitespace untouched.
        let text = format!(
            "{}{}{}",
            stanza_open(),
            invoke(
                "a",
                &format!(
                    "<{BAR}DSML{BAR}parameter name=\"x\" string=\"true\"> padded value </{BAR}DSML{BAR}parameter >"
                )
            ),
            stanza_close(),
        );
        let (calls, _) = parse_dsml_tool_calls(&text).expect("parse");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["x"], Value::String(" padded value ".to_string()));
    }

    #[test]
    fn kind_word_boundaries_stay_strict() {
        // `invokes` / `DSMLX` are not DSML tags: no parse, no false gate.
        let text = format!(
            "<{BAR}DSML{BAR}invokes name=\"a\">{}</{BAR}DSML{BAR}invokes>",
            param("x", true, "1")
        );
        assert!(parse_dsml_tool_calls(&text).is_none());
        assert!(!contains_dsml_tool_calls(&format!(
            "<{BAR}DSMLX{BAR}tool_calls></{BAR}DSMLX{BAR}tool_calls>"
        )));
    }

    #[test]
    fn malformed_constructs_fail_closed() {
        // Missing stanza close.
        assert!(
            parse_dsml_tool_calls(&format!(
                "{}{}",
                stanza_open(),
                invoke("a", &param("x", true, "1"))
            ))
            .is_none()
        );
        // Missing invoke close.
        assert!(
            parse_dsml_tool_calls(&format!(
                "{}<{BAR}DSML{BAR}invoke name=\"a\">{}{}",
                stanza_open(),
                param("x", true, "1"),
                stanza_close(),
            ))
            .is_none()
        );
        // Invalid tool name.
        assert!(
            parse_dsml_tool_calls(&stanza(&invoke("bad name!", &param("x", true, "1")))).is_none()
        );
        // Empty stanza.
        assert!(parse_dsml_tool_calls(&stanza("")).is_none());
        // No markers at all.
        assert!(parse_dsml_tool_calls("plain answer").is_none());
    }
}
