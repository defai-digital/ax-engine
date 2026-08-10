//! DeepSeek DSML tool-call parsing.
//!
//! DeepSeek models emit tool calls as DSML stanzas delimited by dedicated
//! fullwidth-bar control markers (ds4 reference parity). Markers are built
//! from `U+FF5C` escapes so the source stays byte-exact without embedding
//! fullwidth literals.

use serde_json::{Map, Value};

use super::schema::OpenAiFunctionCall;
use super::tool_names;

pub(crate) const DSML_TOOL_CALLS_OPEN: &str = "<\u{FF5C}DSML\u{FF5C}tool_calls>";
pub(crate) const DSML_TOOL_CALLS_CLOSE: &str = "</\u{FF5C}DSML\u{FF5C}tool_calls>";
const DSML_INVOKE_OPEN_PREFIX: &str = "<\u{FF5C}DSML\u{FF5C}invoke";
const DSML_INVOKE_CLOSE: &str = "</\u{FF5C}DSML\u{FF5C}invoke>";
const DSML_PARAMETER_OPEN_PREFIX: &str = "<\u{FF5C}DSML\u{FF5C}parameter";
const DSML_PARAMETER_CLOSE: &str = "</\u{FF5C}DSML\u{FF5C}parameter>";

/// Extract DSML tool calls from model output.
///
/// Returns the parsed calls plus the leftover assistant content (text
/// outside the stanzas, ds4 keeps it). Any malformed construct fails closed
/// with `None`, leaving the caller to surface the raw text unchanged.
pub(crate) fn parse_dsml_tool_calls(text: &str) -> Option<(Vec<OpenAiFunctionCall>, String)> {
    let first = text.find(DSML_TOOL_CALLS_OPEN)?;
    let mut calls = Vec::new();
    let mut leftover = text[..first].trim().to_string();
    let mut cursor = first;
    loop {
        let stanza_start = cursor + text[cursor..].find(DSML_TOOL_CALLS_OPEN)?;
        let body_start = stanza_start + DSML_TOOL_CALLS_OPEN.len();
        let stanza_close = body_start + text[body_start..].find(DSML_TOOL_CALLS_CLOSE)?;
        let before = calls.len();
        parse_dsml_stanza(&text[body_start..stanza_close], &mut calls)?;
        if calls.len() == before {
            return None;
        }
        let after = stanza_close + DSML_TOOL_CALLS_CLOSE.len();
        match text[after..].find(DSML_TOOL_CALLS_OPEN) {
            Some(next_rel) => {
                let between = text[after..after + next_rel].trim();
                if !between.is_empty() {
                    if !leftover.is_empty() {
                        leftover.push('\n');
                    }
                    leftover.push_str(between);
                }
                cursor = after + next_rel;
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
    while let Some(rel) = body[cursor..].find(DSML_INVOKE_OPEN_PREFIX) {
        let open_start = cursor + rel;
        let open_end = open_start + body[open_start..].find('>')?;
        let open_tag = &body[open_start..=open_end];
        let name = dsml_attr(open_tag, "name")?;
        if !tool_names::is_valid(&name) {
            return None;
        }
        let invoke_close_rel = body[open_start..].find(DSML_INVOKE_CLOSE)?;
        let inner = &body[open_end + 1..open_start + invoke_close_rel];
        let arguments = parse_dsml_parameters(inner)?;
        calls.push(OpenAiFunctionCall { name, arguments });
        cursor = open_start + invoke_close_rel + DSML_INVOKE_CLOSE.len();
    }
    Some(())
}

fn parse_dsml_parameters(inner: &str) -> Option<String> {
    let mut args = Map::new();
    let mut cursor = 0;
    while let Some(rel) = inner[cursor..].find(DSML_PARAMETER_OPEN_PREFIX) {
        let open_start = cursor + rel;
        let open_end = open_start + inner[open_start..].find('>')?;
        let open_tag = &inner[open_start..=open_end];
        let name = dsml_attr(open_tag, "name")?;
        let is_string = dsml_attr(open_tag, "string").is_some_and(|value| value == "true");
        let value_start = open_end + 1;
        let close_rel = inner[value_start..].find(DSML_PARAMETER_CLOSE)?;
        let raw = &inner[value_start..value_start + close_rel];
        let value = if is_string {
            Value::String(raw.to_string())
        } else {
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        args.insert(name, value);
        cursor = value_start + close_rel + DSML_PARAMETER_CLOSE.len();
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
        format!("{DSML_TOOL_CALLS_OPEN}{body}{DSML_TOOL_CALLS_CLOSE}")
    }

    #[test]
    fn markers_use_fullwidth_bar() {
        assert!(DSML_TOOL_CALLS_OPEN.chars().any(|c| c == '\u{FF5C}'));
        assert!(!DSML_TOOL_CALLS_OPEN.contains('|'));
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
    fn malformed_constructs_fail_closed() {
        // Missing stanza close.
        assert!(
            parse_dsml_tool_calls(&format!(
                "{}{}",
                DSML_TOOL_CALLS_OPEN,
                invoke("a", &param("x", true, "1"))
            ))
            .is_none()
        );
        // Missing invoke close.
        assert!(
            parse_dsml_tool_calls(&format!(
                "{}<{BAR}DSML{BAR}invoke name=\"a\">{}{DSML_TOOL_CALLS_CLOSE}",
                DSML_TOOL_CALLS_OPEN,
                param("x", true, "1")
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
