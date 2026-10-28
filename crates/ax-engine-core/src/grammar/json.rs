//! JSON grammar compiled from the server's fail-closed JSON Schema subset, and
//! a byte-level incremental matcher over it.
//!
//! The matcher is a pushdown recognizer whose frames are `Copy`, so cloning a
//! state is a memcpy of a short stack. It enforces structure, types, `enum` /
//! `const`, `required`, `additionalProperties: false`, string lengths and item
//! counts. Numeric bounds stay with the post-hoc validator.

use serde_json::Value;

/// Longest whitespace run accepted between JSON tokens: room for deep
/// pretty-printed indentation. A run holds at most one newline, as its first
/// byte, so a sampled model cannot drift into blank lines.
pub const MAX_WHITESPACE_RUN: u16 = 64;
/// Whitespace accepted before the top-level value (a newline after
/// `</think>`, say).
pub const MAX_LEADING_WHITESPACE: u16 = 2;
/// Longest number (digits, sign, dot and exponent) accepted.
const MAX_NUMBER_BYTES: u8 = 40;
/// Object schemas may name at most this many properties (one `u128` bitset).
pub const MAX_OBJECT_PROPERTIES: usize = 128;
/// `enum` / `const` literal sets are tracked in one `u64`.
pub const MAX_LITERALS_PER_NODE: usize = 64;
/// Bound on schema nesting while compiling.
const MAX_SCHEMA_DEPTH: usize = 64;

const TYPE_OBJECT: u8 = 1;
const TYPE_ARRAY: u8 = 2;
const TYPE_STRING: u8 = 4;
const TYPE_NUMBER: u8 = 8;
const TYPE_INTEGER: u8 = 16;
const TYPE_BOOLEAN: u8 = 32;
const TYPE_NULL: u8 = 64;
const TYPE_ANY: u8 = 127;

/// Literal table slots shared by every grammar.
const LIT_TRUE: u32 = 0;
const LIT_FALSE: u32 = 1;
const LIT_NULL: u32 = 2;

type NodeId = u32;

#[derive(Clone, Debug, PartialEq)]
struct ObjectSpec {
    /// Property names sorted by bytes, each with its schema.
    keys: Vec<(Vec<u8>, NodeId)>,
    required: u128,
    /// Schema for properties outside `keys`; `None` when they are forbidden.
    additional: Option<NodeId>,
}

#[derive(Clone, Debug, PartialEq)]
enum Node {
    Typed {
        types: u8,
        object: ObjectSpec,
        items: NodeId,
        min_items: u32,
        max_items: u32,
        min_chars: u32,
        max_chars: u32,
    },
    /// Exact serialized values (`enum` / `const`), a range of the literal table.
    Literals { first: u32, count: u32 },
}

/// A compiled grammar. Immutable and shared by every matcher of a request.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonGrammar {
    nodes: Vec<Node>,
    literals: Vec<Vec<u8>>,
    root: NodeId,
    completion_budget: bool,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum JsonGrammarError {
    #[error("schema{path}: {message}")]
    Unsupported { path: String, message: String },
}

fn unsupported(path: &str, message: impl Into<String>) -> JsonGrammarError {
    JsonGrammarError::Unsupported {
        path: path.to_string(),
        message: message.into(),
    }
}

impl JsonGrammar {
    /// Validate the subset whose constraints the matcher enforces without a
    /// post-hoc validator. Metadata is ignored; unsupported assertions fail closed.
    pub fn validate_strict_schema(schema: &Value) -> Result<(), JsonGrammarError> {
        fn visit(schema: &Value, path: &str, depth: usize) -> Result<(), JsonGrammarError> {
            if depth > MAX_SCHEMA_DEPTH {
                return Err(unsupported(path, "schema nesting is too deep"));
            }
            if schema == &Value::Bool(true) {
                return Ok(());
            }
            let object = schema
                .as_object()
                .ok_or_else(|| unsupported(path, "expected a schema object"))?;
            let literal = object.contains_key("const") || object.contains_key("enum");
            if object.contains_key("const") && object.contains_key("enum") {
                return Err(unsupported(
                    path,
                    "const and enum cannot be combined in strict mode",
                ));
            }
            for (key, value) in object {
                let child = format!("{path}/{key}");
                match key.as_str() {
                    "title" | "description" | "$schema" => {}
                    "const" | "enum" => {}
                    _ if literal => {
                        return Err(unsupported(
                            &child,
                            "literal constraints cannot be combined with other assertions in strict mode",
                        ));
                    }
                    "type" => {
                        if value.as_array().is_some_and(Vec::is_empty) {
                            return Err(unsupported(&child, "type must not be empty"));
                        }
                    }
                    "properties" => {
                        let properties = value
                            .as_object()
                            .ok_or_else(|| unsupported(&child, "expected properties object"))?;
                        for (name, nested) in properties {
                            visit(nested, &format!("{child}/{name}"), depth + 1)?;
                        }
                    }
                    "required" => {
                        let names = value
                            .as_array()
                            .ok_or_else(|| unsupported(&child, "expected required array"))?;
                        if names.iter().any(|name| !name.is_string()) {
                            return Err(unsupported(&child, "required entries must be strings"));
                        }
                    }
                    "additionalProperties" if value.is_boolean() => {}
                    "items" => visit(value, &child, depth + 1)?,
                    "minItems" | "maxItems" | "minLength" | "maxLength" => {
                        if value
                            .as_u64()
                            .is_none_or(|bound| bound >= u64::from(u32::MAX))
                        {
                            return Err(unsupported(
                                &child,
                                "bound must be a nonnegative integer below 4294967295",
                            ));
                        }
                    }
                    _ => {
                        return Err(unsupported(
                            &child,
                            "constraint is not enforced by strict decoding",
                        ));
                    }
                }
            }
            if object.get("enum").is_some_and(|value| !value.is_array()) {
                return Err(unsupported(path, "enum must be an array"));
            }
            Ok(())
        }
        visit(schema, "", 0)
    }

    /// Any JSON object (`response_format: json_object`).
    pub fn any_object() -> Self {
        let mut grammar = Self::empty();
        let any = grammar.push_any();
        let object = ObjectSpec {
            keys: Vec::new(),
            required: 0,
            additional: Some(any),
        };
        grammar.root = grammar.push(Node::Typed {
            types: TYPE_OBJECT,
            object,
            items: any,
            min_items: 0,
            max_items: u32::MAX,
            min_chars: 0,
            max_chars: u32::MAX,
        });
        grammar
    }

    /// Compile a schema already accepted by the server's subset validator.
    /// Shapes the matcher cannot enforce exactly return an error so the
    /// caller keeps post-hoc validation only.
    pub fn from_schema(schema: &Value) -> Result<Self, JsonGrammarError> {
        let mut grammar = Self::empty();
        grammar.root = grammar.compile(schema, "", 0)?;
        Ok(grammar)
    }

    fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            literals: vec![b"true".to_vec(), b"false".to_vec(), b"null".to_vec()],
            root: 0,
            completion_budget: false,
        }
    }

    fn push(&mut self, node: Node) -> NodeId {
        self.nodes.push(node);
        (self.nodes.len() - 1) as NodeId
    }

    fn push_any(&mut self) -> NodeId {
        // The Any node refers to itself for object values and array items.
        let id = self.nodes.len() as NodeId;
        self.push(Node::Typed {
            types: TYPE_ANY,
            object: ObjectSpec {
                keys: Vec::new(),
                required: 0,
                additional: Some(id),
            },
            items: id,
            min_items: 0,
            max_items: u32::MAX,
            min_chars: 0,
            max_chars: u32::MAX,
        })
    }

    fn node(&self, id: NodeId) -> &Node {
        // Ids are only ever produced by `push`, so they are in range.
        &self.nodes[id as usize]
    }

    fn literal(&self, index: u32) -> &[u8] {
        &self.literals[index as usize]
    }

    fn compile(
        &mut self,
        schema: &Value,
        path: &str,
        depth: usize,
    ) -> Result<NodeId, JsonGrammarError> {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(unsupported(
                path,
                "schema nesting is too deep for constrained decoding",
            ));
        }
        let object = match schema {
            Value::Bool(true) => return Ok(self.push_any()),
            Value::Object(object) => object,
            _ => return Err(unsupported(path, "expected a schema object")),
        };
        // `enum` and `const` are stricter than every other keyword here, so
        // they alone decide the accepted bytes; the post-hoc validator still
        // checks the rest.
        if let Some(constant) = object.get("const") {
            let options = std::slice::from_ref(constant);
            let options = match object.get("enum") {
                Some(Value::Array(values)) => {
                    let kept: Vec<Value> = values
                        .iter()
                        .filter(|value| *value == constant)
                        .cloned()
                        .collect();
                    if kept.is_empty() {
                        return Err(unsupported(path, "const is not one of enum"));
                    }
                    return self.push_literals(&kept, path);
                }
                _ => options,
            };
            return self.push_literals(options, path);
        }
        if let Some(Value::Array(options)) = object.get("enum") {
            return self.push_literals(options, path);
        }

        let types = match object.get("type") {
            None => TYPE_ANY,
            Some(Value::String(name)) => type_bit(name, path)?,
            Some(Value::Array(names)) => {
                let mut bits = 0;
                for name in names {
                    let Some(name) = name.as_str() else {
                        return Err(unsupported(path, "type entries must be strings"));
                    };
                    bits |= type_bit(name, path)?;
                }
                bits
            }
            Some(_) => return Err(unsupported(path, "type must be a string or array")),
        };

        let any = self.push_any();
        let mut keys = Vec::new();
        if let Some(Value::Object(properties)) = object.get("properties") {
            if properties.len() > MAX_OBJECT_PROPERTIES {
                return Err(unsupported(
                    path,
                    format!("more than {MAX_OBJECT_PROPERTIES} properties"),
                ));
            }
            for (name, subschema) in properties {
                if !key_is_plain(name) {
                    return Err(unsupported(
                        path,
                        format!("property name {name:?} needs JSON escapes"),
                    ));
                }
                let child =
                    self.compile(subschema, &format!("{path}/properties/{name}"), depth + 1)?;
                keys.push((name.as_bytes().to_vec(), child));
            }
        }
        let additional = match object.get("additionalProperties") {
            Some(Value::Bool(false)) => None,
            _ => Some(any),
        };
        if let Some(Value::Array(required)) = object.get("required") {
            for name in required.iter().filter_map(Value::as_str) {
                if keys.iter().any(|(key, _)| key == name.as_bytes()) {
                    continue;
                }
                if additional.is_none() {
                    return Err(unsupported(
                        path,
                        format!("required property {name} is forbidden by additionalProperties"),
                    ));
                }
                if !key_is_plain(name) || keys.len() >= MAX_OBJECT_PROPERTIES {
                    return Err(unsupported(path, format!("required property {name:?}")));
                }
                keys.push((name.as_bytes().to_vec(), any));
            }
        }
        keys.sort_by(|left, right| left.0.cmp(&right.0));
        let mut required_bits = 0u128;
        if let Some(Value::Array(required)) = object.get("required") {
            for name in required.iter().filter_map(Value::as_str) {
                if let Some(index) = keys.iter().position(|(key, _)| key == name.as_bytes()) {
                    required_bits |= 1u128 << index;
                }
            }
        }
        let items = match object.get("items") {
            Some(items) => self.compile(items, &format!("{path}/items"), depth + 1)?,
            None => any,
        };
        let bound = |keyword: &str| -> Option<u32> {
            object
                .get(keyword)
                .and_then(Value::as_u64)
                .map(|value| value.min(u64::from(u32::MAX - 1)) as u32)
        };
        let min_items = bound("minItems").unwrap_or(0);
        let max_items = bound("maxItems").unwrap_or(u32::MAX);
        let min_chars = bound("minLength").unwrap_or(0);
        let max_chars = bound("maxLength").unwrap_or(u32::MAX);
        if min_items > max_items || min_chars > max_chars {
            return Err(unsupported(
                path,
                "minimum exceeds maximum; nothing can match",
            ));
        }
        Ok(self.push(Node::Typed {
            types,
            object: ObjectSpec {
                keys,
                required: required_bits,
                additional,
            },
            items,
            min_items,
            max_items,
            min_chars,
            max_chars,
        }))
    }

    fn push_literals(&mut self, options: &[Value], path: &str) -> Result<NodeId, JsonGrammarError> {
        if options.is_empty() || options.len() > MAX_LITERALS_PER_NODE {
            return Err(unsupported(
                path,
                format!("enum needs 1..={MAX_LITERALS_PER_NODE} options"),
            ));
        }
        let first = self.literals.len() as u32;
        for option in options {
            let text = serde_json::to_string(option)
                .map_err(|error| unsupported(path, format!("enum option: {error}")))?;
            self.literals.push(text.into_bytes());
        }
        Ok(self.push(Node::Literals {
            first,
            count: options.len() as u32,
        }))
    }

    /// Start a matcher at the root value.
    pub fn matcher(&self) -> JsonMatcher {
        JsonMatcher {
            stack: vec![Frame::Value { node: self.root }],
            whitespace_run: 0,
        }
    }

    /// Require a bounded tokenizer-level completion proof before each answer
    /// token. The immutable policy travels with the request's compiled grammar.
    pub fn with_completion_budget(mut self, enabled: bool) -> Self {
        self.completion_budget = enabled;
        self
    }

    pub fn completion_budget(&self) -> bool {
        self.completion_budget
    }
}

fn type_bit(name: &str, path: &str) -> Result<u8, JsonGrammarError> {
    Ok(match name {
        "object" => TYPE_OBJECT,
        "array" => TYPE_ARRAY,
        "string" => TYPE_STRING,
        "number" => TYPE_NUMBER | TYPE_INTEGER,
        "integer" => TYPE_INTEGER,
        "boolean" => TYPE_BOOLEAN,
        "null" => TYPE_NULL,
        other => return Err(unsupported(path, format!("unsupported type {other}"))),
    })
}

/// Property names the matcher handles: no character that JSON must escape.
fn key_is_plain(name: &str) -> bool {
    name.bytes()
        .all(|byte| byte >= 0x20 && byte != b'"' && byte != b'\\')
}

/// Whitespace between JSON tokens. A carriage return is valid JSON but no
/// model needs it, and admitting it lets sampled requests drift into
/// `\r` runs.
fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n')
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ObjectState {
    AfterOpen,
    /// Inside a key: names `lo..hi` of the sorted list still share its
    /// prefix (an empty range is an additional property).
    InKey {
        lo: u8,
        hi: u8,
        len: u16,
        utf8: u8,
    },
    /// After the closing quote of a key; `prop` is the matched property or
    /// `u8::MAX` for an additional property.
    AwaitColon {
        prop: u8,
    },
    AfterValue,
    AfterComma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ArrayState {
    Open,
    Value,
    Comma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum NumberState {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum StringEscape {
    None,
    Start,
    Unicode {
        remaining: u8,
        value: u16,
        low_surrogate: bool,
    },
    LowBackslash,
    LowU,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Frame {
    Value {
        node: NodeId,
    },
    Object {
        node: NodeId,
        used: u128,
        state: ObjectState,
    },
    Array {
        node: NodeId,
        count: u32,
        state: ArrayState,
    },
    Str {
        min: u32,
        max: u32,
        chars: u32,
        escape: StringEscape,
        utf8: u8,
    },
    Number {
        integer: bool,
        state: NumberState,
        len: u8,
    },
    Literal {
        first: u32,
        alive: u64,
        pos: u16,
    },
}

enum Step {
    Consumed,
    /// The frame finished with this byte.
    Done,
    /// The frame finished before this byte; offer the byte to the parent.
    Refeed,
    Replace(Frame),
    Push(Frame),
    Reject,
}

/// Incremental matcher state. Clone it to explore alternatives.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JsonMatcher {
    stack: Vec<Frame>,
    whitespace_run: u16,
}

impl JsonMatcher {
    /// The value is complete, so end of sequence is acceptable.
    pub fn is_complete(&self, grammar: &JsonGrammar) -> bool {
        match self.stack.as_slice() {
            [] => true,
            [only] => frame_can_end(grammar, only),
            _ => false,
        }
    }

    /// A copy that accepts exactly the same continuations, with counters
    /// that no longer matter folded away: the character count of a string
    /// with no `maxLength` once it has reached `minLength`. Equal keys mean
    /// equal allowed-token masks, so masks can be cached by key.
    pub fn mask_key(&self) -> JsonMatcher {
        let mut key = self.clone();
        for frame in &mut key.stack {
            if let Frame::Str {
                min, max, chars, ..
            } = frame
                && *max == u32::MAX
            {
                *chars = (*chars).min(*min);
            }
            // Once no declared property shares this prefix, its length can
            // never make it a declared property again. Keep UTF-8 progress,
            // but merge arbitrary additional-key text into one language state.
            if let Frame::Object {
                state: ObjectState::InKey { lo, hi, len, .. },
                ..
            } = frame
                && lo == hi
            {
                *lo = 0;
                *hi = 0;
                *len = 0;
            }
        }
        key
    }

    /// No further byte can be accepted.
    pub fn is_closed(&self) -> bool {
        self.stack.is_empty()
    }

    /// When the matcher is inside a string that accepts every unescaped
    /// printable character, the number of characters it may still take.
    /// Token masks use it to accept quote- and escape-free tokens without
    /// walking them byte by byte.
    pub fn free_text_remaining(&self, grammar: &JsonGrammar) -> Option<u32> {
        match self.stack.last()? {
            Frame::Str {
                max,
                chars,
                escape: StringEscape::None,
                utf8: 0,
                ..
            } => Some(max.saturating_sub(*chars)),
            Frame::Object {
                node,
                state: ObjectState::InKey { utf8: 0, .. },
                ..
            } => match grammar.node(*node) {
                Node::Typed { object, .. } if object.additional.is_some() => Some(u32::MAX),
                _ => None,
            },
            _ => None,
        }
    }

    /// Consume a sequence; on rejection the state is left partially advanced.
    pub fn advance_bytes(&mut self, grammar: &JsonGrammar, bytes: &[u8]) -> bool {
        bytes.iter().all(|byte| self.advance(grammar, *byte))
    }

    /// Consume one byte. Returns false, leaving the state unspecified, when
    /// the byte is not allowed.
    pub fn advance(&mut self, grammar: &JsonGrammar, byte: u8) -> bool {
        loop {
            let depth = self.stack.len();
            let Some(top) = self.stack.last_mut() else {
                return false;
            };
            // Whitespace inside strings and keys is content, not a run.
            let content = matches!(
                top,
                Frame::Str { .. }
                    | Frame::Object {
                        state: ObjectState::InKey { .. },
                        ..
                    }
            );
            let leading = matches!(top, Frame::Value { .. }) && depth == 1;
            match step_frame(grammar, top, byte) {
                Step::Consumed => {}
                Step::Done => {
                    self.stack.pop();
                }
                Step::Refeed => {
                    self.stack.pop();
                    continue;
                }
                Step::Replace(frame) => {
                    *top = frame;
                }
                Step::Push(frame) => self.stack.push(frame),
                Step::Reject => return false,
            }
            if is_whitespace(byte) && !content {
                if byte == b'\n' && self.whitespace_run > 0 && !leading {
                    return false;
                }
                self.whitespace_run += 1;
                let limit = if leading {
                    MAX_LEADING_WHITESPACE
                } else {
                    MAX_WHITESPACE_RUN
                };
                if self.whitespace_run > limit {
                    return false;
                }
            } else {
                self.whitespace_run = 0;
            }
            return true;
        }
    }
}

fn frame_can_end(grammar: &JsonGrammar, frame: &Frame) -> bool {
    match frame {
        Frame::Number { state, .. } => matches!(
            state,
            NumberState::Zero | NumberState::Int | NumberState::Frac | NumberState::ExpDigits
        ),
        Frame::Literal { first, alive, pos } => literal_complete(grammar, *first, *alive, *pos),
        _ => false,
    }
}

fn literal_complete(grammar: &JsonGrammar, first: u32, alive: u64, pos: u16) -> bool {
    (0..64u32)
        .filter(|bit| alive & (1u64 << bit) != 0)
        .any(|bit| grammar.literal(first + bit).len() == usize::from(pos))
}

/// Start the value `node` with `byte`; the returned frame has consumed it.
fn start_value(grammar: &JsonGrammar, node: NodeId, byte: u8) -> Option<Frame> {
    match grammar.node(node) {
        Node::Literals { first, count } => {
            let mut alive = 0u64;
            for bit in 0..*count {
                if grammar.literal(first + bit).first() == Some(&byte) {
                    alive |= 1u64 << bit;
                }
            }
            (alive != 0).then_some(Frame::Literal {
                first: *first,
                alive,
                pos: 1,
            })
        }
        Node::Typed {
            types,
            min_chars,
            max_chars,
            ..
        } => {
            let types = *types;
            match byte {
                b'{' if types & TYPE_OBJECT != 0 => Some(Frame::Object {
                    node,
                    used: 0,
                    state: ObjectState::AfterOpen,
                }),
                b'[' if types & TYPE_ARRAY != 0 => Some(Frame::Array {
                    node,
                    count: 0,
                    state: ArrayState::Open,
                }),
                b'"' if types & TYPE_STRING != 0 => Some(Frame::Str {
                    min: *min_chars,
                    max: *max_chars,
                    chars: 0,
                    escape: StringEscape::None,
                    utf8: 0,
                }),
                b'-' | b'0'..=b'9' if types & (TYPE_NUMBER | TYPE_INTEGER) != 0 => {
                    Some(Frame::Number {
                        integer: types & TYPE_NUMBER == 0,
                        state: match byte {
                            b'-' => NumberState::Minus,
                            b'0' => NumberState::Zero,
                            _ => NumberState::Int,
                        },
                        len: 1,
                    })
                }
                b't' | b'f' if types & TYPE_BOOLEAN != 0 => Some(Frame::Literal {
                    first: if byte == b't' { LIT_TRUE } else { LIT_FALSE },
                    alive: 1,
                    pos: 1,
                }),
                b'n' if types & TYPE_NULL != 0 => Some(Frame::Literal {
                    first: LIT_NULL,
                    alive: 1,
                    pos: 1,
                }),
                _ => None,
            }
        }
    }
}

fn object_spec(grammar: &JsonGrammar, node: NodeId) -> Option<&ObjectSpec> {
    match grammar.node(node) {
        Node::Typed { object, .. } => Some(object),
        Node::Literals { .. } => None,
    }
}

/// Track a UTF-8 sequence inside a string or key. Returns the new pending
/// continuation state and whether the byte starts a character. States 4..=7
/// constrain the second byte of a sequence to valid Unicode scalar values.
fn utf8_step(pending: u8, byte: u8) -> Option<(u8, bool)> {
    match (pending, byte) {
        (0, 0x20..=0x7f) => Some((0, true)),
        (0, 0xc2..=0xdf) => Some((1, true)),
        (0, 0xe0) => Some((4, true)),
        (0, 0xed) => Some((5, true)),
        (0, 0xe1..=0xec | 0xee..=0xef) => Some((2, true)),
        (0, 0xf0) => Some((6, true)),
        (0, 0xf4) => Some((7, true)),
        (0, 0xf1..=0xf3) => Some((3, true)),
        (4, 0xa0..=0xbf) | (5, 0x80..=0x9f) => Some((1, false)),
        (6, 0x90..=0xbf) | (7, 0x80..=0x8f) => Some((2, false)),
        (1..=3, 0x80..=0xbf) => Some((pending - 1, false)),
        _ => None,
    }
}

fn step_frame(grammar: &JsonGrammar, frame: &mut Frame, byte: u8) -> Step {
    match *frame {
        Frame::Value { node } => {
            if is_whitespace(byte) {
                return Step::Consumed;
            }
            match start_value(grammar, node, byte) {
                Some(started) => {
                    // A one-byte literal set may already be complete, but it
                    // can still be a prefix of a longer one; keep the frame.
                    Step::Replace(started)
                }
                None => Step::Reject,
            }
        }
        Frame::Str {
            min,
            max,
            chars,
            escape,
            utf8,
        } => {
            let mut next = (chars, escape, utf8);
            match escape {
                StringEscape::None => {
                    if utf8 == 0 && byte == b'"' {
                        return if chars >= min {
                            Step::Done
                        } else {
                            Step::Reject
                        };
                    }
                    if utf8 == 0 && byte == b'\\' {
                        if chars >= max {
                            return Step::Reject;
                        }
                        next = (chars + 1, StringEscape::Start, 0);
                    } else {
                        let Some((pending, starts)) = utf8_step(utf8, byte) else {
                            return Step::Reject;
                        };
                        if starts && chars >= max {
                            return Step::Reject;
                        }
                        next = (chars + u32::from(starts), StringEscape::None, pending);
                    }
                }
                StringEscape::Start => match byte {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                        next.1 = StringEscape::None;
                    }
                    b'u' => {
                        next.1 = StringEscape::Unicode {
                            remaining: 4,
                            value: 0,
                            low_surrogate: false,
                        };
                    }
                    _ => return Step::Reject,
                },
                StringEscape::Unicode {
                    remaining,
                    value,
                    low_surrogate,
                } => {
                    let digit = match byte {
                        b'0'..=b'9' => u16::from(byte - b'0'),
                        b'a'..=b'f' => u16::from(byte - b'a' + 10),
                        b'A'..=b'F' => u16::from(byte - b'A' + 10),
                        _ => return Step::Reject,
                    };
                    let value = (value << 4) | digit;
                    // Reject a prefix as soon as no valid code unit can
                    // complete it, before the token mask reaches a dead end.
                    if low_surrogate {
                        if (remaining == 4 && value != 0xd)
                            || (remaining == 3 && !(0xdc..=0xdf).contains(&value))
                        {
                            return Step::Reject;
                        }
                    } else if remaining == 3 && (0xdc..=0xdf).contains(&value) {
                        return Step::Reject;
                    }
                    next.1 = if remaining > 1 {
                        StringEscape::Unicode {
                            remaining: remaining - 1,
                            value,
                            low_surrogate,
                        }
                    } else if low_surrogate {
                        if !(0xdc00..=0xdfff).contains(&value) {
                            return Step::Reject;
                        }
                        StringEscape::None
                    } else {
                        match value {
                            0xd800..=0xdbff => StringEscape::LowBackslash,
                            0xdc00..=0xdfff => return Step::Reject,
                            _ => StringEscape::None,
                        }
                    };
                }
                StringEscape::LowBackslash => {
                    if byte != b'\\' {
                        return Step::Reject;
                    }
                    next.1 = StringEscape::LowU;
                }
                StringEscape::LowU => {
                    if byte != b'u' {
                        return Step::Reject;
                    }
                    // The high surrogate already counted this scalar value.
                    next.1 = StringEscape::Unicode {
                        remaining: 4,
                        value: 0,
                        low_surrogate: true,
                    };
                }
            }
            *frame = Frame::Str {
                min,
                max,
                chars: next.0,
                escape: next.1,
                utf8: next.2,
            };
            Step::Consumed
        }
        Frame::Number {
            integer,
            state,
            len,
        } => {
            use NumberState::*;
            let next = match (state, byte) {
                (Minus, b'0') => Some(Zero),
                (Minus, b'1'..=b'9') => Some(Int),
                (Int, b'0'..=b'9') => Some(Int),
                (Zero | Int, b'.') if !integer => Some(Dot),
                (Zero | Int, b'e' | b'E') if !integer => Some(Exp),
                (Dot | Frac, b'0'..=b'9') => Some(Frac),
                (Frac, b'e' | b'E') => Some(Exp),
                (Exp, b'+' | b'-') => Some(ExpSign),
                (Exp | ExpSign | ExpDigits, b'0'..=b'9') => Some(ExpDigits),
                _ => None,
            };
            match next {
                Some(state) if len < MAX_NUMBER_BYTES => {
                    *frame = Frame::Number {
                        integer,
                        state,
                        len: len + 1,
                    };
                    Step::Consumed
                }
                Some(_) => Step::Reject,
                None if frame_can_end(grammar, frame) => Step::Refeed,
                None => Step::Reject,
            }
        }
        Frame::Literal { first, alive, pos } => {
            let mut next_alive = 0u64;
            let mut longer = false;
            for bit in 0..64u32 {
                if alive & (1u64 << bit) == 0 {
                    continue;
                }
                let literal = grammar.literal(first + bit);
                if literal.get(usize::from(pos)) == Some(&byte) {
                    next_alive |= 1u64 << bit;
                    if literal.len() > usize::from(pos) + 1 {
                        longer = true;
                    }
                }
            }
            if next_alive == 0 {
                return if literal_complete(grammar, first, alive, pos) {
                    Step::Refeed
                } else {
                    Step::Reject
                };
            }
            if !longer {
                return Step::Done;
            }
            *frame = Frame::Literal {
                first,
                alive: next_alive,
                pos: pos + 1,
            };
            Step::Consumed
        }
        Frame::Array { node, count, state } => {
            let Node::Typed {
                items,
                min_items,
                max_items,
                ..
            } = grammar.node(node)
            else {
                return Step::Reject;
            };
            let value_allowed = count < *max_items;
            match state {
                ArrayState::Open | ArrayState::Comma => {
                    if is_whitespace(byte) {
                        return Step::Consumed;
                    }
                    if byte == b']' && state == ArrayState::Open {
                        return if count >= *min_items {
                            Step::Done
                        } else {
                            Step::Reject
                        };
                    }
                    if !value_allowed {
                        return Step::Reject;
                    }
                    let Some(started) = start_value(grammar, *items, byte) else {
                        return Step::Reject;
                    };
                    *frame = Frame::Array {
                        node,
                        count: count + 1,
                        state: ArrayState::Value,
                    };
                    Step::Push(started)
                }
                ArrayState::Value => match byte {
                    _ if is_whitespace(byte) => Step::Consumed,
                    b',' if value_allowed => {
                        *frame = Frame::Array {
                            node,
                            count,
                            state: ArrayState::Comma,
                        };
                        Step::Consumed
                    }
                    b']' if count >= *min_items => Step::Done,
                    _ => Step::Reject,
                },
            }
        }
        Frame::Object { node, used, state } => {
            let Some(spec) = object_spec(grammar, node) else {
                return Step::Reject;
            };
            let free = spec.additional.is_some();
            let all_bits = if spec.keys.len() >= 128 {
                u128::MAX
            } else {
                (1u128 << spec.keys.len()) - 1
            };
            let more_keys = free || used != all_bits;
            let set_state = |frame: &mut Frame, state| {
                *frame = Frame::Object { node, used, state };
            };
            match state {
                ObjectState::AfterOpen | ObjectState::AfterComma => {
                    if is_whitespace(byte) {
                        return Step::Consumed;
                    }
                    if byte == b'}' && state == ObjectState::AfterOpen {
                        return if spec.required & !used == 0 {
                            Step::Done
                        } else {
                            Step::Reject
                        };
                    }
                    if byte != b'"' || !more_keys {
                        return Step::Reject;
                    }
                    set_state(
                        frame,
                        ObjectState::InKey {
                            lo: 0,
                            hi: spec.keys.len() as u8,
                            len: 0,
                            utf8: 0,
                        },
                    );
                    Step::Consumed
                }
                ObjectState::InKey { lo, hi, len, utf8 } => {
                    let unused_in = |lo: u8, hi: u8, exact: Option<u16>| {
                        (lo..hi).find(|index| {
                            used & (1u128 << index) == 0
                                && match exact {
                                    Some(len) => {
                                        spec.keys[usize::from(*index)].0.len() == usize::from(len)
                                    }
                                    None => true,
                                }
                        })
                    };
                    if utf8 == 0 && byte == b'"' {
                        let prop = unused_in(lo, hi, Some(len));
                        let prop = match prop {
                            Some(index) => index,
                            // An unknown name, or a declared name used
                            // already: with additional properties allowed
                            // both are additional (duplicate keys are valid
                            // JSON). Refusing the repeat would trap the key,
                            // because a free key admits any plain text.
                            None if free => u8::MAX,
                            None => return Step::Reject,
                        };
                        set_state(frame, ObjectState::AwaitColon { prop });
                        return Step::Consumed;
                    }
                    if utf8 == 0 && (byte == b'\\' || byte < 0x20) {
                        return Step::Reject;
                    }
                    let Some((pending, _)) = utf8_step(utf8, byte) else {
                        return Step::Reject;
                    };
                    // Narrow the sorted range to names with this byte here.
                    let at = usize::from(len);
                    let matches =
                        |index: u8| spec.keys[usize::from(index)].0.get(at) == Some(&byte);
                    let new_lo = (lo..hi).find(|index| matches(*index)).unwrap_or(hi);
                    let new_hi = (new_lo..hi).find(|index| !matches(*index)).unwrap_or(hi);
                    let (new_lo, new_hi) = if new_lo >= new_hi {
                        (0, 0)
                    } else {
                        (new_lo, new_hi)
                    };
                    if !free && unused_in(new_lo, new_hi, None).is_none() {
                        return Step::Reject;
                    }
                    set_state(
                        frame,
                        ObjectState::InKey {
                            lo: new_lo,
                            hi: new_hi,
                            len: len.saturating_add(1),
                            utf8: pending,
                        },
                    );
                    Step::Consumed
                }
                ObjectState::AwaitColon { prop } => {
                    if is_whitespace(byte) {
                        return Step::Consumed;
                    }
                    if byte != b':' {
                        return Step::Reject;
                    }
                    let (value_node, used) = if prop == u8::MAX {
                        match spec.additional {
                            Some(additional) => (additional, used),
                            None => return Step::Reject,
                        }
                    } else {
                        (spec.keys[usize::from(prop)].1, used | (1u128 << prop))
                    };
                    *frame = Frame::Object {
                        node,
                        used,
                        state: ObjectState::AfterValue,
                    };
                    Step::Push(Frame::Value { node: value_node })
                }
                ObjectState::AfterValue => match byte {
                    _ if is_whitespace(byte) => Step::Consumed,
                    b',' if more_keys => {
                        set_state(frame, ObjectState::AfterComma);
                        Step::Consumed
                    }
                    b'}' if spec.required & !used == 0 => Step::Done,
                    _ => Step::Reject,
                },
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn accepts(grammar: &JsonGrammar, text: &str) -> bool {
        let mut matcher = grammar.matcher();
        matcher.advance_bytes(grammar, text.as_bytes()) && matcher.is_complete(grammar)
    }

    fn prefix_ok(grammar: &JsonGrammar, text: &str) -> bool {
        grammar.matcher().advance_bytes(grammar, text.as_bytes())
    }

    #[test]
    fn additional_key_cache_keys_preserve_declared_and_utf8_continuations() {
        let grammar = JsonGrammar::from_schema(&json!({
            "type":"object", "properties":{"known":{"type":"boolean"}},
            "required":["known"]
        }))
        .unwrap();
        for prefix in [&b"{\"extra"[..], &b"{\"k"[..], &b"{\"\xe6"[..]] {
            let mut original = grammar.matcher();
            assert!(original.advance_bytes(&grammar, prefix));
            let canonical = original.mask_key();
            for suffix in [
                &b"\":0,\"known\":true}"[..],
                &b"nown\":true}"[..],
                &b"\xb0\xb4\":null,\"known\":true}"[..],
            ] {
                let mut before = original.clone();
                let mut after = canonical.clone();
                let accepted = before.advance_bytes(&grammar, suffix);
                assert_eq!(accepted, after.advance_bytes(&grammar, suffix));
                if accepted {
                    assert_eq!(before.is_complete(&grammar), after.is_complete(&grammar));
                }
            }
        }
    }

    #[test]
    fn any_object_accepts_valid_json_objects_only() {
        let grammar = JsonGrammar::any_object();
        for text in [
            "{}",
            "{\"a\": 1}",
            "{\n  \"a\": [1, 2.5, -3e+2, true, null],\n  \"b\": {\"c\": \"x\\\"y\\u00e9\"}\n}",
            "{\"k\":\"日本\"}",
        ] {
            assert!(accepts(&grammar, text), "{text}");
        }
        for text in [
            "[]",
            "1",
            "{\"a\" 1}",
            "{\"a\":01}",
            "{\"a\":1,}",
            "{\"a\":tru}",
            "{} ",
        ] {
            assert!(!accepts(&grammar, text), "{text}");
        }
        assert!(prefix_ok(&grammar, "{\"a\": 1"));
        assert!(!accepts(&grammar, "{\"a\": 1"));
    }

    #[test]
    fn schema_enforces_types_required_and_closed_objects() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 3},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"enum": ["a", "bb"]}, "maxItems": 2},
                "ok": {"type": "boolean"}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        });
        let grammar = JsonGrammar::from_schema(&schema).unwrap();
        assert!(accepts(&grammar, "{\"name\":\"abc\",\"age\":30}"));
        assert!(accepts(
            &grammar,
            "{\"age\":-1, \"name\":\"x\", \"tags\":[\"bb\",\"a\"], \"ok\":false}"
        ));
        // Missing required, too long, float for integer, unknown key, enum miss.
        assert!(!accepts(&grammar, "{\"name\":\"abc\"}"));
        assert!(!prefix_ok(&grammar, "{\"name\":\"abcd"));
        assert!(!prefix_ok(&grammar, "{\"age\":1.5"));
        assert!(!prefix_ok(&grammar, "{\"nam\""));
        assert!(!prefix_ok(&grammar, "{\"extra"));
        assert!(!prefix_ok(&grammar, "{\"tags\":[\"c"));
        assert!(!prefix_ok(&grammar, "{\"tags\":[\"a\",\"a\",\""));
        // Duplicate keys are refused for closed objects, and accepted as
        // additional properties for open ones.
        let open = JsonGrammar::from_schema(&json!({
            "type": "object", "properties": {"name": {"type": "string"}}
        }))
        .unwrap();
        assert!(accepts(&open, "{\"name\":\"x\",\"name\":1}"));
        assert!(!prefix_ok(&grammar, "{\"age\":1,\"age\""));
        // Empty name violates minLength.
        assert!(!prefix_ok(&grammar, "{\"name\":\"\""));
    }

    #[test]
    fn strings_and_keys_reject_invalid_utf8_at_the_first_invalid_byte() {
        let grammar = JsonGrammar::any_object();
        for prefix in [b"{\"".as_slice(), b"{\"key\":\"".as_slice()] {
            for invalid in [
                b"\xe0\x80".as_slice(),
                b"\xed\xa0".as_slice(),
                b"\xf0\x80".as_slice(),
                b"\xf4\x90".as_slice(),
            ] {
                let mut matcher = grammar.matcher();
                assert!(matcher.advance_bytes(&grammar, prefix));
                assert!(matcher.advance(&grammar, invalid[0]));
                assert!(!matcher.advance(&grammar, invalid[1]), "{invalid:?}");
            }
            for valid in [
                b"\xc2\x80".as_slice(),
                b"\xe0\xa0\x80".as_slice(),
                b"\xed\x9f\xbf".as_slice(),
                b"\xef\xbf\xbf".as_slice(),
                b"\xf0\x90\x80\x80".as_slice(),
                b"\xf4\x8f\xbf\xbf".as_slice(),
            ] {
                let mut matcher = grammar.matcher();
                assert!(matcher.advance_bytes(&grammar, prefix));
                for byte in valid {
                    assert!(matcher.advance(&grammar, *byte), "{valid:?}");
                }
                assert!(matcher.advance(&grammar, b'"'));
            }
        }
    }

    #[test]
    fn unicode_escapes_require_paired_surrogates_and_count_scalar_values() {
        let grammar = JsonGrammar::from_schema(&json!({"type": "string"})).unwrap();
        for text in [
            r#""\ud83d""#,
            r#""\ud83dx"#,
            r#""\ude00"#,
            r#""\ud83d\u0041"#,
            r#""\ud83d\ud83d"#,
            r#""\udc"#,
            r#""\ud83d\u0"#,
            r#""\ud83d\ud8"#,
        ] {
            assert!(!prefix_ok(&grammar, text), "{text}");
        }
        let one = JsonGrammar::from_schema(&json!({
            "type": "string", "minLength": 1, "maxLength": 1
        }))
        .unwrap();
        for text in [r#""\ud83d\ude00""#, r#""\uD83D\uDE00""#, r#""\u0041""#] {
            assert!(accepts(&one, text), "{text}");
            let parsed: String = serde_json::from_str(text).unwrap();
            assert_eq!(parsed.chars().count(), 1);
        }
        let two = JsonGrammar::from_schema(&json!({
            "type": "string", "minLength": 2, "maxLength": 2
        }))
        .unwrap();
        assert!(!accepts(&two, r#""\ud83d\ude00""#));
        assert!(accepts(&two, r#""\ud83d\ude00a""#));
    }

    #[test]
    fn literals_and_numbers_end_on_the_next_delimiter() {
        let grammar = JsonGrammar::from_schema(&json!({"enum": [1, 12, "x"]})).unwrap();
        assert!(accepts(&grammar, "1"));
        assert!(accepts(&grammar, "12"));
        assert!(accepts(&grammar, "\"x\""));
        assert!(!accepts(&grammar, "13"));
        let object = JsonGrammar::from_schema(&json!({
            "type": "object",
            "properties": {"n": {"type": "number"}, "e": {"enum": [1, 12]}},
            "additionalProperties": false
        }))
        .unwrap();
        assert!(accepts(&object, "{\"n\":10.25e-3,\"e\":1}"));
        assert!(accepts(&object, "{\"e\":12 }"));
        assert!(!prefix_ok(&object, "{\"e\":2"));
    }

    #[test]
    fn free_text_shortcut_reports_remaining_characters() {
        let grammar = JsonGrammar::from_schema(&json!({
            "type": "object",
            "properties": {"s": {"type": "string", "maxLength": 5}},
            "additionalProperties": false
        }))
        .unwrap();
        let mut matcher = grammar.matcher();
        assert!(matcher.advance_bytes(&grammar, b"{\"s\":\"ab"));
        assert_eq!(matcher.free_text_remaining(&grammar), Some(3));
        assert!(matcher.advance_bytes(&grammar, b"\\"));
        assert_eq!(matcher.free_text_remaining(&grammar), None);
        let free = JsonGrammar::any_object();
        let mut matcher = free.matcher();
        assert!(matcher.advance_bytes(&free, b"{\"ke"));
        assert_eq!(matcher.free_text_remaining(&free), Some(u32::MAX));
    }

    #[test]
    fn whitespace_runs_are_bounded() {
        let grammar = JsonGrammar::any_object();
        let spaces = " ".repeat(usize::from(MAX_WHITESPACE_RUN));
        assert!(prefix_ok(&grammar, &format!("{{{spaces}")));
        assert!(!prefix_ok(&grammar, &format!("{{{spaces} ")));
        // A run holds one newline, first: indentation, not blank lines.
        assert!(prefix_ok(&grammar, "{\n        \"a\":\n\t1"));
        assert!(!prefix_ok(&grammar, "{\n\n"));
        assert!(!prefix_ok(&grammar, "{ \n"));
        // Before the top-level value only a short run is accepted, and a
        // carriage return never is.
        assert!(prefix_ok(&grammar, "\n\n{"));
        assert!(!prefix_ok(&grammar, "\n\n\n"));
        assert!(!prefix_ok(&grammar, "{\r"));
        // Spaces inside strings are content.
        assert!(prefix_ok(&grammar, &format!("{{\"a\":\"{spaces}{spaces}")));
    }

    #[test]
    fn unenforceable_schemas_are_refused() {
        assert!(JsonGrammar::from_schema(&json!({"properties": {"a\"b": {}}})).is_err());
        assert!(
            JsonGrammar::from_schema(&json!({
                "type": "object", "required": ["x"], "additionalProperties": false
            }))
            .is_err()
        );
        assert!(
            JsonGrammar::from_schema(&json!({"type": "string", "minLength": 3, "maxLength": 2}))
                .is_err()
        );
    }
}
