//! `response_format: json_schema` Phase A: request-shape parsing and post-hoc
//! output validation against a fail-closed JSON Schema subset (ADR-040 D3).
//!
//! The subset is an allowlist. A schema using any keyword outside it rejects
//! the request up front — partial validation must never masquerade as full
//! validation. Constrained decoding (Phase B) will reuse this request shape.

use std::sync::Arc;

use axum::Json;
use axum::http::StatusCode;
use serde_json::{Map, Value};

use crate::errors::{ErrorResponse, error_response};

/// Schema keywords the Phase A validator enforces.
const SUPPORTED_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
];

/// Annotation keywords accepted but not enforced.
const ANNOTATION_KEYWORDS: &[&str] = &["title", "description", "$schema", "default"];

#[derive(Clone, Debug)]
pub(crate) struct JsonSchemaContract {
    pub(crate) name: Option<String>,
    pub(crate) schema: Value,
}

/// Recognize `{"type":"json_schema","json_schema":{"name"?,"schema","strict"?}}`.
/// Returns `Ok(None)` for absent/other response formats (`text`, `json_object`),
/// a 400 for a malformed `json_schema` wrapper, and a 400
/// `unsupported_json_schema` for a schema outside the supported subset.
pub(crate) fn parse_json_schema_response_format(
    response_format: Option<&Value>,
) -> Result<Option<Arc<JsonSchemaContract>>, (StatusCode, Json<ErrorResponse>)> {
    let Some(Value::Object(object)) = response_format else {
        return Ok(None);
    };
    if object.get("type").and_then(Value::as_str).map(str::trim) != Some("json_schema") {
        return Ok(None);
    }
    let Some(wrapper) = object.get("json_schema").and_then(Value::as_object) else {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "response_format type json_schema requires a json_schema object with a schema field"
                .to_string(),
        ));
    };
    let Some(schema) = wrapper.get("schema") else {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "response_format json_schema is missing the schema field".to_string(),
        ));
    };
    validate_schema_supported(schema, "").map_err(|message| {
        error_response(StatusCode::BAD_REQUEST, "unsupported_json_schema", message)
    })?;
    Ok(Some(Arc::new(JsonSchemaContract {
        name: wrapper
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string),
        schema: schema.clone(),
    })))
}

/// Walk the schema and reject any keyword outside the Phase A subset.
fn validate_schema_supported(schema: &Value, path: &str) -> Result<(), String> {
    let object = match schema {
        Value::Object(object) => object,
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => {
            return Err(format!(
                "schema{path}: the `false` schema is not supported; use additionalProperties or enum instead"
            ));
        }
        _ => return Err(format!("schema{path}: expected a schema object")),
    };
    for (keyword, value) in object {
        let keyword = keyword.as_str();
        if ANNOTATION_KEYWORDS.contains(&keyword) {
            continue;
        }
        if !SUPPORTED_KEYWORDS.contains(&keyword) {
            return Err(format!(
                "schema{path}: keyword {keyword} is not in the supported json_schema subset; \
                 supported keywords: {}",
                SUPPORTED_KEYWORDS.join(", ")
            ));
        }
        match keyword {
            "properties" => {
                let Value::Object(properties) = value else {
                    return Err(format!("schema{path}/properties: expected an object"));
                };
                for (name, subschema) in properties {
                    validate_schema_supported(subschema, &format!("{path}/properties/{name}"))?;
                }
            }
            "items" => validate_schema_supported(value, &format!("{path}/items"))?,
            "additionalProperties" if !value.is_boolean() => {
                return Err(format!(
                    "schema{path}/additionalProperties: only boolean values are supported"
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Validate an output value against the schema. First mismatch wins; the
/// error message carries a `/a/b[2]`-style path into the value.
pub(crate) fn validate_value(schema: &Value, value: &Value) -> Result<(), String> {
    validate_value_at(schema, value, "")
}

fn validate_value_at(schema: &Value, value: &Value, path: &str) -> Result<(), String> {
    let object = match schema {
        Value::Object(object) => object,
        // validate_schema_supported already rejected everything else.
        _ => return Ok(()),
    };
    let display_path = || if path.is_empty() { "/" } else { path };

    if let Some(expected) = object.get("type") {
        validate_type(expected, value, display_path())?;
    }
    if let Some(constant) = object.get("const")
        && value != constant
    {
        return Err(format!("output{}: does not equal const", display_path()));
    }
    if let Some(Value::Array(options)) = object.get("enum")
        && !options.iter().any(|option| option == value)
    {
        return Err(format!("output{}: is not one of enum", display_path()));
    }

    if let Some(number) = value.as_f64() {
        validate_numeric_bounds(object, number, display_path())?;
    }
    if let Some(text) = value.as_str() {
        let chars = text.chars().count() as u64;
        if let Some(min) = keyword_u64(object, "minLength")
            && chars < min
        {
            return Err(format!(
                "output{}: string is shorter than minLength {min}",
                display_path()
            ));
        }
        if let Some(max) = keyword_u64(object, "maxLength")
            && chars > max
        {
            return Err(format!(
                "output{}: string is longer than maxLength {max}",
                display_path()
            ));
        }
    }

    if let Some(items) = value.as_array() {
        if let Some(min) = keyword_u64(object, "minItems")
            && (items.len() as u64) < min
        {
            return Err(format!(
                "output{}: array has fewer than minItems {min}",
                display_path()
            ));
        }
        if let Some(max) = keyword_u64(object, "maxItems")
            && (items.len() as u64) > max
        {
            return Err(format!(
                "output{}: array has more than maxItems {max}",
                display_path()
            ));
        }
        if let Some(item_schema) = object.get("items") {
            for (index, item) in items.iter().enumerate() {
                validate_value_at(item_schema, item, &format!("{path}[{index}]"))?;
            }
        }
    }

    if let Some(fields) = value.as_object() {
        validate_object_fields(object, fields, path, display_path())?;
    }

    Ok(())
}

fn validate_object_fields(
    schema: &Map<String, Value>,
    fields: &Map<String, Value>,
    path: &str,
    display_path: &str,
) -> Result<(), String> {
    if let Some(Value::Array(required)) = schema.get("required") {
        for name in required.iter().filter_map(Value::as_str) {
            if !fields.contains_key(name) {
                return Err(format!(
                    "output{display_path}: missing required property {name}"
                ));
            }
        }
    }
    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (name, subschema) in properties {
            if let Some(field) = fields.get(name) {
                validate_value_at(subschema, field, &format!("{path}/{name}"))?;
            }
        }
    }
    if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
        for name in fields.keys() {
            if !properties.is_some_and(|properties| properties.contains_key(name)) {
                return Err(format!(
                    "output{display_path}: unexpected additional property {name}"
                ));
            }
        }
    }
    Ok(())
}

fn validate_type(expected: &Value, value: &Value, display_path: &str) -> Result<(), String> {
    let matches = |name: &str| match name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        // A mathematical integer: 2.0 passes, 2.5 does not.
        "integer" => {
            value.as_i64().is_some()
                || value.as_u64().is_some()
                || value.as_f64().is_some_and(|number| number.fract() == 0.0)
        }
        _ => false,
    };
    let accepted = match expected {
        Value::String(name) => matches(name),
        Value::Array(names) => names.iter().filter_map(Value::as_str).any(matches),
        _ => true,
    };
    if accepted {
        Ok(())
    } else {
        Err(format!(
            "output{display_path}: value does not match type {expected}"
        ))
    }
}

fn validate_numeric_bounds(
    schema: &Map<String, Value>,
    number: f64,
    display_path: &str,
) -> Result<(), String> {
    if let Some(min) = keyword_f64(schema, "minimum")
        && number < min
    {
        return Err(format!(
            "output{display_path}: {number} is below minimum {min}"
        ));
    }
    if let Some(max) = keyword_f64(schema, "maximum")
        && number > max
    {
        return Err(format!(
            "output{display_path}: {number} is above maximum {max}"
        ));
    }
    if let Some(min) = keyword_f64(schema, "exclusiveMinimum")
        && number <= min
    {
        return Err(format!(
            "output{display_path}: {number} is not above exclusiveMinimum {min}"
        ));
    }
    if let Some(max) = keyword_f64(schema, "exclusiveMaximum")
        && number >= max
    {
        return Err(format!(
            "output{display_path}: {number} is not below exclusiveMaximum {max}"
        ));
    }
    Ok(())
}

fn keyword_f64(schema: &Map<String, Value>, keyword: &str) -> Option<f64> {
    schema.get(keyword).and_then(Value::as_f64)
}

fn keyword_u64(schema: &Map<String, Value>, keyword: &str) -> Option<u64> {
    schema.get(keyword).and_then(Value::as_u64)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(format: Value) -> Result<Option<Arc<JsonSchemaContract>>, String> {
        parse_json_schema_response_format(Some(&format))
            .map_err(|(_, Json(error))| error.error.message)
    }

    #[test]
    fn non_json_schema_formats_pass_through() {
        assert!(parse(json!({"type": "json_object"})).expect("ok").is_none());
        assert!(parse(json!({"type": "text"})).expect("ok").is_none());
        assert!(
            parse_json_schema_response_format(None)
                .expect("ok")
                .is_none()
        );
    }

    #[test]
    fn parses_openai_json_schema_shape() {
        let contract = parse(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": true,
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
            }
        }))
        .expect("ok")
        .expect("contract");
        assert_eq!(contract.name.as_deref(), Some("answer"));
    }

    #[test]
    fn missing_schema_is_invalid_request() {
        let error = parse(json!({"type": "json_schema", "json_schema": {"name": "x"}}))
            .expect_err("missing schema must fail");
        assert!(error.contains("schema"), "{error}");
    }

    #[test]
    fn unsupported_keyword_fails_closed_with_name_and_path() {
        let error = parse(json!({
            "type": "json_schema",
            "json_schema": {"schema": {
                "type": "object",
                "properties": {"x": {"pattern": "^a"}}
            }}
        }))
        .expect_err("pattern is outside the subset");
        assert!(error.contains("pattern"), "{error}");
        assert!(error.contains("/properties/x"), "{error}");
    }

    #[test]
    fn non_boolean_additional_properties_fails_closed() {
        let error = parse(json!({
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object", "additionalProperties": {"type": "string"}}}
        }))
        .expect_err("schema-valued additionalProperties is unsupported");
        assert!(error.contains("additionalProperties"), "{error}");
    }

    #[test]
    fn annotations_are_accepted() {
        parse(json!({
            "type": "json_schema",
            "json_schema": {"schema": {
                "type": "object",
                "title": "T",
                "description": "D",
                "properties": {"x": {"type": "string", "description": "field"}}
            }}
        }))
        .expect("annotations must not be rejected")
        .expect("contract");
    }

    fn check(schema: Value, value: Value) -> Result<(), String> {
        validate_value(&schema, &value)
    }

    #[test]
    fn validates_types_required_and_nesting() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "minimum": 0},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2}
            },
            "required": ["name", "count"],
            "additionalProperties": false
        });
        check(
            schema.clone(),
            json!({"name": "a", "count": 2, "tags": ["x"]}),
        )
        .expect("valid value");
        assert!(
            check(schema.clone(), json!({"name": "a"}))
                .expect_err("missing required")
                .contains("count")
        );
        assert!(
            check(schema.clone(), json!({"name": "a", "count": 2.5}))
                .expect_err("non-integer")
                .contains("/count")
        );
        assert!(
            check(schema.clone(), json!({"name": "a", "count": 1, "extra": 1}))
                .expect_err("additional property")
                .contains("extra")
        );
        assert!(
            check(
                schema,
                json!({"name": "a", "count": 1, "tags": ["x", "y", "z"]})
            )
            .expect_err("too many items")
            .contains("maxItems")
        );
    }

    #[test]
    fn integer_accepts_mathematical_integers() {
        check(json!({"type": "integer"}), json!(2.0)).expect("2.0 is an integer");
        check(json!({"type": "integer"}), json!(-3)).expect("-3 is an integer");
        check(json!({"type": "integer"}), json!(2.5)).expect_err("2.5 is not an integer");
    }

    #[test]
    fn enum_and_const_match_by_equality() {
        check(json!({"enum": ["a", "b"]}), json!("a")).expect("enum member");
        check(json!({"enum": ["a", "b"]}), json!("c")).expect_err("not an enum member");
        check(json!({"const": {"k": 1}}), json!({"k": 1})).expect("const equality");
        check(json!({"const": 1}), json!(2)).expect_err("const mismatch");
    }

    #[test]
    fn array_item_mismatch_reports_index_path() {
        let error = check(
            json!({"type": "array", "items": {"type": "integer"}}),
            json!([1, "x"]),
        )
        .expect_err("string in integer array");
        assert!(error.contains("[1]"), "{error}");
    }

    #[test]
    fn type_arrays_accept_any_listed_type() {
        let schema = json!({"type": ["string", "null"]});
        check(schema.clone(), json!("a")).expect("string accepted");
        check(schema.clone(), Value::Null).expect("null accepted");
        check(schema, json!(1)).expect_err("number rejected");
    }
}
