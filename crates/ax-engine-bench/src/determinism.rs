use serde_json::{Value, json};

pub(crate) fn deterministic_runtime_digest(value: &Value) -> Value {
    json!({
        "requests": value.get("requests").cloned().unwrap_or(Value::Null)
    })
}
