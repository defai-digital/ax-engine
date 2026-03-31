use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, anyhow, bail};
use ax_engine_core::model::arch_registry::{
    NativeSupportLevel, NativeSupportResult, check_native_support,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackendKind {
    Native,
    LlamaCpp,
}

impl InferenceBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::LlamaCpp => "llama_cpp",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingPreview {
    pub backend: InferenceBackendKind,
    pub architecture: String,
    pub reason: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum RoutingPreference {
    Auto,
    Native,
    LlamaCpp,
    #[default]
    Off,
}

#[derive(Debug, Clone)]
pub(crate) struct RoutingDecision {
    pub(crate) backend: InferenceBackendKind,
    pub(crate) support: NativeSupportResult,
    pub(crate) reason: Option<String>,
    pub(crate) source: &'static str,
}

#[derive(Debug, Default)]
pub(crate) struct RoutingPolicy {
    global: RoutingPreference,
    arch_overrides: HashMap<String, RoutingPreference>,
    model_overrides: HashMap<String, RoutingPreference>,
}

impl FromStr for RoutingPreference {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "native" => Ok(Self::Native),
            "llama_cpp" | "llama.cpp" | "llama-cpp" | "llama" => Ok(Self::LlamaCpp),
            other => Err(anyhow!(
                "unsupported routing backend '{other}'; expected auto, native, llama_cpp, or off"
            )),
        }
    }
}

pub fn preview_routing(path: impl AsRef<Path>) -> anyhow::Result<RoutingPreview> {
    let decision = RoutingPolicy::from_env()?.resolve(path.as_ref())?;
    Ok(RoutingPreview {
        backend: decision.backend,
        architecture: decision.support.architecture.clone(),
        reason: decision.reason.clone(),
        source: decision.source.to_string(),
    })
}

impl RoutingPolicy {
    pub(crate) fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            global: std::env::var("AX_ROUTING")
                .ok()
                .as_deref()
                .map(RoutingPreference::from_str)
                .transpose()?
                .unwrap_or_default(),
            arch_overrides: parse_overrides(std::env::var("AX_ROUTING_ARCH").ok(), |key| {
                key.trim().to_ascii_lowercase()
            })?,
            model_overrides: parse_overrides(std::env::var("AX_ROUTING_MODEL").ok(), |key| {
                normalize_model_key(key)
            })?,
        })
    }

    pub(crate) fn resolve(&self, path: &Path) -> anyhow::Result<RoutingDecision> {
        let support = check_native_support(path)
            .with_context(|| format!("failed to inspect native support for {}", path.display()))?;
        let model_key = normalize_model_key(path.display().to_string());

        if let Some(preference) = self.model_overrides.get(&model_key) {
            return resolve_preference(*preference, support, "model_override");
        }

        if let Some(preference) = self
            .arch_overrides
            .get(&support.architecture.to_ascii_lowercase())
        {
            return resolve_preference(*preference, support, "arch_override");
        }

        resolve_preference(self.global, support, "global")
    }
}

fn parse_overrides(
    raw: Option<String>,
    normalize_key: impl Fn(String) -> String,
) -> anyhow::Result<HashMap<String, RoutingPreference>> {
    let mut parsed = HashMap::new();
    let Some(raw) = raw else {
        return Ok(parsed);
    };

    for entry in raw
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (key, value) = entry
            .split_once('=')
            .ok_or_else(|| anyhow!("invalid routing override '{entry}'; expected key=value"))?;
        let preference = RoutingPreference::from_str(value)?;
        parsed.insert(normalize_key(key.trim().to_string()), preference);
    }

    Ok(parsed)
}

fn normalize_model_key(input: impl Into<String>) -> String {
    let input = input.into();
    let path = Path::new(&input);
    std::fs::canonicalize(path)
        .map(|canonical| canonical.display().to_string())
        .unwrap_or(input)
}

fn resolve_preference(
    preference: RoutingPreference,
    support: NativeSupportResult,
    source: &'static str,
) -> anyhow::Result<RoutingDecision> {
    match preference {
        RoutingPreference::Auto => {
            if matches!(support.level, NativeSupportLevel::Full) {
                Ok(RoutingDecision {
                    backend: InferenceBackendKind::Native,
                    support,
                    reason: None,
                    source,
                })
            } else {
                let reason = unsupported_reason(&support.level);
                Ok(RoutingDecision {
                    backend: InferenceBackendKind::LlamaCpp,
                    support,
                    reason: Some(reason),
                    source,
                })
            }
        }
        RoutingPreference::LlamaCpp => Ok(RoutingDecision {
            backend: InferenceBackendKind::LlamaCpp,
            support,
            reason: Some(format!("forced by {source}")),
            source,
        }),
        RoutingPreference::Native | RoutingPreference::Off => {
            if matches!(support.level, NativeSupportLevel::Full) {
                return Ok(RoutingDecision {
                    backend: InferenceBackendKind::Native,
                    support,
                    reason: None,
                    source,
                });
            }

            let reason = unsupported_reason(&support.level);
            bail!("{reason}. Hint: set AX_ROUTING=auto to enable llama.cpp fallback");
        }
    }
}

fn unsupported_reason(level: &NativeSupportLevel) -> String {
    match level {
        NativeSupportLevel::Full => "native support available".to_string(),
        NativeSupportLevel::UnsupportedArch { arch } => {
            format!("architecture '{arch}' is not natively supported by ax-engine")
        }
        NativeSupportLevel::PartialQuant { unsupported_types } => format!(
            "model uses unsupported quantization types for native ax-engine: {}",
            unsupported_types
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_preference_parses_known_values() {
        assert_eq!(
            "off".parse::<RoutingPreference>().unwrap(),
            RoutingPreference::Off
        );
        assert_eq!(
            "auto".parse::<RoutingPreference>().unwrap(),
            RoutingPreference::Auto
        );
        assert_eq!(
            "llama_cpp".parse::<RoutingPreference>().unwrap(),
            RoutingPreference::LlamaCpp
        );
    }

    #[test]
    fn test_parse_overrides_parses_pairs() {
        let parsed =
            parse_overrides(Some("mistral=llama_cpp,qwen3=native".to_string()), |k| k).unwrap();
        assert_eq!(parsed["mistral"], RoutingPreference::LlamaCpp);
        assert_eq!(parsed["qwen3"], RoutingPreference::Native);
    }
}
