use std::collections::BTreeSet;
use std::env;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use crate::backend::{HostReport, MetalToolchainReport, ToolStatusReport};

pub(crate) const UNSUPPORTED_HOST_OVERRIDE_ENV: &str = "AX_ALLOW_UNSUPPORTED_HOST";

static DETECTED_SOC: OnceLock<Option<String>> = OnceLock::new();
static METAL_TOOL_STATUS: OnceLock<ToolStatusReport> = OnceLock::new();
static METALLIB_TOOL_STATUS: OnceLock<ToolStatusReport> = OnceLock::new();
static METAL_AR_TOOL_STATUS: OnceLock<ToolStatusReport> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum HostSupport {
    Supported { detected_host: String },
    Unsupported { detected_host: String },
}

pub(crate) fn validate_local_host() -> Result<(), String> {
    if unsupported_host_override_enabled() {
        return Ok(());
    }

    match detect_local_host_support() {
        HostSupport::Supported { .. } => Ok(()),
        HostSupport::Unsupported { detected_host } => Err(detected_host),
    }
}

pub(crate) fn runtime_host_report() -> HostReport {
    let detected_soc = detected_soc_cached().clone();
    let supported_mlx_runtime = matches!(
        classify_host(env::consts::OS, env::consts::ARCH, detected_soc.as_deref()),
        HostSupport::Supported { .. }
    );

    HostReport {
        os: env::consts::OS.to_string(),
        arch: env::consts::ARCH.to_string(),
        detected_soc,
        supported_mlx_runtime,
        unsupported_host_override_active: unsupported_host_override_enabled(),
    }
}

pub(crate) fn runtime_metal_toolchain_report() -> MetalToolchainReport {
    let metal = cached_tool_status(&METAL_TOOL_STATUS, "metal");
    let metallib = cached_tool_status(&METALLIB_TOOL_STATUS, "metallib");
    let metal_ar = cached_tool_status(&METAL_AR_TOOL_STATUS, "metal-ar");
    let fully_available = metal.available && metallib.available;

    MetalToolchainReport {
        fully_available,
        metal,
        metallib,
        metal_ar,
    }
}

fn detect_local_host_support() -> HostSupport {
    classify_host(
        env::consts::OS,
        env::consts::ARCH,
        detected_soc_cached().as_deref(),
    )
}

fn classify_host(os: &str, arch: &str, soc: Option<&str>) -> HostSupport {
    if os != "macos" || arch != "aarch64" {
        return HostSupport::Unsupported {
            detected_host: format!("{os}/{arch}"),
        };
    }

    let Some(soc) = soc.map(str::trim).filter(|soc| !soc.is_empty()) else {
        return HostSupport::Unsupported {
            detected_host: "unknown Apple Silicon".to_string(),
        };
    };

    let Some(generation) = parse_apple_m_series_generation(soc) else {
        return HostSupport::Unsupported {
            detected_host: soc.to_string(),
        };
    };

    if generation >= 2 {
        HostSupport::Supported {
            detected_host: soc.to_string(),
        }
    } else {
        HostSupport::Unsupported {
            detected_host: soc.to_string(),
        }
    }
}

fn parse_apple_m_series_generation(soc: &str) -> Option<u32> {
    let digits = soc
        .trim()
        .strip_prefix("Apple M")?
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn detect_soc() -> Option<String> {
    if env::consts::OS != "macos" {
        return None;
    }
    command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
}

fn detected_soc_cached() -> &'static Option<String> {
    DETECTED_SOC.get_or_init(detect_soc)
}

fn cached_tool_status(cache: &'static OnceLock<ToolStatusReport>, tool: &str) -> ToolStatusReport {
    cache.get_or_init(|| detect_xcrun_tool(tool)).clone()
}

fn detect_xcrun_tool(tool: &str) -> ToolStatusReport {
    if env::consts::OS != "macos" {
        return ToolStatusReport::default();
    }

    for developer_dir in xcrun_developer_dir_candidates() {
        let mut command = Command::new("xcrun");
        command.args([tool, "--version"]);
        if let Some(developer_dir) = developer_dir.as_deref() {
            command.env("DEVELOPER_DIR", developer_dir);
        }

        let Ok(output) = command.output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }

        return ToolStatusReport {
            available: true,
            version: first_non_empty_output(&output.stdout, &output.stderr),
        };
    }

    ToolStatusReport::default()
}

fn xcrun_developer_dir_candidates() -> Vec<Option<String>> {
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();

    for candidate in [
        env::var("DEVELOPER_DIR").ok(),
        Some("/Applications/Xcode.app/Contents/Developer".to_string())
            .filter(|path| Path::new(path).is_dir()),
        command_stdout("xcode-select", &["-p"]).filter(|path| Path::new(path).is_dir()),
    ] {
        let Some(candidate) = candidate else {
            continue;
        };
        if seen.insert(candidate.clone()) {
            candidates.push(Some(candidate));
        }
    }

    candidates.push(None);
    candidates
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn first_non_empty_output(stdout: &[u8], stderr: &[u8]) -> Option<String> {
    decode_non_empty(stdout).or_else(|| decode_non_empty(stderr))
}

fn decode_non_empty(bytes: &[u8]) -> Option<String> {
    let text = String::from_utf8(bytes.to_vec()).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn unsupported_host_override_enabled() -> bool {
    env::var(UNSUPPORTED_HOST_OVERRIDE_ENV)
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::{
        HostSupport, classify_host, parse_apple_m_series_generation, runtime_host_report,
        runtime_metal_toolchain_report,
    };

    #[test]
    fn parses_apple_m_series_generation() {
        assert_eq!(parse_apple_m_series_generation("Apple M4"), Some(4));
        assert_eq!(parse_apple_m_series_generation("Apple M5 Max"), Some(5));
        assert_eq!(parse_apple_m_series_generation("Apple M10 Ultra"), Some(10));
        assert_eq!(parse_apple_m_series_generation("Apple M3 Pro"), Some(3));
        assert_eq!(parse_apple_m_series_generation("Intel Core i9"), None);
        assert_eq!(parse_apple_m_series_generation("unknown"), None);
    }

    #[test]
    fn classifies_supported_m2_or_newer_host() {
        assert_eq!(
            classify_host("macos", "aarch64", Some("Apple M2 Max")),
            HostSupport::Supported {
                detected_host: "Apple M2 Max".to_string(),
            }
        );
        assert_eq!(
            classify_host("macos", "aarch64", Some("Apple M3 Max")),
            HostSupport::Supported {
                detected_host: "Apple M3 Max".to_string(),
            }
        );
        assert_eq!(
            classify_host("macos", "aarch64", Some("Apple M4 Max")),
            HostSupport::Supported {
                detected_host: "Apple M4 Max".to_string(),
            }
        );
    }

    #[test]
    fn classifies_m1_host_as_unsupported() {
        assert_eq!(
            classify_host("macos", "aarch64", Some("Apple M1 Pro")),
            HostSupport::Unsupported {
                detected_host: "Apple M1 Pro".to_string(),
            }
        );
    }

    #[test]
    fn classifies_non_macos_hosts_as_unsupported() {
        assert_eq!(
            classify_host("linux", "aarch64", Some("Apple M4 Max")),
            HostSupport::Unsupported {
                detected_host: "linux/aarch64".to_string(),
            }
        );
    }

    #[test]
    fn classifies_unknown_soc_as_unsupported() {
        assert_eq!(
            classify_host("macos", "aarch64", None),
            HostSupport::Unsupported {
                detected_host: "unknown Apple Silicon".to_string(),
            }
        );
        assert_eq!(
            classify_host("macos", "aarch64", Some("Apple Silicon")),
            HostSupport::Unsupported {
                detected_host: "Apple Silicon".to_string(),
            }
        );
    }

    #[test]
    fn runtime_host_report_matches_current_platform() {
        let report = runtime_host_report();

        assert_eq!(report.os, std::env::consts::OS);
        assert_eq!(report.arch, std::env::consts::ARCH);
    }

    #[test]
    fn runtime_metal_toolchain_report_is_self_consistent() {
        let report = runtime_metal_toolchain_report();

        assert_eq!(
            report.fully_available,
            report.metal.available && report.metallib.available
        );
    }
}
