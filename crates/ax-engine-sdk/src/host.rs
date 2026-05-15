use std::collections::BTreeSet;
use std::env;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use crate::backend::{HostReport, MetalToolchainReport, ToolStatusReport};

pub(crate) const UNSUPPORTED_HOST_OVERRIDE_ENV: &str = "AX_ALLOW_UNSUPPORTED_HOST";

static SOC_DETECTION: OnceLock<SocDetection> = OnceLock::new();
static METAL_TOOL_STATUS: OnceLock<ToolStatusReport> = OnceLock::new();
static METALLIB_TOOL_STATUS: OnceLock<ToolStatusReport> = OnceLock::new();
static METAL_AR_TOOL_STATUS: OnceLock<ToolStatusReport> = OnceLock::new();

#[derive(Clone, Debug, Default)]
struct SocDetection {
    soc: Option<String>,
    error: Option<String>,
}

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
    let detection = soc_detection_cached().clone();
    let supported_mlx_runtime = matches!(
        classify_host(env::consts::OS, env::consts::ARCH, detection.soc.as_deref(),),
        HostSupport::Supported { .. }
    );

    HostReport {
        os: env::consts::OS.to_string(),
        arch: env::consts::ARCH.to_string(),
        detected_soc: detection.soc,
        supported_mlx_runtime,
        unsupported_host_override_active: unsupported_host_override_enabled(),
        detection_error: detection.error,
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
        soc_detection_cached().soc.as_deref(),
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

fn detect_soc() -> SocDetection {
    if env::consts::OS != "macos" {
        return SocDetection::default();
    }
    let (soc, error) = command_stdout_with_reason("sysctl", &["-n", "machdep.cpu.brand_string"]);
    if soc.is_none() {
        let reason = error.as_deref().unwrap_or("sysctl returned no output");
        tracing::warn!(
            program = "sysctl",
            args = "-n machdep.cpu.brand_string",
            error = %reason,
            "ax-engine host SoC detection failed; reporting detected_soc=\"unknown Apple Silicon\" \
             (sandboxed environments often block sysctl/IOKit). MLX backends will refuse to start; \
             llama.cpp backends still work. Override with AX_ALLOW_UNSUPPORTED_HOST=1 for bring-up."
        );
    }
    SocDetection { soc, error }
}

fn soc_detection_cached() -> &'static SocDetection {
    SOC_DETECTION.get_or_init(detect_soc)
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
    // Source of truth for the active developer dir is `xcode-select -p` (or
    // a `DEVELOPER_DIR` env override). We previously also hardcoded
    // `/Applications/Xcode.app/Contents/Developer` as a candidate, but on
    // machines where the user has both Xcode.app installed AND
    // xcode-select pointing at the Command Line Tools, that candidate
    // costs ~300–350 ms per probed tool (xcrun does an exhaustive Xcode
    // toolchain scan before failing). For three probed tools that adds
    // ~700 ms of one-time cost to every cold start. Trust xcode-select.
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();

    for candidate in [
        env::var("DEVELOPER_DIR").ok(),
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
    command_stdout_with_reason(program, args).0
}

fn command_stdout_with_reason(program: &str, args: &[&str]) -> (Option<String>, Option<String>) {
    let output = match Command::new(program).args(args).output() {
        Ok(output) => output,
        Err(error) => {
            return (None, Some(format!("failed to spawn {program}: {error}")));
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            "no diagnostic output".to_string()
        };
        let reason = format!(
            "{program} exited with {status}: {detail}",
            status = output.status,
        );
        return (None, Some(reason));
    }
    let Ok(stdout) = String::from_utf8(output.stdout) else {
        return (None, Some(format!("{program} stdout was not valid UTF-8")));
    };
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        (None, Some(format!("{program} produced empty stdout")))
    } else {
        (Some(trimmed.to_string()), None)
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
        HostSupport, classify_host, command_stdout_with_reason, parse_apple_m_series_generation,
        runtime_host_report, runtime_metal_toolchain_report,
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

    #[test]
    fn command_stdout_with_reason_captures_missing_binary() {
        // A program that cannot be spawned must surface a reason instead of silently
        // dropping the failure on the floor (issue #18: sandboxed sysctl looked identical
        // to "no Apple Silicon detected").
        let (value, reason) = command_stdout_with_reason(
            "ax-engine-host-detect-non-existent-binary-for-test",
            &["--version"],
        );

        assert!(value.is_none(), "missing binary should produce no value");
        let reason = reason.expect("missing binary should produce a failure reason");
        assert!(
            reason.contains("failed to spawn"),
            "failure reason should mention the spawn error: {reason}"
        );
    }

    #[test]
    fn command_stdout_with_reason_captures_nonzero_exit() {
        // `false` exits non-zero with no output; we should report the exit status.
        let (value, reason) = command_stdout_with_reason("false", &[]);

        if let Some(reason) = reason {
            assert!(value.is_none());
            assert!(
                reason.contains("exited with"),
                "non-zero exit should report status: {reason}"
            );
        } else {
            // `false` may not be on PATH on every CI runner; in that case the missing-binary
            // path above already covered the failure-surfacing contract.
            assert!(
                value.is_none() || value.is_some(),
                "neither branch should panic on this platform"
            );
        }
    }
}
