use serde_json::{Value, json};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

#[derive(Clone, Copy)]
struct ModelProfile {
    label: &'static str,
    preset: Option<&'static str>,
    repo_id: &'static str,
    aliases: &'static [&'static str],
    downloadable: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MtpDownloadKind {
    QwenSidecar {
        mtp_source: &'static str,
    },
    GemmaAssistant {
        assistant_repo_id: &'static str,
        target_model_id: &'static str,
        assistant_model_id: &'static str,
        max_depth: u32,
    },
    GlmSidecar {
        mtp_source: &'static str,
    },
    /// Fallback for models where no MTP sidecar or assistant packager is available.
    #[allow(dead_code)]
    DirectOnly {
        reason: &'static str,
    },
}

#[derive(Clone, Copy, Debug)]
struct MtpDownloadTarget {
    label: &'static str,
    repo_id: &'static str,
    aliases: &'static [&'static str],
    kind: MtpDownloadKind,
}

const MODEL_PROFILES: &[ModelProfile] = &[
    ModelProfile {
        label: "gemma4-e2b",
        preset: Some("gemma4-e2b"),
        repo_id: "mlx-community/gemma-4-e2b-it-4bit",
        aliases: &[
            "gemma4-e2b",
            "gemma-4-e2b",
            "gemma-4-e2b-it",
            "gemma4-e2b-4bit",
        ],
        downloadable: true,
    },
    ModelProfile {
        label: "gemma4-e2b-5bit",
        preset: None,
        repo_id: "mlx-community/gemma-4-e2b-it-5bit",
        aliases: &["gemma4-e2b-5bit", "gemma-4-e2b-5bit", "gemma-4-e2b-it-5bit"],
        downloadable: true,
    },
    ModelProfile {
        label: "gemma4-e2b-6bit",
        preset: None,
        repo_id: "mlx-community/gemma-4-e2b-it-6bit",
        aliases: &["gemma4-e2b-6bit", "gemma-4-e2b-6bit", "gemma-4-e2b-it-6bit"],
        downloadable: true,
    },
    ModelProfile {
        label: "gemma4-e2b-8bit",
        preset: None,
        repo_id: "mlx-community/gemma-4-e2b-it-8bit",
        aliases: &["gemma4-e2b-8bit", "gemma-4-e2b-8bit", "gemma-4-e2b-it-8bit"],
        downloadable: true,
    },
    ModelProfile {
        label: "gemma4-12b",
        preset: Some("gemma4-12b"),
        repo_id: "mlx-community/gemma-4-12B-it-4bit",
        aliases: &[
            "gemma4-12b",
            "gemma-4-12b",
            "gemma-4-12b-it",
            "gemma4-12b-4bit",
        ],
        downloadable: true,
    },
    ModelProfile {
        label: "gemma4-12b-6bit",
        preset: None,
        repo_id: "mlx-community/gemma-4-12B-it-6bit",
        aliases: &["gemma4-12b-6bit", "gemma-4-12b-6bit", "gemma-4-12b-it-6bit"],
        downloadable: true,
    },
    ModelProfile {
        label: "gemma4-26b",
        preset: Some("gemma4-26b"),
        repo_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
        aliases: &[
            "gemma4-26b",
            "gemma-4-26b",
            "gemma-4-26b-a4b-it",
            "gemma4-26b-4bit",
        ],
        downloadable: true,
    },
    ModelProfile {
        label: "gemma4-31b",
        preset: Some("gemma4-31b"),
        repo_id: "mlx-community/gemma-4-31b-it-4bit",
        aliases: &[
            "gemma4-31b",
            "gemma-4-31b",
            "gemma-4-31b-it",
            "gemma4-31b-4bit",
        ],
        downloadable: true,
    },
    ModelProfile {
        label: "glm4.7-flash-4bit",
        preset: Some("glm4.7-flash-4bit"),
        repo_id: "mlx-community/GLM-4.7-Flash-4bit",
        aliases: &[
            "glm4.7-flash-4bit",
            "glm47-flash-4bit",
            "glm4-moe-lite",
            "glm4_moe_lite",
            "glm-4.7-flash-4bit",
            "glm-4-7-flash-4bit",
        ],
        downloadable: false,
    },
    ModelProfile {
        label: "qwen3.6-27b",
        preset: None,
        repo_id: "mlx-community/Qwen3.6-27B-4bit",
        aliases: &[
            "qwen3.6-27b",
            "qwen36-27b",
            "qwen3-6-27b",
            "qwen3.6-27b-4bit",
            "qwen36-27b-4bit",
        ],
        downloadable: true,
    },
    ModelProfile {
        label: "qwen3.6-27b-5bit",
        preset: None,
        repo_id: "mlx-community/Qwen3.6-27B-5bit",
        aliases: &["qwen3.6-27b-5bit", "qwen36-27b-5bit", "qwen3-6-27b-5bit"],
        downloadable: true,
    },
    ModelProfile {
        label: "qwen3.6-27b-6bit",
        preset: None,
        repo_id: "mlx-community/Qwen3.6-27B-6bit",
        aliases: &["qwen3.6-27b-6bit", "qwen36-27b-6bit", "qwen3-6-27b-6bit"],
        downloadable: true,
    },
    ModelProfile {
        label: "qwen3.6-27b-8bit",
        preset: None,
        repo_id: "mlx-community/Qwen3.6-27B-8bit",
        aliases: &["qwen3.6-27b-8bit", "qwen36-27b-8bit", "qwen3-6-27b-8bit"],
        downloadable: true,
    },
    ModelProfile {
        label: "qwen3.6-35b",
        preset: Some("qwen3.6-35b"),
        repo_id: "mlx-community/Qwen3.6-35B-A3B-4bit",
        aliases: &[
            "qwen3.6-35b",
            "qwen36-35b",
            "qwen3-6-35b",
            "qwen3.6-35b-a3b",
            "qwen36-35b-a3b",
        ],
        downloadable: true,
    },
];

const MTP_DOWNLOAD_TARGETS: &[MtpDownloadTarget] = &[
    MtpDownloadTarget {
        label: "qwen3.6-27b-6bit",
        repo_id: "mlx-community/Qwen3.6-27B-6bit",
        aliases: &[
            "qwen3.6-27b-6bit",
            "qwen36-27b-6bit",
            "qwen3-6-27b-6bit",
            "qwen3.6-27b",
            "qwen36-27b",
        ],
        kind: MtpDownloadKind::QwenSidecar {
            mtp_source: "Qwen/Qwen3.6-27B",
        },
    },
    MtpDownloadTarget {
        label: "qwen3.6-35b-a3b",
        repo_id: "mlx-community/Qwen3.6-35B-A3B-6bit",
        aliases: &[
            "qwen3.6-35b-a3b",
            "qwen3.6-35b-a3b-6bit",
            "qwen36-35b-a3b",
            "qwen36-35b",
            "qwen3.6-35b",
        ],
        kind: MtpDownloadKind::QwenSidecar {
            mtp_source: "Qwen/Qwen3.6-35B-A3B",
        },
    },
    MtpDownloadTarget {
        label: "gemma-4-12b",
        repo_id: "mlx-community/gemma-4-12B-it-6bit",
        aliases: &[
            "gemma-4-12b",
            "gemma-4-12b-it",
            "gemma-4-12b-6bit",
            "gemma4-12b",
            "gemma4-12b-6bit",
        ],
        kind: MtpDownloadKind::GemmaAssistant {
            assistant_repo_id: "mlx-community/gemma-4-12B-it-assistant-6bit",
            target_model_id: "gemma-4-12b-it",
            assistant_model_id: "gemma-4-12b-it-assistant",
            max_depth: 2,
        },
    },
    MtpDownloadTarget {
        label: "gemma-4-26b",
        repo_id: "mlx-community/gemma-4-26b-a4b-it-6bit",
        aliases: &[
            "gemma-4-26b",
            "gemma-4-26b-a4b",
            "gemma-4-26b-a4b-it",
            "gemma-4-26b-6bit",
            "gemma4-26b",
            "gemma4-26b-6bit",
        ],
        kind: MtpDownloadKind::GemmaAssistant {
            assistant_repo_id: "google/gemma-4-26b-a4b-it-assistant",
            target_model_id: "gemma-4-26b-a4b-it",
            assistant_model_id: "gemma-4-26b-a4b-it-assistant",
            max_depth: 1,
        },
    },
    MtpDownloadTarget {
        label: "gemma-4-31b",
        repo_id: "mlx-community/gemma-4-31b-it-6bit",
        aliases: &[
            "gemma-4-31b",
            "gemma-4-31b-it",
            "gemma-4-31b-6bit",
            "gemma4-31b",
            "gemma4-31b-6bit",
        ],
        kind: MtpDownloadKind::GemmaAssistant {
            assistant_repo_id: "google/gemma-4-31b-it-assistant",
            target_model_id: "gemma-4-31b-it",
            assistant_model_id: "gemma-4-31b-it-assistant",
            max_depth: 1,
        },
    },
    MtpDownloadTarget {
        label: "glm-4.7-flash",
        repo_id: "mlx-community/GLM-4.7-Flash-6bit",
        aliases: &[
            "glm-4.7-flash",
            "glm-4.7-flash-6bit",
            "glm4.7-flash",
            "glm47-flash",
            "glm-4-7-flash",
        ],
        kind: MtpDownloadKind::GlmSidecar {
            mtp_source: "zai-org/GLM-4.7-Flash",
        },
    },
];

fn main() -> ExitCode {
    match run(env::args_os().skip(1).collect()) {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}

fn run(args: Vec<OsString>) -> Result<u8, String> {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print_usage();
        return Ok(0);
    }
    match args[0].to_string_lossy().as_ref() {
        "serve" => cmd_serve(&args[1..]),
        "download" => cmd_download(&args[1..]),
        "download-mtp" => cmd_download_mtp(&args[1..]),
        "models" => cmd_models(&args[1..]),
        "doctor" => cmd_doctor(&args[1..]),
        "convert-mtplx" => cmd_convert_mtplx(&args[1..]),
        unknown => Err(format!(
            "unknown command: {unknown}\n\nRun `ax-engine --help` for usage."
        )),
    }
}

fn print_usage() {
    println!(
        "Usage:\n  ax-engine serve <model-dir-or-alias> [--host <host>] [--port <port>] [--download] [--dry-run] [--json] [-- <ax-engine-server args>]\n  ax-engine download [<alias-or-repo-id>] [--dest <path>] [--force] [--list] [--json]\n  ax-engine download-mtp <6bit-mtp-target> [--output <dir>] [--force] [--quantize 4|8] [--mtp-depth-max <n>] [--group-size <n>] [--fair-base-only] [--json]\n  ax-engine models list [--models-dir <path>] [--json]\n  ax-engine models info <alias-or-path> [--json]\n  ax-engine models rm <path> [--dry-run] [--yes] [--json]\n  ax-engine doctor [--json] [--verbose] [--mlx-model-artifacts-dir <path>]\n  ax-engine convert-mtplx <base-model> --mtp-source <repo> [--output <dir>] [--quantize 4|8] [--mtp-depth-max <n>] [--group-size <n>] [--fair-base-only] [--json]"
    );
}

fn cmd_doctor(args: &[OsString]) -> Result<u8, String> {
    let args = parse_doctor_args(args)?;
    if args.help {
        return Ok(0);
    }
    let mut bench_args = vec![OsString::from("doctor")];
    if args.verbose {
        if args.json {
            bench_args.push(OsString::from("--json"));
        }
        bench_args.extend(args.bench_args);
        return exec_or_status(find_executable("ax-engine-bench"), &bench_args);
    }
    bench_args.push(OsString::from("--json"));
    bench_args.extend(args.bench_args);
    let (code, bench_report, stderr) = run_bench_doctor_json(&bench_args)?;
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }
    if code != 0 {
        return Ok(code);
    }
    let report = user_doctor_report(&bench_report);
    if args.json {
        print_json(&report);
    } else {
        println!("{}", format_user_doctor_report(&report));
    }
    Ok(
        if report.get("result").and_then(Value::as_str) == Some("not_ready") {
            1
        } else {
            0
        },
    )
}

#[derive(Debug)]
struct DoctorArgs {
    json: bool,
    verbose: bool,
    help: bool,
    bench_args: Vec<OsString>,
}

fn parse_doctor_args(args: &[OsString]) -> Result<DoctorArgs, String> {
    let mut json = false;
    let mut verbose = false;
    let mut bench_args = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--json" => json = true,
            "--verbose" => verbose = true,
            "--mlx-model-artifacts-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--mlx-model-artifacts-dir requires a value".to_string())?
                    .clone();
                bench_args.push(OsString::from("--mlx-model-artifacts-dir"));
                bench_args.push(value);
            }
            "--help" | "-h" => {
                println!(
                    "Usage:\n  ax-engine doctor [--json] [--verbose] [--mlx-model-artifacts-dir <path>]\n\nDefault output is an end-user readiness summary. Use --verbose for the detailed ax-engine-bench doctor report."
                );
                return Ok(DoctorArgs {
                    json,
                    verbose,
                    help: true,
                    bench_args,
                });
            }
            flag if flag.starts_with('-') => return Err(format!("unknown doctor option: {flag}")),
            _ => return Err("doctor does not accept positional arguments".into()),
        }
        index += 1;
    }
    Ok(DoctorArgs {
        json,
        verbose,
        help: false,
        bench_args,
    })
}

fn run_bench_doctor_json(args: &[OsString]) -> Result<(u8, Value, String), String> {
    let bench = find_executable("ax-engine-bench");
    let output = Command::new(&bench)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run {}: {err}", bench.display()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report = serde_json::from_str::<Value>(stdout.trim()).map_err(|err| {
        format!(
            "ax-engine-bench doctor did not emit valid JSON: {err}\nstdout:\n{}",
            stdout.trim()
        )
    })?;
    Ok((
        output.status.code().unwrap_or(1).try_into().unwrap_or(1),
        report,
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

fn user_doctor_report(bench: &Value) -> Value {
    let server = probe_binary("ax-engine-server");
    let bench_bin = probe_binary("ax-engine-bench");
    let host_system = host_system_summary();
    let bench_status = value_str(bench, &["status"]).unwrap_or("unknown");
    let mlx_ready = value_bool(bench, &["mlx_runtime_ready"]).unwrap_or(false);
    let model_status = value_str(bench, &["model_artifacts", "status"]).unwrap_or("unknown");
    let model_selected = value_bool(bench, &["model_artifacts", "selected"]).unwrap_or(false);
    let model_path = value_str(bench, &["model_artifacts", "path"]);
    let issues = bench
        .get("issues")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let model_issues = bench
        .get("model_artifacts")
        .and_then(|value| value.get("issues"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let result = if !server.available || !bench_bin.available || bench_status == "not_ready" {
        "not_ready"
    } else if bench_status == "bringup_only" {
        "degraded"
    } else {
        "ready"
    };

    let mut next_actions = Vec::new();
    if !server.available {
        next_actions.push("Reinstall ax-engine so ax-engine-server is on PATH.".to_string());
    } else if !bench_bin.available {
        next_actions.push("Reinstall ax-engine so ax-engine-bench is on PATH.".to_string());
    } else if !mlx_ready {
        next_actions.push("Fix the host or Metal runtime issues listed below.".to_string());
    } else if model_status == "not_ready" {
        if let Some(path) = model_path {
            next_actions.push(format!("ax-engine-bench generate-manifest {path} --json"));
            next_actions.push(format!("ax-engine doctor --mlx-model-artifacts-dir {path}"));
        } else {
            next_actions
                .push("Pass --mlx-model-artifacts-dir <model-dir> to inspect a model.".to_string());
        }
    } else if model_selected {
        if let Some(path) = model_path {
            next_actions.push(format!("ax-engine serve {path} --port 8080"));
        } else {
            next_actions.push("ax-engine serve <model-dir> --port 8080".to_string());
        }
    } else {
        next_actions.push("ax-engine serve qwen36-35b --download --port 8080".to_string());
        next_actions.push("ax-engine models list".to_string());
    }

    json!({
        "schema_version": "ax.engine.doctor.v1",
        "result": result,
        "ready_for": ready_for(result, model_status),
        "install": {
            "version": env!("CARGO_PKG_VERSION"),
            "mode": value_str(bench, &["workflow", "mode"]).unwrap_or("unknown"),
            "cwd": value_str(bench, &["workflow", "cwd"]).unwrap_or("unknown"),
        },
        "host": host_system,
        "checks": [
            check("server_binary", server.available, server.detail),
            check("bench_binary", bench_bin.available, bench_bin.detail),
            check("host", value_bool(bench, &["host", "supported_mlx_runtime"]).unwrap_or(false), host_detail(bench)),
            check("metal_toolchain", value_bool(bench, &["metal_toolchain", "fully_available"]).unwrap_or(false), metal_detail(bench)),
            check("mlx_runtime", mlx_ready, bench_status.to_string()),
            json!({
                "id": "model",
                "status": model_status,
                "selected": model_selected,
                "path": model_path,
            }),
        ],
        "issues": issues,
        "model_issues": model_issues,
        "next_actions": next_actions,
        "details_command": "ax-engine-bench doctor",
        "source": {
            "schema_version": value_str(bench, &["schema_version"]).unwrap_or("unknown"),
            "status": bench_status,
            "details_command": "ax-engine-bench doctor --json",
        },
    })
}

#[derive(Debug)]
struct BinaryProbe {
    available: bool,
    detail: String,
}

fn probe_binary(name: &str) -> BinaryProbe {
    let bin = find_executable(name);
    match Command::new(&bin)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => BinaryProbe {
            available: true,
            detail: format!("{} ok", bin.display()),
        },
        Ok(status) => BinaryProbe {
            available: false,
            detail: format!("{} exited with status {}", bin.display(), status),
        },
        Err(err) => BinaryProbe {
            available: false,
            detail: format!("{}: {err}", bin.display()),
        },
    }
}

fn host_system_summary() -> Value {
    let os = value_or_unknown(env::consts::OS);
    let arch = value_or_unknown(env::consts::ARCH);
    let hardware_profile = command_stdout("system_profiler", &["SPHardwareDataType"]);
    let os_version = detect_os_version();
    let os_build = detect_os_build();
    let ram_bytes =
        detect_memory_bytes().or_else(|| hardware_profile.as_deref().and_then(parse_memory_bytes));
    let cpu_cores = detect_cpu_cores(hardware_profile.as_deref());
    json!({
        "os": os,
        "arch": arch,
        "os_version": os_version,
        "os_build": os_build,
        "ram_bytes": ram_bytes,
        "ram_gib": ram_bytes.map(bytes_to_gib),
        "cpu_cores": cpu_cores,
        "gpu_cores": detect_gpu_cores(),
    })
}

fn value_or_unknown(value: &str) -> &str {
    if value.is_empty() { "unknown" } else { value }
}

fn detect_os_version() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sw_vers", &["-productVersion"]),
        _ => None,
    }
}

fn detect_os_build() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sw_vers", &["-buildVersion"]),
        _ => None,
    }
}

fn detect_memory_bytes() -> Option<u64> {
    match env::consts::OS {
        "macos" => command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.parse::<u64>().ok()),
        _ => None,
    }
}

fn detect_cpu_cores(hardware_profile: Option<&str>) -> Value {
    let physical = command_stdout("sysctl", &["-n", "hw.physicalcpu"])
        .and_then(|value| value.parse::<u64>().ok())
        .or_else(|| hardware_profile.and_then(parse_physical_cpu_cores));
    let logical = command_stdout("sysctl", &["-n", "hw.logicalcpu"])
        .and_then(|value| value.parse::<u64>().ok());
    let mut performance = None;
    let mut efficiency = None;
    let mut types = serde_json::Map::new();

    for level in ["0", "1", "2", "3"] {
        let name_key = format!("hw.perflevel{level}.name");
        let cpu_key = format!("hw.perflevel{level}.physicalcpu");
        let Some(name) = command_stdout("sysctl", &["-n", &name_key]) else {
            continue;
        };
        let cores =
            command_stdout("sysctl", &["-n", &cpu_key]).and_then(|value| value.parse::<u64>().ok());
        let normalized = name.to_ascii_lowercase();
        if normalized.contains("performance") {
            performance = cores;
        } else if normalized.contains("efficiency") {
            efficiency = cores;
        }
        if let Some(cores) = cores {
            types.insert(normalized.replace(' ', "_"), json!(cores));
        }
    }

    let summary = hardware_profile.and_then(parse_cpu_core_summary);
    if types.is_empty()
        && let Some(summary) = summary.as_deref()
    {
        for (label, cores) in parse_cpu_core_types(summary) {
            let normalized = label.to_ascii_lowercase().replace(' ', "_");
            if normalized.contains("performance") && performance.is_none() {
                performance = Some(cores);
            } else if normalized.contains("efficiency") && efficiency.is_none() {
                efficiency = Some(cores);
            }
            types.insert(normalized, json!(cores));
        }
    }

    json!({
        "physical": physical,
        "logical": logical,
        "performance": performance,
        "efficiency": efficiency,
        "summary": summary,
        "types": types,
    })
}

fn parse_memory_bytes(output: &str) -> Option<u64> {
    for line in output.lines() {
        let trimmed = line.trim();
        let Some(value) = trimmed.strip_prefix("Memory:") else {
            continue;
        };
        let mut parts = value.split_whitespace();
        let amount = parts.next()?.parse::<u64>().ok()?;
        let unit = parts.next()?.to_ascii_lowercase();
        return match unit.as_str() {
            "gb" | "gib" => amount.checked_mul(1024 * 1024 * 1024),
            "mb" | "mib" => amount.checked_mul(1024 * 1024),
            _ => None,
        };
    }
    None
}

fn parse_physical_cpu_cores(output: &str) -> Option<u64> {
    let summary = parse_cpu_core_summary(output)?;
    summary
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u64>().ok())
}

fn parse_cpu_core_summary(output: &str) -> Option<String> {
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("Total Number of Cores:") {
            return Some(value.trim().to_string());
        }
    }
    None
}

fn parse_cpu_core_types(summary: &str) -> Vec<(String, u64)> {
    let Some(start) = summary.find('(') else {
        return Vec::new();
    };
    let Some(end) = summary[start + 1..].find(')') else {
        return Vec::new();
    };
    let inside = &summary[start + 1..start + 1 + end];
    inside
        .split(" and ")
        .filter_map(|part| {
            let mut words = part.split_whitespace();
            let cores = words.next()?.parse::<u64>().ok()?;
            let label = words.collect::<Vec<_>>().join(" ");
            if label.is_empty() {
                None
            } else {
                Some((label, cores))
            }
        })
        .collect()
}

fn detect_gpu_cores() -> Option<u64> {
    let output = command_stdout("system_profiler", &["SPDisplaysDataType"])?;
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("Total Number of Cores:") {
            return value.trim().parse::<u64>().ok();
        }
    }
    None
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

fn bytes_to_gib(bytes: u64) -> u64 {
    bytes / (1024 * 1024 * 1024)
}

fn ready_for(result: &str, model_status: &str) -> Vec<&'static str> {
    if result == "not_ready" {
        Vec::new()
    } else if model_status == "ready" {
        vec!["serve", "python_sdk", "model_checks"]
    } else {
        vec!["serve", "python_sdk"]
    }
}

fn check(id: &str, pass: bool, detail: String) -> Value {
    json!({
        "id": id,
        "status": if pass { "pass" } else { "fail" },
        "detail": detail,
    })
}

fn host_detail(report: &Value) -> String {
    format!(
        "{} ({}/{})",
        value_str(report, &["host", "detected_soc"]).unwrap_or("unknown Apple Silicon"),
        value_str(report, &["host", "os"]).unwrap_or("unknown"),
        value_str(report, &["host", "arch"]).unwrap_or("unknown")
    )
}

fn metal_detail(report: &Value) -> String {
    if value_bool(report, &["metal_toolchain", "fully_available"]).unwrap_or(false) {
        "Metal compiler and metallib available".to_string()
    } else {
        "Metal compiler or metallib missing".to_string()
    }
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn value_str<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    value_at(value, path)?.as_str()
}

fn value_bool(value: &Value, path: &[&str]) -> Option<bool> {
    value_at(value, path)?.as_bool()
}

fn format_user_doctor_report(report: &Value) -> String {
    let mut lines = vec![
        "AX Engine doctor".to_string(),
        String::new(),
        format!(
            "Result: {}",
            report
                .get("result")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .replace('_', " ")
        ),
        String::new(),
        "Install:".to_string(),
        format!(
            "  version: {}",
            value_str(report, &["install", "version"]).unwrap_or("unknown")
        ),
        format!(
            "  mode: {}",
            value_str(report, &["install", "mode"]).unwrap_or("unknown")
        ),
        format!(
            "  host: {} {} ({})",
            value_str(report, &["host", "os"]).unwrap_or("unknown"),
            value_str(report, &["host", "os_version"]).unwrap_or("unknown"),
            value_str(report, &["host", "arch"]).unwrap_or("unknown")
        ),
        format!(
            "  RAM: {}",
            report
                .get("host")
                .and_then(|host| host.get("ram_gib"))
                .and_then(Value::as_u64)
                .map(|gib| format!("{gib} GiB"))
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "  CPU cores: {}",
            format_cpu_cores(report.get("host").and_then(|host| host.get("cpu_cores")))
        ),
        format!(
            "  GPU cores: {}",
            report
                .get("host")
                .and_then(|host| host.get("gpu_cores"))
                .and_then(Value::as_u64)
                .map(|cores| cores.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
        String::new(),
        "Checks:".to_string(),
    ];
    if let Some(checks) = report.get("checks").and_then(Value::as_array) {
        for check in checks {
            let id = check.get("id").and_then(Value::as_str).unwrap_or("unknown");
            let status = check
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let detail = check.get("detail").and_then(Value::as_str);
            if let Some(detail) = detail {
                lines.push(format!("  {id}: {status} - {detail}"));
            } else {
                let selected = check
                    .get("selected")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let path = check.get("path").and_then(Value::as_str).unwrap_or("none");
                lines.push(format!(
                    "  {id}: {status} (selected: {selected}, path: {path})"
                ));
            }
        }
    }
    lines.push(String::new());
    lines.push("Issues:".to_string());
    append_string_array(&mut lines, report.get("issues").and_then(Value::as_array));
    lines.push(String::new());
    lines.push("Model issues:".to_string());
    append_string_array(
        &mut lines,
        report.get("model_issues").and_then(Value::as_array),
    );
    lines.push(String::new());
    lines.push("Next:".to_string());
    append_string_array(
        &mut lines,
        report.get("next_actions").and_then(Value::as_array),
    );
    lines.push(String::new());
    lines.push(format!(
        "More details: {}",
        report
            .get("details_command")
            .and_then(Value::as_str)
            .unwrap_or("ax-engine-bench doctor")
    ));
    lines.join("\n")
}

fn format_cpu_cores(cpu_cores: Option<&Value>) -> String {
    let Some(cpu_cores) = cpu_cores else {
        return "unknown".to_string();
    };
    if let Some(summary) = cpu_cores.get("summary").and_then(Value::as_str) {
        return summary.to_string();
    }
    let physical = cpu_cores.get("physical").and_then(Value::as_u64);
    let logical = cpu_cores.get("logical").and_then(Value::as_u64);
    let performance = cpu_cores.get("performance").and_then(Value::as_u64);
    let efficiency = cpu_cores.get("efficiency").and_then(Value::as_u64);
    match (physical, logical, performance, efficiency) {
        (Some(physical), Some(logical), Some(performance), Some(efficiency)) => {
            format!("{physical} physical / {logical} logical ({performance}P+{efficiency}E)")
        }
        (Some(physical), Some(logical), _, _) => format!("{physical} physical / {logical} logical"),
        (Some(physical), _, _, _) => format!("{physical} physical"),
        _ => "unknown".to_string(),
    }
}

fn append_string_array(lines: &mut Vec<String>, values: Option<&Vec<Value>>) {
    let Some(values) = values else {
        lines.push("  none".to_string());
        return;
    };
    if values.is_empty() {
        lines.push("  none".to_string());
    } else {
        for value in values {
            if let Some(text) = value.as_str() {
                lines.push(format!("  {text}"));
            }
        }
    }
}

#[derive(Debug)]
struct ServeArgs {
    model: OsString,
    host: String,
    port: String,
    hf_cache_root: Option<OsString>,
    download: bool,
    dry_run: bool,
    json: bool,
    passthrough: Vec<OsString>,
}

fn cmd_serve(args: &[OsString]) -> Result<u8, String> {
    let args = parse_serve_args(args)?;
    let server = find_executable("ax-engine-server");
    let target = args.model.to_string_lossy();
    let target_path = expand_home(&target);
    let mut argv = vec![
        OsString::from("--host"),
        OsString::from(&args.host),
        OsString::from("--port"),
        OsString::from(&args.port),
    ];

    let resolved = if target_path.exists() {
        let model = absolute_path(&target_path);
        argv.extend([
            OsString::from("--mlx"),
            OsString::from("--mlx-model-artifacts-dir"),
            model.clone().into_os_string(),
        ]);
        json!({
            "kind": "local_dir",
            "model": model.to_string_lossy(),
        })
    } else {
        let profile = profile_for_model(&target);
        let preset = profile.and_then(|profile| profile.preset);
        if args.download && !args.dry_run {
            let (code, mut summary, stderr) = run_download_summary(&target, None, false, profile)?;
            if code != 0 || summary.get("status").and_then(Value::as_str) != Some("ready") {
                if !stderr.is_empty() {
                    eprint!("{stderr}");
                }
                if !summary.is_null() {
                    print_download_summary(&summary);
                }
                return Err(format!(
                    "model download did not produce ready AX artifacts; run: ax-engine download {target}"
                ));
            }
            let Some(dest) = summary.get("dest").and_then(Value::as_str) else {
                return Err("download helper returned ready status without a dest".into());
            };
            let model_dir = absolute_path(&expand_home(dest));
            argv.push(OsString::from("--mlx"));
            if let Some(preset) = preset {
                argv.extend([OsString::from("--preset"), OsString::from(preset)]);
                summary["preset"] = json!(preset);
            }
            argv.extend([
                OsString::from("--mlx-model-artifacts-dir"),
                model_dir.clone().into_os_string(),
            ]);
            json!({
                "kind": "downloaded",
                "model": target.as_ref(),
                "repo_id": summary.get("repo_id").cloned().unwrap_or(Value::Null),
                "path": model_dir.to_string_lossy(),
                "preset": preset,
                "download": {
                    "status": summary.get("status").cloned().unwrap_or(Value::Null),
                    "manifest_present": summary.get("manifest_present").cloned().unwrap_or(Value::Null),
                },
            })
        } else {
            let Some(preset) = preset else {
                let hint = if target.contains('/') {
                    format!(" or run: ax-engine serve {target} --download")
                } else {
                    String::new()
                };
                return Err(format!(
                    "unknown model alias or missing local directory: {target:?}; pass a model directory or one of {}{hint}",
                    server_preset_labels().join(", ")
                ));
            };
            argv.extend([
                OsString::from("--mlx"),
                OsString::from("--preset"),
                OsString::from(preset),
                OsString::from("--resolve-model-artifacts"),
                OsString::from("hf-cache"),
            ]);
            if let Some(root) = &args.hf_cache_root {
                argv.extend([OsString::from("--hf-cache-root"), root.clone()]);
            }
            json!({
                "kind": "preset",
                "model": target.as_ref(),
                "preset": preset,
                "resolution": "hf-cache",
                "download": if args.download {
                    json!({
                        "enabled": true,
                        "repo_id": profile.map(|profile| profile.repo_id),
                        "dry_run": true,
                    })
                } else {
                    Value::Null
                },
            })
        }
    };

    argv.extend(args.passthrough);
    let server_argv = std::iter::once(server.as_os_str().to_string_lossy().to_string())
        .chain(argv.iter().map(|arg| arg.to_string_lossy().to_string()))
        .collect::<Vec<_>>();
    let plan = json!({
        "schema_version": "ax.local_serve_plan.v1",
        "command": "serve",
        "input": target.as_ref(),
        "resolved": resolved,
        "server": {
            "url": format!("http://{}:{}", args.host, args.port),
            "argv": server_argv,
        },
    });

    if args.json {
        print_json(&plan);
    } else {
        println!("AX Engine server: http://{}:{}", args.host, args.port);
        println!("Command:");
        println!("  {}", server_argv.join(" "));
    }
    if args.dry_run {
        Ok(0)
    } else {
        exec_or_status(server, &argv)
    }
}

fn parse_serve_args(args: &[OsString]) -> Result<ServeArgs, String> {
    let mut before_separator = Vec::new();
    let mut passthrough = Vec::new();
    let mut after_separator = false;
    for arg in args {
        if !after_separator && arg == "--" {
            after_separator = true;
            continue;
        }
        if after_separator {
            passthrough.push(arg.clone());
        } else {
            before_separator.push(arg.clone());
        }
    }

    let mut model = None;
    let mut host = "127.0.0.1".to_string();
    let mut port = "8080".to_string();
    let mut hf_cache_root = None;
    let mut download = false;
    let mut dry_run = false;
    let mut json = false;
    let mut index = 0;
    while index < before_separator.len() {
        let arg = before_separator[index].to_string_lossy();
        match arg.as_ref() {
            "--host" => {
                index += 1;
                host = require_value(&before_separator, index, "--host")?;
            }
            "--port" => {
                index += 1;
                port = require_value(&before_separator, index, "--port")?;
            }
            "--hf-cache-root" => {
                index += 1;
                hf_cache_root = Some(
                    before_separator
                        .get(index)
                        .ok_or_else(|| "--hf-cache-root requires a value".to_string())?
                        .clone(),
                );
            }
            "--download" => download = true,
            "--dry-run" => dry_run = true,
            "--json" => json = true,
            flag if flag.starts_with('-') => return Err(format!("unknown serve option: {flag}")),
            _ => {
                if model.replace(before_separator[index].clone()).is_some() {
                    return Err("serve accepts exactly one model argument".into());
                }
            }
        }
        index += 1;
    }
    let model = model.ok_or_else(|| "serve requires a model directory or alias".to_string())?;
    Ok(ServeArgs {
        model,
        host,
        port,
        hf_cache_root,
        download,
        dry_run,
        json,
        passthrough,
    })
}

fn cmd_models(args: &[OsString]) -> Result<u8, String> {
    let Some(command) = args.first() else {
        return Err(models_usage());
    };
    match command.to_string_lossy().as_ref() {
        "list" => cmd_models_list(&args[1..]),
        "info" => cmd_models_info(&args[1..]),
        "rm" => cmd_models_rm(&args[1..]),
        "--help" | "-h" => {
            println!("{}", models_usage());
            Ok(0)
        }
        unknown => Err(format!(
            "unknown models command: {unknown}\n\n{}",
            models_usage()
        )),
    }
}

fn models_usage() -> String {
    "Usage:\n  ax-engine models list [--models-dir <path>] [--json]\n  ax-engine models info <alias-or-path> [--json]\n  ax-engine models rm <path> [--dry-run] [--yes] [--json]".to_string()
}

fn cmd_models_list(args: &[OsString]) -> Result<u8, String> {
    let mut models_dir = env::var_os("AX_ENGINE_MODELS_DIR").map(PathBuf::from);
    let mut json_output = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--models-dir" => {
                index += 1;
                models_dir = Some(expand_home(&require_value(args, index, "--models-dir")?));
            }
            "--json" => json_output = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown models list option: {flag}"));
            }
            _ => return Err("models list does not accept positional arguments".into()),
        }
        index += 1;
    }

    let payload = models_list_payload(models_dir.as_deref());
    if json_output {
        print_json(&payload);
    } else {
        println!("{}", format_models_list(&payload));
    }
    Ok(0)
}

fn cmd_models_info(args: &[OsString]) -> Result<u8, String> {
    let mut target = None;
    let mut json_output = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--json" => json_output = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown models info option: {flag}"));
            }
            _ => {
                if target.replace(arg.to_string()).is_some() {
                    return Err("models info accepts exactly one alias or path".into());
                }
            }
        }
        index += 1;
    }
    let target = target.ok_or_else(|| "models info requires an alias or path".to_string())?;
    let payload = model_info_payload(&target)?;
    if json_output {
        print_json(&payload);
    } else {
        println!("{}", format_model_info(&payload));
    }
    Ok(0)
}

fn cmd_models_rm(args: &[OsString]) -> Result<u8, String> {
    let mut target = None;
    let mut dry_run = false;
    let mut yes = false;
    let mut json_output = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--dry-run" => dry_run = true,
            "--yes" => yes = true,
            "--json" => json_output = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown models rm option: {flag}"));
            }
            _ => {
                if target.replace(arg.to_string()).is_some() {
                    return Err("models rm accepts exactly one path".into());
                }
            }
        }
        index += 1;
    }
    let target = target.ok_or_else(|| "models rm requires a local model path".to_string())?;
    if profile_for_model(&target).is_some() {
        return Err(
            "models rm refuses aliases; pass an explicit local model directory path".into(),
        );
    }
    let path = absolute_path(&expand_home(&target));
    let effective_dry_run = dry_run || !yes;
    let report = validate_model_rm_target(&path, effective_dry_run)?;
    if !effective_dry_run {
        fs::remove_dir_all(&path)
            .map_err(|err| format!("failed to remove {}: {err}", path.display()))?;
    }
    let payload = json!({
        "schema_version": "ax.models_rm.v1",
        "command": "models rm",
        "path": path.to_string_lossy(),
        "dry_run": effective_dry_run,
        "removed": !effective_dry_run,
        "safety": report,
    });
    if json_output {
        print_json(&payload);
    } else if !effective_dry_run {
        println!("Removed {}", path.display());
    } else {
        println!("Dry run: would remove {}", path.display());
        println!("Pass --yes to remove this local artifact directory.");
    }
    Ok(0)
}

fn models_list_payload(models_dir: Option<&Path>) -> Value {
    json!({
        "schema_version": "ax.models_list.v1",
        "supported_aliases": MODEL_PROFILES.iter().map(model_profile_payload).collect::<Vec<_>>(),
        "local_artifacts": models_dir.map(local_model_artifacts_payload).unwrap_or_else(|| {
            json!({
                "source": "not_selected",
                "env": "AX_ENGINE_MODELS_DIR",
                "items": [],
            })
        }),
    })
}

fn model_profile_payload(profile: &ModelProfile) -> Value {
    json!({
        "kind": "supported_alias",
        "label": profile.label,
        "repo_id": profile.repo_id,
        "preset": profile.preset,
        "downloadable": profile.downloadable,
        "aliases": profile.aliases,
    })
}

fn local_model_artifacts_payload(root: &Path) -> Value {
    let root = absolute_path(root);
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir(&root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir()
                && let Some(item) = local_model_artifact_payload(&path)
            {
                items.push(item);
            }
        }
    }
    json!({
        "source": "models_dir",
        "path": root.to_string_lossy(),
        "items": items,
    })
}

fn local_model_artifact_payload(path: &Path) -> Option<Value> {
    let manifest_present = path.join("model-manifest.json").is_file();
    let config_present = path.join("config.json").is_file();
    if !manifest_present && !config_present {
        return None;
    }
    Some(json!({
        "kind": "local_artifact",
        "path": absolute_path(path).to_string_lossy(),
        "manifest_present": manifest_present,
        "config_present": config_present,
    }))
}

fn model_info_payload(target: &str) -> Result<Value, String> {
    if let Some(profile) = profile_for_model(target) {
        return Ok(json!({
            "schema_version": "ax.models_info.v1",
            "query": target,
            "kind": "supported_alias",
            "profile": model_profile_payload(&profile),
        }));
    }
    let path = expand_home(target);
    if path.exists() {
        let path = absolute_path(&path);
        return Ok(json!({
            "schema_version": "ax.models_info.v1",
            "query": target,
            "kind": "local_artifact",
            "path": path.to_string_lossy(),
            "manifest_present": path.join("model-manifest.json").is_file(),
            "config_present": path.join("config.json").is_file(),
            "hf_cache_path": is_hf_cache_path(&path),
        }));
    }
    if target.contains('/') {
        return Ok(json!({
            "schema_version": "ax.models_info.v1",
            "query": target,
            "kind": "repo_id",
            "repo_id": target,
            "managed_alias": false,
        }));
    }
    Err(format!(
        "unknown model alias or missing local path: {target:?}; run `ax-engine models list`"
    ))
}

fn validate_model_rm_target(path: &Path, dry_run: bool) -> Result<Value, String> {
    if !path.exists() {
        return Err(format!(
            "models rm target does not exist: {}",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!(
            "models rm target is not a directory: {}",
            path.display()
        ));
    }
    if is_hf_cache_path(path) {
        return Err(format!(
            "models rm refuses Hugging Face cache paths; remove cache entries with huggingface-cli instead: {}",
            path.display()
        ));
    }
    if path.parent().is_none() {
        return Err("models rm refuses filesystem root".into());
    }
    let manifest_present = path.join("model-manifest.json").is_file();
    let config_present = path.join("config.json").is_file();
    if !manifest_present && !config_present {
        return Err(format!(
            "models rm target does not look like an AX/MLX artifact directory: {}",
            path.display()
        ));
    }
    Ok(json!({
        "dry_run": dry_run,
        "manifest_present": manifest_present,
        "config_present": config_present,
        "hf_cache_path": false,
    }))
}

fn is_hf_cache_path(path: &Path) -> bool {
    let text = path.to_string_lossy();
    text.contains("/huggingface/hub/")
        || text.contains("/.cache/huggingface/")
        || path.components().any(|component| {
            component
                .as_os_str()
                .to_string_lossy()
                .starts_with("models--")
        })
}

fn format_models_list(payload: &Value) -> String {
    let mut lines = vec!["Supported aliases:".to_string()];
    if let Some(targets) = payload.get("supported_aliases").and_then(Value::as_array) {
        for target in targets {
            lines.push(format!(
                "  - {} -> {}",
                target
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                target
                    .get("repo_id")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
            ));
        }
    }
    lines.push("Local artifacts:".into());
    let local = &payload["local_artifacts"];
    if local.get("source").and_then(Value::as_str) == Some("not_selected") {
        lines.push("  - set AX_ENGINE_MODELS_DIR or pass --models-dir".into());
    } else if let Some(items) = local.get("items").and_then(Value::as_array) {
        if items.is_empty() {
            lines.push("  - none found".into());
        } else {
            for item in items {
                lines.push(format!(
                    "  - {}",
                    item.get("path")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                ));
            }
        }
    }
    lines.join("\n")
}

fn format_model_info(payload: &Value) -> String {
    match payload.get("kind").and_then(Value::as_str) {
        Some("supported_alias") => {
            let profile = &payload["profile"];
            format!(
                "Supported alias: {}\nRepo: {}\nPreset: {}",
                profile
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                profile
                    .get("repo_id")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                profile
                    .get("preset")
                    .and_then(Value::as_str)
                    .unwrap_or("none")
            )
        }
        Some("local_artifact") => format!(
            "Local artifact: {}\nmodel-manifest.json: {}\nconfig.json: {}\nHF cache path: {}",
            payload
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            payload
                .get("manifest_present")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            payload
                .get("config_present")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            payload
                .get("hf_cache_path")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        ),
        Some("repo_id") => format!(
            "Repo id: {}\nManaged alias: false",
            payload
                .get("repo_id")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        ),
        _ => "Unknown model".to_string(),
    }
}

#[derive(Debug)]
struct DownloadArgs {
    model: Option<String>,
    dest: Option<String>,
    force: bool,
    list: bool,
    json: bool,
}

#[derive(Debug)]
struct DownloadMtpArgs {
    model: String,
    output: Option<String>,
    force: bool,
    quantize: Option<String>,
    mtp_depth_max: Option<String>,
    group_size: String,
    fair_base_only: bool,
    json: bool,
}

fn cmd_download(args: &[OsString]) -> Result<u8, String> {
    let args = parse_download_args(args)?;
    if args.list {
        if args.json {
            print_json(&download_options_payload());
        } else {
            println!("{}", format_download_options());
        }
        return Ok(0);
    }
    let Some(model) = args.model else {
        if args.json {
            print_json(&download_options_payload());
        } else {
            println!("missing model alias or repo id\n");
            println!("{}", format_download_options());
        }
        return Ok(2);
    };

    let profile = profile_for_model(&model);
    let (code, summary, stderr) =
        run_download_summary(&model, args.dest.as_deref(), args.force, profile)?;
    if args.json {
        if !summary.is_null() {
            print_json(&summary);
        }
        if !stderr.is_empty() {
            eprint!("{stderr}");
        }
        return Ok(code);
    }
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }
    if summary.is_null() {
        return Err("download helper did not emit an ax.download_model.v1 summary".into());
    }
    print_download_summary(&summary);
    Ok(code)
}

fn cmd_download_mtp(args: &[OsString]) -> Result<u8, String> {
    let args = parse_download_mtp_args(args)?;
    let target = mtp_download_target_for_model(&args.model)
        .ok_or_else(|| format_unknown_download_mtp_target(&args.model))?;
    let (download_code, download_summary, download_stderr) =
        run_download_summary(target.repo_id, None, args.force, None)?;
    if !download_stderr.is_empty() {
        eprint!("{download_stderr}");
    }
    if download_code != 0 || download_summary.get("status").and_then(Value::as_str) != Some("ready")
    {
        if args.json && !download_summary.is_null() {
            print_json(&json!({
                "schema_version": "ax.download_mtp.v1",
                "command": "download-mtp",
                "base_model": args.model,
                "repo_id": target.repo_id,
                "download": download_summary,
                "status": "download_failed",
            }));
            return Ok(download_code);
        }
        if !download_summary.is_null() {
            print_download_summary(&download_summary);
        }
        return Err(format!(
            "base model download did not produce ready AX artifacts; run: ax-engine download {}",
            args.model
        ));
    }
    let Some(base_dir) = download_summary
        .get("dest")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return Err("download helper returned ready status without a dest".into());
    };
    if !args.json {
        print_download_summary(&download_summary);
    }

    match target.kind {
        MtpDownloadKind::QwenSidecar { mtp_source } => {
            let convert_args = ConvertArgs {
                base_model: base_dir.clone(),
                mtp_source: mtp_source.to_string(),
                output: args.output.clone(),
                quantize: args.quantize.clone(),
                mtp_depth_max: args.mtp_depth_max.clone(),
                group_size: args.group_size.clone(),
                fair_base_only: args.fair_base_only,
                json: args.json,
            };
            run_convert_mtplx(
                &convert_args,
                "download-mtp",
                "ax.download_mtp.v1",
                Some(download_summary),
            )
        }
        MtpDownloadKind::GemmaAssistant { .. } => run_download_gemma_assistant_mtp(
            target,
            &args,
            &base_dir,
            target.kind,
            download_summary,
        ),
        MtpDownloadKind::GlmSidecar { mtp_source } => {
            run_download_glm_mtp_sidecar(target, &args, &base_dir, mtp_source, download_summary)
        }
        MtpDownloadKind::DirectOnly { reason } => {
            if args.json {
                print_json(&json!({
                    "schema_version": "ax.download_mtp.v1",
                    "command": "download-mtp",
                    "status": "direct_only",
                    "base_model": &args.model,
                    "repo_id": target.repo_id,
                    "output_dir": base_dir,
                    "reason": reason,
                    "download": download_summary,
                }));
            } else {
                println!("MTP status: direct-only");
                println!("{reason}");
                println!("Next:");
                println!("  ax-engine serve {base_dir}");
            }
            Ok(0)
        }
    }
}

fn parse_download_args(args: &[OsString]) -> Result<DownloadArgs, String> {
    let mut model = None;
    let mut dest = None;
    let mut force = false;
    let mut list = false;
    let mut json = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--dest" => {
                index += 1;
                dest = Some(require_value(args, index, "--dest")?);
            }
            "--force" => force = true,
            "--list" => list = true,
            "--json" => json = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown download option: {flag}"));
            }
            _ => {
                if model.replace(arg.to_string()).is_some() {
                    return Err("download accepts at most one model argument".into());
                }
            }
        }
        index += 1;
    }
    Ok(DownloadArgs {
        model,
        dest,
        force,
        list,
        json,
    })
}

fn parse_download_mtp_args(args: &[OsString]) -> Result<DownloadMtpArgs, String> {
    let mut model = None;
    let mut output = None;
    let mut force = false;
    let mut quantize = None;
    let mut mtp_depth_max = None;
    let mut group_size = "64".to_string();
    let mut fair_base_only = false;
    let mut json = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--output" => {
                index += 1;
                output = Some(require_value(args, index, "--output")?);
            }
            "--force" => force = true,
            "--quantize" => {
                index += 1;
                let value = require_value(args, index, "--quantize")?;
                if value != "4" && value != "8" {
                    return Err("--quantize must be 4 or 8".into());
                }
                quantize = Some(value);
            }
            "--mtp-depth-max" => {
                index += 1;
                mtp_depth_max = Some(require_value(args, index, "--mtp-depth-max")?);
            }
            "--group-size" => {
                index += 1;
                group_size = require_value(args, index, "--group-size")?;
            }
            "--fair-base-only" => fair_base_only = true,
            "--json" => json = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown download-mtp option: {flag}"));
            }
            _ => {
                if model.replace(arg.to_string()).is_some() {
                    return Err("download-mtp accepts exactly one model argument".into());
                }
            }
        }
        index += 1;
    }
    Ok(DownloadMtpArgs {
        model: model.ok_or_else(|| "download-mtp requires a model".to_string())?,
        output,
        force,
        quantize,
        mtp_depth_max,
        group_size,
        fair_base_only,
        json,
    })
}

fn run_download_summary(
    model: &str,
    dest: Option<&str>,
    force: bool,
    profile: Option<ModelProfile>,
) -> Result<(u8, Value, String), String> {
    let (repo_id, profile) = download_repo_id(model, profile)?;
    let helper = find_helper(
        "AX_ENGINE_DOWNLOAD_HELPER",
        "ax-engine-download-model.py",
        "download_model.py",
    )?;
    let mut command = Command::new(python());
    command.arg(helper).arg(repo_id).arg("--json");
    if let Some(dest) = dest {
        command.arg("--dest").arg(dest);
    }
    if force {
        command.arg("--force");
    }
    let output = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run download helper: {err}"))?;
    let mut summary =
        parse_summary_json(&String::from_utf8_lossy(&output.stdout)).unwrap_or(Value::Null);
    if let Value::Object(map) = &mut summary {
        map.insert("input".into(), json!(model));
        if let Some(profile) = profile {
            map.insert("alias".into(), json!(profile.label));
            if let Some(preset) = profile.preset {
                map.insert("preset".into(), json!(preset));
            }
        }
    }
    Ok((
        output.status.code().unwrap_or(1).try_into().unwrap_or(1),
        summary,
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

fn run_download_gemma_assistant_mtp(
    target: MtpDownloadTarget,
    args: &DownloadMtpArgs,
    base_dir: &str,
    kind: MtpDownloadKind,
    target_download: Value,
) -> Result<u8, String> {
    let MtpDownloadKind::GemmaAssistant {
        assistant_repo_id,
        target_model_id,
        assistant_model_id,
        max_depth,
    } = kind
    else {
        return Err("internal error: expected Gemma assistant MTP target".into());
    };
    let (assistant_code, assistant_summary, assistant_stderr) =
        run_download_summary(assistant_repo_id, None, args.force, None)?;
    if !assistant_stderr.is_empty() {
        eprint!("{assistant_stderr}");
    }
    if !assistant_download_usable(assistant_code, &assistant_summary) {
        if args.json && !assistant_summary.is_null() {
            print_json(&json!({
                "schema_version": "ax.download_mtp.v1",
                "command": "download-mtp",
                "status": "assistant_download_failed",
                "base_model": &args.model,
                "repo_id": target.repo_id,
                "assistant_repo_id": assistant_repo_id,
                "download": target_download,
                "assistant_download": assistant_summary,
            }));
            return Ok(assistant_code);
        }
        if !assistant_summary.is_null() {
            print_download_summary(&assistant_summary);
        }
        return Err(format!(
            "assistant model download did not produce ready AX artifacts; run: ax-engine download {assistant_repo_id}"
        ));
    }
    let Some(assistant_dir) = assistant_summary.get("dest").and_then(Value::as_str) else {
        return Err("assistant download helper returned ready status without a dest".into());
    };
    if !args.json {
        print_download_summary(&assistant_summary);
    }

    let helper = find_helper(
        "AX_ENGINE_PREPARE_GEMMA4_ASSISTANT_MTP_HELPER",
        "ax-engine-prepare-gemma4-assistant-mtp.py",
        "prepare_gemma4_assistant_mtp.py",
    )?;
    let default_depth = max_depth.to_string();
    let depth = args.mtp_depth_max.as_deref().unwrap_or(&default_depth);
    let mut prepare_cmd = Command::new(python());
    prepare_cmd
        .arg(&helper)
        .args(["--target", base_dir, "--assistant", assistant_dir])
        .args(["--target-model-id", target_model_id])
        .args(["--assistant-model-id", assistant_model_id])
        .args(["--max-depth", depth]);
    if let Some(output) = &args.output {
        prepare_cmd.args(["--output", output]);
    }
    let prepare_output = prepare_cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run prepare_gemma4_assistant_mtp helper: {err}"))?;
    let prepare_stdout = String::from_utf8_lossy(&prepare_output.stdout).into_owned();
    let prepare_stderr = String::from_utf8_lossy(&prepare_output.stderr).into_owned();
    if !args.json {
        print!("{prepare_stdout}");
        eprint!("{prepare_stderr}");
    }
    if !prepare_output.status.success() {
        if args.json {
            eprint!("{prepare_stderr}");
        }
        return Ok(prepare_output
            .status
            .code()
            .unwrap_or(1)
            .try_into()
            .unwrap_or(1));
    }
    let output_dir =
        parse_output_dir(&prepare_stdout, args.output.as_deref()).ok_or_else(|| {
            "prepare_gemma4_assistant_mtp.py succeeded but output dir could not be determined"
                .to_string()
        })?;

    if args.json {
        print_json(&json!({
            "schema_version": "ax.download_mtp.v1",
            "command": "download-mtp",
            "status": "ready",
            "kind": "gemma_assistant_mtp",
            "base_model": &args.model,
            "repo_id": target.repo_id,
            "assistant_repo_id": assistant_repo_id,
            "target_model_id": target_model_id,
            "assistant_model_id": assistant_model_id,
            "max_depth": depth.parse::<u32>().unwrap_or(max_depth),
            "output_dir": output_dir,
            "download": target_download,
            "assistant_download": assistant_summary,
        }));
    }
    Ok(0)
}

fn run_download_glm_mtp_sidecar(
    target: MtpDownloadTarget,
    args: &DownloadMtpArgs,
    base_dir: &str,
    mtp_source: &str,
    target_download: Value,
) -> Result<u8, String> {
    let helper = find_helper(
        "AX_ENGINE_PREPARE_GLM_MTP_SIDECAR_HELPER",
        "ax-engine-prepare-glm-mtp-sidecar.py",
        "prepare_glm_mtp_sidecar.py",
    )?;
    let depth = args.mtp_depth_max.as_deref().unwrap_or("1");
    let mut prepare_cmd = Command::new(python());
    prepare_cmd
        .arg(&helper)
        .args(["--hf-repo", mtp_source])
        .args(["--base", base_dir])
        .args(["--mtp-depth-max", depth])
        .args(["--group-size", &args.group_size]);
    if let Some(output) = &args.output {
        prepare_cmd.args(["--output", output]);
    }
    if let Some(quantize) = &args.quantize {
        prepare_cmd.args(["--quantize", quantize]);
    }
    let prepare_output = prepare_cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run prepare_glm_mtp_sidecar helper: {err}"))?;
    let prepare_stdout = String::from_utf8_lossy(&prepare_output.stdout).into_owned();
    let prepare_stderr = String::from_utf8_lossy(&prepare_output.stderr).into_owned();
    if !args.json {
        print!("{prepare_stdout}");
        eprint!("{prepare_stderr}");
    }
    if !prepare_output.status.success() {
        if args.json {
            eprint!("{prepare_stderr}");
        }
        return Ok(prepare_output
            .status
            .code()
            .unwrap_or(1)
            .try_into()
            .unwrap_or(1));
    }
    let output_dir =
        parse_output_dir(&prepare_stdout, args.output.as_deref()).ok_or_else(|| {
            "prepare_glm_mtp_sidecar.py succeeded but output dir could not be determined"
                .to_string()
        })?;

    if args.json {
        print_json(&json!({
            "schema_version": "ax.download_mtp.v1",
            "command": "download-mtp",
            "status": "ready",
            "kind": "glm_sidecar_mtp",
            "base_model": &args.model,
            "repo_id": target.repo_id,
            "mtp_source": mtp_source,
            "mtp_depth_max": depth.parse::<u32>().unwrap_or(1),
            "output_dir": output_dir,
            "download": target_download,
        }));
    }
    Ok(0)
}

fn assistant_download_usable(code: u8, summary: &Value) -> bool {
    let status = summary.get("status").and_then(Value::as_str);
    if code == 0 && status == Some("ready") {
        return true;
    }
    status == Some("manifest_missing")
        && summary
            .get("config_present")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        && summary
            .get("safetensors_count")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
}

#[derive(Debug)]
struct ConvertArgs {
    base_model: String,
    mtp_source: String,
    output: Option<String>,
    quantize: Option<String>,
    mtp_depth_max: Option<String>,
    group_size: String,
    fair_base_only: bool,
    json: bool,
}

fn cmd_convert_mtplx(args: &[OsString]) -> Result<u8, String> {
    let args = parse_convert_args(args)?;
    run_convert_mtplx(&args, "convert-mtplx", "ax.convert_mtplx.v1", None)
}

fn run_convert_mtplx(
    args: &ConvertArgs,
    command_name: &str,
    schema_version: &str,
    download_summary: Option<Value>,
) -> Result<u8, String> {
    let prepare = find_helper(
        "AX_ENGINE_PREPARE_MTP_SIDECAR_HELPER",
        "ax-engine-prepare-mtp-sidecar.py",
        "prepare_mtp_sidecar.py",
    )?;
    let check = find_helper(
        "AX_ENGINE_CHECK_MTP_SIDECAR_HELPER",
        "ax-engine-check-mtp-sidecar-provenance.py",
        "check_mtp_sidecar_provenance.py",
    )?;
    let depth = args
        .mtp_depth_max
        .clone()
        .unwrap_or_else(|| default_mtp_depth_max(&args.base_model, &args.mtp_source).to_string());

    let mut prepare_cmd = Command::new(python());
    prepare_cmd.arg(&prepare).args([
        "--hf-repo",
        &args.mtp_source,
        "--base",
        &args.base_model,
        "--mtp-depth-max",
        &depth,
        "--group-size",
        &args.group_size,
    ]);
    if let Some(output) = &args.output {
        prepare_cmd.args(["--output", output]);
    }
    if let Some(quantize) = &args.quantize {
        prepare_cmd.args(["--quantize", quantize]);
    }
    let prepare_output = prepare_cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run prepare_mtp_sidecar helper: {err}"))?;
    let prepare_stdout = String::from_utf8_lossy(&prepare_output.stdout).into_owned();
    let prepare_stderr = String::from_utf8_lossy(&prepare_output.stderr).into_owned();
    if !args.json {
        print!("{prepare_stdout}");
        eprint!("{prepare_stderr}");
    }
    if !prepare_output.status.success() {
        if args.json {
            eprint!("{prepare_stderr}");
        }
        return Ok(prepare_output
            .status
            .code()
            .unwrap_or(1)
            .try_into()
            .unwrap_or(1));
    }
    let output_dir =
        parse_output_dir(&prepare_stdout, args.output.as_deref()).ok_or_else(|| {
            "prepare_mtp_sidecar.py succeeded but output dir could not be determined".to_string()
        })?;

    let mut check_cmd = Command::new(python());
    check_cmd.arg(&check).arg(&output_dir).arg("--json");
    if args.fair_base_only {
        check_cmd.arg("--fair-base-only");
    }
    let provenance = check_cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run sidecar provenance checker: {err}"))?;
    let provenance_stdout = String::from_utf8_lossy(&provenance.stdout).into_owned();
    let provenance_stderr = String::from_utf8_lossy(&provenance.stderr).into_owned();
    if !args.json {
        print!("{provenance_stdout}");
        eprint!("{provenance_stderr}");
    }
    if !provenance.status.success() {
        if args.json {
            eprint!("{provenance_stderr}");
        }
        return Ok(provenance
            .status
            .code()
            .unwrap_or(1)
            .try_into()
            .unwrap_or(1));
    }

    if args.json {
        let provenance_summary = serde_json::from_str::<Value>(&provenance_stdout)
            .unwrap_or_else(|_| json!({ "raw": provenance_stdout }));
        let mut summary = json!({
            "schema_version": schema_version,
            "command": command_name,
            "base_model": &args.base_model,
            "mtp_source": &args.mtp_source,
            "mtp_depth_max": depth.parse::<u32>().unwrap_or(1),
            "output_dir": output_dir,
            "provenance": provenance_summary,
        });
        if let Some(download_summary) = download_summary {
            summary["download"] = download_summary;
        }
        print_json(&summary);
    }
    Ok(0)
}

fn parse_convert_args(args: &[OsString]) -> Result<ConvertArgs, String> {
    let mut base_model = None;
    let mut mtp_source = None;
    let mut output = None;
    let mut quantize = None;
    let mut mtp_depth_max = None;
    let mut group_size = "64".to_string();
    let mut fair_base_only = false;
    let mut json = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--mtp-source" => {
                index += 1;
                mtp_source = Some(require_value(args, index, "--mtp-source")?);
            }
            "--output" => {
                index += 1;
                output = Some(require_value(args, index, "--output")?);
            }
            "--quantize" => {
                index += 1;
                let value = require_value(args, index, "--quantize")?;
                if value != "4" && value != "8" {
                    return Err("--quantize must be 4 or 8".into());
                }
                quantize = Some(value);
            }
            "--mtp-depth-max" => {
                index += 1;
                mtp_depth_max = Some(require_value(args, index, "--mtp-depth-max")?);
            }
            "--group-size" => {
                index += 1;
                group_size = require_value(args, index, "--group-size")?;
            }
            "--fair-base-only" => fair_base_only = true,
            "--json" => json = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown convert-mtplx option: {flag}"));
            }
            _ => {
                if base_model.replace(arg.to_string()).is_some() {
                    return Err("convert-mtplx accepts exactly one base model argument".into());
                }
            }
        }
        index += 1;
    }
    Ok(ConvertArgs {
        base_model: base_model.ok_or_else(|| "convert-mtplx requires a base model".to_string())?,
        mtp_source: mtp_source.ok_or_else(|| "convert-mtplx requires --mtp-source".to_string())?,
        output,
        quantize,
        mtp_depth_max,
        group_size,
        fair_base_only,
        json,
    })
}

fn download_repo_id(
    value: &str,
    profile: Option<ModelProfile>,
) -> Result<(&'static str, Option<ModelProfile>), String> {
    if let Some(profile) = profile {
        if !profile.downloadable {
            return Err(format!(
                "{} is not managed by ax-engine download; use an explicit repo id or one of these targets:\n{}",
                profile.label,
                format_download_options()
            ));
        }
        return Ok((profile.repo_id, Some(profile)));
    }
    if value.contains('/') {
        let leaked: &'static str = Box::leak(value.to_string().into_boxed_str());
        Ok((leaked, None))
    } else {
        Err(format!(
            "unknown model alias or repo id: {value:?}; pass a Hugging Face repo id or one of these targets:\n{}",
            format_download_options()
        ))
    }
}

fn mtp_download_target_for_model(value: &str) -> Option<MtpDownloadTarget> {
    let normalized = normalize_alias(value);
    MTP_DOWNLOAD_TARGETS.iter().copied().find(|target| {
        target
            .aliases
            .iter()
            .any(|alias| normalize_alias(alias) == normalized)
            || normalize_alias(target.repo_id) == normalized
    })
}

fn format_unknown_download_mtp_target(value: &str) -> String {
    format!(
        "unknown download-mtp target: {value:?}; use one of these 6-bit targets:\n{}",
        format_download_mtp_targets()
    )
}

fn format_download_mtp_targets() -> String {
    let mut lines = Vec::new();
    for target in MTP_DOWNLOAD_TARGETS {
        let kind = match target.kind {
            MtpDownloadKind::QwenSidecar { .. } => "qwen-sidecar-mtp",
            MtpDownloadKind::GemmaAssistant { .. } => "gemma-assistant-mtp",
            MtpDownloadKind::GlmSidecar { .. } => "glm-sidecar-mtp",
            MtpDownloadKind::DirectOnly { .. } => "direct-only",
        };
        lines.push(format!(
            "  - {} -> {} ({kind}; aliases: {})",
            target.label,
            target.repo_id,
            target.aliases.join(", ")
        ));
    }
    lines.join("\n")
}

fn download_options_payload() -> Value {
    json!({
        "schema_version": "ax.download_options.v1",
        "default_destination": {
            "kind": "huggingface_hub_cache",
            "env": ["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"],
            "dest_semantics": "--dest copies the resolved snapshot to an explicit directory",
        },
        "targets": MODEL_PROFILES.iter().filter(|profile| profile.downloadable).map(|profile| {
            json!({
                "alias": profile.label,
                "repo_id": profile.repo_id,
                "preset": profile.preset,
                "aliases": profile.aliases,
            })
        }).collect::<Vec<_>>(),
        "examples": [
            "ax-engine download qwen36-35b",
            "ax-engine download qwen36-27b-8bit",
            "ax-engine download gemma4-12b",
            "ax-engine download gemma4-e2b-6bit",
            "ax-engine download mlx-community/Qwen3.6-35B-A3B-4bit --json",
        ],
    })
}

fn format_download_options() -> String {
    let mut lines = vec!["Available Qwen3.6 and Gemma 4 MLX download targets:".to_string()];
    for profile in MODEL_PROFILES.iter().filter(|profile| profile.downloadable) {
        let aliases = profile.aliases.join(", ");
        lines.push(format!(
            "  - {} -> {} (aliases: {})",
            profile.label, profile.repo_id, aliases
        ));
    }
    lines.push("Examples:".into());
    lines.push("  ax-engine download qwen36-35b".into());
    lines.push("  ax-engine download qwen36-27b-8bit".into());
    lines.push("  ax-engine download gemma4-12b".into());
    lines.push("  ax-engine download gemma4-e2b-6bit".into());
    lines.push("  ax-engine download mlx-community/Qwen3.6-35B-A3B-4bit --json".into());
    lines.join("\n")
}

fn print_download_summary(summary: &Value) {
    let status = summary
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let repo_id = summary
        .get("repo_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let dest = summary.get("dest").and_then(Value::as_str).unwrap_or("");
    println!("AX Engine model: {repo_id}");
    println!("Status: {status}");
    if !dest.is_empty() {
        println!("Path: {dest}");
    }
    if let Some(errors) = summary.get("errors").and_then(Value::as_array) {
        for error in errors {
            if let Some(error) = error.as_str() {
                eprintln!("Error: {error}");
            }
        }
    }
    if status == "ready" && !dest.is_empty() {
        println!("Next:");
        println!("  ax-engine serve {dest}");
    } else if !dest.is_empty() {
        println!("Next:");
        println!("  ax-engine-bench generate-manifest {dest}");
    }
}

fn parse_summary_json(stdout: &str) -> Option<Value> {
    if let Ok(value @ Value::Object(_)) = serde_json::from_str::<Value>(stdout.trim()) {
        return Some(value);
    }
    stdout.lines().rev().find_map(|line| {
        let value = serde_json::from_str::<Value>(line.trim()).ok()?;
        if value.get("schema_version").and_then(Value::as_str) == Some("ax.download_model.v1") {
            Some(value)
        } else {
            None
        }
    })
}

fn profile_for_model(value: &str) -> Option<ModelProfile> {
    let normalized = normalize_alias(value);
    MODEL_PROFILES.iter().copied().find(|profile| {
        profile
            .aliases
            .iter()
            .any(|alias| normalize_alias(alias) == normalized)
    })
}

fn server_preset_labels() -> Vec<&'static str> {
    let mut labels = MODEL_PROFILES
        .iter()
        .filter_map(|profile| profile.preset)
        .collect::<Vec<_>>();
    labels.sort_unstable();
    labels.dedup();
    labels
}

fn normalize_alias(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

fn find_executable(name: &str) -> PathBuf {
    if let Ok(current) = env::current_exe()
        && let Some(dir) = current.parent()
    {
        let sibling = dir.join(name);
        if sibling.is_file() {
            return sibling;
        }
    }
    PathBuf::from(name)
}

fn find_helper(env_name: &str, installed_name: &str, source_name: &str) -> Result<PathBuf, String> {
    if let Some(path) = env::var_os(env_name) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }
    if let Ok(current) = env::current_exe()
        && let Some(dir) = current.parent()
    {
        for name in [installed_name, source_name] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    let mut roots = Vec::new();
    if let Some(root) = env::var_os("AX_ENGINE_REPO_ROOT") {
        roots.push(PathBuf::from(root));
    }
    if let Ok(cwd) = env::current_dir() {
        roots.push(cwd.clone());
        roots.extend(cwd.ancestors().skip(1).map(Path::to_path_buf));
    }
    for root in roots {
        for candidate in [
            root.join("scripts").join(source_name),
            root.join(source_name),
        ] {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err(format!(
        "cannot locate {source_name}. Reinstall ax-engine, set {env_name}, or run from a source checkout."
    ))
}

fn python() -> OsString {
    env::var_os("AX_ENGINE_PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

fn expand_home(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/")
        && let Some(home) = env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(value)
}

fn absolute_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn require_value(args: &[OsString], index: usize, flag: &str) -> Result<String, String> {
    args.get(index)
        .map(|value| value.to_string_lossy().to_string())
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_json(value: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(value).expect("JSON serialization cannot fail")
    );
}

fn parse_output_dir(stdout: &str, explicit: Option<&str>) -> Option<String> {
    if let Some(explicit) = explicit {
        return Some(
            absolute_path(&expand_home(explicit))
                .to_string_lossy()
                .into(),
        );
    }
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("Output dir:") {
            return Some(rest.trim().to_string());
        }
    }
    let mut saw_sidecar_ready = false;
    for line in stdout.lines() {
        if saw_sidecar_ready {
            let value = line.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
        saw_sidecar_ready = line.trim() == "Sidecar ready at:";
    }
    None
}

fn default_mtp_depth_max(base_model: &str, mtp_source: &str) -> u32 {
    let label = format!("{base_model} {mtp_source}").to_ascii_lowercase();
    if label.contains("qwen3.6-27b") || label.contains("qwen3-6-27b") {
        3
    } else {
        1
    }
}

#[cfg(unix)]
fn exec_or_status(program: PathBuf, args: &[OsString]) -> Result<u8, String> {
    use std::os::unix::process::CommandExt;
    let err = Command::new(&program).args(args).exec();
    Err(format!("failed to exec {}: {err}", program.display()))
}

#[cfg(not(unix))]
fn exec_or_status(program: PathBuf, args: &[OsString]) -> Result<u8, String> {
    let status = Command::new(&program)
        .args(args)
        .status()
        .map_err(|err| format!("failed to run {}: {err}", program.display()))?;
    Ok(status.code().unwrap_or(1).try_into().unwrap_or(1))
}

#[allow(dead_code)]
fn _os_str(value: &str) -> &OsStr {
    OsStr::new(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_options_json_matches_contract() {
        let payload = download_options_payload();
        assert_eq!(payload["schema_version"], "ax.download_options.v1");
        assert!(
            payload["targets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|target| target["alias"] == "gemma4-12b")
        );
    }

    #[test]
    fn alias_resolution_matches_python_cli_contract() {
        let profile = profile_for_model("qwen36-35b").unwrap();
        assert_eq!(profile.preset, Some("qwen3.6-35b"));
        assert_eq!(profile.repo_id, "mlx-community/Qwen3.6-35B-A3B-4bit");
        let profile = profile_for_model("gemma-4-12b-it").unwrap();
        assert_eq!(profile.preset, Some("gemma4-12b"));
    }

    #[test]
    fn download_mtp_targets_cover_requested_6bit_models() {
        let cases = [
            (
                "qwen3.6-27b-6bit",
                "mlx-community/Qwen3.6-27B-6bit",
                MtpDownloadKind::QwenSidecar {
                    mtp_source: "Qwen/Qwen3.6-27B",
                },
            ),
            (
                "qwen3.6-35b-a3b",
                "mlx-community/Qwen3.6-35B-A3B-6bit",
                MtpDownloadKind::QwenSidecar {
                    mtp_source: "Qwen/Qwen3.6-35B-A3B",
                },
            ),
            (
                "gemma-4-12b",
                "mlx-community/gemma-4-12B-it-6bit",
                MtpDownloadKind::GemmaAssistant {
                    assistant_repo_id: "mlx-community/gemma-4-12B-it-assistant-6bit",
                    target_model_id: "gemma-4-12b-it",
                    assistant_model_id: "gemma-4-12b-it-assistant",
                    max_depth: 2,
                },
            ),
            (
                "gemma-4-26b",
                "mlx-community/gemma-4-26b-a4b-it-6bit",
                MtpDownloadKind::GemmaAssistant {
                    assistant_repo_id: "google/gemma-4-26b-a4b-it-assistant",
                    target_model_id: "gemma-4-26b-a4b-it",
                    assistant_model_id: "gemma-4-26b-a4b-it-assistant",
                    max_depth: 1,
                },
            ),
            (
                "gemma-4-31b",
                "mlx-community/gemma-4-31b-it-6bit",
                MtpDownloadKind::GemmaAssistant {
                    assistant_repo_id: "google/gemma-4-31b-it-assistant",
                    target_model_id: "gemma-4-31b-it",
                    assistant_model_id: "gemma-4-31b-it-assistant",
                    max_depth: 1,
                },
            ),
            (
                "glm-4.7-flash",
                "mlx-community/GLM-4.7-Flash-6bit",
                MtpDownloadKind::GlmSidecar {
                    mtp_source: "zai-org/GLM-4.7-Flash",
                },
            ),
        ];
        for (alias, repo_id, kind) in cases {
            let target = mtp_download_target_for_model(alias).unwrap();
            assert_eq!(target.repo_id, repo_id);
            assert!(target.repo_id.ends_with("6bit"));
            assert_eq!(target.kind, kind);
        }
        assert!(mtp_download_target_for_model("qwen3-coder-next").is_none());
    }

    #[test]
    fn parse_download_mtp_args_matches_convert_knobs() {
        let args = parse_download_mtp_args(&[
            OsString::from("qwen36-35b"),
            OsString::from("--output"),
            OsString::from("/tmp/qwen-mtp"),
            OsString::from("--force"),
            OsString::from("--quantize"),
            OsString::from("4"),
            OsString::from("--mtp-depth-max"),
            OsString::from("1"),
            OsString::from("--group-size"),
            OsString::from("128"),
            OsString::from("--fair-base-only"),
            OsString::from("--json"),
        ])
        .unwrap();
        assert_eq!(args.model, "qwen36-35b");
        assert_eq!(args.output.as_deref(), Some("/tmp/qwen-mtp"));
        assert!(args.force);
        assert_eq!(args.quantize.as_deref(), Some("4"));
        assert_eq!(args.mtp_depth_max.as_deref(), Some("1"));
        assert_eq!(args.group_size, "128");
        assert!(args.fair_base_only);
        assert!(args.json);
    }

    #[test]
    fn models_info_distinguishes_aliases_from_repo_ids() {
        let alias = model_info_payload("gemma4-12b").unwrap();
        assert_eq!(alias["kind"], "supported_alias");
        assert_eq!(
            alias["profile"]["repo_id"],
            "mlx-community/gemma-4-12B-it-4bit"
        );

        let repo = model_info_payload("mlx-community/custom-model").unwrap();
        assert_eq!(repo["kind"], "repo_id");
        assert_eq!(repo["managed_alias"], false);
    }

    #[test]
    fn models_list_reports_local_artifacts_from_explicit_root() {
        let root = unique_temp_dir("ax-engine-models-list");
        let model_dir = root.join("local-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model-manifest.json"), "{}").unwrap();

        let payload = models_list_payload(Some(&root));
        let items = payload["local_artifacts"]["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["kind"], "local_artifact");
        assert_eq!(items[0]["manifest_present"], true);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn models_rm_refuses_hugging_face_cache_paths() {
        let root = unique_temp_dir("ax-engine-models-rm");
        let cache_model = root
            .join("huggingface")
            .join("hub")
            .join("models--org--model");
        fs::create_dir_all(&cache_model).unwrap();
        fs::write(cache_model.join("config.json"), "{}").unwrap();

        let error = validate_model_rm_target(&cache_model, true)
            .expect_err("HF cache paths must be removed with cache tooling");
        assert!(error.contains("Hugging Face cache"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn models_rm_allows_dry_run_for_local_artifact_directories() {
        let root = unique_temp_dir("ax-engine-models-rm-local");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();

        let report = validate_model_rm_target(&root, true).unwrap();
        assert_eq!(report["dry_run"], true);
        assert_eq!(report["config_present"], true);
        assert!(root.exists(), "dry-run validation must not remove files");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parse_output_dir_handles_prepare_messages() {
        assert_eq!(
            parse_output_dir("Sidecar ready at:\n  /tmp/model\n", None).as_deref(),
            Some("/tmp/model")
        );
        assert_eq!(
            parse_output_dir("Output dir: /tmp/other\n", None).as_deref(),
            Some("/tmp/other")
        );
    }

    #[test]
    fn parse_doctor_args_preserves_summary_and_verbose_modes() {
        let args = parse_doctor_args(&[
            OsString::from("--json"),
            OsString::from("--mlx-model-artifacts-dir"),
            OsString::from("/models/gemma4-12b"),
        ])
        .unwrap();
        assert!(args.json);
        assert!(!args.verbose);
        assert!(!args.help);
        assert_eq!(
            args.bench_args,
            vec![
                OsString::from("--mlx-model-artifacts-dir"),
                OsString::from("/models/gemma4-12b")
            ]
        );

        let args = parse_doctor_args(&[OsString::from("--verbose")]).unwrap();
        assert!(!args.json);
        assert!(args.verbose);
    }

    #[test]
    fn user_doctor_text_highlights_status_checks_and_next_steps() {
        let report = json!({
            "result": "ready",
            "install": {"version": "6.4.3", "mode": "installed_tools"},
            "host": {
                "os": "macos",
                "arch": "aarch64",
                "os_version": "15.5",
                "ram_gib": 64,
                "cpu_cores": {
                    "physical": 16,
                    "logical": 16,
                    "performance": 12,
                    "efficiency": 4,
                    "summary": "16 (4 Efficiency and 12 Performance)",
                    "types": {
                        "efficiency": 4,
                        "performance": 12
                    }
                },
                "gpu_cores": 40
            },
            "checks": [
                {"id": "server_binary", "status": "pass", "detail": "ax-engine-server ok"},
                {"id": "model", "status": "not_selected", "selected": false, "path": null}
            ],
            "issues": [],
            "model_issues": [],
            "next_actions": ["ax-engine serve qwen36-35b --download --port 8080"],
            "details_command": "ax-engine-bench doctor"
        });
        let output = format_user_doctor_report(&report);
        assert!(output.contains("AX Engine doctor"));
        assert!(output.contains("Result: ready"));
        assert!(output.contains("host: macos 15.5 (aarch64)"));
        assert!(output.contains("RAM: 64 GiB"));
        assert!(output.contains("CPU cores: 16 (4 Efficiency and 12 Performance)"));
        assert!(output.contains("GPU cores: 40"));
        assert!(output.contains("server_binary: pass - ax-engine-server ok"));
        assert!(output.contains("model: not_selected"));
        assert!(output.contains("ax-engine serve qwen36-35b --download --port 8080"));
        assert!(output.contains("More details: ax-engine-bench doctor"));
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("{label}-{nanos}"))
    }
}
