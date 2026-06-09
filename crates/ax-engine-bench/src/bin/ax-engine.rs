use serde_json::{Value, json};
use std::env;
use std::ffi::{OsStr, OsString};
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
        "doctor" => cmd_doctor(&args[1..]),
        "convert-mtplx" => cmd_convert_mtplx(&args[1..]),
        unknown => Err(format!(
            "unknown command: {unknown}\n\nRun `ax-engine --help` for usage."
        )),
    }
}

fn print_usage() {
    println!(
        "Usage:\n  ax-engine serve <model-dir-or-alias> [--host <host>] [--port <port>] [--download] [--dry-run] [--json] [-- <ax-engine-server args>]\n  ax-engine download [<alias-or-repo-id>] [--dest <path>] [--force] [--list] [--json]\n  ax-engine doctor [--json] [--mlx-model-artifacts-dir <path>]\n  ax-engine convert-mtplx <base-model> --mtp-source <repo> [--output <dir>] [--quantize 4|8] [--mtp-depth-max <n>] [--group-size <n>] [--fair-base-only] [--json]"
    );
}

fn cmd_doctor(args: &[OsString]) -> Result<u8, String> {
    let mut argv = vec![OsString::from("doctor")];
    argv.extend(args.iter().cloned());
    exec_or_status(find_executable("ax-engine-bench"), &argv)
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

#[derive(Debug)]
struct DownloadArgs {
    model: Option<String>,
    dest: Option<String>,
    force: bool,
    list: bool,
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
        print_json(&json!({
            "schema_version": "ax.convert_mtplx.v1",
            "command": "convert-mtplx",
            "base_model": args.base_model,
            "mtp_source": args.mtp_source,
            "mtp_depth_max": depth.parse::<u32>().unwrap_or(1),
            "output_dir": output_dir,
            "provenance": provenance_summary,
        }));
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
}
