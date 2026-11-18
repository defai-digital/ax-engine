//! Experimental reference-free NCM CLI; no normal serve/chat admission.
use std::process::ExitCode;
#[cfg(feature = "mlx-native")]
fn run() -> Result<(), String> {
    use ax_engine_sdk::ncm::{NcmGenerateRequest, NcmOutputMode, NcmSamplingMode, NcmSession};
    use std::{collections::BTreeSet, path::PathBuf, time::Duration};
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "Usage: ncm-infer <model-dir> (--sampling greedy | --sampling seeded-v1 --seed <u64>) [--memory-budget-mib 2048] [--timeout-ms 60000] [--json]\nExperimental unconditional F32 TinyStories only."
        );
        return Ok(());
    }
    let path = PathBuf::from(args.first().ok_or("missing model directory; see --help")?);
    let mut memory = 2048u64;
    let mut timeout = 60000u64;
    let mut sampling = None;
    let mut seed = None;
    let mut json = false;
    let mut seen = BTreeSet::new();
    let mut i = 1;
    while i < args.len() {
        let flag = args[i].to_str().ok_or("invalid argument encoding")?;
        if !seen.insert(flag) {
            return Err(format!("duplicate option {flag}"));
        }
        if flag == "--json" {
            json = true;
            i += 1;
            continue;
        }
        if ![
            "--sampling",
            "--seed",
            "--memory-budget-mib",
            "--timeout-ms",
        ]
        .contains(&flag)
        {
            return Err(format!("unsupported option {flag}"));
        }
        let value = args
            .get(i + 1)
            .and_then(|v| v.to_str())
            .ok_or("missing option value")?;
        match flag {
            "--sampling" => sampling = Some(value),
            "--seed" => seed = Some(value.parse().map_err(|_| "invalid unsigned seed")?),
            "--memory-budget-mib" => memory = value.parse().map_err(|_| "invalid memory budget")?,
            "--timeout-ms" => timeout = value.parse().map_err(|_| "invalid timeout")?,
            _ => {}
        }
        i += 2;
    }
    let sampling = match (sampling, seed) {
        (Some("greedy"), None) => NcmSamplingMode::Greedy,
        (Some("greedy"), Some(_)) => return Err("--seed requires --sampling seeded-v1".into()),
        (Some("seeded-v1"), Some(seed)) => NcmSamplingMode::SeededV1 { seed },
        (Some("seeded-v1"), None) => return Err("--sampling seeded-v1 requires --seed".into()),
        _ => return Err("--sampling must be greedy or seeded-v1".into()),
    };
    let bytes = memory
        .checked_mul(1024 * 1024)
        .ok_or("memory budget overflow")?;
    let session = NcmSession::load_experimental(&path, bytes, Duration::from_millis(timeout))
        .map_err(|e| e.to_string())?;
    let response = session
        .generate_unconditional(NcmGenerateRequest {
            sampling,
            output_mode: NcmOutputMode::FullSequence128,
        })
        .map_err(|e| e.to_string())?
        .finish()
        .map_err(|e| e.to_string())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&response).map_err(|e| e.to_string())?
        );
    } else {
        print!("{}", response.text);
    }
    Ok(())
}
#[cfg(not(feature = "mlx-native"))]
fn run() -> Result<(), String> {
    Err("ncm-infer requires the mlx-native feature".into())
}
fn main() -> ExitCode {
    match std::panic::catch_unwind(run) {
        Ok(Ok(())) => ExitCode::SUCCESS,
        Ok(Err(e)) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
        Err(_) => ExitCode::from(3),
    }
}
