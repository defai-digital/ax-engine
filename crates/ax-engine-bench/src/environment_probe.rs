use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

use crate::error::CliError;

pub(crate) fn detect_system_model() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sysctl", &["-n", "hw.model"]),
        _ => None,
    }
}

pub(crate) fn detect_soc() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"]),
        _ => None,
    }
}

pub(crate) fn detect_memory_bytes() -> Option<u64> {
    match env::consts::OS {
        "macos" => command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.parse::<u64>().ok()),
        _ => None,
    }
}

pub(crate) fn detect_os_version() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sw_vers", &["-productVersion"]),
        _ => None,
    }
}

pub(crate) fn detect_os_build() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sw_vers", &["-buildVersion"]),
        _ => None,
    }
}

pub(crate) fn detect_kernel_release() -> Option<String> {
    command_stdout("uname", &["-r"])
}

pub(crate) fn default_metal_driver() -> &'static str {
    match env::consts::OS {
        "macos" => "system-default",
        _ => "unavailable",
    }
}

pub(crate) fn bytes_to_gib(bytes: u64) -> u64 {
    bytes / (1024 * 1024 * 1024)
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

pub(crate) fn file_fingerprint_fnv1a64(path: &Path) -> Result<String, CliError> {
    let bytes = fs::read(path).map_err(|error| {
        CliError::Runtime(format!(
            "failed to read {} for benchmark provenance fingerprint: {error}",
            path.display()
        ))
    })?;
    Ok(format!("{:016x}", fnv1a64(&bytes)))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS_64: u64 = 0xcbf29ce484222325;
    const FNV_PRIME_64: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS_64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }
    hash
}
