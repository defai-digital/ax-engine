//! Host hardware summary shown on the Home screen and used by the RAM-fit
//! badge.  Probed once at startup via `sysctl`/`df` shell-outs so the crate
//! needs no new dependencies; every field degrades to `None` when a probe
//! fails (non-macOS host, missing tool, unparsable output).

use std::path::Path;
use std::process::Command;

pub(super) struct HardwareInfo {
    /// e.g. "Apple M5 Max".
    pub chip: Option<String>,
    pub total_ram_bytes: Option<u64>,
    /// Free space on the volume holding the HF cache (the download target).
    pub free_disk_bytes: Option<u64>,
}

impl HardwareInfo {
    pub fn probe() -> Self {
        let cache_root = crate::default_hf_cache_root();
        HardwareInfo {
            chip: sysctl_string("machdep.cpu.brand_string"),
            total_ram_bytes: sysctl_string("hw.memsize").and_then(|s| s.parse().ok()),
            free_disk_bytes: free_disk_bytes(&cache_root),
        }
    }

    #[cfg(test)]
    pub fn for_tests() -> Self {
        HardwareInfo {
            chip: Some("Test Chip".into()),
            total_ram_bytes: Some(64 * 1024 * 1024 * 1024),
            free_disk_bytes: Some(500 * 1024 * 1024 * 1024),
        }
    }
}

fn sysctl_string(key: &str) -> Option<String> {
    let output = Command::new("sysctl").args(["-n", key]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() { None } else { Some(value) }
}

/// Free bytes on the volume containing `path` (walking up to the nearest
/// existing ancestor), via `df -k`.
pub(super) fn free_disk_bytes(path: &Path) -> Option<u64> {
    let mut candidate = path;
    while !candidate.exists() {
        candidate = candidate.parent()?;
    }
    let output = Command::new("df").arg("-k").arg(candidate).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    parse_df_available_kib(&text).map(|kib| kib * 1024)
}

/// The "Available" column (KiB) from `df -k` output: fourth field of the data
/// row.  Handles the multi-line wrap `df` uses for long device names by
/// concatenating all lines after the header.
pub(super) fn parse_df_available_kib(output: &str) -> Option<u64> {
    let mut lines = output.lines();
    let header = lines.next()?;
    let avail_col = header.split_whitespace().position(|col| {
        col.eq_ignore_ascii_case("avail") || col.eq_ignore_ascii_case("available")
    })?;
    let data = lines.collect::<Vec<_>>().join(" ");
    data.split_whitespace().nth(avail_col)?.parse().ok()
}
