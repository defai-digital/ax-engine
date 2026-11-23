// SPDX-License-Identifier: Apache-2.0

//! Default reader scope measured on the pinned resident Flash Next MXFP4 export.

use std::fs::{File, Metadata};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const CONFIG_SHA256: &str = "fb3c637ce4a240453c454fbc1211fa1e893d0af12df54a9df19c9c017fa59237";
const AXQ_MANIFEST_SHA256: &str =
    "727cfc8014dafe7d894f3ded5b5f1351f133595e195891d59202425e0f468a3c";
const SIDECAR_HEADER_SHA256: &str =
    "d2170b4aafb916374d809b091ddde58288db99ce9652ee5f77a2b5d076ab6cd9";
const SIDECAR_LEN: u64 = 5_214_305_544;
const TESTED_MEMORY_BYTES: u64 = 128 * 1024 * 1024 * 1024;
const STARTUP_DATA: &str = "/System/Volumes/Data";
const DISK_PROBE_TIMEOUT: Duration = Duration::from_millis(1250);

/// Bind the automatic choice to the tested on-disk layout on the opened descriptor.
pub(super) fn matches_header(file_len: u64, header: &[u8]) -> bool {
    file_len == SIDECAR_LEN
        && header.len() == 3840
        && ax_engine_core::sha256_hex(header) == SIDECAR_HEADER_SHA256
}

pub(super) fn enabled(
    root: &Path,
    sidecar: &Metadata,
    matches_header: bool,
    experts_paged: bool,
) -> bool {
    use_prefetch(
        crate::fastpath::flash_next_mtp_sidecar_prefetch_override(),
        matches_header,
        experts_paged,
        || on_startup_data_volume(sidecar) && matches_export(root) && tested_internal_startup(),
    )
}

fn use_prefetch(
    explicit: Option<bool>,
    matches_header: bool,
    experts_paged: bool,
    installed_on_tested_platform: impl FnOnce() -> bool,
) -> bool {
    explicit.unwrap_or_else(|| matches_header && !experts_paged && installed_on_tested_platform())
}

fn on_startup_data_volume(sidecar: &Metadata) -> bool {
    std::fs::metadata(STARTUP_DATA).is_ok_and(|data| data.dev() == sidecar.dev())
}

fn matches_export(root: &Path) -> bool {
    // Metadata fingerprints identify the measured export's format and layout.
    matches_metadata(&root.join("config.json"), 250_621, CONFIG_SHA256)
        && matches_metadata(
            &root.join("axquant_manifest.json"),
            11_345,
            AXQ_MANIFEST_SHA256,
        )
}

fn matches_metadata(path: &Path, expected_size: u64, expected_sha256: &str) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    if !metadata.is_file() || metadata.len() != expected_size {
        return false;
    }
    let Ok(file) = File::open(path) else {
        return false;
    };
    let mut bytes = Vec::new();
    file.take(expected_size + 1).read_to_end(&mut bytes).is_ok()
        && bytes.len() as u64 == expected_size
        && ax_engine_core::sha256_hex(&bytes) == expected_sha256
}

fn tested_host(brand: Option<&str>, memory: Option<u64>) -> bool {
    brand == Some("Apple M5 Max") && memory == Some(TESTED_MEMORY_BYTES)
}

fn tested_internal_startup() -> bool {
    static ELIGIBLE: OnceLock<bool> = OnceLock::new();
    *ELIGIBLE.get_or_init(|| {
        let brand = crate::hardware::sysctl_string(&["-n", "machdep.cpu.brand_string"]);
        tested_host(
            brand.as_deref(),
            crate::expert_stream::unified_memory_bytes(),
        ) && startup_disk_info().is_some_and(|plist| internal_solid_state(&plist))
    })
}

fn startup_disk_info() -> Option<Vec<u8>> {
    let mut child = Command::new("/usr/sbin/diskutil")
        .args(["info", "-plist", STARTUP_DATA])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + DISK_PROBE_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                return child.wait_with_output().ok().map(|output| output.stdout);
            }
            Ok(Some(_)) => return None,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(None) | Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    None
}

fn internal_solid_state(plist: &[u8]) -> bool {
    std::str::from_utf8(plist).is_ok_and(|plist| {
        ["<key>Internal</key>", "<key>SolidState</key>"]
            .into_iter()
            .all(|key| {
                let mut values = plist.split(key).skip(1);
                values
                    .next()
                    .is_some_and(|value| value.trim_start().starts_with("<true/>"))
                    && values.next().is_none()
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_reader_selection_takes_precedence() {
        let calls = std::cell::Cell::new(0);
        let probe = || {
            calls.set(calls.get() + 1);
            true
        };
        assert!(use_prefetch(Some(true), false, true, probe));
        assert!(!use_prefetch(Some(false), true, false, probe));
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn requires_matching_layout_and_resident_experts_before_probing() {
        let calls = std::cell::Cell::new(0);
        let probe = || {
            calls.set(calls.get() + 1);
            true
        };
        assert!(!use_prefetch(None, false, false, probe));
        assert!(!use_prefetch(None, true, true, probe));
        assert_eq!(calls.get(), 0);
        assert!(use_prefetch(None, true, false, probe));
        assert_eq!(calls.get(), 1);
        assert!(!use_prefetch(None, true, false, || false));
    }

    #[test]
    fn default_is_limited_to_the_measured_host() {
        assert!(tested_host(Some("Apple M5 Max"), Some(TESTED_MEMORY_BYTES)));
        for (brand, memory) in [
            (Some("Apple M5 Max"), None),
            (Some("Apple M5 Max"), Some(TESTED_MEMORY_BYTES / 2)),
            (Some("Apple M4 Pro"), Some(TESTED_MEMORY_BYTES)),
            (Some("Apple M5"), Some(TESTED_MEMORY_BYTES)),
            (None, Some(TESTED_MEMORY_BYTES)),
        ] {
            assert!(!tested_host(brand, memory));
        }
    }

    #[test]
    fn disk_probe_requires_the_internal_solid_state_booleans() {
        assert!(internal_solid_state(
            b"<key>SolidState</key>\n\t<true/>\n<key>Internal</key><true/>"
        ));
        for report in [
            b"<key>Internal</key><false/><key>SolidState</key><true/>".as_slice(),
            b"<key>Internal</key><true/><key>SolidState</key><false/>",
            b"<key>Internal</key><true/>",
            b"<key>Internal</key><true/><key>SolidState</key><string>true</string>",
            b"<key>Internal</key><true/><key>Internal</key><false/><key>SolidState</key><true/>",
        ] {
            assert!(!internal_solid_state(report));
        }
    }
}
