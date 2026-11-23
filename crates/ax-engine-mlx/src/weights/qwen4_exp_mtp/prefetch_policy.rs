// SPDX-License-Identifier: Apache-2.0

//! Default reader scope measured on the pinned resident Flash Next MXFP4 export.

use std::fs::{Metadata, OpenOptions};
use std::io::{ErrorKind, Read};
use std::os::fd::OwnedFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

const CONFIG_SHA256: &str = "fb3c637ce4a240453c454fbc1211fa1e893d0af12df54a9df19c9c017fa59237";
const AXQ_MANIFEST_SHA256: &str =
    "727cfc8014dafe7d894f3ded5b5f1351f133595e195891d59202425e0f468a3c";
const SIDECAR_HEADER_SHA256: &str =
    "d2170b4aafb916374d809b091ddde58288db99ce9652ee5f77a2b5d076ab6cd9";
const SIDECAR_LEN: u64 = 5_214_305_544;
const TESTED_MEMORY_BYTES: u64 = 128 * 1024 * 1024 * 1024;
const STARTUP_DATA: &str = "/System/Volumes/Data";
const DISK_PROBE_TIMEOUT: Duration = Duration::from_millis(1250);
const MAX_DISK_REPORT_BYTES: usize = 64 * 1024;

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
    let Ok(file) = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
        .open(path)
    else {
        return false;
    };
    if !file
        .metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() == expected_size)
    {
        return false;
    }
    let mut bytes = Vec::new();
    file.take(expected_size + 1).read_to_end(&mut bytes).is_ok()
        && bytes.len() as u64 == expected_size
        && ax_engine_core::sha256_hex(&bytes) == expected_sha256
}

fn tested_host(brand: Option<&str>, memory: Option<u64>) -> bool {
    brand == Some("Apple M5 Max") && memory == Some(TESTED_MEMORY_BYTES)
}

fn tested_internal_startup() -> bool {
    static ELIGIBLE: Mutex<Option<bool>> = Mutex::new(None);
    let Some(mut cached) = ELIGIBLE.try_lock() else {
        return false;
    };
    if let Some(eligible) = *cached {
        return eligible;
    }
    *cached = probe_tested_internal_startup();
    cached.unwrap_or(false)
}

fn probe_tested_internal_startup() -> Option<bool> {
    let brand = crate::hardware::sysctl_string(&["-n", "machdep.cpu.brand_string"])?;
    let memory = crate::expert_stream::unified_memory_bytes()?;
    if !tested_host(Some(&brand), Some(memory)) {
        return Some(false);
    }
    let report =
        bounded_probe(Command::new("/usr/sbin/diskutil").args(["info", "-plist", STARTUP_DATA]))?;
    internal_solid_state(&report)
}

fn bounded_probe(command: &mut Command) -> Option<Vec<u8>> {
    let (mut reader, writer) = UnixStream::pair().ok()?;
    reader.set_nonblocking(true).ok()?;
    let child_stdout: OwnedFd = writer.into();
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::from(child_stdout))
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // A reusable Command retains the configured stdout descriptor after spawn.
    command.stdout(Stdio::null());
    let output = collect_probe(&mut child, &mut reader, Instant::now() + DISK_PROBE_TIMEOUT);
    if output.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    output
}

fn collect_probe(child: &mut Child, reader: &mut UnixStream, deadline: Instant) -> Option<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0; 4096];
    let mut eof = false;
    while Instant::now() < deadline {
        let mut received = false;
        if !eof {
            match reader.read(&mut buffer) {
                Ok(0) => eof = true,
                Ok(n) if output.len() + n <= MAX_DISK_REPORT_BYTES => {
                    output.extend_from_slice(&buffer[..n]);
                    received = true;
                }
                Err(error)
                    if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => {}
                Ok(_) | Err(_) => return None,
            }
        }
        if let Some(status) = child.try_wait().ok()? {
            if !status.success() {
                return None;
            }
            if eof {
                return Some(output);
            }
        }
        if !received {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    None
}

fn internal_solid_state(plist: &[u8]) -> Option<bool> {
    fn property(plist: &str, key: &str) -> Option<bool> {
        let mut values = plist.split(key).skip(1);
        let value = values.next()?.trim_start();
        if values.next().is_some() {
            return None;
        }
        if value.starts_with("<true/>") {
            return Some(true);
        }
        value.starts_with("<false/>").then_some(false)
    }
    let plist = std::str::from_utf8(plist).ok()?;
    let internal = property(plist, "<key>Internal</key>")?;
    let solid_state = property(plist, "<key>SolidState</key>")?;
    Some(internal && solid_state)
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
        assert_eq!(
            internal_solid_state(b"<key>SolidState</key>\n\t<true/>\n<key>Internal</key><true/>"),
            Some(true)
        );
        assert_eq!(
            internal_solid_state(b"<key>Internal</key><false/><key>SolidState</key><true/>"),
            Some(false)
        );
        assert_eq!(
            internal_solid_state(b"<key>Internal</key><true/><key>SolidState</key><false/>"),
            Some(false)
        );
        for report in [
            b"<key>Internal</key><true/>".as_slice(),
            b"<key>Internal</key><true/><key>SolidState</key><string>true</string>",
            b"<key>Internal</key><true/><key>Internal</key><false/><key>SolidState</key><true/>",
        ] {
            assert_eq!(internal_solid_state(report), None);
        }
    }

    #[test]
    fn probe_drains_stdout_and_caps_the_result() {
        assert_eq!(
            bounded_probe(Command::new("/bin/echo").arg("disk probe")),
            Some(b"disk probe\n".to_vec())
        );
        assert!(bounded_probe(&mut Command::new("/usr/bin/yes")).is_none());
    }

    #[test]
    fn probe_finishes_when_a_writer_holds_stdout_open() {
        let (mut reader, _writer) = UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        let mut child = Command::new("/usr/bin/true").spawn().unwrap();
        let deadline = Instant::now() + Duration::from_millis(40);
        assert!(collect_probe(&mut child, &mut reader, deadline).is_none());
        assert!(child.wait().unwrap().success());
    }

    #[test]
    fn metadata_fingerprint_reads_only_regular_files() {
        let temporary = super::super::tests::temp_root("prefetch-fingerprint");
        let dir = &temporary.0;
        let file = dir.join("config.json");
        std::fs::write(&file, b"test").unwrap();
        let expected = ax_engine_core::sha256_hex(b"test");
        assert!(matches_metadata(&file, 4, &expected));
        assert!(!matches_metadata(&file, 3, &expected));

        let link = dir.join("linked-config.json");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        assert!(!matches_metadata(&link, 4, &expected));

        let fifo = dir.join("named-pipe");
        assert!(
            Command::new("/usr/bin/mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        assert!(!matches_metadata(
            &fifo,
            0,
            &ax_engine_core::sha256_hex(b"")
        ));
    }
}
