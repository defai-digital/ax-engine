// SPDX-License-Identifier: Apache-2.0

//! Automatic buffered reads for the measured Flash Next resident profile.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read};
use std::os::fd::OwnedFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

const STARTUP_DATA: &str = "/System/Volumes/Data";
const DISK_PROBE_TIMEOUT: Duration = Duration::from_millis(1250);
const MAX_DISK_REPORT_BYTES: usize = 64 * 1024;
const PUBLISHED_METADATA: [(&str, u64, &str); 2] = [
    (
        "config.json",
        250_621,
        "fb3c637ce4a240453c454fbc1211fa1e893d0af12df54a9df19c9c017fa59237",
    ),
    (
        "model.safetensors.index.json",
        290_934,
        "37344288777e7e78b35901cb1d572dad0dc640cb9cedb90024860b932a657696",
    ),
];

// Layout fingerprints exclude weights. Publisher payload integrity remains a
// delivery check and the selected tensor extents remain checked by the loader.
const RESIDENT_SHARDS: [(u8, u64, usize, &str); 15] = [
    (
        1,
        5_283_244_832,
        49_216,
        "9d824a41e33f3456b9ebd9a69754e07492492f5730cf7be99bda90eacad43f64",
    ),
    (
        12,
        5_045_785_762,
        9_866,
        "6240068f3a59dac1864e1be647273baca148f6458d53d903093974edbdf20890",
    ),
    (
        13,
        5_082_667_435,
        16_995,
        "e263c53b4e0cbbb808b66b8f141f6033b499385a5b8834697296e00735b06603",
    ),
    (
        14,
        5_144_030_876,
        21_588,
        "1ff09bc99699af580c78e4eee48af90325fcc627a03e19f9190c3ab266a8b118",
    ),
    (
        15,
        5_144_031_001,
        21_713,
        "2218563a3d16bb60c0bd662e50708839b244e7fe7935368b920ebde13ab8f165",
    ),
    (
        16,
        5_082_667_551,
        17_111,
        "4ccd49ac04e011f657ee1847d4c1eac24fa472ce6a192403f90201033c85090c",
    ),
    (
        17,
        5_144_031_014,
        21_726,
        "bde01813b85c5f85666ccaa17778fcb72d67a76eb8bc0c568e575572b15e27a9",
    ),
    (
        18,
        5_144_030_972,
        21_684,
        "8262c2599e1d00a13dfd41fb108aa5286fe103918b0267018b4dffa9fd35ab3c",
    ),
    (
        19,
        5_082_667_581,
        17_141,
        "59af438c97b36100ccc3a4a246c6ea1c687dab681f288a57466e3ea39a9cdfa9",
    ),
    (
        20,
        5_144_031_040,
        21_752,
        "e00028cac4ea0d205d31493ef34a806c5581ce71e1c6cbcf90258bfcb480f622",
    ),
    (
        21,
        5_144_030_974,
        21_686,
        "9ad48ab8b012f8cb3c7a4a7f65f8eade10919fb97e45d0a95c453fcc61c9490b",
    ),
    (
        22,
        5_086_186_431,
        17_079,
        "b222e6c5b3dd97e8c867be5ef607ad1568acfa7bf1d3e134ed1fca615cc83cab",
    ),
    (
        23,
        5_144_031_102,
        21_814,
        "be333410a9c4faea0c2a9061b58d403d531c91640cc1d20791b119d406c509bf",
    ),
    (
        24,
        5_144_030_984,
        21_696,
        "9ecf90c166ec1c3a15994795ad79ce84e7c89d81f951fbb5bbe747d680d73717",
    ),
    (
        25,
        3_449_309_686,
        9_582,
        "9a61f59c4ce5c1ac45f46b8c3c6c2b1a3e117dd8674b74a12eb0ecefea6d1026",
    ),
];

pub(super) fn eligible<'a>(
    root: &Path,
    canonical_root: &Path,
    files: impl ExactSizeIterator<Item = &'a Path>,
) -> bool {
    let Ok(startup) = std::fs::metadata(STARTUP_DATA) else {
        return false;
    };
    // Check the resolved resident files. The snapshot directory itself can be on
    // a different volume from its blobs. The disk-served table is not part of this read.
    let device = startup.dev();
    files.len() == RESIDENT_SHARDS.len()
        && PUBLISHED_METADATA.iter().all(|&(name, length, digest)| {
            super::resolve_in_root(root, canonical_root, Path::new(name))
                .is_ok_and(|file| matches_metadata(&file, length, digest))
        })
        && matches_resident_shards(root, canonical_root, device, files.collect())
        && tested_internal_startup(device)
}

fn matches_resident_shards(
    root: &Path,
    canonical_root: &Path,
    device: u64,
    mut selected: HashSet<&Path>,
) -> bool {
    RESIDENT_SHARDS
        .iter()
        .all(|&(ordinal, size, header_bytes, digest)| {
            let name = format!("model-{ordinal:05}-of-00025.safetensors");
            super::resolve_in_root(root, canonical_root, Path::new(&name)).is_ok_and(|path| {
                selected.remove(path.as_path())
                    && matches_shard_header(&path, device, size, header_bytes, digest)
            })
        })
        && selected.is_empty()
}

fn open_regular(path: &Path, expected_size: u64, device: Option<u64>) -> Option<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
        .open(path)
        .ok()?;
    let metadata = file.metadata().ok()?;
    (metadata.is_file()
        && metadata.len() == expected_size
        && device.is_none_or(|device| metadata.dev() == device))
    .then_some(file)
}

fn matches_shard_header(path: &Path, device: u64, size: u64, length: usize, digest: &str) -> bool {
    let Some(mut file) = open_regular(path, size, Some(device)) else {
        return false;
    };
    let mut prefix = [0; 8];
    if file.read_exact(&mut prefix).is_err() || u64::from_le_bytes(prefix) != length as u64 {
        return false;
    }
    let mut header = vec![0; length];
    file.read_exact(&mut header).is_ok() && ax_engine_core::sha256_hex(&header) == digest
}

fn matches_metadata(path: &Path, expected_size: u64, expected_sha256: &str) -> bool {
    let Some(file) = open_regular(path, expected_size, None) else {
        return false;
    };
    let mut bytes = Vec::new();
    file.take(expected_size + 1).read_to_end(&mut bytes).is_ok()
        && bytes.len() as u64 == expected_size
        && ax_engine_core::sha256_hex(&bytes) == expected_sha256
}

fn tested_host(report: &str) -> bool {
    report
        .lines()
        .map(str::trim)
        .eq(["Mac17,6", "137438953472", "Apple M5 Max", "16384"])
}

#[derive(Default)]
struct StartupProbe {
    pending: Option<(u64, Receiver<Option<bool>>)>,
    known: Option<(u64, bool)>,
}

impl StartupProbe {
    fn check(
        &mut self,
        device: u64,
        launch: impl FnOnce() -> Option<Receiver<Option<bool>>>,
        timeout: Duration,
    ) -> bool {
        if let Some((known_device, verdict)) = self.known
            && known_device == device
        {
            return verdict;
        }
        let budget = if self.pending.is_some() {
            Duration::ZERO
        } else {
            self.pending = launch().map(|receiver| (device, receiver));
            timeout
        };
        let Some((pending_device, receiver)) = &self.pending else {
            return false;
        };
        let pending_device = *pending_device;
        match receiver.recv_timeout(budget) {
            Err(RecvTimeoutError::Timeout) => false,
            result => {
                self.pending = None;
                let verdict = result.ok().flatten();
                if pending_device != device {
                    return false;
                }
                self.known = verdict.map(|verdict| (device, verdict));
                verdict.unwrap_or(false)
            }
        }
    }
}

fn tested_internal_startup(device: u64) -> bool {
    static PROBE: Mutex<StartupProbe> = Mutex::new(StartupProbe {
        pending: None,
        known: None,
    });
    PROBE.try_lock().is_some_and(|mut probe| {
        probe.check(device, || launch_startup_probe(device), DISK_PROBE_TIMEOUT)
    })
}

fn launch_startup_probe(device: u64) -> Option<Receiver<Option<bool>>> {
    // Keep spawn, I/O, and reaping off the loader thread. A pending worker
    // makes subsequent loads fall back immediately without starting another.
    let (send, receive) = mpsc::channel();
    std::thread::Builder::new()
        .name("flash-next-storage-probe".into())
        .spawn(move || {
            let _ = send.send(probe_tested_internal_startup(device));
        })
        .ok()?;
    Some(receive)
}

fn probe_tested_internal_startup(device: u64) -> Option<bool> {
    if std::fs::metadata(STARTUP_DATA).ok()?.dev() != device {
        return None;
    }
    let mut sysctl = Command::new("/usr/sbin/sysctl");
    sysctl.args([
        "-n",
        "hw.model",
        "hw.memsize",
        "machdep.cpu.brand_string",
        "hw.pagesize",
    ]);
    let report = bounded_probe(sysctl)?;
    if !tested_host(std::str::from_utf8(&report).ok()?) {
        return Some(false);
    }
    let mut command = Command::new("/usr/sbin/diskutil");
    command.args(["info", "-plist", STARTUP_DATA]);
    let report = bounded_probe(command)?;
    if std::fs::metadata(STARTUP_DATA).ok()?.dev() != device {
        return None;
    }
    internal_solid_state(&report)
}

fn bounded_probe(mut command: Command) -> Option<Vec<u8>> {
    let (mut reader, writer) = UnixStream::pair().ok()?;
    reader.set_nonblocking(true).ok()?;
    let child_stdout: OwnedFd = writer.into();
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::from(child_stdout))
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    drop(command);
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
    fn property(dict: &str, key: &str) -> Option<bool> {
        // diskutil's XML plist indents the root dictionary's entries by one tab.
        let mut lines = dict.lines();
        lines.find(|line| *line == key)?;
        let value = lines.next()?;
        if lines.any(|line| line == key) {
            return None;
        }
        match value {
            "\t<true/>" => Some(true),
            "\t<false/>" => Some(false),
            _ => None,
        }
    }
    let plist = std::str::from_utf8(plist).ok()?;
    let (_, dict) = plist.split_once("\n<dict>\n")?;
    let (dict, _) = dict.split_once("\n</dict>")?;
    if !property(dict, "\t<key>Internal</key>")? {
        return Some(false);
    }
    property(dict, "\t<key>SolidState</key>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffered_default_requires_the_measured_host_and_page_geometry() {
        let matched = "Mac17,6\n137438953472\nApple M5 Max\n16384";
        assert!(tested_host(matched));
        for (from, to) in [
            ("Mac17,6", "Mac16,5"),
            ("137438953472", "68719476736"),
            ("Apple M5 Max", "Apple M4 Pro"),
            ("16384", "4096"),
        ] {
            assert!(!tested_host(&matched.replace(from, to)));
        }
    }

    #[test]
    fn disk_probe_requires_top_level_internal_solid_state_booleans() {
        let check = |values| {
            let plist = format!("<plist version=\"1.0\">\n<dict>\n{values}\n</dict>\n</plist>");
            internal_solid_state(plist.as_bytes())
        };
        let internal = "\t<key>Internal</key>\n\t<true/>";
        let external = "\t<key>Internal</key>\n\t<false/>";
        let solid_state = "\t<key>SolidState</key>\n\t<true/>";
        let rotating = "\t<key>SolidState</key>\n\t<false/>";
        let nested = concat!(
            "\t<key>APFSPhysicalStores</key>\n\t<array>\n\t\t<dict>\n",
            "\t\t\t<key>Internal</key>\n\t\t\t<true/>\n",
            "\t\t\t<key>SolidState</key>\n\t\t\t<true/>\n\t\t</dict>\n\t</array>",
        );
        assert_eq!(check(format!("{solid_state}\n{internal}")), Some(true));
        assert_eq!(
            check(format!("{internal}\n{nested}\n{solid_state}")),
            Some(true)
        );
        assert_eq!(check(format!("{internal}\n{rotating}")), Some(false));
        assert_eq!(check(external.to_owned()), Some(false));
        for values in [
            nested.to_owned(),
            internal.to_owned(),
            format!("{internal}\n\t<key>SolidState</key>\n\t<string>true</string>"),
            format!("{internal}\n{external}\n{solid_state}"),
        ] {
            assert_eq!(check(values), None);
        }
    }

    #[test]
    fn storage_probe_timeout_keeps_a_single_pending_worker() {
        let mut probe = StartupProbe::default();
        let (send, pending) = mpsc::channel();
        assert!(!probe.check(7, || Some(pending), Duration::ZERO));
        assert!(!probe.check(7, || unreachable!(), Duration::from_secs(10)));
        send.send(Some(true)).unwrap();
        assert!(probe.check(7, || unreachable!(), Duration::ZERO));
        assert!(probe.check(7, || unreachable!(), Duration::ZERO));
    }

    #[test]
    fn storage_probe_retries_after_an_inconclusive_worker() {
        let mut probe = StartupProbe::default();
        let (send, failed) = mpsc::channel();
        send.send(None).unwrap();
        assert!(!probe.check(7, || Some(failed), Duration::ZERO));
        let (send, external) = mpsc::channel();
        send.send(Some(false)).unwrap();
        assert!(!probe.check(7, || Some(external), Duration::ZERO));
        assert!(!probe.check(7, || unreachable!(), Duration::ZERO));
    }

    #[test]
    fn storage_probe_does_not_reuse_a_different_mount_verdict() {
        let mut probe = StartupProbe::default();
        let (send, pending) = mpsc::channel();
        assert!(!probe.check(7, || Some(pending), Duration::ZERO));
        send.send(Some(true)).unwrap();
        assert!(!probe.check(8, || unreachable!(), Duration::ZERO));
        let (send, replacement) = mpsc::channel();
        send.send(Some(false)).unwrap();
        assert!(!probe.check(8, || Some(replacement), Duration::ZERO));
        assert!(!probe.check(8, || unreachable!(), Duration::ZERO));
    }

    #[test]
    fn shard_header_fingerprint_rejects_a_different_header_or_device() {
        let dir = super::super::tests::norm_contract_dir("buffered-shard-fingerprint");
        let shard = dir.join("weights");
        let header = b"original";
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend(header);
        data.extend([0; 8]);
        std::fs::write(&shard, &data).unwrap();
        let device = std::fs::metadata(&shard).unwrap().dev();
        let size = data.len() as u64;
        let digest = ax_engine_core::sha256_hex(header);
        assert!(matches_shard_header(
            &shard,
            device,
            size,
            header.len(),
            &digest
        ));
        assert!(!matches_shard_header(
            &shard,
            device ^ 1,
            size,
            header.len(),
            &digest
        ));
        data[8] ^= 1;
        std::fs::write(&shard, &data).unwrap();
        assert!(!matches_shard_header(
            &shard,
            device,
            size,
            header.len(),
            &digest
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn probe_drains_stdout_and_caps_the_result() {
        let mut echo = Command::new("/bin/echo");
        echo.arg("disk probe");
        assert_eq!(bounded_probe(echo), Some(b"disk probe\n".to_vec()));
        assert!(bounded_probe(Command::new("/usr/bin/yes")).is_none());
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
        let dir = super::super::tests::norm_contract_dir("buffered-fingerprint");
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
        std::fs::remove_dir_all(dir).unwrap();
    }
}
