//! Live host metrics for the TUI Home panel (Mac / Apple Silicon).
//!
//! ## What macOS actually exposes (no sudo)
//!
//! | Metric | Source | Notes |
//! |---|---|---|
//! | Unified memory used/free/total | `vm_stat` + `hw.memsize` | not discrete VRAM |
//! | CPU average % | `ps -A -o %cpu=` / ncpu | host-wide (fast) |
//! | Load average · P/E cores | `vm.loadavg`, `hw.perflevel*` | |
//! | GPU util % | `ioreg` IOAccelerator `Device Utilization %` | AGX snapshot |
//! | GPU memory in use | `ioreg` `In use system memory` | from unified pool |
//! | Chip · GPU cores | `ioreg` model / `gpu-core-count` | static identity |
//! | Free disk | `df` on HF cache root | |
//! | Top RSS processes | `ps` | host processes |
//!
//! Omitted: discrete VRAM totals, fan, board power, encode engines
//! (`powermetrics` needs root).
//!
//! ## Sampling model
//!
//! All subprocess probes run on a **background thread**. The UI tick only
//! launches a sample (at most one in flight) and applies finished snapshots.
//! That keeps the event loop free — `top -l 1` alone can take ~1s on macOS
//! and must never block key handling or redraw.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

/// Samples kept for chart history (~4 min at one sample / 2 s).
pub(super) const HISTORY_LEN: usize = 120;

/// How often host probes append a chart point (slower = calmer trend line).
const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

/// One row in the process strip.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct TopProc {
    pub pid: u32,
    pub rss_bytes: u64,
    pub name: String,
}

/// Completed host probe payload (built off the UI thread).
#[derive(Clone, Debug, Default)]
struct MetricsSnapshot {
    total_ram_bytes: Option<u64>,
    used_ram_bytes: Option<u64>,
    free_ram_bytes: Option<u64>,
    cpu_percent: Option<f64>,
    logical_cpus: Option<u32>,
    perf_cpus: Option<u32>,
    eff_cpus: Option<u32>,
    load_1m: Option<f64>,
    free_disk_bytes: Option<u64>,
    top_procs: Vec<TopProc>,
    chip_name: Option<String>,
    gpu_cores: Option<u32>,
    gpu_percent: Option<f64>,
    gpu_mem_bytes: Option<u64>,
}

/// Inputs captured for a background sample (owned so the thread is `'static`).
struct SampleRequest {
    cache_root: PathBuf,
    need_identity: bool,
    total_ram_bytes: Option<u64>,
    logical_cpus: Option<u32>,
    perf_cpus: Option<u32>,
    eff_cpus: Option<u32>,
    chip_name: Option<String>,
    gpu_cores: Option<u32>,
}

#[derive(Debug, Default)]
pub(super) struct LiveMetrics {
    pub total_ram_bytes: Option<u64>,
    /// App-pressure used ≈ active + wired + compressor pages.
    pub used_ram_bytes: Option<u64>,
    /// free + speculative + purgeable pages (approx “available”).
    pub free_ram_bytes: Option<u64>,
    /// Average CPU utilization 0.0–100.0 across logical CPUs.
    pub cpu_percent: Option<f64>,
    pub logical_cpus: Option<u32>,
    /// Performance / efficiency core counts when available.
    pub perf_cpus: Option<u32>,
    pub eff_cpus: Option<u32>,
    pub load_1m: Option<f64>,
    /// Free space on the HF-cache volume (download target).
    pub free_disk_bytes: Option<u64>,
    /// Installed model footprint (bytes) — AX serving headroom proxy.
    pub models_bytes: u64,
    /// Top RSS processes.
    pub top_procs: Vec<TopProc>,

    /// Chip label from AGX / CPU brand (e.g. "Apple M5 Max").
    pub chip_name: Option<String>,
    /// GPU core count from `gpu-core-count`.
    pub gpu_cores: Option<u32>,
    /// Device utilization 0–100 from IOAccelerator.
    pub gpu_percent: Option<f64>,
    /// GPU “In use system memory” (bytes from unified pool, not VRAM).
    pub gpu_mem_bytes: Option<u64>,

    pub mem_history: VecDeque<u64>,
    pub cpu_history: VecDeque<u64>,
    pub gpu_history: VecDeque<u64>,
    pub models_history: VecDeque<u64>,

    last_sample: Option<Instant>,
    identity_probed: bool,
    /// In-flight background sample; UI never blocks on probes.
    pending: Option<Receiver<MetricsSnapshot>>,
}

impl LiveMetrics {
    pub fn new(total_ram_bytes: Option<u64>) -> Self {
        // Cheap sysctls only at construction — subprocess probes are deferred
        // to the first background sample so App::new stays snappy.
        LiveMetrics {
            total_ram_bytes,
            logical_cpus: sysctl_u32("hw.logicalcpu"),
            perf_cpus: sysctl_u32("hw.perflevel0.logicalcpu")
                .or_else(|| sysctl_u32("hw.perflevel0.physicalcpu")),
            eff_cpus: sysctl_u32("hw.perflevel1.logicalcpu")
                .or_else(|| sysctl_u32("hw.perflevel1.physicalcpu")),
            load_1m: parse_loadavg_1m(&sysctl_string("vm.loadavg").unwrap_or_default()),
            free_disk_bytes: None,
            ..Default::default()
        }
    }

    #[cfg(test)]
    pub fn for_tests() -> Self {
        let mut m = LiveMetrics {
            total_ram_bytes: Some(64 * 1024 * 1024 * 1024),
            used_ram_bytes: Some(24 * 1024 * 1024 * 1024),
            free_ram_bytes: Some(40 * 1024 * 1024 * 1024),
            cpu_percent: Some(18.5),
            logical_cpus: Some(12),
            perf_cpus: Some(8),
            eff_cpus: Some(4),
            load_1m: Some(1.2),
            free_disk_bytes: Some(500 * 1024 * 1024 * 1024),
            models_bytes: 8 * 1024 * 1024 * 1024,
            chip_name: Some("Apple M-series".into()),
            gpu_cores: Some(40),
            gpu_percent: Some(22.0),
            gpu_mem_bytes: Some(1500 * 1024 * 1024),
            top_procs: (1..=10)
                .map(|i| TopProc {
                    pid: i,
                    rss_bytes: (11 - i) as u64 * 200 * 1024 * 1024,
                    name: if i == 1 {
                        "Code".into()
                    } else if i == 2 {
                        "ax-engine-server".into()
                    } else {
                        format!("proc{i}")
                    },
                })
                .collect(),
            identity_probed: true,
            ..Default::default()
        };
        // Histories end at the same values as the live gauges (18.5 / 22 / 37.5%).
        for v in [20u64, 25, 30, 28, 35, 40, 38] {
            m.mem_history.push_back(v);
            m.cpu_history.push_back(v.saturating_sub(10));
            m.gpu_history.push_back(v.saturating_sub(5));
            m.models_history.push_back(12);
        }
        if let Some(last) = m.mem_history.back_mut() {
            *last = 38; // ~24/64 used
        }
        if let Some(last) = m.cpu_history.back_mut() {
            *last = 19;
        }
        if let Some(last) = m.gpu_history.back_mut() {
            *last = 22;
        }
        m
    }

    pub fn mem_ratio(&self) -> Option<f64> {
        let used = self.used_ram_bytes?;
        let total = self.total_ram_bytes?;
        if total == 0 {
            return None;
        }
        Some((used as f64 / total as f64).clamp(0.0, 1.0))
    }

    pub fn cpu_ratio(&self) -> Option<f64> {
        self.cpu_percent.map(|p| (p / 100.0).clamp(0.0, 1.0))
    }

    pub fn gpu_ratio(&self) -> Option<f64> {
        self.gpu_percent.map(|p| (p / 100.0).clamp(0.0, 1.0))
    }

    pub fn models_ratio(&self) -> Option<f64> {
        let total = self.total_ram_bytes?;
        if total == 0 {
            return None;
        }
        Some((self.models_bytes as f64 / total as f64).clamp(0.0, 1.0))
    }

    /// Pressure band from a 0–1 ratio (green / yellow / red gauges).
    pub fn pressure_band(ratio: f64) -> PressureBand {
        if ratio >= 0.85 {
            PressureBand::High
        } else if ratio >= 0.55 {
            PressureBand::Medium
        } else {
            PressureBand::Low
        }
    }

    /// Advance samplers; returns true only when new data actually landed
    /// (the TUI repaints on that signal, not on every poll cycle).
    ///
    /// Never blocks: subprocess probes run on a helper thread.
    pub fn tick(&mut self, models_bytes: u64, cache_root: &Path) -> bool {
        self.models_bytes = models_bytes;

        // Apply a finished sample first (non-blocking).
        if let Some(rx) = &self.pending {
            match rx.try_recv() {
                Ok(snap) => {
                    self.pending = None;
                    self.apply_snapshot(snap);
                    self.push_histories();
                    self.last_sample = Some(Instant::now());
                    return true;
                }
                Err(TryRecvError::Empty) => {
                    // Still sampling — do not launch another.
                    return false;
                }
                Err(TryRecvError::Disconnected) => {
                    self.pending = None;
                }
            }
        }

        let now = Instant::now();
        if self
            .last_sample
            .is_some_and(|t| now.duration_since(t) < SAMPLE_INTERVAL)
        {
            return false;
        }

        let need_identity = !self.identity_probed;
        if need_identity {
            // Avoid re-queueing identity probes if the worker is slow.
            self.identity_probed = true;
        }
        let request = SampleRequest {
            cache_root: cache_root.to_path_buf(),
            need_identity,
            total_ram_bytes: self.total_ram_bytes,
            logical_cpus: self.logical_cpus,
            perf_cpus: self.perf_cpus,
            eff_cpus: self.eff_cpus,
            chip_name: self.chip_name.clone(),
            gpu_cores: self.gpu_cores,
        };
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        thread::spawn(move || {
            let snap = collect_snapshot(request);
            let _ = tx.send(snap);
        });
        false
    }

    fn apply_snapshot(&mut self, snap: MetricsSnapshot) {
        if snap.total_ram_bytes.is_some() {
            self.total_ram_bytes = snap.total_ram_bytes;
        }
        if snap.used_ram_bytes.is_some() {
            self.used_ram_bytes = snap.used_ram_bytes;
        }
        if snap.free_ram_bytes.is_some() {
            self.free_ram_bytes = snap.free_ram_bytes;
        }
        if snap.cpu_percent.is_some() {
            self.cpu_percent = snap.cpu_percent;
        }
        if snap.logical_cpus.is_some() {
            self.logical_cpus = snap.logical_cpus;
        }
        if snap.perf_cpus.is_some() {
            self.perf_cpus = snap.perf_cpus;
        }
        if snap.eff_cpus.is_some() {
            self.eff_cpus = snap.eff_cpus;
        }
        if snap.load_1m.is_some() {
            self.load_1m = snap.load_1m;
        }
        if snap.free_disk_bytes.is_some() {
            self.free_disk_bytes = snap.free_disk_bytes;
        }
        if !snap.top_procs.is_empty() {
            self.top_procs = snap.top_procs;
        }
        if snap.chip_name.is_some() {
            self.chip_name = snap.chip_name;
        }
        if snap.gpu_cores.is_some() {
            self.gpu_cores = snap.gpu_cores;
        }
        if snap.gpu_percent.is_some() {
            self.gpu_percent = snap.gpu_percent;
        }
        if snap.gpu_mem_bytes.is_some() {
            self.gpu_mem_bytes = snap.gpu_mem_bytes;
        }
    }

    fn push_histories(&mut self) {
        // Lockstep histories: always push all three so series share an x-axis.
        // On probe miss, **hold last value** (never invent a dip to 0) so the
        // chart does not show false gaps. First sample may still be 0 until
        // that metric has been seen once.
        let mem = self
            .mem_ratio()
            .map(|r| (r * 100.0).round().clamp(0.0, 100.0) as u64)
            .or_else(|| self.mem_history.back().copied())
            .unwrap_or(0);
        let cpu = self
            .cpu_percent
            .map(|p| p.round().clamp(0.0, 100.0) as u64)
            .or_else(|| self.cpu_history.back().copied())
            .unwrap_or(0);
        let gpu = self
            .gpu_percent
            .map(|p| p.round().clamp(0.0, 100.0) as u64)
            .or_else(|| self.gpu_history.back().copied())
            .unwrap_or(0);
        push_hist(&mut self.mem_history, mem);
        push_hist(&mut self.cpu_history, cpu);
        push_hist(&mut self.gpu_history, gpu);
        if let Some(r) = self.models_ratio() {
            push_hist(&mut self.models_history, (r * 100.0).round() as u64);
        }
    }
}

/// Run every host probe off the UI thread.
fn collect_snapshot(req: SampleRequest) -> MetricsSnapshot {
    let mut snap = MetricsSnapshot {
        total_ram_bytes: req.total_ram_bytes,
        logical_cpus: req.logical_cpus,
        perf_cpus: req.perf_cpus,
        eff_cpus: req.eff_cpus,
        chip_name: req.chip_name,
        gpu_cores: req.gpu_cores,
        ..Default::default()
    };

    if snap.total_ram_bytes.is_none() {
        snap.total_ram_bytes = sysctl_string("hw.memsize").and_then(|s| s.parse().ok());
    }
    if snap.logical_cpus.is_none() {
        snap.logical_cpus = sysctl_u32("hw.logicalcpu");
    }
    if req.need_identity {
        if snap.perf_cpus.is_none() {
            snap.perf_cpus = sysctl_u32("hw.perflevel0.logicalcpu")
                .or_else(|| sysctl_u32("hw.perflevel0.physicalcpu"));
        }
        if snap.eff_cpus.is_none() {
            snap.eff_cpus = sysctl_u32("hw.perflevel1.logicalcpu")
                .or_else(|| sysctl_u32("hw.perflevel1.physicalcpu"));
        }
        if snap.chip_name.is_none()
            && let Some(out) = Command::new("ioreg")
                .args(["-r", "-d", "1", "-c", "IOAccelerator"])
                .output()
                .ok()
                .filter(|o| o.status.success())
        {
            let text = String::from_utf8_lossy(&out.stdout);
            let sample = parse_ioreg_gpu(&text);
            snap.chip_name = sample.chip_name;
            if snap.gpu_cores.is_none() {
                snap.gpu_cores = sample.gpu_cores;
            }
            // Reuse this ioreg output for util numbers when identity was needed.
            if sample.gpu_percent.is_some() {
                snap.gpu_percent = sample.gpu_percent;
            }
            if sample.gpu_mem_bytes.is_some() {
                snap.gpu_mem_bytes = sample.gpu_mem_bytes;
            }
        }
        if snap.chip_name.is_none() {
            snap.chip_name = sysctl_string("machdep.cpu.brand_string");
        }
    }

    // Memory
    if let Some(out) = Command::new("vm_stat")
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let text = String::from_utf8_lossy(&out.stdout);
        snap.used_ram_bytes = parse_vm_stat_used_bytes(&text);
        snap.free_ram_bytes = parse_vm_stat_free_bytes(&text);
    }

    // CPU — prefer `ps` over `top -l 1`. On macOS, `top -l 1` routinely takes
    // ~0.5–1.5s even with `-n 0 -s 0`, which freezes a synchronous UI loop.
    if let Some(out) = Command::new("ps")
        .args(["-A", "-o", "%cpu="])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let text = String::from_utf8_lossy(&out.stdout);
        let ncpu = snap.logical_cpus.unwrap_or(1).max(1) as f64;
        snap.cpu_percent = parse_ps_cpu_percent(&text, ncpu);
    }

    // GPU util (skip second ioreg if identity pass already filled it).
    if (snap.gpu_percent.is_none() || snap.gpu_mem_bytes.is_none())
        && let Some(out) = Command::new("ioreg")
            .args(["-r", "-d", "1", "-c", "IOAccelerator"])
            .output()
            .ok()
            .filter(|o| o.status.success())
    {
        let text = String::from_utf8_lossy(&out.stdout);
        let sample = parse_ioreg_gpu(&text);
        if snap.gpu_percent.is_none() {
            snap.gpu_percent = sample.gpu_percent;
        }
        if snap.gpu_mem_bytes.is_none() {
            snap.gpu_mem_bytes = sample.gpu_mem_bytes;
        }
        if snap.chip_name.is_none() {
            snap.chip_name = sample.chip_name;
        }
        if snap.gpu_cores.is_none() {
            snap.gpu_cores = sample.gpu_cores;
        }
    }

    // Top processes
    if let Some(out) = Command::new("ps")
        .args(["-A", "-o", "rss=,pid=,comm="])
        .output()
        .ok()
        .filter(|o| o.status.success())
    {
        let text = String::from_utf8_lossy(&out.stdout);
        snap.top_procs = parse_ps_top_rss(&text, 10);
    }

    snap.load_1m = parse_loadavg_1m(&sysctl_string("vm.loadavg").unwrap_or_default());
    snap.free_disk_bytes = free_disk_bytes(&req.cache_root);
    snap
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PressureBand {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, Default)]
pub(super) struct GpuSample {
    pub chip_name: Option<String>,
    pub gpu_cores: Option<u32>,
    pub gpu_percent: Option<f64>,
    pub gpu_mem_bytes: Option<u64>,
}

fn push_hist(hist: &mut VecDeque<u64>, value: u64) {
    hist.push_back(value);
    while hist.len() > HISTORY_LEN {
        hist.pop_front();
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

fn sysctl_u32(key: &str) -> Option<u32> {
    sysctl_string(key)?.parse().ok()
}

fn free_disk_bytes(path: &Path) -> Option<u64> {
    crate::tui::hardware::free_disk_bytes(path)
}

/// Parse `vm.loadavg` forms: `{ 1.2 1.3 1.4 }` or `1.2 1.3 1.4`.
pub(super) fn parse_loadavg_1m(raw: &str) -> Option<f64> {
    let cleaned = raw
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim();
    cleaned.split_whitespace().next()?.parse().ok()
}

/// Parse macOS `top -l 1` host CPU line into busy percent 0–100.
///
/// Kept for unit tests and offline parsing; the live sampler uses `ps`
/// instead because `top -l 1` is multi-hundred-ms on typical Macs.
///
/// Accepts forms like:
/// `CPU usage: 1.42% user, 5.71% sys, 92.85% idle`
/// Prefer `100 - idle` when idle is present; else `user + sys`.
#[cfg(test)]
pub(super) fn parse_top_cpu_percent(raw: &str) -> Option<f64> {
    for line in raw.lines() {
        let line = line.trim();
        if !line.starts_with("CPU usage:") && !line.to_ascii_lowercase().starts_with("cpu usage:") {
            continue;
        }
        let idle = parse_top_field_percent(line, "idle");
        if let Some(idle) = idle {
            return Some((100.0 - idle).clamp(0.0, 100.0));
        }
        let user = parse_top_field_percent(line, "user").unwrap_or(0.0);
        let sys = parse_top_field_percent(line, "sys")
            .or_else(|| parse_top_field_percent(line, "system"))
            .unwrap_or(0.0);
        if user > 0.0 || sys > 0.0 {
            return Some((user + sys).clamp(0.0, 100.0));
        }
    }
    None
}

#[cfg(test)]
fn parse_top_field_percent(line: &str, field: &str) -> Option<f64> {
    // Match "12.3% user" / "12.3% idle" (order is value then unit then label).
    let lower = line.to_ascii_lowercase();
    let field = field.to_ascii_lowercase();
    let needle = format!("% {field}");
    let idx = lower.find(&needle)?;
    // Walk left from '%' to collect the number.
    let before = &line[..idx];
    let num_start = before
        .rfind(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .map(|i| i + 1)
        .unwrap_or(0);
    before[num_start..].trim().parse().ok()
}

/// Parse `ps -A -o %cpu=` into average utilization 0–100.
pub(super) fn parse_ps_cpu_percent(raw: &str, ncpu: f64) -> Option<f64> {
    let mut sum = 0.0_f64;
    let mut any = false;
    for tok in raw.split_whitespace() {
        if let Ok(v) = tok.parse::<f64>() {
            sum += v;
            any = true;
        }
    }
    if !any || ncpu <= 0.0 {
        return None;
    }
    Some((sum / ncpu).clamp(0.0, 100.0))
}

/// Parse IOAccelerator `ioreg` text for Device Utilization % and identity.
///
/// Looks for keys inside `PerformanceStatistics` and top-level AGX properties.
/// Works without sudo (unlike `powermetrics`).
pub(super) fn parse_ioreg_gpu(raw: &str) -> GpuSample {
    let mut sample = GpuSample::default();

    // Prefer the first matching value; AGX accelerator is usually first.
    if let Some(v) = find_ioreg_number(raw, "Device Utilization %") {
        sample.gpu_percent = Some(v.clamp(0.0, 100.0));
    } else if let Some(v) = find_ioreg_number(raw, "Renderer Utilization %") {
        // Fallback when Device util is absent on some chips/OS versions.
        sample.gpu_percent = Some(v.clamp(0.0, 100.0));
    }

    if let Some(v) = find_ioreg_number(raw, "In use system memory") {
        // Bytes (integer). Ignore the "(driver)" variant by matching exact key
        // with word-boundary style: we scan for `"In use system memory"=`.
        sample.gpu_mem_bytes = Some(v as u64);
    }

    if let Some(v) = find_ioreg_number(raw, "gpu-core-count") {
        sample.gpu_cores = Some(v as u32);
    }

    if let Some(name) = find_ioreg_string(raw, "model") {
        // Prefer Apple chip model over generic IO names.
        if name.contains("Apple") || name.starts_with('M') {
            sample.chip_name = Some(name);
        } else {
            sample.chip_name.get_or_insert(name);
        }
    }

    sample
}

fn find_ioreg_number(raw: &str, key: &str) -> Option<f64> {
    // ioreg text: "Key"=123  or  "Key" = 123
    let needle = format!("\"{key}\"");
    let mut search = raw;
    while let Some(idx) = search.find(&needle) {
        let after = &search[idx + needle.len()..];
        let after = after.trim_start();
        let after = after.strip_prefix('=')?.trim_start();
        // Skip the driver variant when looking for "In use system memory"
        // because ioreg may also have "In use system memory (driver)".
        // We already matched the exact quoted key, so this is fine.
        let digits: String = after
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        if let Ok(v) = digits.parse::<f64>() {
            return Some(v);
        }
        search = &search[idx + needle.len()..];
    }
    None
}

fn find_ioreg_string(raw: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let idx = raw.find(&needle)?;
    let after = &raw[idx + needle.len()..];
    let after = after.trim_start().strip_prefix('=')?.trim_start();
    let after = after.strip_prefix('"')?;
    let end = after.find('"')?;
    let value = after[..end].trim().to_string();
    if value.is_empty() { None } else { Some(value) }
}

/// Used ≈ (active + wired + compressor) × page_size.
pub(super) fn parse_vm_stat_used_bytes(raw: &str) -> Option<u64> {
    let page_size = parse_vm_stat_page_size(raw)?;
    let active = parse_vm_stat_pages(raw, "Pages active")?;
    let wired = parse_vm_stat_pages(raw, "Pages wired down")
        .or_else(|| parse_vm_stat_pages(raw, "Pages wired"))?;
    let compressor = parse_vm_stat_pages(raw, "Pages occupied by compressor").unwrap_or(0);
    Some(
        active
            .saturating_add(wired)
            .saturating_add(compressor)
            .saturating_mul(page_size),
    )
}

/// Free-ish ≈ (free + speculative + purgeable) × page_size.
pub(super) fn parse_vm_stat_free_bytes(raw: &str) -> Option<u64> {
    let page_size = parse_vm_stat_page_size(raw)?;
    let free = parse_vm_stat_pages(raw, "Pages free")?;
    let speculative = parse_vm_stat_pages(raw, "Pages speculative").unwrap_or(0);
    let purgeable = parse_vm_stat_pages(raw, "Pages purgeable").unwrap_or(0);
    Some(
        free.saturating_add(speculative)
            .saturating_add(purgeable)
            .saturating_mul(page_size),
    )
}

fn parse_vm_stat_page_size(raw: &str) -> Option<u64> {
    for line in raw.lines() {
        if let Some(idx) = line.find("page size of ") {
            let rest = &line[idx + "page size of ".len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = digits.parse::<u64>()
                && n > 0
            {
                return Some(n);
            }
        }
    }
    Some(16_384)
}

fn parse_vm_stat_pages(raw: &str, key: &str) -> Option<u64> {
    for line in raw.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start_matches(':').trim();
            let digits: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .filter(|c| c.is_ascii_digit())
                .collect();
            return digits.parse().ok();
        }
    }
    None
}

/// Parse `ps -A -o rss=,pid=,comm=` (rss KiB), return top `limit` by RSS.
pub(super) fn parse_ps_top_rss(raw: &str, limit: usize) -> Vec<TopProc> {
    let mut rows: Vec<TopProc> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(rss_kib) = parts.next().and_then(|s| s.parse::<u64>().ok()) else {
            continue;
        };
        let Some(pid) = parts.next().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let name_raw = parts.collect::<Vec<_>>().join(" ");
        if name_raw.is_empty() {
            continue;
        }
        let name = Path::new(&name_raw)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(name_raw);
        rows.push(TopProc {
            pid,
            rss_bytes: rss_kib.saturating_mul(1024),
            name,
        });
    }
    rows.sort_by_key(|process| std::cmp::Reverse(process.rss_bytes));
    rows.truncate(limit);
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vm_stat_used_and_free_parse() {
        let raw = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               1000.
Pages active:                             2000.
Pages speculative:                         100.
Pages wired down:                          300.
Pages purgeable:                            50.
Pages occupied by compressor:              100.
";
        assert_eq!(parse_vm_stat_used_bytes(raw), Some(2400 * 16_384));
        assert_eq!(parse_vm_stat_free_bytes(raw), Some(1150 * 16_384));
    }

    #[test]
    fn ps_cpu_averages_across_cores() {
        let raw = "10.0\n20.0\n30.0\n";
        assert!((parse_ps_cpu_percent(raw, 4.0).unwrap() - 15.0).abs() < 1e-6);
    }

    #[test]
    fn top_cpu_parses_user_sys_idle() {
        let raw = "\
Processes: 675 total
Load Avg: 2.17, 2.24, 2.31
CPU usage: 1.42% user, 5.71% sys, 92.85% idle
PhysMem: 33G used
";
        let p = parse_top_cpu_percent(raw).unwrap();
        // 100 - 92.85 = 7.15
        assert!((p - 7.15).abs() < 1e-6, "got {p}");
    }

    #[test]
    fn top_cpu_falls_back_to_user_plus_sys() {
        let raw = "CPU usage: 10.0% user, 5.0% sys\n";
        let p = parse_top_cpu_percent(raw).unwrap();
        assert!((p - 15.0).abs() < 1e-6);
    }

    #[test]
    fn loadavg_parses_braced_form() {
        assert!((parse_loadavg_1m("{ 1.25 1.50 1.75 }").unwrap() - 1.25).abs() < 1e-6);
    }

    #[test]
    fn top_rss_sorts_and_basenames() {
        let raw = "\
  100  10 /Applications/Foo.app/Contents/MacOS/Foo
  500  20 /usr/bin/bar
   50  30 tiny
";
        let tops = parse_ps_top_rss(raw, 2);
        assert_eq!(tops.len(), 2);
        assert_eq!(tops[0].name, "bar");
        assert_eq!(tops[0].rss_bytes, 500 * 1024);
        assert_eq!(tops[1].name, "Foo");
    }

    #[test]
    fn history_caps_at_limit() {
        let mut m = LiveMetrics::for_tests();
        for i in 0..200 {
            push_hist(&mut m.mem_history, i);
        }
        assert_eq!(m.mem_history.len(), HISTORY_LEN);
        assert_eq!(*m.mem_history.back().unwrap(), 199);
    }

    #[test]
    fn pressure_bands_match_thresholds() {
        assert_eq!(LiveMetrics::pressure_band(0.2), PressureBand::Low);
        assert_eq!(LiveMetrics::pressure_band(0.6), PressureBand::Medium);
        assert_eq!(LiveMetrics::pressure_band(0.9), PressureBand::High);
    }

    #[test]
    fn ioreg_gpu_parses_device_util_and_identity() {
        let raw = r#"
+-o AGXAcceleratorG17X  <class AGXAcceleratorG17X>
    {
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=2188197888,"Tiler Utilization %"=3,"Renderer Utilization %"=11,"Device Utilization %"=27,"In use system memory"=1014890496}
      "model" = "Apple M5 Max"
      "gpu-core-count" = 40
    }
"#;
        let s = parse_ioreg_gpu(raw);
        assert!((s.gpu_percent.unwrap() - 27.0).abs() < 1e-6);
        assert_eq!(s.gpu_cores, Some(40));
        assert_eq!(s.chip_name.as_deref(), Some("Apple M5 Max"));
        assert_eq!(s.gpu_mem_bytes, Some(1_014_890_496));
    }

    #[test]
    fn ioreg_falls_back_to_renderer_util() {
        let raw = r#"
"PerformanceStatistics" = {"Renderer Utilization %"=15,"In use system memory"=1000}
"model" = "Apple M4"
"#;
        let s = parse_ioreg_gpu(raw);
        assert!((s.gpu_percent.unwrap() - 15.0).abs() < 1e-6);
        assert_eq!(s.gpu_mem_bytes, Some(1000));
    }

    #[test]
    fn tick_is_nonblocking_and_applies_async_sample() {
        // Regression: sample must not call `top` (or any multi-hundred-ms
        // probe) on the calling thread.
        let mut m = LiveMetrics::new(Some(16 * 1024 * 1024 * 1024));
        let cache = std::env::temp_dir();
        let t0 = Instant::now();
        let first = m.tick(0, &cache);
        // First call only launches the worker.
        assert!(!first, "first tick launches worker, does not apply yet");
        assert!(
            t0.elapsed() < Duration::from_millis(50),
            "tick must not block on host probes (elapsed {:?})",
            t0.elapsed()
        );
        // Wait for the background sample (ps/vm_stat/ioreg, not top).
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut applied = false;
        while Instant::now() < deadline {
            if m.tick(0, &cache) {
                applied = true;
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert!(applied, "background sample should land within 3s");
        // A second immediate tick must not re-sample inside the interval.
        assert!(!m.tick(0, &cache));
    }
}
