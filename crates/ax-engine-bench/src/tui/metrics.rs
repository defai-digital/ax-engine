//! Live host metrics for the TUI Home panel (Mac / Apple Silicon).
//!
//! ## Honest Mac metrics (not an nvtop port)
//!
//! nvtop assumes discrete GPUs (util, VRAM, clocks, multi-device plots).
//! Apple Silicon is **unified memory**; we only sample what macOS exposes
//! cheaply and without privilege:
//!
//! | Metric | Source |
//! |---|---|
//! | Memory used / free / total | `vm_stat` + `hw.memsize` |
//! | CPU average % | `ps -A -o %cpu=` / logical CPUs |
//! | Load average | `vm.loadavg` |
//! | Free disk (model cache) | `df` on HF cache root |
//! | Top memory users | `ps` RSS (host processes, not GPU clients) |
//! | Model headroom | installed weights on disk vs unified RAM |
//!
//! Omitted: discrete GPU util, VRAM, fan, board power, encode engines.

use std::collections::VecDeque;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

/// Samples kept for each sparkline (~60s at ~1 Hz).
pub(super) const HISTORY_LEN: usize = 60;

const SAMPLE_INTERVAL: Duration = Duration::from_millis(900);

/// One row in the htop-style “who is using memory” strip.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct TopProc {
    pub pid: u32,
    pub rss_bytes: u64,
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub(super) struct LiveMetrics {
    pub total_ram_bytes: Option<u64>,
    /// App-pressure used ≈ active + wired + compressor pages.
    pub used_ram_bytes: Option<u64>,
    /// free + speculative + purgeable pages (approx “available”).
    pub free_ram_bytes: Option<u64>,
    /// Average CPU utilization 0.0–100.0 across logical CPUs.
    pub cpu_percent: Option<f64>,
    pub logical_cpus: Option<u32>,
    pub load_1m: Option<f64>,
    /// Free space on the HF-cache volume (download target).
    pub free_disk_bytes: Option<u64>,
    /// Installed model footprint (bytes) — AX serving headroom proxy.
    pub models_bytes: u64,
    /// Top RSS processes (htop-style consumers).
    pub top_procs: Vec<TopProc>,

    pub mem_history: VecDeque<u64>,
    pub cpu_history: VecDeque<u64>,
    pub models_history: VecDeque<u64>,

    last_sample: Option<Instant>,
}

impl LiveMetrics {
    pub fn new(total_ram_bytes: Option<u64>) -> Self {
        LiveMetrics {
            total_ram_bytes,
            logical_cpus: sysctl_u32("hw.logicalcpu"),
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
            load_1m: Some(1.2),
            free_disk_bytes: Some(500 * 1024 * 1024 * 1024),
            models_bytes: 8 * 1024 * 1024 * 1024,
            top_procs: vec![
                TopProc {
                    pid: 1,
                    rss_bytes: 3 * 1024 * 1024 * 1024,
                    name: "Code".into(),
                },
                TopProc {
                    pid: 2,
                    rss_bytes: 1500 * 1024 * 1024,
                    name: "ax-engine-server".into(),
                },
            ],
            ..Default::default()
        };
        for v in [20, 25, 30, 28, 35, 40, 38] {
            m.mem_history.push_back(v);
            m.cpu_history.push_back(v.saturating_sub(10));
            m.models_history.push_back(12);
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

    pub fn models_ratio(&self) -> Option<f64> {
        let total = self.total_ram_bytes?;
        if total == 0 {
            return None;
        }
        Some((self.models_bytes as f64 / total as f64).clamp(0.0, 1.0))
    }

    /// nvidia-htop style pressure band from a 0–1 ratio.
    pub fn pressure_band(ratio: f64) -> PressureBand {
        if ratio >= 0.85 {
            PressureBand::High
        } else if ratio >= 0.55 {
            PressureBand::Medium
        } else {
            PressureBand::Low
        }
    }

    /// Refresh probes when the sample interval has elapsed.
    pub fn tick(&mut self, models_bytes: u64, cache_root: &Path) {
        self.models_bytes = models_bytes;
        let now = Instant::now();
        if self
            .last_sample
            .is_some_and(|t| now.duration_since(t) < SAMPLE_INTERVAL)
        {
            if let Some(r) = self.models_ratio() {
                push_hist(&mut self.models_history, (r * 100.0).round() as u64);
            }
            return;
        }
        self.last_sample = Some(now);
        if self.total_ram_bytes.is_none() {
            self.total_ram_bytes = sysctl_string("hw.memsize").and_then(|s| s.parse().ok());
        }
        if self.logical_cpus.is_none() {
            self.logical_cpus = sysctl_u32("hw.logicalcpu");
        }
        self.sample_memory();
        self.sample_cpu();
        self.sample_top_procs();
        self.load_1m = parse_loadavg_1m(&sysctl_string("vm.loadavg").unwrap_or_default());
        self.free_disk_bytes = free_disk_bytes(cache_root);

        if let Some(r) = self.mem_ratio() {
            push_hist(&mut self.mem_history, (r * 100.0).round() as u64);
        }
        if let Some(p) = self.cpu_percent {
            push_hist(&mut self.cpu_history, p.round().clamp(0.0, 100.0) as u64);
        }
        if let Some(r) = self.models_ratio() {
            push_hist(&mut self.models_history, (r * 100.0).round() as u64);
        }
    }

    fn sample_memory(&mut self) {
        let output = Command::new("vm_stat").output().ok();
        if let Some(out) = output
            && out.status.success()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            self.used_ram_bytes = parse_vm_stat_used_bytes(&text);
            self.free_ram_bytes = parse_vm_stat_free_bytes(&text);
        }
    }

    fn sample_cpu(&mut self) {
        let output = Command::new("ps").args(["-A", "-o", "%cpu="]).output().ok();
        let Some(out) = output.filter(|o| o.status.success()) else {
            return;
        };
        let text = String::from_utf8_lossy(&out.stdout);
        let ncpu = self.logical_cpus.unwrap_or(1).max(1) as f64;
        self.cpu_percent = parse_ps_cpu_percent(&text, ncpu);
    }

    fn sample_top_procs(&mut self) {
        // rss is KiB on macOS; sort by RSS descending, take a few.
        let output = Command::new("ps")
            .args(["-A", "-o", "rss=,pid=,comm="])
            .output()
            .ok();
        let Some(out) = output.filter(|o| o.status.success()) else {
            return;
        };
        let text = String::from_utf8_lossy(&out.stdout);
        self.top_procs = parse_ps_top_rss(&text, 5);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PressureBand {
    Low,
    Medium,
    High,
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
        // Basename only for long paths.
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
    fn pressure_bands_match_nvidia_htop_style() {
        assert_eq!(LiveMetrics::pressure_band(0.2), PressureBand::Low);
        assert_eq!(LiveMetrics::pressure_band(0.6), PressureBand::Medium);
        assert_eq!(LiveMetrics::pressure_band(0.9), PressureBand::High);
    }
}
