//! Live host metrics for the TUI Home panel.
//!
//! ## Widget choice (ratatui best practice)
//!
//! | Widget | Role in this panel |
//! |---|---|
//! | [`Gauge`] | **Now** — absolute fill for used/total or 0–100% |
//! | [`Sparkline`] | **Trend** — short history (last ~60 samples) |
//! | `BarChart` | Not used for live host load (categorical, not time-series) |
//! | `Chart` | Not used on Home (axes eat vertical space; fine for a future Metrics tab) |
//!
//! This mirrors the official ratatui demo layout: gauges for current level,
//! sparklines for recent shape. Sampling is best-effort and shell-based so
//! the TUI stays free of new sysinfo/FFI deps; every field degrades to
//! `None` / empty history when a probe fails.

use std::collections::VecDeque;
use std::process::Command;
use std::time::{Duration, Instant};

/// Samples kept for each sparkline (~60s at 1 Hz).
pub(super) const HISTORY_LEN: usize = 60;

/// Minimum gap between expensive probes (ps / vm_stat are cheap; still throttle).
const SAMPLE_INTERVAL: Duration = Duration::from_millis(900);

#[derive(Clone, Debug, Default)]
pub(super) struct LiveMetrics {
    /// Physical RAM total (bytes), from `hw.memsize` or parent `HardwareInfo`.
    pub total_ram_bytes: Option<u64>,
    /// Approximate used RAM (bytes). See [`parse_vm_stat_used_bytes`].
    pub used_ram_bytes: Option<u64>,
    /// Average CPU utilization 0.0–100.0 across logical CPUs.
    pub cpu_percent: Option<f64>,
    /// Logical CPU count (for load context).
    pub logical_cpus: Option<u32>,
    /// 1-minute load average.
    pub load_1m: Option<f64>,
    /// Installed model footprint (bytes) — AX-specific “unified memory” proxy.
    pub models_bytes: u64,

    pub mem_history: VecDeque<u64>,
    pub cpu_history: VecDeque<u64>,
    /// Models footprint as % of total RAM, 0–100.
    pub models_history: VecDeque<u64>,

    last_sample: Option<Instant>,
}

impl LiveMetrics {
    pub fn new(total_ram_bytes: Option<u64>) -> Self {
        LiveMetrics {
            total_ram_bytes,
            logical_cpus: sysctl_u32("hw.logicalcpu"),
            load_1m: parse_loadavg_1m(&sysctl_string("vm.loadavg").unwrap_or_default()),
            ..Default::default()
        }
    }

    #[cfg(test)]
    pub fn for_tests() -> Self {
        let mut m = LiveMetrics {
            total_ram_bytes: Some(64 * 1024 * 1024 * 1024),
            used_ram_bytes: Some(24 * 1024 * 1024 * 1024),
            cpu_percent: Some(18.5),
            logical_cpus: Some(12),
            load_1m: Some(1.2),
            models_bytes: 8 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        for v in [20, 25, 30, 28, 35, 40, 38] {
            m.mem_history.push_back(v);
            m.cpu_history.push_back(v.saturating_sub(10));
            m.models_history.push_back(12);
        }
        m
    }

    /// Memory used / total as 0.0–1.0, if known.
    pub fn mem_ratio(&self) -> Option<f64> {
        let used = self.used_ram_bytes?;
        let total = self.total_ram_bytes?;
        if total == 0 {
            return None;
        }
        Some((used as f64 / total as f64).clamp(0.0, 1.0))
    }

    /// CPU as 0.0–1.0 for gauges.
    pub fn cpu_ratio(&self) -> Option<f64> {
        self.cpu_percent.map(|p| (p / 100.0).clamp(0.0, 1.0))
    }

    /// Installed models as fraction of total RAM (unified-memory headroom proxy).
    pub fn models_ratio(&self) -> Option<f64> {
        let total = self.total_ram_bytes?;
        if total == 0 {
            return None;
        }
        Some((self.models_bytes as f64 / total as f64).clamp(0.0, 1.0))
    }

    /// Refresh probes when the sample interval has elapsed.
    pub fn tick(&mut self, models_bytes: u64) {
        self.models_bytes = models_bytes;
        let now = Instant::now();
        if self
            .last_sample
            .is_some_and(|t| now.duration_since(t) < SAMPLE_INTERVAL)
        {
            // Still update models history cheaply when footprint changes.
            if let Some(r) = self.models_ratio() {
                push_hist(&mut self.models_history, (r * 100.0).round() as u64);
            }
            return;
        }
        self.last_sample = Some(now);
        self.sample_memory();
        self.sample_cpu();
        self.load_1m = parse_loadavg_1m(&sysctl_string("vm.loadavg").unwrap_or_default());
        if self.logical_cpus.is_none() {
            self.logical_cpus = sysctl_u32("hw.logicalcpu");
        }
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
        if self.total_ram_bytes.is_none() {
            self.total_ram_bytes = sysctl_string("hw.memsize").and_then(|s| s.parse().ok());
        }
        let output = Command::new("vm_stat").output().ok();
        if let Some(out) = output
            && out.status.success()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            self.used_ram_bytes = parse_vm_stat_used_bytes(&text);
        }
    }

    fn sample_cpu(&mut self) {
        // Sum of per-process %cpu is “% of one core”; divide by logical CPUs.
        let output = Command::new("ps").args(["-A", "-o", "%cpu="]).output().ok();
        let Some(out) = output.filter(|o| o.status.success()) else {
            return;
        };
        let text = String::from_utf8_lossy(&out.stdout);
        let ncpu = self.logical_cpus.unwrap_or(1).max(1) as f64;
        self.cpu_percent = parse_ps_cpu_percent(&text, ncpu);
    }
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

/// Parse `vm.loadavg` forms: `{ 1.2 1.3 1.4 }` or `1.2 1.3 1.4`.
pub(super) fn parse_loadavg_1m(raw: &str) -> Option<f64> {
    let cleaned = raw
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim();
    cleaned.split_whitespace().next()?.parse().ok()
}

/// Parse `ps -A -o %cpu=` output into average utilization 0–100.
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

/// Estimate used RAM from `vm_stat` output.
///
/// Uses: `(active + wired + compressor) * page_size`.
/// This matches common “app-visible pressure” heuristics (not exactly Activity
/// Monitor's "Memory Used", which includes more categories).
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

fn parse_vm_stat_page_size(raw: &str) -> Option<u64> {
    // "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
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
    // Fallback common on Apple Silicon.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vm_stat_used_parses_fixture() {
        let raw = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               1000.
Pages active:                             2000.
Pages inactive:                           500.
Pages wired down:                         300.
Pages occupied by compressor:             100.
";
        // (2000+300+100)*16384
        assert_eq!(parse_vm_stat_used_bytes(raw), Some(2400 * 16_384));
    }

    #[test]
    fn ps_cpu_averages_across_cores() {
        let raw = "10.0\n20.0\n30.0\n";
        // sum=60, ncpu=4 → 15%
        assert!((parse_ps_cpu_percent(raw, 4.0).unwrap() - 15.0).abs() < 1e-6);
    }

    #[test]
    fn loadavg_parses_braced_form() {
        assert!((parse_loadavg_1m("{ 1.25 1.50 1.75 }").unwrap() - 1.25).abs() < 1e-6);
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
}
