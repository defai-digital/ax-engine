use std::time::{Duration, Instant};

/// Inference metrics collected during generation.
#[derive(Debug, Default)]
pub struct InferenceMetrics {
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub prefill_duration: Duration,
    pub decode_duration: Duration,
    pub peak_rss_bytes: u64,
    pub kv_pages_used: u64,
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn prefill_tok_per_sec(&self) -> f64 {
        if self.prefill_duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.prefill_tokens as f64 / self.prefill_duration.as_secs_f64()
    }

    pub fn decode_tok_per_sec(&self) -> f64 {
        if self.decode_duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.decode_tokens as f64 / self.decode_duration.as_secs_f64()
    }

    /// Snapshot current RSS and update peak if higher.
    pub fn update_peak_rss(&mut self) {
        let rss = current_rss_bytes();
        if rss > self.peak_rss_bytes {
            self.peak_rss_bytes = rss;
        }
    }
}

/// Timer helper for tracking operation durations.
pub struct OpTimer {
    start: Instant,
}

impl OpTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Get current RSS (Resident Set Size) in bytes via macOS mach API.
///
/// Returns 0 on failure.
pub fn current_rss_bytes() -> u64 {
    // SAFETY: calling mach kernel API — standard macOS system call.
    unsafe {
        let mut info: mach_task_basic_info_data_t = std::mem::zeroed();
        let mut count = (std::mem::size_of::<mach_task_basic_info_data_t>()
            / std::mem::size_of::<natural_t>()) as mach_msg_type_number_t;

        let kr = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as task_info_t,
            &mut count,
        );

        if kr == KERN_SUCCESS {
            info.resident_size
        } else {
            0
        }
    }
}

// macOS mach kernel types and functions for RSS tracking.
// Declared here to avoid pulling in a full mach crate.
#[allow(non_camel_case_types)]
type mach_port_t = u32;
#[allow(non_camel_case_types)]
type kern_return_t = i32;
#[allow(non_camel_case_types)]
type natural_t = u32;
#[allow(non_camel_case_types)]
type mach_msg_type_number_t = u32;
#[allow(non_camel_case_types)]
type task_info_t = *mut i32;
#[allow(non_camel_case_types)]
type task_flavor_t = u32;
#[allow(non_camel_case_types)]
type mach_vm_size_t = u64;
#[allow(non_camel_case_types)]
type integer_t = i32;
#[allow(non_camel_case_types)]
type policy_t = i32;
#[allow(non_camel_case_types)]
type time_value_t = [i32; 2]; // tv_sec, tv_usec

const MACH_TASK_BASIC_INFO: task_flavor_t = 20;
const KERN_SUCCESS: kern_return_t = 0;

#[repr(C)]
#[allow(non_camel_case_types)]
struct mach_task_basic_info_data_t {
    virtual_size: mach_vm_size_t,
    resident_size: mach_vm_size_t,
    resident_size_max: mach_vm_size_t,
    user_time: time_value_t,
    system_time: time_value_t,
    policy: policy_t,
    suspend_count: integer_t,
}

unsafe extern "C" {
    fn mach_task_self() -> mach_port_t;
    fn task_info(
        target_task: mach_port_t,
        flavor: task_flavor_t,
        task_info_out: task_info_t,
        task_info_outCnt: *mut mach_msg_type_number_t,
    ) -> kern_return_t;
}

/// Latency histogram for tracking per-token decode timing.
///
/// Collects individual sample durations and computes P50/P95/P99 percentiles.
pub struct LatencyHistogram {
    samples: Vec<Duration>,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            samples: Vec::with_capacity(4096),
        }
    }

    /// Record a single sample.
    pub fn record(&mut self, d: Duration) {
        self.samples.push(d);
    }

    /// Number of samples recorded.
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    /// Compute a percentile (0.0–1.0). Returns None if no samples.
    pub fn percentile(&self, p: f64) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }

        let mut sorted: Vec<Duration> = self.samples.clone();
        sorted.sort_unstable();

        let idx = ((p * (sorted.len() - 1) as f64).round()) as usize;
        let idx = idx.min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// P50 (median) latency.
    pub fn p50(&self) -> Option<Duration> {
        self.percentile(0.50)
    }

    /// P95 latency.
    pub fn p95(&self) -> Option<Duration> {
        self.percentile(0.95)
    }

    /// P99 latency.
    pub fn p99(&self) -> Option<Duration> {
        self.percentile(0.99)
    }

    /// Mean latency.
    pub fn mean(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let total: Duration = self.samples.iter().sum();
        Some(total / self.samples.len() as u32)
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-operation timing breakdown.
///
/// Tracks cumulative time spent in key operations during inference.
#[derive(Debug, Default, Clone)]
pub struct OpBreakdown {
    pub gpu: Duration,
    pub gpu_encode: Duration,
    pub gpu_execute: Duration,
    pub gpu_execute_layers: Duration,
    pub gpu_execute_output: Duration,
    pub gpu_readback: Duration,
    pub gpu_encode_layer_norm: Duration,
    pub gpu_encode_layer_qkv: Duration,
    pub gpu_encode_layer_rope: Duration,
    pub gpu_encode_layer_kv_append: Duration,
    pub gpu_encode_layer_attention: Duration,
    pub gpu_encode_layer_out_proj: Duration,
    pub gpu_encode_layer_ffn: Duration,
    pub gpu_encode_layer_residual: Duration,
    pub matmul: Duration,
    pub matmul_input_proj: Duration,
    pub matmul_output_proj: Duration,
    pub matmul_lm_head: Duration,
    pub attention: Duration,
    pub recurrent: Duration,
    pub dequant: Duration,
    pub rope: Duration,
    pub norm: Duration,
    pub sampling: Duration,
}

impl OpBreakdown {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total time across all tracked operations.
    pub fn total(&self) -> Duration {
        self.gpu
            + self.matmul
            + self.attention
            + self.recurrent
            + self.dequant
            + self.rope
            + self.norm
            + self.sampling
    }

    /// Format as a breakdown string.
    pub fn summary(&self) -> String {
        let total = self.total().as_secs_f64();
        if total == 0.0 {
            return "no op timing data".to_string();
        }
        format!(
            "gpu {:.1}% | gpu-enc {:.1}% | gpu-exec {:.1}% | gpu-rb {:.1}% | matmul {:.1}% | attn {:.1}% | recur {:.1}% | dequant {:.1}% | rope {:.1}% | norm {:.1}% | sample {:.1}%",
            self.gpu.as_secs_f64() / total * 100.0,
            self.gpu_encode.as_secs_f64() / total * 100.0,
            self.gpu_execute.as_secs_f64() / total * 100.0,
            self.gpu_readback.as_secs_f64() / total * 100.0,
            self.matmul.as_secs_f64() / total * 100.0,
            self.attention.as_secs_f64() / total * 100.0,
            self.recurrent.as_secs_f64() / total * 100.0,
            self.dequant.as_secs_f64() / total * 100.0,
            self.rope.as_secs_f64() / total * 100.0,
            self.norm.as_secs_f64() / total * 100.0,
            self.sampling.as_secs_f64() / total * 100.0,
        )
    }

    /// Accumulate another breakdown's timings into this one.
    pub fn accumulate(&mut self, other: &OpBreakdown) {
        self.gpu += other.gpu;
        self.gpu_encode += other.gpu_encode;
        self.gpu_execute += other.gpu_execute;
        self.gpu_execute_layers += other.gpu_execute_layers;
        self.gpu_execute_output += other.gpu_execute_output;
        self.gpu_readback += other.gpu_readback;
        self.gpu_encode_layer_norm += other.gpu_encode_layer_norm;
        self.gpu_encode_layer_qkv += other.gpu_encode_layer_qkv;
        self.gpu_encode_layer_rope += other.gpu_encode_layer_rope;
        self.gpu_encode_layer_kv_append += other.gpu_encode_layer_kv_append;
        self.gpu_encode_layer_attention += other.gpu_encode_layer_attention;
        self.gpu_encode_layer_out_proj += other.gpu_encode_layer_out_proj;
        self.gpu_encode_layer_ffn += other.gpu_encode_layer_ffn;
        self.gpu_encode_layer_residual += other.gpu_encode_layer_residual;
        self.matmul += other.matmul;
        self.matmul_input_proj += other.matmul_input_proj;
        self.matmul_output_proj += other.matmul_output_proj;
        self.matmul_lm_head += other.matmul_lm_head;
        self.attention += other.attention;
        self.recurrent += other.recurrent;
        self.dequant += other.dequant;
        self.rope += other.rope;
        self.norm += other.norm;
        self.sampling += other.sampling;
    }

    /// Reset all timings to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_metrics_tok_per_sec() {
        let m = InferenceMetrics {
            prefill_tokens: 100,
            prefill_duration: Duration::from_secs(2),
            decode_tokens: 50,
            decode_duration: Duration::from_secs(5),
            ..Default::default()
        };
        assert!((m.prefill_tok_per_sec() - 50.0).abs() < 0.01);
        assert!((m.decode_tok_per_sec() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_inference_metrics_zero_duration() {
        let m = InferenceMetrics::new();
        assert_eq!(m.prefill_tok_per_sec(), 0.0);
        assert_eq!(m.decode_tok_per_sec(), 0.0);
    }

    #[test]
    fn test_rss_returns_nonzero() {
        // On macOS this should return the actual RSS of this process
        let rss = current_rss_bytes();
        assert!(rss > 0, "RSS should be > 0, got {rss}");
    }

    #[test]
    fn test_update_peak_rss() {
        let mut m = InferenceMetrics::new();
        assert_eq!(m.peak_rss_bytes, 0);
        m.update_peak_rss();
        assert!(m.peak_rss_bytes > 0, "peak RSS should be updated");
    }

    #[test]
    fn test_latency_histogram_empty() {
        let h = LatencyHistogram::new();
        assert_eq!(h.count(), 0);
        assert!(h.p50().is_none());
        assert!(h.p95().is_none());
        assert!(h.p99().is_none());
        assert!(h.mean().is_none());
    }

    #[test]
    fn test_latency_histogram_single() {
        let mut h = LatencyHistogram::new();
        h.record(Duration::from_millis(10));
        assert_eq!(h.count(), 1);
        assert_eq!(h.p50(), Some(Duration::from_millis(10)));
        assert_eq!(h.p95(), Some(Duration::from_millis(10)));
        assert_eq!(h.p99(), Some(Duration::from_millis(10)));
    }

    #[test]
    fn test_latency_histogram_percentiles() {
        let mut h = LatencyHistogram::new();
        // Add 100 samples: 1ms, 2ms, ..., 100ms
        for i in 1..=100 {
            h.record(Duration::from_millis(i));
        }
        assert_eq!(h.count(), 100);

        // P50 should be ~50ms
        let p50 = h.p50().unwrap();
        assert!(
            p50 >= Duration::from_millis(49) && p50 <= Duration::from_millis(51),
            "P50 = {p50:?}"
        );

        // P95 should be ~95ms
        let p95 = h.p95().unwrap();
        assert!(
            p95 >= Duration::from_millis(94) && p95 <= Duration::from_millis(96),
            "P95 = {p95:?}"
        );

        // P99 should be ~99ms
        let p99 = h.p99().unwrap();
        assert!(
            p99 >= Duration::from_millis(98) && p99 <= Duration::from_millis(100),
            "P99 = {p99:?}"
        );
    }

    #[test]
    fn test_latency_histogram_mean() {
        let mut h = LatencyHistogram::new();
        h.record(Duration::from_millis(10));
        h.record(Duration::from_millis(20));
        h.record(Duration::from_millis(30));
        let mean = h.mean().unwrap();
        assert_eq!(mean, Duration::from_millis(20));
    }

    #[test]
    fn test_latency_histogram_clear() {
        let mut h = LatencyHistogram::new();
        h.record(Duration::from_millis(10));
        assert_eq!(h.count(), 1);
        h.clear();
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn test_op_breakdown_total() {
        let ops = OpBreakdown {
            matmul: Duration::from_millis(50),
            attention: Duration::from_millis(30),
            dequant: Duration::from_millis(20),
            ..Default::default()
        };
        assert_eq!(ops.total(), Duration::from_millis(100));
    }

    #[test]
    fn test_op_breakdown_summary() {
        let ops = OpBreakdown {
            matmul: Duration::from_millis(50),
            attention: Duration::from_millis(30),
            dequant: Duration::from_millis(20),
            ..Default::default()
        };
        let s = ops.summary();
        assert!(s.contains("matmul 50.0%"), "got: {s}");
        assert!(s.contains("attn 30.0%"), "got: {s}");
        assert!(s.contains("dequant 20.0%"), "got: {s}");
    }

    #[test]
    fn test_op_breakdown_empty_summary() {
        let ops = OpBreakdown::new();
        assert_eq!(ops.summary(), "no op timing data");
    }

    #[test]
    fn test_op_timer() {
        let timer = OpTimer::start();
        // Just verify it doesn't panic and returns a duration
        let elapsed = timer.elapsed();
        assert!(elapsed < Duration::from_secs(1));
    }
}
