//! Hardware-oriented microbenchmarks for AX Engine.
//!
//! These suites sit between end-to-end model benchmarks and isolated shader
//! experiments. They are intended to validate hardware hypotheses before
//! production-path changes are made.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use ax_engine_metal::profile::ProfileKernelMode;
use ax_engine_metal::{
    AttentionDecodeCandidate, AttentionDispatchConfig, AttentionKernels, DequantDispatchConfig,
    DequantKernels, ElementwiseKernels, KernelMode, KernelProfile, MetalBuffer, MetalDevice,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicrobenchSuite {
    Cpu,
    Gpu,
    Uma,
    Sync,
    All,
}

#[derive(Debug, Clone, Copy)]
pub struct MicrobenchConfig {
    pub suite: MicrobenchSuite,
    pub iterations: usize,
    pub suite_runs: usize,
}

impl Default for MicrobenchConfig {
    fn default() -> Self {
        Self {
            suite: MicrobenchSuite::All,
            iterations: 10,
            suite_runs: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrobenchMeasurement {
    pub name: String,
    pub unit: String,
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrobenchSuiteResult {
    pub suite: String,
    pub iterations: usize,
    #[serde(default = "default_suite_runs")]
    pub suite_runs: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recommendations: Option<Vec<MicrobenchRecommendation>>,
    pub measurements: Vec<MicrobenchMeasurement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrobenchReport {
    pub suites: Vec<MicrobenchSuiteResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_kernel_profile: Option<KernelProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_kernel_profile_evidence: Option<Vec<MicrobenchProfileEvidence>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrobenchRecommendation {
    pub domain: String,
    pub quant: String,
    pub variant: String,
    pub m: usize,
    pub k: usize,
    pub best_ms: f64,
    pub speedup_vs_auto: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrobenchProfileEvidence {
    pub rule: String,
    pub promoted: bool,
    pub suite_runs: usize,
    pub required_wins: usize,
    pub observed_wins: usize,
    pub min_speedup: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_speedup: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
}

fn default_suite_runs() -> usize {
    1
}

impl MicrobenchReport {
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn suggested_kernel_profile(&self) -> KernelProfile {
        let mut profile = KernelProfile {
            model: "microbench-recommendation".to_string(),
            source: "ax-bench microbench".to_string(),
            generated: "derived".to_string(),
            ..KernelProfile::default()
        };

        if let Some(gpu_suite) = self.suites.iter().find(|suite| suite.suite == "gpu") {
            apply_microbench_recommendations(&mut profile, gpu_suite);
        }

        profile
    }

    pub fn suggested_kernel_profile_evidence(&self) -> Vec<MicrobenchProfileEvidence> {
        self.suites
            .iter()
            .find(|suite| suite.suite == "gpu")
            .map(suggested_profile_evidence_for_suite)
            .unwrap_or_default()
    }

    pub fn suggested_kernel_profile_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(
            &self.suggested_kernel_profile(),
        )?)
    }

    pub fn print_summary(&self) {
        for suite in &self.suites {
            eprintln!();
            eprintln!("=== {} ===", suite.suite);
            if let Some(device) = &suite.device {
                eprintln!("device: {device}");
            }
            if suite.suite_runs > 1 {
                eprintln!("suite_runs: {}", suite.suite_runs);
            }
            if let Some(recommendations) = &suite.recommendations
                && !recommendations.is_empty()
            {
                eprintln!("winner summary:");
                for recommendation in aggregated_recommendations(recommendations) {
                    eprintln!(
                        "{:<18} {:>5}x{:<5} {:<8} {:>8.3} ms   {:>6.3}x vs auto",
                        recommendation.quant,
                        recommendation.m,
                        recommendation.k,
                        recommendation.variant,
                        recommendation.best_ms,
                        recommendation.speedup_vs_auto,
                    );
                }
                eprintln!("details:");
            }
            let observational_quants = suite_observational_quants(suite);
            if !observational_quants.is_empty() {
                eprintln!(
                    "observational_only_quants: {}",
                    observational_quants.join(", ")
                );
            }
            for measurement in &suite.measurements {
                if measurement.name.contains(".best.")
                    || measurement.name.contains(".best_speedup_vs_auto")
                {
                    continue;
                }
                if let Some(note) = &measurement.note {
                    eprintln!(
                        "{:<36} {:>12.3} {:<8} {}",
                        measurement.name, measurement.value, measurement.unit, note
                    );
                } else {
                    eprintln!(
                        "{:<36} {:>12.3} {}",
                        measurement.name, measurement.value, measurement.unit
                    );
                }
            }
        }
        if let Some(profile) = &self.suggested_kernel_profile {
            eprintln!();
            eprintln!("=== suggested_profile ===");
            if let Some(q4) = profile.decode_matvec.get("q4_k") {
                eprintln!(
                    "q4_k: tg={} rows_per_simdgroup={}",
                    q4.threadgroup_size, q4.rows_per_simdgroup
                );
            }
            if let Some(q6) = profile.decode_matvec.get("q6_k") {
                eprintln!(
                    "q6_k: tg={} rows_per_simdgroup={}",
                    q6.threadgroup_size, q6.rows_per_simdgroup
                );
            }
            eprintln!(
                "decode_attention: splitk_threshold={} sdpa_default={} hd128_n2_default={}",
                profile.attention_decode.splitk_threshold,
                profile.attention_decode.sdpa_default.unwrap_or(false),
                profile.attention_decode.hd128_n2_default.unwrap_or(false),
            );
            eprintln!(
                "attention_prefill: fa2_mode={:?} fa2_hd128_mode={:?}",
                profile.attention_prefill.fa2_mode, profile.attention_prefill.fa2_hd128_mode,
            );
        }
        if let Some(evidence) = &self.suggested_kernel_profile_evidence
            && !evidence.is_empty()
        {
            eprintln!();
            eprintln!("=== suggested_profile_evidence ===");
            for item in evidence {
                let variant = item.variant.as_deref().unwrap_or("-");
                let avg_speedup = item
                    .avg_speedup
                    .map(|value| format!("{value:.3}x"))
                    .unwrap_or_else(|| "-".to_string());
                let min_speedup = format!("{:.3}x", item.min_speedup);
                eprintln!(
                    "{:<24} promoted={} variant={} observed_wins={} required_wins={} min={} avg={}",
                    item.rule,
                    item.promoted,
                    variant,
                    item.observed_wins,
                    item.required_wins,
                    min_speedup,
                    avg_speedup,
                );
            }
        }
    }
}

pub fn run_microbench(config: &MicrobenchConfig) -> anyhow::Result<MicrobenchReport> {
    let suite_runs = config.suite_runs.max(1);
    let mut all_runs = Vec::with_capacity(suite_runs);
    for _ in 0..suite_runs {
        let mut suites = Vec::new();
        match config.suite {
            MicrobenchSuite::Cpu => suites.push(run_cpu_suite(config.iterations)),
            MicrobenchSuite::Gpu => suites.push(run_gpu_suite(config.iterations)?),
            MicrobenchSuite::Uma => suites.push(run_uma_suite(config.iterations)?),
            MicrobenchSuite::Sync => suites.push(run_sync_suite(config.iterations)?),
            MicrobenchSuite::All => {
                suites.push(run_cpu_suite(config.iterations));
                suites.push(run_gpu_suite(config.iterations)?);
                suites.push(run_uma_suite(config.iterations)?);
                suites.push(run_sync_suite(config.iterations)?);
            }
        }
        all_runs.push(suites);
    }
    let mut suites = Vec::new();
    if let Some(first_run) = all_runs.first() {
        for idx in 0..first_run.len() {
            let runs = all_runs
                .iter()
                .map(|suite_set| suite_set[idx].clone())
                .collect::<Vec<_>>();
            suites.push(aggregate_suite_runs(runs));
        }
    }
    let suggested_kernel_profile = suites.iter().any(|suite| suite.suite == "gpu").then(|| {
        let report = MicrobenchReport {
            suites: suites.clone(),
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };
        report.suggested_kernel_profile()
    });
    let suggested_kernel_profile_evidence =
        suites.iter().any(|suite| suite.suite == "gpu").then(|| {
            let report = MicrobenchReport {
                suites: suites.clone(),
                suggested_kernel_profile: None,
                suggested_kernel_profile_evidence: None,
            };
            report.suggested_kernel_profile_evidence()
        });
    Ok(MicrobenchReport {
        suites,
        suggested_kernel_profile,
        suggested_kernel_profile_evidence,
    })
}

fn run_cpu_suite(iterations: usize) -> MicrobenchSuiteResult {
    let mut measurements = Vec::new();

    for kib in [32usize, 512, 8 * 1024] {
        let size_bytes = kib * 1024;
        let data: Vec<u8> = (0..size_bytes).map(|i| (i & 0xff) as u8).collect();
        let repeats = repeats_for_target_bytes(size_bytes, 256 * 1024 * 1024);
        let start = Instant::now();
        let mut accum = 0u64;
        for _ in 0..iterations {
            for _ in 0..repeats {
                accum = accum.wrapping_add(sum_bytes(&data));
            }
        }
        black_box(accum);
        let secs = start.elapsed().as_secs_f64().max(f64::EPSILON);
        let bytes = (size_bytes * repeats * iterations) as f64;
        measurements.push(MicrobenchMeasurement {
            name: format!("cpu.cache_sweep.{kib}KiB"),
            unit: "GB/s".to_string(),
            value: bytes / secs / 1e9,
            note: Some("sequential sum throughput".to_string()),
        });
    }

    for len in [1_024usize, 32_768, 262_144] {
        let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
        let repeats = repeats_for_target_elems(len, 8_000_000);
        let start = Instant::now();
        let mut dot = 0.0f32;
        for _ in 0..iterations {
            for _ in 0..repeats {
                dot += dot_f32(&a, &b);
            }
        }
        black_box(dot);
        let secs = start.elapsed().as_secs_f64().max(f64::EPSILON);
        let ops = (2 * len * repeats * iterations) as f64;
        measurements.push(MicrobenchMeasurement {
            name: format!("cpu.dot_f32.{len}"),
            unit: "GFLOP/s".to_string(),
            value: ops / secs / 1e9,
            note: Some("mul+add counted as 2 FLOPs".to_string()),
        });
    }

    MicrobenchSuiteResult {
        suite: "cpu".to_string(),
        iterations,
        suite_runs: 1,
        device: None,
        recommendations: None,
        measurements,
    }
}

fn run_gpu_suite(iterations: usize) -> anyhow::Result<MicrobenchSuiteResult> {
    let gpu = MetalDevice::new()?;
    let info = gpu.info();
    let kernels = DequantKernels::new(&gpu)?;
    let attention = AttentionKernels::new(&gpu)?;
    let elementwise = ElementwiseKernels::new(&gpu)?;

    let sync_ms = average_ms(iterations, || gpu.execute_sync(|_| Ok(())))?;
    let concurrent_ms = average_ms(iterations, || gpu.execute_sync_concurrent(|_| Ok(())))?;

    let mut measurements = vec![
        MicrobenchMeasurement {
            name: "gpu.empty_execute_sync".to_string(),
            unit: "ms".to_string(),
            value: sync_ms,
            note: Some("empty encoder submit+wait overhead".to_string()),
        },
        MicrobenchMeasurement {
            name: "gpu.empty_execute_sync_concurrent".to_string(),
            unit: "ms".to_string(),
            value: concurrent_ms,
            note: Some("empty concurrent encoder submit+wait overhead".to_string()),
        },
    ];
    let mut recommendations = Vec::new();

    // Keep the generic decode shapes, then include the dominant Qwen 3.5
    // input-projection sizes for 9B and 27B so microbench output is directly
    // actionable for the current focus models.
    let shapes = [
        MatvecShape { m: 1_024, k: 4_096 },
        MatvecShape { m: 4_096, k: 4_096 },
        MatvecShape { m: 8_192, k: 4_096 },
        MatvecShape {
            m: 12_288,
            k: 4_096,
        },
        MatvecShape { m: 4_096, k: 8_192 },
        MatvecShape { m: 1_024, k: 5_120 },
        MatvecShape { m: 5_120, k: 5_120 },
        MatvecShape { m: 8_192, k: 5_120 },
        MatvecShape {
            m: 10_240,
            k: 5_120,
        },
        MatvecShape {
            m: 12_288,
            k: 5_120,
        },
        MatvecShape {
            m: 17_408,
            k: 5_120,
        },
        MatvecShape {
            m: 25_600,
            k: 5_120,
        },
        MatvecShape { m: 5_120, k: 8_192 },
        MatvecShape {
            m: 5_120,
            k: 25_600,
        },
    ];
    for shape in shapes {
        let q4 = run_quant_shape_sweep(
            iterations,
            &gpu,
            &kernels,
            QuantCase::Q4K,
            shape,
            quant_case_variants(QuantCase::Q4K),
        )?;
        measurements.extend(q4.measurements);
        if let Some(recommendation) = q4.recommendation {
            recommendations.push(recommendation);
        }

        let q6 = run_quant_shape_sweep(
            iterations,
            &gpu,
            &kernels,
            QuantCase::Q6K,
            shape,
            quant_case_variants(QuantCase::Q6K),
        )?;
        measurements.extend(q6.measurements);
        if let Some(recommendation) = q6.recommendation {
            recommendations.push(recommendation);
        }

        let q5 = run_quant_shape_sweep(
            iterations,
            &gpu,
            &kernels,
            QuantCase::Q5K,
            shape,
            quant_case_variants(QuantCase::Q5K),
        )?;
        measurements.extend(q5.measurements);
        if let Some(recommendation) = q5.recommendation {
            recommendations.push(recommendation);
        }
    }

    let q5k_prefill_shapes = [
        Q5KPrefillBatchShape {
            model: "8b_q5k",
            label: "attn_qkv",
            m: 4096,
            n: 512,
            k: 4096,
        },
        Q5KPrefillBatchShape {
            model: "8b_q5k",
            label: "ffn_up",
            m: 14336,
            n: 512,
            k: 4096,
        },
        Q5KPrefillBatchShape {
            model: "8b_q5k",
            label: "attn_qkv_small_window",
            m: 4096,
            n: 8,
            k: 4096,
        },
        Q5KPrefillBatchShape {
            model: "70b_mixed_q5k",
            label: "attn_qkv",
            m: 8192,
            n: 512,
            k: 8192,
        },
        Q5KPrefillBatchShape {
            model: "70b_mixed_q5k",
            label: "ffn_up",
            m: 28672,
            n: 512,
            k: 8192,
        },
        Q5KPrefillBatchShape {
            model: "70b_mixed_q5k",
            label: "attn_qkv_small_window",
            m: 8192,
            n: 8,
            k: 8192,
        },
    ];
    for shape in q5k_prefill_shapes {
        measurements.extend(run_q5k_prefill_batch_case(
            iterations,
            &gpu,
            &kernels,
            &elementwise,
            shape,
            q5k_prefill_batch_variants(shape),
        )?);
    }

    let exact_prefill_batch_shapes = [
        QuantPrefillBatchShape {
            model: "8b_q4k",
            label: "attn_qkv",
            m: 4096,
            n: 512,
            k: 4096,
        },
        QuantPrefillBatchShape {
            model: "8b_q4k",
            label: "ffn_down",
            m: 4096,
            n: 512,
            k: 14336,
        },
        QuantPrefillBatchShape {
            model: "qwen3_14b",
            label: "attn_qkv",
            m: 5120,
            n: 512,
            k: 5120,
        },
        QuantPrefillBatchShape {
            model: "qwen3_14b",
            label: "ffn_down",
            m: 5120,
            n: 512,
            k: 17408,
        },
        QuantPrefillBatchShape {
            model: "qwen3_32b",
            label: "attn_qkv_fused",
            m: 10240,
            n: 512,
            k: 5120,
        },
        QuantPrefillBatchShape {
            model: "qwen3_32b",
            label: "attn_wo",
            m: 5120,
            n: 512,
            k: 8192,
        },
        QuantPrefillBatchShape {
            model: "qwen3_32b",
            label: "ffn_down",
            m: 5120,
            n: 512,
            k: 25600,
        },
        QuantPrefillBatchShape {
            model: "gemma3_12b",
            label: "attn_qkv",
            m: 4096,
            n: 512,
            k: 3840,
        },
        QuantPrefillBatchShape {
            model: "gemma3_12b",
            label: "ffn_down",
            m: 3840,
            n: 512,
            k: 15360,
        },
    ];
    for quant in [QuantCase::Q4K, QuantCase::Q6K] {
        for shape in exact_prefill_batch_shapes {
            measurements.extend(run_quant_prefill_batch_case(
                iterations,
                &gpu,
                &kernels,
                &elementwise,
                quant,
                shape,
                &[
                    QuantPrefillBatchVariant::F32,
                    QuantPrefillBatchVariant::F16In,
                    QuantPrefillBatchVariant::F16InBn32,
                ],
            )?);
        }
    }

    let exact_prefill_pair_shapes = [
        QuantPrefillPairShape {
            model: "8b_q4k",
            label: "ffn_gate_up",
            m: 14336,
            n: 512,
            k: 4096,
        },
        QuantPrefillPairShape {
            model: "qwen3_14b",
            label: "ffn_gate_up",
            m: 17408,
            n: 512,
            k: 5120,
        },
        QuantPrefillPairShape {
            model: "qwen3_32b",
            label: "ffn_gate_up",
            m: 25600,
            n: 512,
            k: 5120,
        },
        QuantPrefillPairShape {
            model: "gemma3_12b",
            label: "ffn_gate_up",
            m: 15360,
            n: 512,
            k: 3840,
        },
    ];
    for quant in [QuantCase::Q4K, QuantCase::Q6K] {
        for shape in exact_prefill_pair_shapes {
            measurements.extend(run_quant_prefill_pair_case(
                iterations,
                &gpu,
                &kernels,
                &elementwise,
                quant,
                shape,
                &[
                    QuantPrefillPairVariant::SeparateF16In,
                    QuantPrefillPairVariant::PairF16In,
                ],
            )?);
        }
    }

    let attention_shapes = [
        AttentionShape {
            head_dim: 128,
            attend_len: 256,
            n_heads: 32,
            n_kv_heads: 8,
            kv_f16: true,
        },
        AttentionShape {
            head_dim: 128,
            attend_len: 1024,
            n_heads: 32,
            n_kv_heads: 8,
            kv_f16: true,
        },
        AttentionShape {
            head_dim: 256,
            attend_len: 256,
            n_heads: 32,
            n_kv_heads: 8,
            kv_f16: true,
        },
        AttentionShape {
            head_dim: 256,
            attend_len: 1024,
            n_heads: 32,
            n_kv_heads: 8,
            kv_f16: true,
        },
    ];
    for shape in attention_shapes {
        let variants: &[AttentionVariant] = if shape.head_dim == 256 {
            &[
                AttentionVariant::Auto,
                AttentionVariant::Baseline,
                AttentionVariant::SplitK,
                AttentionVariant::Sdpa,
            ]
        } else {
            &[
                AttentionVariant::Auto,
                AttentionVariant::Baseline,
                AttentionVariant::Hd128N2,
            ]
        };
        let result = run_attention_shape_sweep(iterations, &gpu, &attention, shape, variants)?;
        measurements.extend(result.measurements);
        if let Some(recommendation) = result.recommendation {
            recommendations.push(recommendation);
        }
    }

    let prefill_shapes = [
        PrefillShape {
            mode: PrefillModeCase::Local,
            head_dim: 128,
            n_tokens: 512,
            base_seq_len: 0,
            sliding_window: 0,
            n_heads: 32,
            n_kv_heads: 8,
            kv_f16: false,
        },
        PrefillShape {
            mode: PrefillModeCase::Cached,
            head_dim: 256,
            n_tokens: 512,
            base_seq_len: 512,
            sliding_window: 0,
            n_heads: 32,
            n_kv_heads: 8,
            kv_f16: true,
        },
    ];
    for shape in prefill_shapes {
        let variants: &[PrefillVariant] = match shape.mode {
            PrefillModeCase::Local => &[
                PrefillVariant::Auto,
                PrefillVariant::Baseline,
                PrefillVariant::Fa2Hd128,
            ],
            PrefillModeCase::Cached => &[
                PrefillVariant::Auto,
                PrefillVariant::Baseline,
                PrefillVariant::Fa2,
            ],
        };
        let result = run_prefill_shape_sweep(iterations, &gpu, &attention, shape, variants)?;
        measurements.extend(result.measurements);
        if let Some(recommendation) = result.recommendation {
            recommendations.push(recommendation);
        }
    }

    Ok(MicrobenchSuiteResult {
        suite: "gpu".to_string(),
        iterations,
        suite_runs: 1,
        device: Some(info.name),
        recommendations: Some(recommendations),
        measurements,
    })
}

fn run_uma_suite(iterations: usize) -> anyhow::Result<MicrobenchSuiteResult> {
    let gpu = MetalDevice::new()?;
    let info = gpu.info();

    let alloc_size = 4 * 1024 * 1024usize;
    let alloc_ms = average_ms(iterations, || {
        let buf = MetalBuffer::new(gpu.device(), alloc_size)?;
        black_box(buf.len());
        Ok(())
    })?;

    let mut shared = MetalBuffer::new(gpu.device(), 16 * 1024 * 1024)?;
    let write_secs = {
        let start = Instant::now();
        for iter in 0..iterations {
            unsafe {
                let slice = shared.as_mut_slice::<u8>();
                let value = (iter & 0xff) as u8;
                slice.fill(value);
                black_box(slice[0]);
            }
        }
        start.elapsed().as_secs_f64().max(f64::EPSILON)
    };
    let total_bytes = (shared.len() * iterations) as f64;

    Ok(MicrobenchSuiteResult {
        suite: "uma".to_string(),
        iterations,
        suite_runs: 1,
        device: Some(info.name),
        recommendations: None,
        measurements: vec![
            MicrobenchMeasurement {
                name: "uma.shared_buffer_alloc_4MiB".to_string(),
                unit: "ms".to_string(),
                value: alloc_ms,
                note: Some("fresh StorageModeShared allocation".to_string()),
            },
            MicrobenchMeasurement {
                name: "uma.shared_buffer_write_16MiB".to_string(),
                unit: "GB/s".to_string(),
                value: total_bytes / write_secs / 1e9,
                note: Some("CPU write throughput to shared Metal buffer".to_string()),
            },
        ],
    })
}

fn run_sync_suite(iterations: usize) -> anyhow::Result<MicrobenchSuiteResult> {
    let gpu = MetalDevice::new()?;
    let info = gpu.info();

    let execute_sync_ms = average_ms(iterations, || gpu.execute_sync(|_| Ok(())))?;
    let encode_commit_wait_ms = average_ms(iterations, || {
        let pending = gpu.encode_frame(|_| Ok(()))?;
        let inflight = gpu.commit_frame(pending);
        gpu.wait_frame(inflight)
    })?;

    Ok(MicrobenchSuiteResult {
        suite: "sync".to_string(),
        iterations,
        suite_runs: 1,
        device: Some(info.name),
        recommendations: None,
        measurements: vec![
            MicrobenchMeasurement {
                name: "sync.execute_sync".to_string(),
                unit: "ms".to_string(),
                value: execute_sync_ms,
                note: Some("single API submit+wait path".to_string()),
            },
            MicrobenchMeasurement {
                name: "sync.encode_commit_wait".to_string(),
                unit: "ms".to_string(),
                value: encode_commit_wait_ms,
                note: Some("split frame lifecycle overhead".to_string()),
            },
        ],
    })
}

fn aggregate_suite_runs(runs: Vec<MicrobenchSuiteResult>) -> MicrobenchSuiteResult {
    let first = runs.first().expect("at least one suite run");
    let mut measurements_by_name: BTreeMap<String, (String, Option<String>, f64, usize)> =
        BTreeMap::new();
    let measurement_order = first
        .measurements
        .iter()
        .map(|measurement| measurement.name.clone())
        .collect::<Vec<_>>();
    let mut recommendations = Vec::new();

    for run in &runs {
        if let Some(run_recommendations) = &run.recommendations {
            recommendations.extend(run_recommendations.iter().cloned());
        }
        for measurement in &run.measurements {
            let entry = measurements_by_name
                .entry(measurement.name.clone())
                .or_insert_with(|| {
                    (
                        measurement.unit.clone(),
                        measurement.note.clone(),
                        0.0,
                        0usize,
                    )
                });
            entry.2 += measurement.value;
            entry.3 += 1;
        }
    }

    let measurements = measurement_order
        .into_iter()
        .filter_map(|name| {
            measurements_by_name
                .remove(&name)
                .map(|(unit, note, sum, count)| MicrobenchMeasurement {
                    name,
                    unit,
                    value: sum / count.max(1) as f64,
                    note,
                })
        })
        .collect();

    MicrobenchSuiteResult {
        suite: first.suite.clone(),
        iterations: first.iterations,
        suite_runs: runs.len(),
        device: first.device.clone(),
        recommendations: (!recommendations.is_empty()).then_some(recommendations),
        measurements,
    }
}

fn average_ms<F>(iterations: usize, mut f: F) -> anyhow::Result<f64>
where
    F: FnMut() -> anyhow::Result<()>,
{
    let start = Instant::now();
    for _ in 0..iterations {
        f()?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / iterations.max(1) as f64)
}

fn repeats_for_target_bytes(size_bytes: usize, target_bytes: usize) -> usize {
    (target_bytes / size_bytes.max(1)).max(1)
}

fn repeats_for_target_elems(len: usize, target_elems: usize) -> usize {
    (target_elems / len.max(1)).max(1)
}

fn sum_bytes(data: &[u8]) -> u64 {
    let mut sum = 0u64;
    for &value in data {
        sum = sum.wrapping_add(value as u64);
    }
    black_box(sum)
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for (&lhs, &rhs) in a.iter().zip(b.iter()) {
        acc += lhs * rhs;
    }
    black_box(acc)
}

#[derive(Debug, Clone, Copy)]
enum QuantCase {
    Q4K,
    Q5K,
    Q6K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantVariant {
    Auto,
    Nr2,
}

#[derive(Debug, Clone, Copy)]
struct MatvecShape {
    m: usize,
    k: usize,
}

#[derive(Debug, Clone)]
struct QuantMatvecCaseResult {
    avg_ms: f64,
    effective_gbps: f64,
    command_buffers: f64,
    buffer_barriers: f64,
    candidate_label: &'static str,
    stability_label: &'static str,
}

struct QuantShapeSweepResult {
    recommendation: Option<MicrobenchRecommendation>,
    measurements: Vec<MicrobenchMeasurement>,
}

#[derive(Debug, Clone, Copy)]
struct Q5KPrefillBatchShape {
    model: &'static str,
    label: &'static str,
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Q5KPrefillBatchVariant {
    Base,
    F16In,
    Small,
}

#[derive(Debug, Clone, Copy)]
struct QuantPrefillBatchShape {
    model: &'static str,
    label: &'static str,
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantPrefillBatchVariant {
    F32,
    F16In,
    F16InBn32,
}

#[derive(Debug, Clone, Copy)]
struct QuantPrefillPairShape {
    model: &'static str,
    label: &'static str,
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantPrefillPairVariant {
    SeparateF16In,
    PairF16In,
}

fn quant_case_is_observational_only(quant: QuantCase) -> bool {
    matches!(quant, QuantCase::Q5K)
}

fn quant_case_supports_recommendation(quant: QuantCase) -> bool {
    !quant_case_is_observational_only(quant)
}

fn quant_variant_supported(quant: QuantCase, variant: QuantVariant) -> bool {
    matches!(
        (quant, variant),
        (QuantCase::Q4K, QuantVariant::Auto)
            | (QuantCase::Q4K, QuantVariant::Nr2)
            | (QuantCase::Q5K, QuantVariant::Auto)
            | (QuantCase::Q6K, QuantVariant::Auto)
            | (QuantCase::Q6K, QuantVariant::Nr2)
    )
}

fn quant_case_variants(quant: QuantCase) -> &'static [QuantVariant] {
    match quant {
        QuantCase::Q4K => &[QuantVariant::Auto, QuantVariant::Nr2],
        QuantCase::Q5K => &[QuantVariant::Auto],
        QuantCase::Q6K => &[QuantVariant::Auto, QuantVariant::Nr2],
    }
}

fn quant_measurement_note(
    quant: QuantCase,
    candidate_label: &str,
    stability_label: &str,
) -> String {
    format!(
        "fused decode matvec wall time; candidate={} tier={}{}",
        candidate_label,
        stability_label,
        if quant_case_is_observational_only(quant) {
            " observational_only=baseline"
        } else {
            ""
        }
    )
}

fn q5k_prefill_batch_measurement_note(variant: Q5KPrefillBatchVariant) -> String {
    let candidate = match variant {
        Q5KPrefillBatchVariant::Base => "q5_k.batch_base",
        Q5KPrefillBatchVariant::F16In => "q5_k.batch_f16in",
        Q5KPrefillBatchVariant::Small => "q5_k.batch_small",
    };
    format!(
        "q5_k prefill batch matmul wall time; candidate={} observational_only=route_study",
        candidate
    )
}

fn quant_prefill_batch_measurement_note(
    quant: QuantCase,
    variant: QuantPrefillBatchVariant,
) -> String {
    let candidate = match variant {
        QuantPrefillBatchVariant::F32 => "f32",
        QuantPrefillBatchVariant::F16In => "f16in",
        QuantPrefillBatchVariant::F16InBn32 => "f16in_bn32",
    };
    format!(
        "exact-shape prefill batch matmul wall time; quant={} candidate={} observational_only=exact_shape",
        quant_case_name(quant),
        candidate
    )
}

fn quant_prefill_pair_measurement_note(
    quant: QuantCase,
    variant: QuantPrefillPairVariant,
) -> String {
    let candidate = match variant {
        QuantPrefillPairVariant::SeparateF16In => "separate_f16in",
        QuantPrefillPairVariant::PairF16In => "pair_f16in",
    };
    format!(
        "exact-shape prefill FFN pair wall time; quant={} candidate={} observational_only=exact_shape",
        quant_case_name(quant),
        candidate
    )
}

fn q5k_prefill_batch_variants(shape: Q5KPrefillBatchShape) -> &'static [Q5KPrefillBatchVariant] {
    if (4..=8).contains(&shape.n) {
        &[
            Q5KPrefillBatchVariant::Base,
            Q5KPrefillBatchVariant::F16In,
            Q5KPrefillBatchVariant::Small,
        ]
    } else {
        &[Q5KPrefillBatchVariant::Base, Q5KPrefillBatchVariant::F16In]
    }
}

fn suite_observational_quants(suite: &MicrobenchSuiteResult) -> Vec<&'static str> {
    let mut quants = Vec::new();
    if suite
        .measurements
        .iter()
        .any(|m| m.name.starts_with("gpu.decode_matvec.q5_k."))
    {
        quants.push("q5_k");
    }
    if suite
        .measurements
        .iter()
        .any(|m| m.name.starts_with("gpu.prefill_matmul.q5_k."))
    {
        quants.push("q5_k_prefill");
    }
    quants
}

#[derive(Debug, Clone, Copy)]
enum AttentionVariant {
    Auto,
    Baseline,
    SplitK,
    Sdpa,
    Hd128N2,
}

#[derive(Debug, Clone, Copy)]
struct AttentionShape {
    head_dim: usize,
    attend_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    kv_f16: bool,
}

#[derive(Debug, Clone)]
struct AttentionDecodeCaseResult {
    avg_ms: f64,
    effective_kv_gbps: f64,
    command_buffers: f64,
    buffer_barriers: f64,
    candidate_label: &'static str,
    stability_label: &'static str,
}

#[derive(Debug, Clone, Copy)]
enum PrefillVariant {
    Auto,
    Baseline,
    Fa2,
    Fa2Hd128,
}

#[derive(Debug, Clone, Copy)]
enum PrefillModeCase {
    Local,
    Cached,
}

#[derive(Debug, Clone, Copy)]
struct PrefillShape {
    mode: PrefillModeCase,
    head_dim: usize,
    n_tokens: usize,
    base_seq_len: usize,
    sliding_window: usize,
    n_heads: usize,
    n_kv_heads: usize,
    kv_f16: bool,
}

#[derive(Debug, Clone)]
struct AttentionPrefillCaseResult {
    avg_ms: f64,
    effective_attn_gbps: f64,
    command_buffers: f64,
    buffer_barriers: f64,
    candidate_label: &'static str,
    stability_label: &'static str,
}

fn run_quant_shape_sweep(
    iterations: usize,
    gpu: &MetalDevice,
    kernels: &DequantKernels,
    quant: QuantCase,
    shape: MatvecShape,
    variants: &[QuantVariant],
) -> anyhow::Result<QuantShapeSweepResult> {
    let mut results = Vec::with_capacity(variants.len());
    for &variant in variants {
        let result = run_quant_matvec_case(iterations, gpu, kernels, quant, variant, shape)?;
        results.push((variant, result));
    }

    let auto_ms = results
        .iter()
        .find(|(variant, _)| matches!(variant, QuantVariant::Auto))
        .map(|(_, result)| result.avg_ms)
        .ok_or_else(|| anyhow::anyhow!("auto variant missing from GPU shape sweep"))?;

    let mut measurements = Vec::new();
    for (variant, result) in &results {
        let prefix = format!(
            "gpu.decode_matvec.{}.{}.{}x{}",
            quant_case_name(quant),
            quant_variant_name(*variant),
            shape.m,
            shape.k
        );
        measurements.push(MicrobenchMeasurement {
            name: prefix.clone(),
            unit: "ms".to_string(),
            value: result.avg_ms,
            note: Some(quant_measurement_note(
                quant,
                result.candidate_label,
                result.stability_label,
            )),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.effective_weight_bw"),
            unit: "GB/s".to_string(),
            value: result.effective_gbps,
            note: Some("weight bytes only; ignores x/y traffic".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.command_buffers"),
            unit: "avg".to_string(),
            value: result.command_buffers,
            note: Some("Metal command buffers per iteration".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.buffer_barriers"),
            unit: "avg".to_string(),
            value: result.buffer_barriers,
            note: Some("explicit buffer barriers per iteration".to_string()),
        });
        if !matches!(variant, QuantVariant::Auto) {
            measurements.push(MicrobenchMeasurement {
                name: format!("{prefix}.speedup_vs_auto"),
                unit: "x".to_string(),
                value: auto_ms / result.avg_ms.max(f64::EPSILON),
                note: Some("values > 1.0 mean faster than auto".to_string()),
            });
        }
    }

    let (best_variant, best_result) = results
        .iter()
        .min_by(|(_, lhs), (_, rhs)| lhs.avg_ms.total_cmp(&rhs.avg_ms))
        .ok_or_else(|| anyhow::anyhow!("no GPU shape sweep results produced"))?;
    if should_emit_quant_recommendation(quant, variants.len()) {
        let best_prefix = format!(
            "gpu.decode_matvec.{}.best.{}x{}",
            quant_case_name(quant),
            shape.m,
            shape.k
        );
        measurements.push(MicrobenchMeasurement {
            name: best_prefix.clone(),
            unit: "ms".to_string(),
            value: best_result.avg_ms,
            note: Some(format!("winner={}", quant_variant_name(*best_variant))),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{best_prefix}_speedup_vs_auto"),
            unit: "x".to_string(),
            value: auto_ms / best_result.avg_ms.max(f64::EPSILON),
            note: Some(format!(
                "winner={} values > 1.0 mean faster than auto",
                quant_variant_name(*best_variant)
            )),
        });
    }

    Ok(QuantShapeSweepResult {
        recommendation: should_emit_quant_recommendation(quant, variants.len()).then(|| {
            MicrobenchRecommendation {
                domain: "decode_matvec".to_string(),
                quant: quant_case_name(quant).to_string(),
                variant: quant_variant_name(*best_variant).to_string(),
                m: shape.m,
                k: shape.k,
                best_ms: best_result.avg_ms,
                speedup_vs_auto: auto_ms / best_result.avg_ms.max(f64::EPSILON),
            }
        }),
        measurements,
    })
}

fn should_emit_quant_recommendation(quant: QuantCase, variant_count: usize) -> bool {
    quant_case_supports_recommendation(quant) && variant_count > 1
}

fn run_attention_shape_sweep(
    iterations: usize,
    gpu: &MetalDevice,
    kernels: &AttentionKernels,
    shape: AttentionShape,
    variants: &[AttentionVariant],
) -> anyhow::Result<QuantShapeSweepResult> {
    let mut results = Vec::with_capacity(variants.len());
    for &variant in variants {
        let result = run_attention_decode_case(iterations, gpu, kernels, shape, variant)?;
        results.push((variant, result));
    }

    let auto_ms = results
        .iter()
        .find(|(variant, _)| matches!(variant, AttentionVariant::Auto))
        .map(|(_, result)| result.avg_ms)
        .ok_or_else(|| anyhow::anyhow!("auto variant missing from attention shape sweep"))?;

    let mut measurements = Vec::new();
    for (variant, result) in &results {
        let prefix = format!(
            "gpu.decode_attention.{}.{}.ctx{}",
            attention_shape_name(shape),
            attention_variant_name(*variant),
            shape.attend_len
        );
        measurements.push(MicrobenchMeasurement {
            name: prefix.clone(),
            unit: "ms".to_string(),
            value: result.avg_ms,
            note: Some(format!(
                "decode attention wall time; candidate={} tier={}",
                result.candidate_label, result.stability_label
            )),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.effective_kv_bw"),
            unit: "GB/s".to_string(),
            value: result.effective_kv_gbps,
            note: Some("K/V cache bytes only; ignores q/o traffic".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.command_buffers"),
            unit: "avg".to_string(),
            value: result.command_buffers,
            note: Some("Metal command buffers per iteration".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.buffer_barriers"),
            unit: "avg".to_string(),
            value: result.buffer_barriers,
            note: Some("explicit buffer barriers per iteration".to_string()),
        });
        if !matches!(variant, AttentionVariant::Auto) {
            measurements.push(MicrobenchMeasurement {
                name: format!("{prefix}.speedup_vs_auto"),
                unit: "x".to_string(),
                value: auto_ms / result.avg_ms.max(f64::EPSILON),
                note: Some("values > 1.0 mean faster than auto".to_string()),
            });
        }
    }

    let (best_variant, best_result) = results
        .iter()
        .min_by(|(_, lhs), (_, rhs)| lhs.avg_ms.total_cmp(&rhs.avg_ms))
        .ok_or_else(|| anyhow::anyhow!("no attention shape sweep results produced"))?;
    let best_prefix = format!(
        "gpu.decode_attention.{}.best.ctx{}",
        attention_shape_name(shape),
        shape.attend_len
    );
    measurements.push(MicrobenchMeasurement {
        name: best_prefix.clone(),
        unit: "ms".to_string(),
        value: best_result.avg_ms,
        note: Some(format!("winner={}", attention_variant_name(*best_variant))),
    });
    measurements.push(MicrobenchMeasurement {
        name: format!("{best_prefix}_speedup_vs_auto"),
        unit: "x".to_string(),
        value: auto_ms / best_result.avg_ms.max(f64::EPSILON),
        note: Some(format!(
            "winner={} values > 1.0 mean faster than auto",
            attention_variant_name(*best_variant)
        )),
    });

    Ok(QuantShapeSweepResult {
        recommendation: Some(MicrobenchRecommendation {
            domain: "decode_attention".to_string(),
            quant: attention_shape_name(shape).to_string(),
            variant: attention_variant_name(*best_variant).to_string(),
            m: shape.head_dim,
            k: shape.attend_len,
            best_ms: best_result.avg_ms,
            speedup_vs_auto: auto_ms / best_result.avg_ms.max(f64::EPSILON),
        }),
        measurements,
    })
}

fn run_prefill_shape_sweep(
    iterations: usize,
    gpu: &MetalDevice,
    kernels: &AttentionKernels,
    shape: PrefillShape,
    variants: &[PrefillVariant],
) -> anyhow::Result<QuantShapeSweepResult> {
    let mut results = Vec::with_capacity(variants.len());
    for &variant in variants {
        let result = run_attention_prefill_case(iterations, gpu, kernels, shape, variant)?;
        results.push((variant, result));
    }

    let auto_ms = results
        .iter()
        .find(|(variant, _)| matches!(variant, PrefillVariant::Auto))
        .map(|(_, result)| result.avg_ms)
        .ok_or_else(|| anyhow::anyhow!("auto variant missing from prefill shape sweep"))?;

    let mut measurements = Vec::new();
    for (variant, result) in &results {
        let prefix = format!(
            "gpu.prefill_attention.{}.{}.tokens{}",
            prefill_shape_name(shape),
            prefill_variant_name(*variant),
            shape.n_tokens
        );
        measurements.push(MicrobenchMeasurement {
            name: prefix.clone(),
            unit: "ms".to_string(),
            value: result.avg_ms,
            note: Some(format!(
                "prefill attention wall time; candidate={} tier={}",
                result.candidate_label, result.stability_label
            )),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.effective_attn_bw"),
            unit: "GB/s".to_string(),
            value: result.effective_attn_gbps,
            note: Some("attention q/k/v traffic estimate only".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.command_buffers"),
            unit: "avg".to_string(),
            value: result.command_buffers,
            note: Some("Metal command buffers per iteration".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.buffer_barriers"),
            unit: "avg".to_string(),
            value: result.buffer_barriers,
            note: Some("explicit buffer barriers per iteration".to_string()),
        });
        if !matches!(variant, PrefillVariant::Auto) {
            measurements.push(MicrobenchMeasurement {
                name: format!("{prefix}.speedup_vs_auto"),
                unit: "x".to_string(),
                value: auto_ms / result.avg_ms.max(f64::EPSILON),
                note: Some("values > 1.0 mean faster than auto".to_string()),
            });
        }
    }

    let (best_variant, best_result) = results
        .iter()
        .min_by(|(_, lhs), (_, rhs)| lhs.avg_ms.total_cmp(&rhs.avg_ms))
        .ok_or_else(|| anyhow::anyhow!("no prefill shape sweep results produced"))?;
    let best_prefix = format!(
        "gpu.prefill_attention.{}.best.tokens{}",
        prefill_shape_name(shape),
        shape.n_tokens
    );
    measurements.push(MicrobenchMeasurement {
        name: best_prefix.clone(),
        unit: "ms".to_string(),
        value: best_result.avg_ms,
        note: Some(format!("winner={}", prefill_variant_name(*best_variant))),
    });
    measurements.push(MicrobenchMeasurement {
        name: format!("{best_prefix}_speedup_vs_auto"),
        unit: "x".to_string(),
        value: auto_ms / best_result.avg_ms.max(f64::EPSILON),
        note: Some(format!(
            "winner={} values > 1.0 mean faster than auto",
            prefill_variant_name(*best_variant)
        )),
    });

    Ok(QuantShapeSweepResult {
        recommendation: Some(MicrobenchRecommendation {
            domain: "prefill_attention".to_string(),
            quant: prefill_shape_name(shape).to_string(),
            variant: prefill_variant_name(*best_variant).to_string(),
            m: shape.head_dim,
            k: shape.n_tokens,
            best_ms: best_result.avg_ms,
            speedup_vs_auto: auto_ms / best_result.avg_ms.max(f64::EPSILON),
        }),
        measurements,
    })
}

fn run_quant_matvec_case(
    iterations: usize,
    gpu: &MetalDevice,
    kernels: &DequantKernels,
    quant: QuantCase,
    variant: QuantVariant,
    shape: MatvecShape,
) -> anyhow::Result<QuantMatvecCaseResult> {
    anyhow::ensure!(
        quant_variant_supported(quant, variant),
        "unsupported {} microbench variant {}",
        quant_case_name(quant),
        quant_variant_name(variant),
    );
    let m = shape.m;
    let k = shape.k;
    let weight_bytes = match quant {
        QuantCase::Q4K => make_q4k_matrix_bytes(m, k),
        QuantCase::Q5K => make_q5k_matrix_bytes(m, k),
        QuantCase::Q6K => make_q6k_matrix_bytes(m, k),
    };
    let x: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
    let (candidate_label, stability_label) =
        quant_variant_candidate(kernels, quant, variant, m as u32);
    let y = MetalBuffer::new(gpu.device(), m * std::mem::size_of::<f32>())?;
    let buf_a = MetalBuffer::from_bytes(gpu.device(), &weight_bytes)?;
    let buf_x = MetalBuffer::from_slice(gpu.device(), &x)?;

    // Warm the kernel and pipelines once before timed iterations.
    run_quant_variant(
        kernels, gpu, quant, variant, &buf_a, &buf_x, &y, m as u32, k as u32,
    )?;

    gpu.reset_perf_counters();
    let start = Instant::now();
    for _ in 0..iterations {
        run_quant_variant(
            kernels, gpu, quant, variant, &buf_a, &buf_x, &y, m as u32, k as u32,
        )?;
    }
    black_box(unsafe { y.as_slice::<f32>()[0] });
    let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
    let counters = gpu.perf_counters();
    let avg_ms = elapsed * 1000.0 / iterations.max(1) as f64;
    let effective_gbps = (weight_bytes.len() * iterations) as f64 / elapsed / 1e9;

    Ok(QuantMatvecCaseResult {
        avg_ms,
        effective_gbps,
        command_buffers: counters.command_buffers as f64 / iterations.max(1) as f64,
        buffer_barriers: counters.buffer_barriers as f64 / iterations.max(1) as f64,
        candidate_label,
        stability_label,
    })
}

fn run_q5k_prefill_batch_case(
    iterations: usize,
    gpu: &MetalDevice,
    kernels: &DequantKernels,
    elementwise: &ElementwiseKernels,
    shape: Q5KPrefillBatchShape,
    variants: &[Q5KPrefillBatchVariant],
) -> anyhow::Result<Vec<MicrobenchMeasurement>> {
    let weight_bytes = make_q5k_matrix_bytes(shape.m, shape.k);
    let batch_input: Vec<f32> = (0..shape.n * shape.k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
        .collect();
    let buf_a = MetalBuffer::from_bytes(gpu.device(), &weight_bytes)?;
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input)?;
    let buf_b_f16 = MetalBuffer::new(
        gpu.device(),
        batch_input.len() * std::mem::size_of::<half::f16>(),
    )?;
    let buf_c = MetalBuffer::new(gpu.device(), shape.n * shape.m * std::mem::size_of::<f32>())?;

    let mut measurements = Vec::new();
    let mut base_avg_ms = None;
    for &variant in variants {
        run_q5k_prefill_batch_variant(
            gpu,
            kernels,
            elementwise,
            variant,
            &buf_a,
            &buf_b,
            &buf_b_f16,
            &buf_c,
            shape.m as u32,
            shape.n as u32,
            shape.k as u32,
        )?;

        gpu.reset_perf_counters();
        let start = Instant::now();
        for _ in 0..iterations {
            run_q5k_prefill_batch_variant(
                gpu,
                kernels,
                elementwise,
                variant,
                &buf_a,
                &buf_b,
                &buf_b_f16,
                &buf_c,
                shape.m as u32,
                shape.n as u32,
                shape.k as u32,
            )?;
        }
        black_box(unsafe { buf_c.as_slice::<f32>()[0] });
        let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
        let counters = gpu.perf_counters();
        let avg_ms = elapsed * 1000.0 / iterations.max(1) as f64;
        let effective_gbps = ((weight_bytes.len() + batch_input.len() * std::mem::size_of::<f32>())
            * iterations) as f64
            / elapsed
            / 1e9;
        let prefix = format!(
            "gpu.prefill_matmul.q5_k.{}.{}.{}.tokens{}.{}x{}",
            q5k_prefill_batch_variant_name(variant),
            shape.model,
            shape.label,
            shape.n,
            shape.m,
            shape.k
        );

        measurements.push(MicrobenchMeasurement {
            name: prefix.clone(),
            unit: "ms".to_string(),
            value: avg_ms,
            note: Some(q5k_prefill_batch_measurement_note(variant)),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.effective_io_bw"),
            unit: "GB/s".to_string(),
            value: effective_gbps,
            note: Some("weights + dense input bytes only".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.command_buffers"),
            unit: "avg".to_string(),
            value: counters.command_buffers as f64 / iterations.max(1) as f64,
            note: Some("Metal command buffers per iteration".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.buffer_barriers"),
            unit: "avg".to_string(),
            value: counters.buffer_barriers as f64 / iterations.max(1) as f64,
            note: Some("explicit buffer barriers per iteration".to_string()),
        });
        if matches!(variant, Q5KPrefillBatchVariant::Base) {
            base_avg_ms = Some(avg_ms);
        } else if let Some(base_avg_ms) = base_avg_ms {
            measurements.push(MicrobenchMeasurement {
                name: format!("{prefix}.speedup_vs_base"),
                unit: "x".to_string(),
                value: base_avg_ms / avg_ms.max(f64::EPSILON),
                note: Some("values > 1.0 mean faster than q5_k.batch_base".to_string()),
            });
        }
    }

    Ok(measurements)
}

fn quant_prefill_batch_variant_name(variant: QuantPrefillBatchVariant) -> &'static str {
    match variant {
        QuantPrefillBatchVariant::F32 => "f32",
        QuantPrefillBatchVariant::F16In => "f16in",
        QuantPrefillBatchVariant::F16InBn32 => "f16in_bn32",
    }
}

fn quant_prefill_pair_variant_name(variant: QuantPrefillPairVariant) -> &'static str {
    match variant {
        QuantPrefillPairVariant::SeparateF16In => "separate_f16in",
        QuantPrefillPairVariant::PairF16In => "pair_f16in",
    }
}

fn quant_prefill_dispatch_config(variant: QuantPrefillBatchVariant) -> DequantDispatchConfig {
    let mut config = DequantDispatchConfig::default();
    if matches!(variant, QuantPrefillBatchVariant::F16InBn32) {
        config.batch_f16in_use_bn32 = true;
    }
    config
}

fn run_quant_prefill_batch_case(
    iterations: usize,
    gpu: &MetalDevice,
    kernels: &DequantKernels,
    elementwise: &ElementwiseKernels,
    quant: QuantCase,
    shape: QuantPrefillBatchShape,
    variants: &[QuantPrefillBatchVariant],
) -> anyhow::Result<Vec<MicrobenchMeasurement>> {
    let weight_bytes = match quant {
        QuantCase::Q4K => make_q4k_matrix_bytes(shape.m, shape.k),
        QuantCase::Q6K => make_q6k_matrix_bytes(shape.m, shape.k),
        QuantCase::Q5K => anyhow::bail!("q5_k exact-shape prefill batch microbench is unsupported"),
    };
    let batch_input: Vec<f32> = (0..shape.n * shape.k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
        .collect();
    let buf_a = MetalBuffer::from_bytes(gpu.device(), &weight_bytes)?;
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input)?;
    let buf_b_f16 = MetalBuffer::new(
        gpu.device(),
        batch_input.len() * std::mem::size_of::<half::f16>(),
    )?;
    let buf_c = MetalBuffer::new(gpu.device(), shape.n * shape.m * std::mem::size_of::<f32>())?;

    let mut measurements = Vec::new();
    let mut base_avg_ms = None;
    for &variant in variants {
        run_quant_prefill_batch_variant(
            gpu,
            kernels,
            elementwise,
            quant,
            variant,
            &buf_a,
            &buf_b,
            &buf_b_f16,
            &buf_c,
            shape.m as u32,
            shape.n as u32,
            shape.k as u32,
        )?;

        gpu.reset_perf_counters();
        let start = Instant::now();
        for _ in 0..iterations {
            run_quant_prefill_batch_variant(
                gpu,
                kernels,
                elementwise,
                quant,
                variant,
                &buf_a,
                &buf_b,
                &buf_b_f16,
                &buf_c,
                shape.m as u32,
                shape.n as u32,
                shape.k as u32,
            )?;
        }
        black_box(unsafe { buf_c.as_slice::<f32>()[0] });
        let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
        let counters = gpu.perf_counters();
        let avg_ms = elapsed * 1000.0 / iterations.max(1) as f64;
        let effective_gbps = ((weight_bytes.len()
            + batch_input.len() * std::mem::size_of::<f32>()
            + shape.n * shape.m * std::mem::size_of::<f32>())
            * iterations) as f64
            / elapsed
            / 1e9;
        let prefix = format!(
            "gpu.prefill_matmul.{}.{}.{}.{}.tokens{}.{}x{}",
            quant_case_name(quant),
            quant_prefill_batch_variant_name(variant),
            shape.model,
            shape.label,
            shape.n,
            shape.m,
            shape.k
        );

        measurements.push(MicrobenchMeasurement {
            name: prefix.clone(),
            unit: "ms".to_string(),
            value: avg_ms,
            note: Some(quant_prefill_batch_measurement_note(quant, variant)),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.effective_io_bw"),
            unit: "GB/s".to_string(),
            value: effective_gbps,
            note: Some("weights + dense input/output bytes only".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.command_buffers"),
            unit: "avg".to_string(),
            value: counters.command_buffers as f64 / iterations.max(1) as f64,
            note: Some("Metal command buffers per iteration".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.buffer_barriers"),
            unit: "avg".to_string(),
            value: counters.buffer_barriers as f64 / iterations.max(1) as f64,
            note: Some("explicit buffer barriers per iteration".to_string()),
        });
        if matches!(variant, QuantPrefillBatchVariant::F32) {
            base_avg_ms = Some(avg_ms);
        } else if let Some(base_avg_ms) = base_avg_ms {
            measurements.push(MicrobenchMeasurement {
                name: format!("{prefix}.speedup_vs_f32"),
                unit: "x".to_string(),
                value: base_avg_ms / avg_ms.max(f64::EPSILON),
                note: Some("values > 1.0 mean faster than f32 batch matmul".to_string()),
            });
        }
    }

    Ok(measurements)
}

#[allow(clippy::too_many_arguments)]
fn run_quant_prefill_batch_variant(
    gpu: &MetalDevice,
    kernels: &DequantKernels,
    elementwise: &ElementwiseKernels,
    quant: QuantCase,
    variant: QuantPrefillBatchVariant,
    buf_a: &MetalBuffer,
    buf_b: &MetalBuffer,
    buf_b_f16: &MetalBuffer,
    buf_c: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) -> anyhow::Result<()> {
    gpu.execute_sync(|encoder| {
        match variant {
            QuantPrefillBatchVariant::F32 => match quant {
                QuantCase::Q4K => {
                    kernels.encode_fused_batch_q4_k(encoder, buf_a, buf_b, buf_c, m, n, k)
                }
                QuantCase::Q6K => {
                    kernels.encode_fused_batch_q6_k(encoder, buf_a, buf_b, buf_c, m, n, k)
                }
                QuantCase::Q5K => {
                    anyhow::bail!("unsupported q5_k exact-shape prefill batch variant")
                }
            },
            QuantPrefillBatchVariant::F16In | QuantPrefillBatchVariant::F16InBn32 => {
                elementwise.encode_cast_f32_to_f16(encoder, buf_b, buf_b_f16, n * k);
                let config = quant_prefill_dispatch_config(variant);
                match quant {
                    QuantCase::Q4K => kernels.encode_fused_batch_q4_k_f16in_with_config(
                        encoder, buf_a, buf_b_f16, buf_c, m, n, k, config,
                    ),
                    QuantCase::Q6K => kernels.encode_fused_batch_q6_k_f16in_with_config(
                        encoder, buf_a, buf_b_f16, buf_c, m, n, k, config,
                    ),
                    QuantCase::Q5K => {
                        anyhow::bail!("unsupported q5_k exact-shape prefill batch variant")
                    }
                }
            }
        }
        Ok(())
    })
}

fn run_quant_prefill_pair_case(
    iterations: usize,
    gpu: &MetalDevice,
    kernels: &DequantKernels,
    elementwise: &ElementwiseKernels,
    quant: QuantCase,
    shape: QuantPrefillPairShape,
    variants: &[QuantPrefillPairVariant],
) -> anyhow::Result<Vec<MicrobenchMeasurement>> {
    let weight_bytes = match quant {
        QuantCase::Q4K => make_q4k_matrix_bytes(shape.m, shape.k),
        QuantCase::Q6K => make_q6k_matrix_bytes(shape.m, shape.k),
        QuantCase::Q5K => anyhow::bail!("q5_k exact-shape prefill pair microbench is unsupported"),
    };
    let batch_input: Vec<f32> = (0..shape.n * shape.k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
        .collect();
    let buf_w0 = MetalBuffer::from_bytes(gpu.device(), &weight_bytes)?;
    let buf_w1 = MetalBuffer::from_bytes(gpu.device(), &weight_bytes)?;
    let buf_b = MetalBuffer::from_slice(gpu.device(), &batch_input)?;
    let buf_b_f16 = MetalBuffer::new(
        gpu.device(),
        batch_input.len() * std::mem::size_of::<half::f16>(),
    )?;
    let buf_c0 = MetalBuffer::new(gpu.device(), shape.n * shape.m * std::mem::size_of::<f32>())?;
    let buf_c1 = MetalBuffer::new(gpu.device(), shape.n * shape.m * std::mem::size_of::<f32>())?;

    let mut measurements = Vec::new();
    let mut separate_avg_ms = None;
    for &variant in variants {
        run_quant_prefill_pair_variant(
            gpu,
            kernels,
            elementwise,
            quant,
            variant,
            &buf_w0,
            &buf_w1,
            &buf_b,
            &buf_b_f16,
            &buf_c0,
            &buf_c1,
            shape.m as u32,
            shape.n as u32,
            shape.k as u32,
        )?;

        gpu.reset_perf_counters();
        let start = Instant::now();
        for _ in 0..iterations {
            run_quant_prefill_pair_variant(
                gpu,
                kernels,
                elementwise,
                quant,
                variant,
                &buf_w0,
                &buf_w1,
                &buf_b,
                &buf_b_f16,
                &buf_c0,
                &buf_c1,
                shape.m as u32,
                shape.n as u32,
                shape.k as u32,
            )?;
        }
        black_box(unsafe { buf_c0.as_slice::<f32>()[0] });
        black_box(unsafe { buf_c1.as_slice::<f32>()[0] });
        let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
        let counters = gpu.perf_counters();
        let avg_ms = elapsed * 1000.0 / iterations.max(1) as f64;
        let effective_gbps = ((2 * weight_bytes.len()
            + batch_input.len() * std::mem::size_of::<f32>()
            + 2 * shape.n * shape.m * std::mem::size_of::<f32>())
            * iterations) as f64
            / elapsed
            / 1e9;
        let prefix = format!(
            "gpu.prefill_pair.{}.{}.{}.{}.tokens{}.{}x{}",
            quant_case_name(quant),
            quant_prefill_pair_variant_name(variant),
            shape.model,
            shape.label,
            shape.n,
            shape.m,
            shape.k
        );

        measurements.push(MicrobenchMeasurement {
            name: prefix.clone(),
            unit: "ms".to_string(),
            value: avg_ms,
            note: Some(quant_prefill_pair_measurement_note(quant, variant)),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.effective_io_bw"),
            unit: "GB/s".to_string(),
            value: effective_gbps,
            note: Some("two weights + dense input/output bytes only".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.command_buffers"),
            unit: "avg".to_string(),
            value: counters.command_buffers as f64 / iterations.max(1) as f64,
            note: Some("Metal command buffers per iteration".to_string()),
        });
        measurements.push(MicrobenchMeasurement {
            name: format!("{prefix}.buffer_barriers"),
            unit: "avg".to_string(),
            value: counters.buffer_barriers as f64 / iterations.max(1) as f64,
            note: Some("explicit buffer barriers per iteration".to_string()),
        });
        if matches!(variant, QuantPrefillPairVariant::SeparateF16In) {
            separate_avg_ms = Some(avg_ms);
        } else if let Some(separate_avg_ms) = separate_avg_ms {
            measurements.push(MicrobenchMeasurement {
                name: format!("{prefix}.speedup_vs_separate_f16in"),
                unit: "x".to_string(),
                value: separate_avg_ms / avg_ms.max(f64::EPSILON),
                note: Some(
                    "values > 1.0 mean faster than two separate f16in projections".to_string(),
                ),
            });
        }
    }

    Ok(measurements)
}

#[allow(clippy::too_many_arguments)]
fn run_quant_prefill_pair_variant(
    gpu: &MetalDevice,
    kernels: &DequantKernels,
    elementwise: &ElementwiseKernels,
    quant: QuantCase,
    variant: QuantPrefillPairVariant,
    buf_w0: &MetalBuffer,
    buf_w1: &MetalBuffer,
    buf_b: &MetalBuffer,
    buf_b_f16: &MetalBuffer,
    buf_c0: &MetalBuffer,
    buf_c1: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) -> anyhow::Result<()> {
    gpu.execute_sync(|encoder| {
        elementwise.encode_cast_f32_to_f16(encoder, buf_b, buf_b_f16, n * k);
        match variant {
            QuantPrefillPairVariant::SeparateF16In => {
                let config = DequantDispatchConfig::default();
                match quant {
                    QuantCase::Q4K => {
                        kernels.encode_fused_batch_q4_k_f16in_with_config(
                            encoder, buf_w0, buf_b_f16, buf_c0, m, n, k, config,
                        );
                        kernels.encode_fused_batch_q4_k_f16in_with_config(
                            encoder, buf_w1, buf_b_f16, buf_c1, m, n, k, config,
                        );
                    }
                    QuantCase::Q6K => {
                        kernels.encode_fused_batch_q6_k_f16in_with_config(
                            encoder, buf_w0, buf_b_f16, buf_c0, m, n, k, config,
                        );
                        kernels.encode_fused_batch_q6_k_f16in_with_config(
                            encoder, buf_w1, buf_b_f16, buf_c1, m, n, k, config,
                        );
                    }
                    QuantCase::Q5K => {
                        anyhow::bail!("unsupported q5_k exact-shape prefill pair variant")
                    }
                }
            }
            QuantPrefillPairVariant::PairF16In => match quant {
                QuantCase::Q4K => {
                    kernels.encode_fused_batch_pair_q4_k_f16in(
                        encoder, buf_w0, buf_w1, buf_b_f16, buf_c0, buf_c1, m, n, k,
                    );
                }
                QuantCase::Q6K => {
                    kernels.encode_fused_batch_pair_q6_k_f16in(
                        encoder, buf_w0, buf_w1, buf_b_f16, buf_c0, buf_c1, m, n, k,
                    );
                }
                QuantCase::Q5K => {
                    anyhow::bail!("unsupported q5_k exact-shape prefill pair variant")
                }
            },
        }
        Ok(())
    })
}

fn q5k_prefill_batch_variant_name(variant: Q5KPrefillBatchVariant) -> &'static str {
    match variant {
        Q5KPrefillBatchVariant::Base => "base",
        Q5KPrefillBatchVariant::F16In => "f16in",
        Q5KPrefillBatchVariant::Small => "small",
    }
}

#[allow(clippy::too_many_arguments)]
fn run_q5k_prefill_batch_variant(
    gpu: &MetalDevice,
    kernels: &DequantKernels,
    elementwise: &ElementwiseKernels,
    variant: Q5KPrefillBatchVariant,
    buf_a: &MetalBuffer,
    buf_b: &MetalBuffer,
    buf_b_f16: &MetalBuffer,
    buf_c: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) -> anyhow::Result<()> {
    gpu.execute_sync(|encoder| {
        match variant {
            Q5KPrefillBatchVariant::Base => {
                kernels.encode_fused_batch_q5_k(encoder, buf_a, buf_b, buf_c, m, n, k);
            }
            Q5KPrefillBatchVariant::F16In => {
                elementwise.encode_cast_f32_to_f16(encoder, buf_b, buf_b_f16, n * k);
                kernels.encode_fused_batch_q5_k_f16in(encoder, buf_a, buf_b_f16, buf_c, m, n, k);
            }
            Q5KPrefillBatchVariant::Small => {
                kernels.encode_fused_batch_q5_k_small(encoder, buf_a, buf_b, buf_c, m, n, k);
            }
        }
        Ok(())
    })
}

fn run_attention_decode_case(
    iterations: usize,
    gpu: &MetalDevice,
    kernels: &AttentionKernels,
    shape: AttentionShape,
    variant: AttentionVariant,
) -> anyhow::Result<AttentionDecodeCaseResult> {
    let q_elems = shape.n_heads * shape.head_dim;
    let kv_elems = shape.attend_len * shape.n_kv_heads * shape.head_dim;
    let q: Vec<f32> = (0..q_elems)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.0625)
        .collect();
    let o = MetalBuffer::new(gpu.device(), q_elems * std::mem::size_of::<f32>())?;
    let buf_q = MetalBuffer::from_slice(gpu.device(), &q)?;
    let (buf_k, buf_v, kv_bytes_per_elem) = if shape.kv_f16 {
        let k: Vec<half::f16> = (0..kv_elems)
            .map(|i| half::f16::from_f32(((i % 19) as f32 - 9.0) * 0.03125))
            .collect();
        let v: Vec<half::f16> = (0..kv_elems)
            .map(|i| half::f16::from_f32(((i % 23) as f32 - 11.0) * 0.03125))
            .collect();
        (
            MetalBuffer::from_slice(gpu.device(), &k)?,
            MetalBuffer::from_slice(gpu.device(), &v)?,
            std::mem::size_of::<half::f16>(),
        )
    } else {
        let k: Vec<f32> = (0..kv_elems)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.03125)
            .collect();
        let v: Vec<f32> = (0..kv_elems)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
            .collect();
        (
            MetalBuffer::from_slice(gpu.device(), &k)?,
            MetalBuffer::from_slice(gpu.device(), &v)?,
            std::mem::size_of::<f32>(),
        )
    };

    let config = attention_variant_config(variant);
    let selection = config.decode_candidate_selection(
        shape.kv_f16,
        shape.head_dim as u32,
        shape.attend_len as u32,
    );

    run_attention_variant(
        kernels,
        gpu,
        &buf_q,
        &buf_k,
        &buf_v,
        &o,
        shape,
        selection.candidate,
        config,
    )?;

    gpu.reset_perf_counters();
    let start = Instant::now();
    for _ in 0..iterations {
        run_attention_variant(
            kernels,
            gpu,
            &buf_q,
            &buf_k,
            &buf_v,
            &o,
            shape,
            selection.candidate,
            config,
        )?;
    }
    black_box(unsafe { o.as_slice::<f32>()[0] });
    let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
    let counters = gpu.perf_counters();
    let avg_ms = elapsed * 1000.0 / iterations.max(1) as f64;
    let kv_bytes = 2 * kv_elems * kv_bytes_per_elem;
    let effective_kv_gbps = (kv_bytes * iterations) as f64 / elapsed / 1e9;

    Ok(AttentionDecodeCaseResult {
        avg_ms,
        effective_kv_gbps,
        command_buffers: counters.command_buffers as f64 / iterations.max(1) as f64,
        buffer_barriers: counters.buffer_barriers as f64 / iterations.max(1) as f64,
        candidate_label: selection.label(),
        stability_label: selection.stability.label(),
    })
}

fn run_attention_prefill_case(
    iterations: usize,
    gpu: &MetalDevice,
    kernels: &AttentionKernels,
    shape: PrefillShape,
    variant: PrefillVariant,
) -> anyhow::Result<AttentionPrefillCaseResult> {
    let q_elems = shape.n_tokens * shape.n_heads * shape.head_dim;
    let local_kv_elems = shape.n_tokens * shape.n_kv_heads * shape.head_dim;
    let total_kv_elems = (shape.base_seq_len + shape.n_tokens) * shape.n_kv_heads * shape.head_dim;
    let q: Vec<f32> = (0..q_elems)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.0625)
        .collect();
    let o = MetalBuffer::new(gpu.device(), q_elems * std::mem::size_of::<f32>())?;
    let buf_q = MetalBuffer::from_slice(gpu.device(), &q)?;
    let config = prefill_variant_config(variant);

    match shape.mode {
        PrefillModeCase::Local => {
            let k: Vec<f32> = (0..local_kv_elems)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
                .collect();
            let v: Vec<f32> = (0..local_kv_elems)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
                .collect();
            let buf_k = MetalBuffer::from_slice(gpu.device(), &k)?;
            let buf_v = MetalBuffer::from_slice(gpu.device(), &v)?;
            let selection = config
                .prefill_local_candidate_selection(shape.n_tokens as u32, shape.head_dim as u32);
            let traffic_bytes =
                (q_elems + 2 * local_kv_elems + q_elems) * std::mem::size_of::<f32>();
            kernels.attention_prefill_with_config(
                gpu,
                &buf_q,
                &buf_k,
                &buf_v,
                &o,
                shape.n_tokens as u32,
                shape.n_heads as u32,
                shape.n_kv_heads as u32,
                shape.head_dim as u32,
                config,
            )?;
            gpu.reset_perf_counters();
            let start = Instant::now();
            for _ in 0..iterations {
                kernels.attention_prefill_with_config(
                    gpu,
                    &buf_q,
                    &buf_k,
                    &buf_v,
                    &o,
                    shape.n_tokens as u32,
                    shape.n_heads as u32,
                    shape.n_kv_heads as u32,
                    shape.head_dim as u32,
                    config,
                )?;
            }
            black_box(unsafe { o.as_slice::<f32>()[0] });
            let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
            let counters = gpu.perf_counters();
            Ok(AttentionPrefillCaseResult {
                avg_ms: elapsed * 1000.0 / iterations.max(1) as f64,
                effective_attn_gbps: (traffic_bytes * iterations) as f64 / elapsed / 1e9,
                command_buffers: counters.command_buffers as f64 / iterations.max(1) as f64,
                buffer_barriers: counters.buffer_barriers as f64 / iterations.max(1) as f64,
                candidate_label: selection.label(),
                stability_label: selection.stability.label(),
            })
        }
        PrefillModeCase::Cached => {
            let kv_bytes_per_elem = if shape.kv_f16 {
                std::mem::size_of::<half::f16>()
            } else {
                std::mem::size_of::<f32>()
            };
            let selection = config.prefill_cached_candidate_selection(
                shape.kv_f16,
                shape.n_tokens as u32,
                shape.head_dim as u32,
                shape.base_seq_len as u32,
                shape.sliding_window as u32,
            );
            let traffic_bytes = q_elems * std::mem::size_of::<f32>()
                + 2 * total_kv_elems * kv_bytes_per_elem
                + q_elems * std::mem::size_of::<f32>();
            if shape.kv_f16 {
                let k_cache: Vec<half::f16> = (0..total_kv_elems)
                    .map(|i| half::f16::from_f32(((i % 19) as f32 - 9.0) * 0.03125))
                    .collect();
                let v_cache: Vec<half::f16> = (0..total_kv_elems)
                    .map(|i| half::f16::from_f32(((i % 23) as f32 - 11.0) * 0.03125))
                    .collect();
                let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache)?;
                let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache)?;
                gpu.execute_sync(|encoder| {
                    kernels.encode_attention_prefill_cached_with_config(
                        encoder,
                        &buf_q,
                        &buf_k,
                        &buf_v,
                        &o,
                        true,
                        shape.n_tokens as u32,
                        shape.n_heads as u32,
                        shape.n_kv_heads as u32,
                        shape.head_dim as u32,
                        shape.base_seq_len as u32,
                        shape.sliding_window as u32,
                        config,
                    );
                    Ok(())
                })?;
                gpu.reset_perf_counters();
                let start = Instant::now();
                for _ in 0..iterations {
                    gpu.execute_sync(|encoder| {
                        kernels.encode_attention_prefill_cached_with_config(
                            encoder,
                            &buf_q,
                            &buf_k,
                            &buf_v,
                            &o,
                            true,
                            shape.n_tokens as u32,
                            shape.n_heads as u32,
                            shape.n_kv_heads as u32,
                            shape.head_dim as u32,
                            shape.base_seq_len as u32,
                            shape.sliding_window as u32,
                            config,
                        );
                        Ok(())
                    })?;
                }
                black_box(unsafe { o.as_slice::<f32>()[0] });
                let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
                let counters = gpu.perf_counters();
                Ok(AttentionPrefillCaseResult {
                    avg_ms: elapsed * 1000.0 / iterations.max(1) as f64,
                    effective_attn_gbps: (traffic_bytes * iterations) as f64 / elapsed / 1e9,
                    command_buffers: counters.command_buffers as f64 / iterations.max(1) as f64,
                    buffer_barriers: counters.buffer_barriers as f64 / iterations.max(1) as f64,
                    candidate_label: selection.label(),
                    stability_label: selection.stability.label(),
                })
            } else {
                let k_cache: Vec<f32> = (0..total_kv_elems)
                    .map(|i| ((i % 19) as f32 - 9.0) * 0.03125)
                    .collect();
                let v_cache: Vec<f32> = (0..total_kv_elems)
                    .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
                    .collect();
                let buf_k = MetalBuffer::from_slice(gpu.device(), &k_cache)?;
                let buf_v = MetalBuffer::from_slice(gpu.device(), &v_cache)?;
                gpu.execute_sync(|encoder| {
                    kernels.encode_attention_prefill_cached_with_config(
                        encoder,
                        &buf_q,
                        &buf_k,
                        &buf_v,
                        &o,
                        false,
                        shape.n_tokens as u32,
                        shape.n_heads as u32,
                        shape.n_kv_heads as u32,
                        shape.head_dim as u32,
                        shape.base_seq_len as u32,
                        shape.sliding_window as u32,
                        config,
                    );
                    Ok(())
                })?;
                gpu.reset_perf_counters();
                let start = Instant::now();
                for _ in 0..iterations {
                    gpu.execute_sync(|encoder| {
                        kernels.encode_attention_prefill_cached_with_config(
                            encoder,
                            &buf_q,
                            &buf_k,
                            &buf_v,
                            &o,
                            false,
                            shape.n_tokens as u32,
                            shape.n_heads as u32,
                            shape.n_kv_heads as u32,
                            shape.head_dim as u32,
                            shape.base_seq_len as u32,
                            shape.sliding_window as u32,
                            config,
                        );
                        Ok(())
                    })?;
                }
                black_box(unsafe { o.as_slice::<f32>()[0] });
                let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
                let counters = gpu.perf_counters();
                Ok(AttentionPrefillCaseResult {
                    avg_ms: elapsed * 1000.0 / iterations.max(1) as f64,
                    effective_attn_gbps: (traffic_bytes * iterations) as f64 / elapsed / 1e9,
                    command_buffers: counters.command_buffers as f64 / iterations.max(1) as f64,
                    buffer_barriers: counters.buffer_barriers as f64 / iterations.max(1) as f64,
                    candidate_label: selection.label(),
                    stability_label: selection.stability.label(),
                })
            }
        }
    }
}

fn quant_variant_candidate(
    kernels: &DequantKernels,
    quant: QuantCase,
    variant: QuantVariant,
    m: u32,
) -> (&'static str, &'static str) {
    match (quant, variant) {
        (QuantCase::Q4K, QuantVariant::Auto) => {
            let selection =
                kernels.q4_k_matvec_candidate_with_config(m, DequantDispatchConfig::default());
            (selection.label(), selection.stability.label())
        }
        (QuantCase::Q4K, QuantVariant::Nr2) => ("q4_k.nr2", "profile_preferred"),
        (QuantCase::Q5K, QuantVariant::Auto) => {
            let selection =
                kernels.q5_k_matvec_candidate_with_config(m, DequantDispatchConfig::default());
            (selection.label(), selection.stability.label())
        }
        (QuantCase::Q5K, QuantVariant::Nr2) => {
            unreachable!("unsupported Q5_K microbench variant")
        }
        (QuantCase::Q6K, QuantVariant::Auto) => {
            let selection =
                kernels.q6_k_matvec_candidate_with_config(m, DequantDispatchConfig::default());
            (selection.label(), selection.stability.label())
        }
        (QuantCase::Q6K, QuantVariant::Nr2) => ("q6_k.nr2", "profile_preferred"),
    }
}

fn attention_variant_config(variant: AttentionVariant) -> AttentionDispatchConfig {
    match variant {
        AttentionVariant::Auto => AttentionDispatchConfig::default(),
        AttentionVariant::Baseline => AttentionDispatchConfig::default()
            .with_decode_splitk_mode(KernelMode::Off)
            .with_decode_sdpa_default(false)
            .with_decode_hd128_n2_default(false),
        AttentionVariant::SplitK => AttentionDispatchConfig::default()
            .with_decode_splitk_mode(KernelMode::On)
            .with_decode_sdpa_default(false),
        AttentionVariant::Sdpa => AttentionDispatchConfig::default()
            .with_decode_splitk_mode(KernelMode::Off)
            .with_decode_sdpa_default(true),
        AttentionVariant::Hd128N2 => AttentionDispatchConfig::default()
            .with_decode_splitk_mode(KernelMode::Off)
            .with_decode_hd128_n2_default(true),
    }
}

fn prefill_variant_config(variant: PrefillVariant) -> AttentionDispatchConfig {
    match variant {
        PrefillVariant::Auto => AttentionDispatchConfig::default(),
        PrefillVariant::Baseline => AttentionDispatchConfig::default()
            .with_prefill_fa2_mode(KernelMode::Off)
            .with_prefill_fa2_hd128_mode(KernelMode::Off),
        PrefillVariant::Fa2 => {
            AttentionDispatchConfig::default().with_prefill_fa2_mode(KernelMode::On)
        }
        PrefillVariant::Fa2Hd128 => {
            AttentionDispatchConfig::default().with_prefill_fa2_hd128_mode(KernelMode::On)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_attention_variant(
    kernels: &AttentionKernels,
    gpu: &MetalDevice,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_cache: &MetalBuffer,
    o: &MetalBuffer,
    shape: AttentionShape,
    candidate: AttentionDecodeCandidate,
    config: AttentionDispatchConfig,
) -> anyhow::Result<()> {
    match candidate {
        AttentionDecodeCandidate::SplitKHd128 | AttentionDecodeCandidate::SplitKHd256 => kernels
            .attention_decode_splitk_with_config(
                gpu,
                q,
                k_cache,
                v_cache,
                o,
                shape.kv_f16,
                shape.n_heads as u32,
                shape.n_kv_heads as u32,
                shape.head_dim as u32,
                0,
                shape.attend_len as u32,
                config,
            ),
        _ => kernels.attention_decode_with_config(
            gpu,
            q,
            k_cache,
            v_cache,
            o,
            shape.kv_f16,
            shape.n_heads as u32,
            shape.n_kv_heads as u32,
            shape.head_dim as u32,
            0,
            shape.attend_len as u32,
            config,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_quant_variant(
    kernels: &DequantKernels,
    gpu: &MetalDevice,
    quant: QuantCase,
    variant: QuantVariant,
    buf_a: &MetalBuffer,
    buf_x: &MetalBuffer,
    y: &MetalBuffer,
    m: u32,
    k: u32,
) -> anyhow::Result<()> {
    match (quant, variant) {
        (QuantCase::Q4K, QuantVariant::Auto) => kernels.fused_matvec_q4_k_with_config(
            gpu,
            buf_a,
            buf_x,
            y,
            m,
            k,
            DequantDispatchConfig::default(),
        ),
        (QuantCase::Q5K, QuantVariant::Auto) => kernels.fused_matvec_q5_k_with_config(
            gpu,
            buf_a,
            buf_x,
            y,
            m,
            k,
            DequantDispatchConfig::default(),
        ),
        (QuantCase::Q4K, QuantVariant::Nr2) => {
            kernels.fused_matvec_q4_k_nr2(gpu, buf_a, buf_x, y, m, k)
        }
        (QuantCase::Q6K, QuantVariant::Auto) => kernels.fused_matvec_q6_k_with_config(
            gpu,
            buf_a,
            buf_x,
            y,
            m,
            k,
            DequantDispatchConfig::default(),
        ),
        (QuantCase::Q6K, QuantVariant::Nr2) => {
            kernels.fused_matvec_q6_k_nr2(gpu, buf_a, buf_x, y, m, k)
        }
        (QuantCase::Q5K, QuantVariant::Nr2) => {
            anyhow::bail!("unsupported Q5_K microbench variant")
        }
    }
}

fn make_q4k_matrix_bytes(m: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256), "Q4_K requires k multiple of 256");
    let blocks_per_row = k / 256;
    let mut block = [0u8; 144];
    let one = half::f16::from_f32(1.0).to_le_bytes();
    block[0] = one[0];
    block[1] = one[1];
    block[4] = 1;
    block[5] = 1;
    block[6] = 1;
    block[7] = 1;
    block[16..144].fill(0x33);

    let mut out = Vec::with_capacity(m * blocks_per_row * block.len());
    for _ in 0..(m * blocks_per_row) {
        out.extend_from_slice(&block);
    }
    out
}

fn make_q5k_matrix_bytes(m: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256), "Q5_K requires k multiple of 256");
    let blocks_per_row = k / 256;
    let mut block = [0u8; 176];
    let one = half::f16::from_f32(1.0).to_le_bytes();
    block[0] = one[0];
    block[1] = one[1];
    block[4] = 1;
    block[5] = 1;
    block[6] = 1;
    block[7] = 1;
    block[48..176].fill(0x33);

    let mut out = Vec::with_capacity(m * blocks_per_row * block.len());
    for _ in 0..(m * blocks_per_row) {
        out.extend_from_slice(&block);
    }
    out
}

fn make_q6k_matrix_bytes(m: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256), "Q6_K requires k multiple of 256");
    let blocks_per_row = k / 256;
    let mut block = [0u8; 210];
    let one = half::f16::from_f32(1.0).to_le_bytes();
    block[208] = one[0];
    block[209] = one[1];
    block[192..208].fill(1);
    block[0..128].fill(0x00);
    block[128..192].fill(0xAA);

    let mut out = Vec::with_capacity(m * blocks_per_row * block.len());
    for _ in 0..(m * blocks_per_row) {
        out.extend_from_slice(&block);
    }
    out
}

fn quant_case_name(quant: QuantCase) -> &'static str {
    match quant {
        QuantCase::Q4K => "q4_k",
        QuantCase::Q5K => "q5_k",
        QuantCase::Q6K => "q6_k",
    }
}

fn quant_variant_name(variant: QuantVariant) -> &'static str {
    match variant {
        QuantVariant::Auto => "auto",
        QuantVariant::Nr2 => "nr2",
    }
}

fn attention_shape_name(shape: AttentionShape) -> &'static str {
    match (shape.head_dim, shape.kv_f16) {
        (128, true) => "attn_hd128_f16kv",
        (256, true) => "attn_hd256_f16kv",
        (128, false) => "attn_hd128_f32kv",
        (256, false) => "attn_hd256_f32kv",
        _ => "attn_generic",
    }
}

fn attention_variant_name(variant: AttentionVariant) -> &'static str {
    match variant {
        AttentionVariant::Auto => "auto",
        AttentionVariant::Baseline => "baseline",
        AttentionVariant::SplitK => "splitk",
        AttentionVariant::Sdpa => "sdpa",
        AttentionVariant::Hd128N2 => "hd128_n2",
    }
}

fn prefill_shape_name(shape: PrefillShape) -> &'static str {
    match (shape.mode, shape.head_dim, shape.kv_f16) {
        (PrefillModeCase::Local, 128, _) => "prefill_local_hd128",
        (PrefillModeCase::Local, 256, _) => "prefill_local_hd256",
        (PrefillModeCase::Cached, 256, true) => "prefill_cached_hd256_f16kv",
        (PrefillModeCase::Cached, 256, false) => "prefill_cached_hd256_f32kv",
        (PrefillModeCase::Cached, 128, true) => "prefill_cached_hd128_f16kv",
        (PrefillModeCase::Cached, 128, false) => "prefill_cached_hd128_f32kv",
        (PrefillModeCase::Local, _, _) => "prefill_local_generic",
        (PrefillModeCase::Cached, _, _) => "prefill_cached_generic",
    }
}

fn prefill_variant_name(variant: PrefillVariant) -> &'static str {
    match variant {
        PrefillVariant::Auto => "auto",
        PrefillVariant::Baseline => "baseline",
        PrefillVariant::Fa2 => "fa2",
        PrefillVariant::Fa2Hd128 => "fa2_hd128",
    }
}

fn apply_microbench_recommendations(profile: &mut KernelProfile, suite: &MicrobenchSuiteResult) {
    apply_decode_matvec_recommendations(profile, suite);
    apply_decode_attention_recommendations(profile, suite);
    apply_prefill_attention_recommendations(profile, suite);
}

fn apply_decode_matvec_recommendations(profile: &mut KernelProfile, suite: &MicrobenchSuiteResult) {
    if let Some(q4_variant) = recommended_matvec_variant(suite, "q4_k") {
        apply_decode_matvec_recommendation(profile, "q4_k", &q4_variant);
    }
    if let Some(q6_variant) = recommended_matvec_variant(suite, "q6_k") {
        apply_decode_matvec_recommendation(profile, "q6_k", &q6_variant);
    }
}

fn apply_decode_matvec_recommendation(profile: &mut KernelProfile, quant: &str, variant: &str) {
    let params = profile.decode_matvec.entry(quant.to_string()).or_default();
    let (threadgroup_size, rows_per_simdgroup) = match (quant, variant) {
        ("q4_k", "nr2") => (64, 2),
        ("q4_k", "tg256") => (256, 1),
        ("q4_k", _) => (128, 1),
        ("q6_k", "nr2") => (64, 2),
        ("q6_k", _) => (128, 1),
        _ => return,
    };
    params.threadgroup_size = threadgroup_size;
    params.rows_per_simdgroup = rows_per_simdgroup;
}

fn apply_decode_attention_recommendations(
    profile: &mut KernelProfile,
    suite: &MicrobenchSuiteResult,
) {
    // Keep hd128_n2 as a manual/profile-authored choice for now. A single
    // decode-attention microbench sweep was not predictive enough in
    // end-to-end Qwen3 validation.
    let hd256_ctx256 =
        recommended_shape_variant(suite, "decode_attention", "attn_hd256_f16kv", 256);
    let hd256_ctx1024 =
        recommended_shape_variant(suite, "decode_attention", "attn_hd256_f16kv", 1024);

    if hd256_ctx256.as_ref().is_some_and(|r| r.variant == "sdpa")
        && hd256_ctx1024.as_ref().is_some_and(|r| r.variant == "sdpa")
    {
        profile.attention_decode.sdpa_default = Some(true);
    }

    profile.attention_decode.splitk_threshold = match (hd256_ctx256, hd256_ctx1024) {
        (Some(short), Some(long)) if short.variant == "splitk" && long.variant == "splitk" => 256,
        (_, Some(long)) if long.variant == "splitk" => 1024,
        _ => profile.attention_decode.splitk_threshold,
    };
}

fn has_recommendations_for(suite: &MicrobenchSuiteResult, domain: &str, quant: &str) -> bool {
    suite
        .recommendations
        .as_ref()
        .is_some_and(|recs| recs.iter().any(|r| r.domain == domain && r.quant == quant))
}

fn apply_prefill_attention_recommendations(
    profile: &mut KernelProfile,
    suite: &MicrobenchSuiteResult,
) {
    let local_hd128 =
        recommended_shape_variant(suite, "prefill_attention", "prefill_local_hd128", 512);
    let cached_hd256 = recommended_shape_variant(
        suite,
        "prefill_attention",
        "prefill_cached_hd256_f16kv",
        512,
    );

    // Only override when the suite actually evaluated this domain+quant.
    // When no data exists (e.g., CPU-only suite), preserve the profile default.
    if has_recommendations_for(suite, "prefill_attention", "prefill_local_hd128") {
        profile.attention_prefill.fa2_hd128_mode = if local_hd128
            .as_ref()
            .is_some_and(|r| r.variant == "fa2_hd128")
        {
            ProfileKernelMode::On
        } else {
            ProfileKernelMode::Off
        };
    }

    if has_recommendations_for(suite, "prefill_attention", "prefill_cached_hd256_f16kv") {
        profile.attention_prefill.fa2_mode =
            if cached_hd256.as_ref().is_some_and(|r| r.variant == "fa2") {
                ProfileKernelMode::On
            } else {
                ProfileKernelMode::Off
            };
    }
}

fn dominant_variant(suite: &MicrobenchSuiteResult, domain: &str, quant: &str) -> Option<String> {
    let recommendations = suite.recommendations.as_ref()?;
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for recommendation in recommendations
        .iter()
        .filter(|r| r.domain == domain && r.quant == quant)
    {
        *counts.entry(recommendation.variant.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(variant, _)| variant.to_string())
}

fn recommended_matvec_variant(suite: &MicrobenchSuiteResult, quant: &str) -> Option<String> {
    let variant = dominant_variant(suite, "decode_matvec", quant)?;
    let min_speedup = recommendation_speedup_threshold(suite);
    let required_wins = recommendation_required_wins(suite);
    let wins = suite
        .recommendations
        .as_ref()?
        .iter()
        .filter(|r| {
            r.domain == "decode_matvec"
                && r.quant == quant
                && r.variant == variant
                && r.speedup_vs_auto >= min_speedup
        })
        .count();

    (wins >= required_wins).then_some(variant)
}

fn recommended_shape_variant(
    suite: &MicrobenchSuiteResult,
    domain: &str,
    quant: &str,
    k: usize,
) -> Option<MicrobenchRecommendation> {
    let recommended = recommendation(suite, domain, quant, k)?;
    let observed = suite
        .recommendations
        .as_ref()?
        .iter()
        .filter(|r| r.domain == domain && r.quant == quant && r.k == k)
        .collect::<Vec<_>>();
    let wins = recommendation_stats(
        &observed,
        &recommended.variant,
        recommendation_speedup_threshold(suite),
    )
    .0;

    (wins >= recommendation_required_wins(suite)).then_some(recommended)
}

fn recommendation_speedup_threshold(suite: &MicrobenchSuiteResult) -> f64 {
    if suite.suite_runs <= 1 { 1.08 } else { 1.05 }
}

fn recommendation_required_wins(suite: &MicrobenchSuiteResult) -> usize {
    suite.suite_runs.max(2)
}

fn suggested_profile_evidence_for_suite(
    suite: &MicrobenchSuiteResult,
) -> Vec<MicrobenchProfileEvidence> {
    let min_speedup = recommendation_speedup_threshold(suite);
    let required_wins = recommendation_required_wins(suite);
    vec![
        matvec_evidence(suite, "q4_k", "decode_matvec.q4_k"),
        matvec_evidence(suite, "q6_k", "decode_matvec.q6_k"),
        decode_attention_sdpa_evidence(suite),
        decode_attention_splitk_threshold_evidence(suite),
        prefill_variant_evidence(
            suite,
            "prefill_local_hd128",
            "fa2_hd128",
            "prefill_attention.local_hd128.fa2_hd128_mode",
            recommendation_required_wins(suite),
        ),
        prefill_variant_evidence(
            suite,
            "prefill_cached_hd256_f16kv",
            "fa2",
            "prefill_attention.cached_hd256_f16kv.fa2_mode",
            recommendation_required_wins(suite),
        ),
        MicrobenchProfileEvidence {
            rule: "evidence_thresholds".to_string(),
            promoted: false,
            suite_runs: suite.suite_runs,
            required_wins,
            observed_wins: 0,
            min_speedup,
            avg_speedup: None,
            variant: None,
        },
    ]
}

fn matvec_evidence(
    suite: &MicrobenchSuiteResult,
    quant: &str,
    rule: &str,
) -> MicrobenchProfileEvidence {
    let recommended = recommended_matvec_variant(suite, quant);
    let observed = suite
        .recommendations
        .as_ref()
        .into_iter()
        .flatten()
        .filter(|r| r.domain == "decode_matvec" && r.quant == quant)
        .collect::<Vec<_>>();
    let variant = dominant_variant(suite, "decode_matvec", quant);
    let stats = recommendation_stats(
        &observed,
        variant.as_deref().unwrap_or(""),
        recommendation_speedup_threshold(suite),
    );

    MicrobenchProfileEvidence {
        rule: rule.to_string(),
        promoted: recommended.is_some(),
        suite_runs: suite.suite_runs,
        required_wins: recommendation_required_wins(suite),
        observed_wins: stats.0,
        min_speedup: recommendation_speedup_threshold(suite),
        avg_speedup: stats.1,
        variant,
    }
}

fn decode_attention_sdpa_evidence(suite: &MicrobenchSuiteResult) -> MicrobenchProfileEvidence {
    let short = shape_variant_stats(suite, "decode_attention", "attn_hd256_f16kv", 256, "sdpa");
    let long = shape_variant_stats(suite, "decode_attention", "attn_hd256_f16kv", 1024, "sdpa");
    let required_wins = recommendation_required_wins(suite);

    MicrobenchProfileEvidence {
        rule: "decode_attention.hd256.sdpa_default".to_string(),
        promoted: short.0 >= required_wins && long.0 >= required_wins,
        suite_runs: suite.suite_runs,
        required_wins,
        observed_wins: short.0.min(long.0),
        min_speedup: recommendation_speedup_threshold(suite),
        avg_speedup: match (short.1, long.1) {
            (Some(short_avg), Some(long_avg)) => Some(short_avg.min(long_avg)),
            _ => None,
        },
        variant: Some("sdpa".to_string()),
    }
}

fn decode_attention_splitk_threshold_evidence(
    suite: &MicrobenchSuiteResult,
) -> MicrobenchProfileEvidence {
    let required_wins = recommendation_required_wins(suite);
    let short = shape_variant_stats(suite, "decode_attention", "attn_hd256_f16kv", 256, "splitk");
    let long = shape_variant_stats(
        suite,
        "decode_attention",
        "attn_hd256_f16kv",
        1024,
        "splitk",
    );
    let (promoted, observed_wins, avg_speedup, variant) =
        if short.0 >= required_wins && long.0 >= required_wins {
            (
                true,
                short.0.min(long.0),
                match (short.1, long.1) {
                    (Some(short_avg), Some(long_avg)) => Some(short_avg.min(long_avg)),
                    _ => None,
                },
                Some("splitk@256".to_string()),
            )
        } else if long.0 >= required_wins {
            (true, long.0, long.1, Some("splitk@1024".to_string()))
        } else {
            (
                false,
                short.0.max(long.0),
                long.1.or(short.1),
                Some("splitk".to_string()),
            )
        };

    MicrobenchProfileEvidence {
        rule: "decode_attention.hd256.splitk_threshold".to_string(),
        promoted,
        suite_runs: suite.suite_runs,
        required_wins,
        observed_wins,
        min_speedup: recommendation_speedup_threshold(suite),
        avg_speedup,
        variant,
    }
}

fn prefill_variant_evidence(
    suite: &MicrobenchSuiteResult,
    quant: &str,
    variant: &str,
    rule: &str,
    required_wins: usize,
) -> MicrobenchProfileEvidence {
    let observed = suite
        .recommendations
        .as_ref()
        .into_iter()
        .flatten()
        .filter(|r| r.domain == "prefill_attention" && r.quant == quant)
        .collect::<Vec<_>>();
    let stats = recommendation_stats(&observed, variant, recommendation_speedup_threshold(suite));

    MicrobenchProfileEvidence {
        rule: rule.to_string(),
        promoted: stats.0 >= required_wins,
        suite_runs: suite.suite_runs,
        required_wins,
        observed_wins: stats.0,
        min_speedup: recommendation_speedup_threshold(suite),
        avg_speedup: stats.1,
        variant: Some(variant.to_string()),
    }
}

fn shape_variant_stats(
    suite: &MicrobenchSuiteResult,
    domain: &str,
    quant: &str,
    k: usize,
    variant: &str,
) -> (usize, Option<f64>) {
    let observed = suite
        .recommendations
        .as_ref()
        .into_iter()
        .flatten()
        .filter(|r| r.domain == domain && r.quant == quant && r.k == k)
        .collect::<Vec<_>>();
    recommendation_stats(&observed, variant, recommendation_speedup_threshold(suite))
}

fn recommendation_stats(
    recommendations: &[&MicrobenchRecommendation],
    variant: &str,
    min_speedup: f64,
) -> (usize, Option<f64>) {
    let winners = recommendations
        .iter()
        .filter(|r| r.variant == variant && r.speedup_vs_auto >= min_speedup)
        .copied()
        .collect::<Vec<_>>();
    let observed_wins = winners.len();
    let avg_speedup = (!winners.is_empty())
        .then(|| winners.iter().map(|r| r.speedup_vs_auto).sum::<f64>() / winners.len() as f64);
    (observed_wins, avg_speedup)
}

fn recommendation(
    suite: &MicrobenchSuiteResult,
    domain: &str,
    quant: &str,
    k: usize,
) -> Option<MicrobenchRecommendation> {
    let recommendations = suite.recommendations.as_ref()?;
    let matching = recommendations
        .iter()
        .filter(|r| r.domain == domain && r.quant == quant && r.k == k)
        .cloned()
        .collect::<Vec<_>>();
    aggregate_recommendations(&matching).into_iter().next()
}

fn aggregated_recommendations(
    recommendations: &[MicrobenchRecommendation],
) -> Vec<MicrobenchRecommendation> {
    let mut groups: BTreeMap<(String, String, usize, usize), Vec<MicrobenchRecommendation>> =
        BTreeMap::new();
    let mut order = Vec::new();
    for recommendation in recommendations {
        let key = (
            recommendation.domain.clone(),
            recommendation.quant.clone(),
            recommendation.m,
            recommendation.k,
        );
        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(recommendation.clone());
    }

    order
        .into_iter()
        .filter_map(|key| groups.remove(&key))
        .filter_map(|group| aggregate_recommendations(&group).into_iter().next())
        .collect()
}

fn aggregate_recommendations(
    recommendations: &[MicrobenchRecommendation],
) -> Vec<MicrobenchRecommendation> {
    let mut by_variant: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
    let first = match recommendations.first() {
        Some(first) => first,
        None => return Vec::new(),
    };

    for recommendation in recommendations {
        let entry = by_variant
            .entry(recommendation.variant.clone())
            .or_insert((0, 0.0, 0.0));
        entry.0 += 1;
        entry.1 += recommendation.best_ms;
        entry.2 += recommendation.speedup_vs_auto;
    }

    let (variant, (count, best_ms_sum, speedup_sum)) = by_variant
        .into_iter()
        .max_by(
            |(lhs_variant, (lhs_count, _, lhs_speedup)),
             (rhs_variant, (rhs_count, _, rhs_speedup))| {
                lhs_count
                    .cmp(rhs_count)
                    .then_with(|| lhs_speedup.total_cmp(rhs_speedup))
                    .then_with(|| rhs_variant.cmp(lhs_variant))
            },
        )
        .expect("non-empty recommendation group");

    vec![MicrobenchRecommendation {
        domain: first.domain.clone(),
        quant: first.quant.clone(),
        variant,
        m: first.m,
        k: first.k,
        best_ms: best_ms_sum / count as f64,
        speedup_vs_auto: speedup_sum / count as f64,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggested_kernel_profile_skips_hd128_n2_auto_promotion() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 1,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 8192,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.3,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.5,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q6_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 1024,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.2,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q6_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.2,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd128_f16kv".to_string(),
                        variant: "hd128_n2".to_string(),
                        m: 128,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.4,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.1,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        let q4 = profile.decode_matvec.get("q4_k").unwrap();
        let q6 = profile.decode_matvec.get("q6_k").unwrap();
        assert_eq!(q4.threadgroup_size, 64);
        assert_eq!(q4.rows_per_simdgroup, 2);
        assert_eq!(q6.threadgroup_size, 64);
        assert_eq!(q6.rows_per_simdgroup, 2);
        assert_eq!(profile.attention_decode.hd128_n2_default, None);
        assert_eq!(profile.attention_decode.sdpa_default, None);
    }

    #[test]
    fn test_suggested_kernel_profile_promotes_sdpa_only_after_consistent_hd256_wins() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.11,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.11,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.25,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.25,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_decode.sdpa_default, Some(true));
    }

    #[test]
    fn test_suggested_kernel_profile_skips_sdpa_promotion_for_weak_single_run_margin() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 1,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.06,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.07,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.07,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.08,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_decode.sdpa_default, None);
    }

    #[test]
    fn test_suggested_kernel_profile_allows_sdpa_promotion_after_multi_run_aggregation() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.06,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.07,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.07,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.08,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_decode.sdpa_default, Some(true));
    }

    #[test]
    fn test_suggested_kernel_profile_requires_repeated_shape_wins_for_decode_attention() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.10,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.12,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_decode.sdpa_default, None);
        assert_eq!(profile.attention_decode.splitk_threshold, 512);
    }

    #[test]
    fn test_suggested_kernel_profile_skips_matvec_promotion_without_repeated_margin() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 1,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.04,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 8192,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.02,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q6_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.20,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        // With suite_runs=1, speedup threshold is 1.08 and required_wins is 2.
        // q4_k has speedup 1.04 and 1.02 (both below 1.08): no promotion.
        // q6_k has speedup 1.20 (above 1.08, but only 1 win, needs 2): no promotion.
        assert!(
            !profile.decode_matvec.contains_key("q4_k"),
            "q4_k should not be promoted: speedups below threshold",
        );
        assert!(
            !profile.decode_matvec.contains_key("q6_k"),
            "q6_k should not be promoted: insufficient wins",
        );
    }

    #[test]
    fn test_suggested_kernel_profile_promotes_matvec_after_repeated_margin_wins() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 1,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.20,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 8192,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.15,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q6_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 1024,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.10,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q6_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.08,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        let q4 = profile.decode_matvec.get("q4_k").unwrap();
        let q6 = profile.decode_matvec.get("q6_k").unwrap();
        assert_eq!(q4.threadgroup_size, 64);
        assert_eq!(q4.rows_per_simdgroup, 2);
        assert_eq!(q6.threadgroup_size, 64);
        assert_eq!(q6.rows_per_simdgroup, 2);
    }

    #[test]
    fn test_suggested_kernel_profile_requires_wins_to_scale_with_suite_runs() {
        // With suite_runs=3, required_wins=3 but only 2 recommendations exist.
        // Even though both exceed the speedup threshold, promotion should be skipped.
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 3,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.10,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 8192,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.09,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert!(
            !profile.decode_matvec.contains_key("q4_k"),
            "q4_k should not be promoted: only 2 wins but required_wins=3 for suite_runs=3",
        );
    }

    #[test]
    fn test_suggested_kernel_profile_raises_splitk_threshold_for_long_context_only() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "baseline".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.0,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "baseline".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.0,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "splitk".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.2,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "splitk".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.2,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_decode.splitk_threshold, 1024);
    }

    #[test]
    fn test_suggested_kernel_profile_requires_repeated_long_context_wins_for_splitk_threshold() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![MicrobenchRecommendation {
                    domain: "decode_attention".to_string(),
                    quant: "attn_hd256_f16kv".to_string(),
                    variant: "splitk".to_string(),
                    m: 256,
                    k: 1024,
                    best_ms: 1.0,
                    speedup_vs_auto: 1.12,
                }]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_decode.splitk_threshold, 512);
    }

    #[test]
    fn test_suggested_kernel_profile_promotes_splitk_threshold_after_repeated_long_context_wins() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "splitk".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.11,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "splitk".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.10,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_decode.splitk_threshold, 1024);

        let evidence = report.suggested_kernel_profile_evidence();
        let splitk = evidence
            .iter()
            .find(|item| item.rule == "decode_attention.hd256.splitk_threshold")
            .unwrap();
        assert!(splitk.promoted);
        assert_eq!(splitk.observed_wins, 2);
        assert_eq!(splitk.variant.as_deref(), Some("splitk@1024"));
    }

    #[test]
    fn test_suggested_kernel_profile_promotes_splitk_threshold_256_after_repeated_wins() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "splitk".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.10,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "splitk".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.09,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "splitk".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.11,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "splitk".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.10,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_decode.splitk_threshold, 256);

        let evidence = report.suggested_kernel_profile_evidence();
        let splitk = evidence
            .iter()
            .find(|item| item.rule == "decode_attention.hd256.splitk_threshold")
            .unwrap();
        assert!(splitk.promoted);
        assert_eq!(splitk.observed_wins, 2);
        assert_eq!(splitk.variant.as_deref(), Some("splitk@256"));
    }

    #[test]
    fn test_aggregate_suite_runs_keeps_raw_recommendations_and_averages_measurements() {
        let aggregated = aggregate_suite_runs(vec![
            MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 2,
                suite_runs: 1,
                device: Some("Test GPU".to_string()),
                recommendations: Some(vec![MicrobenchRecommendation {
                    domain: "decode_attention".to_string(),
                    quant: "attn_hd256_f16kv".to_string(),
                    variant: "sdpa".to_string(),
                    m: 256,
                    k: 1024,
                    best_ms: 1.0,
                    speedup_vs_auto: 1.10,
                }]),
                measurements: vec![MicrobenchMeasurement {
                    name: "gpu.empty_execute_sync".to_string(),
                    unit: "ms".to_string(),
                    value: 1.0,
                    note: Some("note".to_string()),
                }],
            },
            MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 2,
                suite_runs: 1,
                device: Some("Test GPU".to_string()),
                recommendations: Some(vec![MicrobenchRecommendation {
                    domain: "decode_attention".to_string(),
                    quant: "attn_hd256_f16kv".to_string(),
                    variant: "sdpa".to_string(),
                    m: 256,
                    k: 1024,
                    best_ms: 0.8,
                    speedup_vs_auto: 1.20,
                }]),
                measurements: vec![MicrobenchMeasurement {
                    name: "gpu.empty_execute_sync".to_string(),
                    unit: "ms".to_string(),
                    value: 3.0,
                    note: Some("note".to_string()),
                }],
            },
        ]);

        assert_eq!(aggregated.suite_runs, 2);
        assert_eq!(aggregated.measurements.len(), 1);
        assert!((aggregated.measurements[0].value - 2.0).abs() < 1e-6);
        assert_eq!(aggregated.recommendations.as_ref().map(Vec::len), Some(2));

        let aggregated_recommendation =
            recommendation(&aggregated, "decode_attention", "attn_hd256_f16kv", 1024).unwrap();
        assert_eq!(aggregated_recommendation.variant, "sdpa");
        assert!((aggregated_recommendation.best_ms - 0.9).abs() < 1e-6);
        assert!((aggregated_recommendation.speedup_vs_auto - 1.15).abs() < 1e-6);
    }

    #[test]
    fn test_suggested_kernel_profile_evidence_reports_thresholds_and_wins() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.07,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 8192,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.09,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.06,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 256,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.08,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.08,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_attention".to_string(),
                        quant: "attn_hd256_f16kv".to_string(),
                        variant: "sdpa".to_string(),
                        m: 256,
                        k: 1024,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.09,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let evidence = report.suggested_kernel_profile_evidence();
        let q4 = evidence
            .iter()
            .find(|item| item.rule == "decode_matvec.q4_k")
            .unwrap();
        assert!(q4.promoted);
        assert_eq!(q4.required_wins, 2);
        assert_eq!(q4.observed_wins, 2);
        assert_eq!(q4.variant.as_deref(), Some("nr2"));
        assert_eq!(q4.min_speedup, 1.05);

        let sdpa = evidence
            .iter()
            .find(|item| item.rule == "decode_attention.hd256.sdpa_default")
            .unwrap();
        assert!(sdpa.promoted);
        assert_eq!(sdpa.observed_wins, 2);
        assert_eq!(sdpa.variant.as_deref(), Some("sdpa"));
    }

    #[test]
    fn test_suggested_kernel_profile_promotes_prefill_fa2_modes_after_repeated_wins() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "prefill_attention".to_string(),
                        quant: "prefill_local_hd128".to_string(),
                        variant: "fa2_hd128".to_string(),
                        m: 128,
                        k: 512,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.06,
                    },
                    MicrobenchRecommendation {
                        domain: "prefill_attention".to_string(),
                        quant: "prefill_local_hd128".to_string(),
                        variant: "fa2_hd128".to_string(),
                        m: 128,
                        k: 512,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.08,
                    },
                    MicrobenchRecommendation {
                        domain: "prefill_attention".to_string(),
                        quant: "prefill_cached_hd256_f16kv".to_string(),
                        variant: "fa2".to_string(),
                        m: 256,
                        k: 512,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.07,
                    },
                    MicrobenchRecommendation {
                        domain: "prefill_attention".to_string(),
                        quant: "prefill_cached_hd256_f16kv".to_string(),
                        variant: "fa2".to_string(),
                        m: 256,
                        k: 512,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.09,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::On);
        assert_eq!(
            profile.attention_prefill.fa2_hd128_mode,
            ProfileKernelMode::On
        );

        let evidence = report.suggested_kernel_profile_evidence();
        assert!(evidence.iter().any(|item| {
            item.rule == "prefill_attention.local_hd128.fa2_hd128_mode" && item.promoted
        }));
        assert!(evidence.iter().any(|item| {
            item.rule == "prefill_attention.cached_hd256_f16kv.fa2_mode" && item.promoted
        }));
    }

    #[test]
    fn test_suggested_kernel_profile_skips_prefill_fa2_promotion_for_weak_single_run_margin() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 1,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "prefill_attention".to_string(),
                        quant: "prefill_local_hd128".to_string(),
                        variant: "fa2_hd128".to_string(),
                        m: 128,
                        k: 512,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.06,
                    },
                    MicrobenchRecommendation {
                        domain: "prefill_attention".to_string(),
                        quant: "prefill_cached_hd256_f16kv".to_string(),
                        variant: "fa2".to_string(),
                        m: 256,
                        k: 512,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.07,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Off);
        assert_eq!(
            profile.attention_prefill.fa2_hd128_mode,
            ProfileKernelMode::Off
        );

        let evidence = report.suggested_kernel_profile_evidence();
        assert!(evidence.iter().any(|item| {
            item.rule == "prefill_attention.local_hd128.fa2_hd128_mode" && !item.promoted
        }));
        assert!(evidence.iter().any(|item| {
            item.rule == "prefill_attention.cached_hd256_f16kv.fa2_mode" && !item.promoted
        }));
    }

    #[test]
    fn test_suggested_kernel_profile_requires_repeated_shape_wins_for_prefill_attention() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "prefill_attention".to_string(),
                        quant: "prefill_local_hd128".to_string(),
                        variant: "fa2_hd128".to_string(),
                        m: 128,
                        k: 512,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.10,
                    },
                    MicrobenchRecommendation {
                        domain: "prefill_attention".to_string(),
                        quant: "prefill_cached_hd256_f16kv".to_string(),
                        variant: "fa2".to_string(),
                        m: 256,
                        k: 512,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.10,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        assert_eq!(profile.attention_prefill.fa2_mode, ProfileKernelMode::Off);
        assert_eq!(
            profile.attention_prefill.fa2_hd128_mode,
            ProfileKernelMode::Off
        );
    }

    #[test]
    fn test_repeats_for_target_bytes_never_zero() {
        assert_eq!(repeats_for_target_bytes(4096, 1024), 1);
    }

    #[test]
    fn test_cpu_suite_has_measurements() {
        let result = run_cpu_suite(1);
        assert!(!result.measurements.is_empty());
    }

    #[test]
    fn test_q4k_matrix_byte_size() {
        let bytes = make_q4k_matrix_bytes(4, 512);
        assert_eq!(bytes.len(), 4 * 2 * 144);
    }

    #[test]
    fn test_q5k_matrix_byte_size() {
        let bytes = make_q5k_matrix_bytes(4, 512);
        assert_eq!(bytes.len(), 4 * 2 * 176);
    }

    #[test]
    fn test_q6k_matrix_byte_size() {
        let bytes = make_q6k_matrix_bytes(4, 512);
        assert_eq!(bytes.len(), 4 * 2 * 210);
    }

    #[test]
    fn test_q5k_quant_case_name() {
        assert_eq!(quant_case_name(QuantCase::Q5K), "q5_k");
    }

    #[test]
    fn test_single_variant_quant_sweep_does_not_emit_recommendation() {
        assert!(!should_emit_quant_recommendation(QuantCase::Q4K, 1));
        assert!(should_emit_quant_recommendation(QuantCase::Q4K, 2));
    }

    #[test]
    fn test_quant_variant_support_matches_current_q5k_and_q6k_contract() {
        assert!(quant_variant_supported(QuantCase::Q5K, QuantVariant::Auto));
        assert!(!quant_variant_supported(QuantCase::Q5K, QuantVariant::Nr2));

        assert!(quant_variant_supported(QuantCase::Q6K, QuantVariant::Auto));
        assert!(quant_variant_supported(QuantCase::Q6K, QuantVariant::Nr2));
    }

    #[test]
    fn test_quant_case_variants_match_supported_contract() {
        for &variant in quant_case_variants(QuantCase::Q4K) {
            assert!(quant_variant_supported(QuantCase::Q4K, variant));
        }
        for &variant in quant_case_variants(QuantCase::Q5K) {
            assert!(quant_variant_supported(QuantCase::Q5K, variant));
        }
        for &variant in quant_case_variants(QuantCase::Q6K) {
            assert!(quant_variant_supported(QuantCase::Q6K, variant));
        }

        assert_eq!(quant_case_variants(QuantCase::Q5K), &[QuantVariant::Auto]);
        assert_eq!(
            quant_case_variants(QuantCase::Q6K),
            &[QuantVariant::Auto, QuantVariant::Nr2]
        );
    }

    #[test]
    fn test_q5k_best_summary_is_suppressed_for_single_variant_sweeps() {
        assert!(!should_emit_quant_recommendation(QuantCase::Q5K, 1));
    }

    #[test]
    fn test_q5k_stays_observational_even_if_multiple_variants_exist() {
        assert!(!should_emit_quant_recommendation(QuantCase::Q5K, 2));
    }

    #[test]
    fn test_q5k_is_marked_observational_only() {
        assert!(quant_case_is_observational_only(QuantCase::Q5K));
        assert!(!quant_case_is_observational_only(QuantCase::Q4K));
        assert!(!quant_case_is_observational_only(QuantCase::Q6K));
        assert!(!quant_case_supports_recommendation(QuantCase::Q5K));
        assert!(quant_case_supports_recommendation(QuantCase::Q4K));
    }

    #[test]
    fn test_q5k_measurement_note_marks_baseline_only() {
        let note = quant_measurement_note(QuantCase::Q5K, "q5_k.base", "stable");
        assert!(note.contains("candidate=q5_k.base"));
        assert!(note.contains("tier=stable"));
        assert!(note.contains("observational_only=baseline"));
    }

    #[test]
    fn test_q4k_measurement_note_stays_normal() {
        let note = quant_measurement_note(QuantCase::Q4K, "q4_k.nr2", "profile_preferred");
        assert!(note.contains("candidate=q4_k.nr2"));
        assert!(note.contains("tier=profile_preferred"));
        assert!(!note.contains("observational_only=baseline"));
    }

    #[test]
    fn test_suite_observational_quants_detects_q5k_measurements() {
        let suite = MicrobenchSuiteResult {
            suite: "gpu".to_string(),
            iterations: 1,
            suite_runs: 1,
            device: None,
            recommendations: None,
            measurements: vec![MicrobenchMeasurement {
                name: "gpu.decode_matvec.q5_k.auto.1024x4096".to_string(),
                unit: "ms".to_string(),
                value: 1.0,
                note: None,
            }],
        };
        assert_eq!(suite_observational_quants(&suite), vec!["q5_k"]);
    }

    #[test]
    fn test_suite_observational_quants_detects_q5k_prefill_measurements() {
        let suite = MicrobenchSuiteResult {
            suite: "gpu".to_string(),
            iterations: 1,
            suite_runs: 1,
            device: None,
            recommendations: None,
            measurements: vec![MicrobenchMeasurement {
                name: "gpu.prefill_matmul.q5_k.base.proj.tokens128.4096x4096".to_string(),
                unit: "ms".to_string(),
                value: 1.0,
                note: None,
            }],
        };
        assert_eq!(suite_observational_quants(&suite), vec!["q5_k_prefill"]);
    }

    #[test]
    fn test_q5k_prefill_measurement_note_marks_route_study_observational_only() {
        let note = q5k_prefill_batch_measurement_note(Q5KPrefillBatchVariant::Base);
        assert!(note.contains("candidate=q5_k.batch_base"));
        assert!(note.contains("observational_only=route_study"));
    }

    #[test]
    fn test_q5k_prefill_small_measurement_note_marks_small_candidate() {
        let note = q5k_prefill_batch_measurement_note(Q5KPrefillBatchVariant::Small);
        assert!(note.contains("candidate=q5_k.batch_small"));
        assert!(note.contains("observational_only=route_study"));
    }

    #[test]
    fn test_q5k_prefill_f16in_measurement_note_marks_f16in_candidate() {
        let note = q5k_prefill_batch_measurement_note(Q5KPrefillBatchVariant::F16In);
        assert!(note.contains("candidate=q5_k.batch_f16in"));
        assert!(note.contains("observational_only=route_study"));
    }

    #[test]
    fn test_q5k_observational_results_do_not_affect_suggested_profile_or_evidence() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 2,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q5_k".to_string(),
                        variant: "auto".to_string(),
                        m: 1024,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.0,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q5_k".to_string(),
                        variant: "auto".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.0,
                    },
                ]),
                measurements: Vec::new(),
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        // Q5_K is now in the default profile (G5 parity fix), so it will be
        // present.  The invariant is that observational Q5_K decode results
        // must NOT produce Q5_K-specific evidence or mutate the default entry.
        let q5_default = ax_engine_metal::profile::MatvecParams::default();
        if let Some(q5) = profile.decode_matvec.get("q5_k") {
            assert_eq!(q5.rows_per_simdgroup, q5_default.rows_per_simdgroup);
        }

        let evidence = report.suggested_kernel_profile_evidence();
        assert!(
            !evidence
                .iter()
                .any(|item| item.rule == "decode_matvec.q5_k")
        );
    }

    #[test]
    fn test_q5k_prefill_observational_measurements_do_not_affect_suggested_profile_or_evidence() {
        let report = MicrobenchReport {
            suites: vec![MicrobenchSuiteResult {
                suite: "gpu".to_string(),
                iterations: 1,
                suite_runs: 1,
                device: None,
                recommendations: Some(vec![
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 4096,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.2,
                    },
                    MicrobenchRecommendation {
                        domain: "decode_matvec".to_string(),
                        quant: "q4_k".to_string(),
                        variant: "nr2".to_string(),
                        m: 8192,
                        k: 4096,
                        best_ms: 1.0,
                        speedup_vs_auto: 1.2,
                    },
                ]),
                measurements: vec![
                    MicrobenchMeasurement {
                        name: "gpu.prefill_matmul.q5_k.base.8b_q5k.attn_qkv.tokens512.4096x4096".to_string(),
                        unit: "ms".to_string(),
                        value: 1.0,
                        note: Some(q5k_prefill_batch_measurement_note(
                            Q5KPrefillBatchVariant::Base,
                        )),
                    },
                    MicrobenchMeasurement {
                        name: "gpu.prefill_matmul.q5_k.base.8b_q5k.ffn_up.tokens512.14336x4096"
                            .to_string(),
                        unit: "ms".to_string(),
                        value: 2.0,
                        note: Some(q5k_prefill_batch_measurement_note(
                            Q5KPrefillBatchVariant::Base,
                        )),
                    },
                    MicrobenchMeasurement {
                        name: "gpu.prefill_matmul.q5_k.small.8b_q5k.attn_qkv_small_window.tokens8.4096x4096"
                            .to_string(),
                        unit: "ms".to_string(),
                        value: 0.8,
                        note: Some(q5k_prefill_batch_measurement_note(
                            Q5KPrefillBatchVariant::Small,
                        )),
                    },
                ],
            }],
            suggested_kernel_profile: None,
            suggested_kernel_profile_evidence: None,
        };

        let profile = report.suggested_kernel_profile();
        // Q5_K is now in the default profile (G5 parity fix), so it will be
        // present.  The invariant is that observational Q5_K prefill results
        // must NOT produce Q5_K-specific evidence or mutate the default entry.
        let q5_default = ax_engine_metal::profile::MatvecParams::default();
        if let Some(q5) = profile.decode_matvec.get("q5_k") {
            assert_eq!(q5.rows_per_simdgroup, q5_default.rows_per_simdgroup);
        }

        let evidence = report.suggested_kernel_profile_evidence();
        assert!(
            !evidence
                .iter()
                .any(|item| item.rule.contains("q5_k") || item.rule.contains("prefill_matmul"))
        );
    }

    #[test]
    fn test_q5k_prefill_batch_variants_add_small_route_only_for_small_n() {
        let small = Q5KPrefillBatchShape {
            model: "8b_q5k",
            label: "attn_qkv_small_window",
            m: 4096,
            n: 8,
            k: 4096,
        };
        let large = Q5KPrefillBatchShape {
            model: "8b_q5k",
            label: "attn_qkv",
            m: 4096,
            n: 512,
            k: 4096,
        };

        assert_eq!(
            q5k_prefill_batch_variants(small),
            &[
                Q5KPrefillBatchVariant::Base,
                Q5KPrefillBatchVariant::F16In,
                Q5KPrefillBatchVariant::Small,
            ]
        );
        assert_eq!(
            q5k_prefill_batch_variants(large),
            &[Q5KPrefillBatchVariant::Base, Q5KPrefillBatchVariant::F16In]
        );
    }

    #[test]
    fn test_quant_variant_name() {
        assert_eq!(quant_variant_name(QuantVariant::Nr2), "nr2");
    }
}
