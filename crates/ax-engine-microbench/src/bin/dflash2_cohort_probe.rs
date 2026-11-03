//! P0 probe for batched multi-lane DFlash2 (design:
//! `.internal/reports/batched-dflash2-design-20261103/`).
//!
//! Times the DFlash2 target verify forward (8 rows per lane, the same scopes as
//! the runner's round) for B lanes three ways:
//!   seq    — B rounds one after another, one eval each (what the runner does);
//!   joint  — the same B lane graphs, one eval (host/sync overlap only);
//!   packed — one sequence of 8·B rows on one cache: the projection
//!            amortization a row-packed verify can reach (upper bound; attention
//!            and gated-delta work over 8·B rows is close to B lanes of 8).
//! The drafter is not included.
//!
//! Run: dflash2-cohort-probe <target_dir> <dflash2_drafter_dir> [ctx,ctx...] [reps]

use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill},
    kv_cache::MlxKVCache,
    model::ModelConfig,
    probes::{
        ProbeLane, dflash2_packed_verify_probe, dflash2_packed_verify_supported,
        dflash2_solo_verify_logits_probe, dflash2_verify_forward_probe,
    },
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::{MlxArray, MlxDtype, argmax, astype, eval};

const ROWS: usize = 8;
const LANES: [usize; 4] = [1, 2, 3, 4];
const WARMUP: usize = 3;

fn ids(seed: u32, n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| 1000 + (seed.wrapping_mul(7919) + i * 104_729) % 120_000)
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

struct Ctx<'a> {
    cfg: &'a ModelConfig,
    weights: &'a ModelWeights,
    base: &'a MlxKVCache,
    offset: usize,
    layers: &'a [usize],
}

impl Ctx<'_> {
    fn lane(&self, seed: u32, rows: usize) -> (MlxKVCache, Vec<MlxArray>) {
        let mut cache = self.base.clone();
        let out = dflash2_verify_forward_probe(
            self.cfg,
            self.weights,
            &ids(seed, rows),
            &mut cache,
            self.offset,
            self.layers,
        );
        (cache, out)
    }

    fn seq(&self, lanes: usize) -> f64 {
        let t = Instant::now();
        for l in 0..lanes {
            let (_cache, out) = self.lane(l as u32, ROWS);
            let refs: Vec<&MlxArray> = out.iter().collect();
            eval(&refs);
        }
        t.elapsed().as_secs_f64() * 1e3
    }

    fn joint(&self, lanes: usize) -> f64 {
        let t = Instant::now();
        let built: Vec<_> = (0..lanes).map(|l| self.lane(l as u32, ROWS)).collect();
        let refs: Vec<&MlxArray> = built.iter().flat_map(|(_, out)| out.iter()).collect();
        eval(&refs);
        t.elapsed().as_secs_f64() * 1e3
    }

    /// Real row-packed verify: B lanes, each on its own cache clone.
    fn lanes(&self, lanes: usize) -> f64 {
        self.lanes_rows(lanes, ROWS)
    }

    /// Real row-packed verify of `lanes` lanes with `rows` rows each.
    fn lanes_rows(&self, lanes: usize, rows: usize) -> f64 {
        let t = Instant::now();
        let mut caches: Vec<MlxKVCache> = (0..lanes).map(|_| self.base.clone()).collect();
        let mut probe: Vec<ProbeLane<'_>> = caches
            .iter_mut()
            .enumerate()
            .map(|(l, cache)| ProbeLane {
                cache,
                token_offset: self.offset,
                ids: ids(l as u32, rows),
            })
            .collect();
        let mut out = dflash2_packed_verify_probe(self.cfg, self.weights, &mut probe, self.layers);
        out[0] = argmax(&out[0], None);
        let refs: Vec<&MlxArray> = out.iter().collect();
        eval(&refs);
        t.elapsed().as_secs_f64() * 1e3
    }

    /// Packed lanes vs solo forwards: (rows whose argmax matches, rows, max |Δlogit|).
    fn check(&self, lanes: usize) -> (usize, usize, f32) {
        let mut caches: Vec<MlxKVCache> = (0..lanes).map(|_| self.base.clone()).collect();
        let mut probe: Vec<ProbeLane<'_>> = caches
            .iter_mut()
            .enumerate()
            .map(|(l, cache)| ProbeLane {
                cache,
                token_offset: self.offset,
                ids: ids(l as u32, ROWS),
            })
            .collect();
        let packed = astype(
            &dflash2_packed_verify_probe(self.cfg, self.weights, &mut probe, &[])[0],
            MlxDtype::Float32,
            None,
        );
        let packed_arg = argmax(&packed, None);
        eval(&[&packed, &packed_arg]);
        let (pv, pa) = (packed.data_f32().to_vec(), packed_arg.data_u32().to_vec());
        let vocab = self.cfg.vocab_size;
        let (mut same, mut max_diff) = (0, 0f32);
        for l in 0..lanes {
            let mut cache = self.base.clone();
            let solo = astype(
                &dflash2_solo_verify_logits_probe(
                    self.cfg,
                    self.weights,
                    &ids(l as u32, ROWS),
                    &mut cache,
                    self.offset,
                ),
                MlxDtype::Float32,
                None,
            );
            let solo_arg = argmax(&solo, None);
            eval(&[&solo, &solo_arg]);
            let (sv, sa) = (solo.data_f32(), solo_arg.data_u32());
            for r in 0..ROWS {
                let row = l * ROWS + r;
                same += usize::from(pa[row] == sa[r]);
                for v in 0..vocab {
                    max_diff = max_diff.max((pv[row * vocab + v] - sv[r * vocab + v]).abs());
                }
            }
        }
        (same, lanes * ROWS, max_diff)
    }

    /// One solo verify of `rows` rows.
    fn solo_rows(&self, rows: usize) -> f64 {
        let t = Instant::now();
        let (_cache, out) = self.lane(0, rows);
        let refs: Vec<&MlxArray> = out.iter().collect();
        eval(&refs);
        t.elapsed().as_secs_f64() * 1e3
    }

    fn packed(&self, lanes: usize) -> f64 {
        let t = Instant::now();
        let (_cache, out) = self.lane(0, ROWS * lanes);
        let refs: Vec<&MlxArray> = out.iter().collect();
        eval(&refs);
        t.elapsed().as_secs_f64() * 1e3
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let usage = "usage: dflash2-cohort-probe <target_dir> <drafter_dir> [ctx,ctx...] [reps]";
    let model_dir = args.next().ok_or(usage)?;
    let drafter_dir = args.next().ok_or(usage)?;
    let contexts: Vec<usize> = args
        .next()
        .unwrap_or_else(|| "512,4096".into())
        .split(',')
        .map(|s| s.parse().map_err(|_| format!("bad context length {s:?}")))
        .collect::<Result<_, _>>()?;
    let reps: usize = args
        .next()
        .map(|s| s.parse().map_err(|_| format!("bad reps {s:?}")))
        .transpose()?
        .unwrap_or(9);

    let drafter_cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&drafter_dir).join("config.json"))
            .map_err(|e| format!("drafter config: {e}"))?,
    )
    .map_err(|e| format!("drafter config: {e}"))?;
    let layers: Vec<usize> = drafter_cfg["dflash_config"]["target_layer_ids"]
        .as_array()
        .ok_or("drafter config has no dflash_config.target_layer_ids")?
        .iter()
        .filter_map(|v| v.as_u64().map(|v| v as usize))
        .collect();

    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|e| format!("failed to load model artifacts: {e}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).map_err(|e| format!("failed to load weights: {e}"))?;
    eprintln!("capture layers {layers:?}, contexts {contexts:?}, reps {reps}");

    for &context in &contexts {
        let prompt = ids(424_242, context);
        let mut base = MlxKVCache::new(cfg.layer_count);
        let mut rng = Xorshift64::new(0);
        chunked_prefill(
            &cfg,
            &weights,
            &prompt,
            &mut base,
            DEFAULT_PREFILL_CHUNK,
            MlxSamplingRequest::new(MlxSamplingParams::greedy(), &prompt),
            &mut rng,
        );
        let ctx = Ctx {
            cfg: &cfg,
            weights: &weights,
            base: &base,
            offset: base.seq_len(),
            layers: &layers,
        };
        let supported = dflash2_packed_verify_supported(&cfg, &weights);
        if env::var_os("AX_COHORT_PROBE_SWEEP").is_some() {
            // Cost curve for the cohort window policy: B lanes x W rows.
            for lanes in 1..=4usize {
                for rows in [2usize, 3, 4, 5, 6, 7, 8] {
                    let f = |c: &Ctx<'_>| {
                        if lanes == 1 {
                            c.solo_rows(rows)
                        } else {
                            c.lanes_rows(lanes, rows)
                        }
                    };
                    for _ in 0..WARMUP {
                        f(&ctx);
                    }
                    let ms = median((0..reps).map(|_| f(&ctx)).collect());
                    println!(
                        "{}",
                        serde_json::json!({"context": context, "sweep_lanes": lanes,
                            "rows_per_lane": rows, "ms": ms})
                    );
                }
            }
            continue;
        }
        if supported {
            for lanes in [1, 4] {
                let (same, rows, max_diff) = ctx.check(lanes);
                println!(
                    "{}",
                    serde_json::json!({"context": context, "check_lanes": lanes,
                        "argmax_same": same, "rows": rows, "max_abs_logit_diff": max_diff})
                );
            }
        } else {
            eprintln!("packed verify does not support this model");
        }
        let mut t1 = None;
        for &lanes in &LANES {
            let mut row = Vec::new();
            for name in ["seq", "joint", "packed", "lanes"] {
                if name == "lanes" && !supported {
                    row.push((name, f64::NAN));
                    continue;
                }
                let f = |c: &Ctx<'_>| match name {
                    "seq" => c.seq(lanes),
                    "joint" => c.joint(lanes),
                    "packed" => c.packed(lanes),
                    _ => c.lanes(lanes),
                };
                for _ in 0..WARMUP {
                    f(&ctx);
                }
                let ms = median((0..reps).map(|_| f(&ctx)).collect());
                row.push((name, ms));
            }
            let base_ms = *t1.get_or_insert(row[0].1);
            let line = serde_json::json!({
                "context": context,
                "lanes": lanes,
                "rows": ROWS * lanes,
                "seq_ms": row[0].1,
                "joint_ms": row[1].1,
                "packed_ms": row[2].1,
                "packed_vs_one_lane": row[2].1 / base_ms,
                "packed_aggregate_gain": lanes as f64 * base_ms / row[2].1,
                "joint_aggregate_gain": lanes as f64 * base_ms / row[1].1,
                "lanes_ms": row[3].1,
                "lanes_vs_one_lane": row[3].1 / base_ms,
                "lanes_aggregate_gain": lanes as f64 * base_ms / row[3].1,
            });
            println!("{line}");
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
