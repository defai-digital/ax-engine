use super::{
    artifact::{ByteDecoder, NcmArtifactDescriptorV1, SCRATCH_BYTES, runtime, validate_unigram},
    codec::{Codec, LENGTH, LEVELS, WIDTH, token_ids, validate_tensor},
    generator::{Generator, area_pool, chosen},
};
use ax_engine_core::{
    ExecutionRunner, GenerationKind, RequestId, RunnerInput, RunnerOutput,
    memory_budget::{estimated_footprint_bytes, memory_plan_os_reserve_bytes},
    ncm::{
        ConceptPhase, ConceptProgress, ConceptUpdate, ConceptWork, NcmError, NcmOutput,
        NcmRequestConfig, NcmSampling,
    },
};
use mlx_sys::{MlxArray, add, load_safetensors};
use parking_lot::Mutex;
use std::{fmt, path::Path, sync::OnceLock, time::Duration};

fn load_ledger() -> &'static Mutex<u64> {
    static LEDGER: OnceLock<Mutex<u64>> = OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(0))
}
struct LoadLease(u64);
impl LoadLease {
    fn acquire(bytes: u64, ceiling: u64) -> Result<Self, NcmError> {
        let mut ledger = load_ledger().lock();
        let total = ledger.checked_add(bytes).ok_or(NcmError::MemoryBudget)?;
        if total > ceiling {
            return Err(NcmError::MemoryBudget);
        }
        *ledger = total;
        Ok(Self(bytes))
    }
}
impl Drop for LoadLease {
    fn drop(&mut self) {
        let mut ledger = load_ledger().lock();
        *ledger = ledger.saturating_sub(self.0);
    }
}

struct Model {
    descriptor: NcmArtifactDescriptorV1,
    codec: Codec,
    generator: Generator,
    decoder: ByteDecoder,
    level0: u32,
}
struct State {
    id: RequestId,
    progress: ConceptProgress,
    latent: Option<MlxArray>,
    output: Option<NcmOutput>,
}

/// One immutable model and an exclusive request/scratch lease. No AR KV state.
pub struct NcmRunner {
    model: Model,
    state: Mutex<Option<State>>,
    _load_lease: LoadLease,
}
impl fmt::Debug for NcmRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NcmRunner")
            .field("identity", &self.model.descriptor.identity)
            .finish_non_exhaustive()
    }
}
impl NcmRunner {
    pub fn load_experimental(path: &Path, memory_budget_bytes: u64) -> Result<Self, NcmError> {
        std::panic::catch_unwind(|| Self::load_inner(path, memory_budget_bytes))
            .map_err(|_| runtime("MLX model loading panicked"))?
    }

    fn load_inner(path: &Path, memory_budget_bytes: u64) -> Result<Self, NcmError> {
        let descriptor = NcmArtifactDescriptorV1::from_research_pin(path)?;
        let required = estimated_footprint_bytes(descriptor.weight_bytes, Some(0))
            .checked_add(SCRATCH_BYTES)
            .ok_or(NcmError::MemoryBudget)?;
        let host = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .map_err(runtime)?;
        if !host.status.success() {
            return Err(runtime("unable to inspect unified memory"));
        }
        let host: u64 = std::str::from_utf8(&host.stdout)
            .map_err(runtime)?
            .trim()
            .parse()
            .map_err(runtime)?;
        if required > memory_budget_bytes
            || memory_budget_bytes > host.saturating_sub(memory_plan_os_reserve_bytes(host))
        {
            return Err(NcmError::MemoryBudget);
        }
        let ceiling = host.saturating_sub(memory_plan_os_reserve_bytes(host));
        if mlx_sys::process_memory_footprint()
            .is_some_and(|p| p.phys_footprint_bytes.saturating_add(required) > ceiling)
        {
            return Err(NcmError::MemoryBudget);
        }
        // Capacity is reserved at load; active scratch leases are subsets, not extra charges.
        let lease = LoadLease::acquire(required, ceiling)?;
        let decoder = ByteDecoder::load(descriptor.file("bpe.json")?)?;
        let inference =
            load_safetensors(descriptor.file("inference.safetensors")?, None).map_err(runtime)?;
        if inference.len() != 1 {
            return Err(runtime("unexpected inference tensor set"));
        }
        let probabilities = inference
            .get("level0_unigram")
            .ok_or_else(|| runtime("missing L0 unigram"))?;
        validate_tensor("level0_unigram", probabilities, &[4096]).map_err(runtime)?;
        let level0 = validate_unigram(probabilities.data_f32())?;
        let codec = Codec::load_verified(
            descriptor.file("codec.safetensors")?,
            descriptor.file("spec.json")?,
        )
        .map_err(runtime)?;
        let generator = Generator::load_verified(descriptor.file("generator.safetensors")?, &codec)
            .map_err(runtime)?;
        Ok(Self {
            model: Model {
                descriptor,
                codec,
                generator,
                decoder,
                level0,
            },
            state: Mutex::new(None),
            _load_lease: lease,
        })
    }
    pub fn identity(&self) -> &str {
        &self.model.descriptor.identity
    }
    pub fn reserved_capacity_bytes(&self) -> u64 {
        self._load_lease.0
    }
    pub fn active_scratch_bytes(&self) -> u64 {
        if self.state.lock().is_some() {
            SCRATCH_BYTES
        } else {
            0
        }
    }
    pub fn into_core(self) -> ax_engine_core::EngineCore {
        let config =
            ax_engine_core::KvManagerConfig::validated(ax_engine_core::CacheGroupId(0), 16, 1);
        let mut core = ax_engine_core::EngineCore::with_runtime_components(
            config,
            self,
            ax_engine_core::DeterministicSampler,
        );
        core.set_generation_kind(GenerationKind::NextConcept);
        core
    }
    pub fn default_timeout() -> Duration {
        Duration::from_secs(60)
    }
}
impl ExecutionRunner for NcmRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        RunnerOutput {
            step_id: input.execution_batch.step_id,
            request_updates: input
                .execution_batch
                .items
                .iter()
                .map(|item| ax_engine_core::RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: 0,
                    output_token: None,
                    output_tokens: vec![],
                    stop_reason: Some(ax_engine_core::StopReason::Error),
                    error: Some("NextConcept requires typed scheduling".into()),
                    diffusion_schedule: None,
                })
                .collect(),
            logits_handles: vec![],
            logits_outputs: vec![],
            kv_write_summary: ax_engine_core::KvWriteSummary {
                tokens_written: 0,
                blocks_touched: 0,
            },
            route_metadata: input.execution_batch.route_metadata,
            execution_status: ax_engine_core::ExecutionStatus::Failed,
        }
    }
    fn begin_ncm(&self, id: RequestId, config: NcmRequestConfig) -> Result<(), NcmError> {
        match config.sampling {
            NcmSampling::Greedy => {}
        }
        let mut state = self.state.try_lock().ok_or(NcmError::Busy)?;
        if state.is_some() {
            return Err(NcmError::Busy);
        }
        *state = Some(State {
            id,
            progress: ConceptProgress::default(),
            latent: None,
            output: None,
        });
        Ok(())
    }
    fn run_ncm(&self, work: ConceptWork) -> Result<ConceptUpdate, NcmError> {
        match work.config.sampling {
            NcmSampling::Greedy => {}
        }
        if work.cancellation.is_cancelled() {
            return Err(NcmError::Cancelled);
        }
        let mut slot = self.state.lock();
        let s = slot.as_mut().ok_or(NcmError::NoRequest)?;
        if s.id != work.request_id || s.progress != work.before {
            return Err(runtime("runner/core progress mismatch"));
        }
        let output = match work.before.phase {
            ConceptPhase::Init => {
                s.latent = Some(self.model.codec.contribution(&[self.model.level0], 0));
                None
            }
            ConceptPhase::Predict(level @ 1..=11) => {
                let level = usize::from(level);
                let latent = s
                    .latent
                    .as_ref()
                    .ok_or_else(|| runtime("missing accumulated latent"))?;
                let pooled = area_pool(latent, LEVELS[level]);
                let forward = self.model.generator.forward_with_capture(
                    &self.model.codec,
                    level,
                    &pooled,
                    false,
                );
                validate_tensor(
                    "generator logits",
                    &forward.logits,
                    &[1, LEVELS[level] as i32, 4096],
                )
                .map_err(runtime)?;
                let ids = chosen(&forward.logits).map_err(runtime)?;
                s.latent = Some(add(
                    latent,
                    &self.model.codec.contribution(&ids, level),
                    None,
                ));
                None
            }
            ConceptPhase::Decode => {
                let logits = self.model.codec.decode_logits(
                    s.latent
                        .as_ref()
                        .ok_or_else(|| runtime("missing final latent"))?,
                );
                validate_tensor("codec logits", &logits, &[1, LENGTH, 8192]).map_err(runtime)?;
                s.output = Some(
                    self.model
                        .decoder
                        .decode(token_ids(&logits).map_err(runtime)?)?,
                );
                None
            }
            ConceptPhase::Commit => Some(
                s.output
                    .take()
                    .ok_or_else(|| runtime("missing decoded output"))?,
            ),
            _ => return Err(runtime("invalid concept work phase")),
        };
        if let Some(latent) = &s.latent {
            validate_tensor("accumulated latent", latent, &[1, LENGTH, WIDTH]).map_err(runtime)?;
        }
        if work.cancellation.is_cancelled() {
            return Err(NcmError::Cancelled);
        }
        s.progress = s.progress.advance()?;
        Ok(ConceptUpdate {
            request_id: s.id,
            progress: s.progress,
            output,
        })
    }
    fn release_request_state(&self, id: RequestId) {
        let mut slot = self.state.lock();
        if slot.as_ref().is_some_and(|s| s.id == id) {
            *slot = None;
        }
    }
    fn native_live_sequence_cap(&self) -> Option<u32> {
        Some(1)
    }
}

#[cfg(test)]
mod resource_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    #[test]
    fn aggregate_capacity_fails_closed_and_returns_on_drop() {
        let initial = *load_ledger().lock();
        let lease = LoadLease::acquire(60, initial + 100).unwrap();
        assert!(matches!(
            LoadLease::acquire(60, initial + 100),
            Err(NcmError::MemoryBudget)
        ));
        assert_eq!(*load_ledger().lock(), initial + 60);
        drop(lease);
        assert_eq!(*load_ledger().lock(), initial);
        let again = LoadLease::acquire(100, initial + 100).unwrap();
        drop(again);
        assert_eq!(*load_ledger().lock(), initial);
    }
}
