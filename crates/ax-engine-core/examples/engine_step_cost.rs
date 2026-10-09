//! CPU-only measurement of `EngineCore::step` bookkeeping cost in the decode
//! phase. The runner does no model work, so each reported `step_us` is
//! scheduler/request-manager/KV overhead only; it is not inference throughput.

use ax_engine_core::{
    CacheGroupId, DeterministicSampler, EngineCore, ExecutionMode, ExecutionRunner,
    ExecutionStatus, KvManagerConfig, KvWriteSummary, ModelId, RequestExecutionUpdate, RequestId,
    RequestSubmission, RunnerInput, RunnerOutput, SamplingParams, SequenceNo,
};
use std::hint::black_box;
use std::time::Instant;

const BLOCK_SIZE_TOKENS: u32 = 16;

/// Constant-time-per-item runner: emits one fixed token per decode item and does
/// no per-block work, so the measurement isolates engine bookkeeping from
/// `DeterministicRunner`'s own block-table sorting.
#[derive(Debug)]
struct NoopRunner;

impl ExecutionRunner for NoopRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        let request_updates = input
            .execution_batch
            .items
            .iter()
            .map(|item| RequestExecutionUpdate {
                request_id: item.request_id,
                tokens_executed: item.scheduled_token_count,
                output_token: (item.mode == ExecutionMode::Decode).then_some(7),
                output_tokens: Vec::new(),
                stop_reason: None,
                error: None,
                diffusion_schedule: None,
            })
            .collect();
        RunnerOutput {
            step_id: input.execution_batch.step_id,
            request_updates,
            logits_handles: Vec::new(),
            logits_outputs: Vec::new(),
            kv_write_summary: KvWriteSummary {
                tokens_written: input.execution_batch.total_scheduled_tokens,
                blocks_touched: input.block_tables.len() as u32,
            },
            route_metadata: input.execution_batch.route_metadata.clone(),
            execution_status: ExecutionStatus::Success,
        }
    }
}
const WARMUP_STEPS: u32 = 8;
const MEASURED_STEPS: u32 = 64;
const TRIALS: u32 = 5;

fn submit(
    engine: &mut EngineCore,
    id: u64,
    context: usize,
    max_output_tokens: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Distinct token streams per request so prefix reuse does not alias prompts.
    let input_tokens = (0..context as u32)
        .map(|t| t.wrapping_mul(31).wrapping_add(id as u32 * 7_919))
        .collect();
    engine.submit(RequestSubmission {
        request_id: RequestId(id),
        model_id: ModelId("qwen3".into()),
        input_tokens,
        multimodal_inputs: Default::default(),
        sampling_params: SamplingParams::default(),
        max_output_tokens,
        arrival_sequence: SequenceNo(id),
        metadata: None,
    })?;
    Ok(())
}

fn is_prefilling(outcome: &ax_engine_core::EngineStepOutcome) -> bool {
    outcome
        .schedule_plan
        .execution_batch
        .as_ref()
        .is_some_and(|batch| {
            batch
                .items
                .iter()
                .any(|item| item.mode == ExecutionMode::Prefill)
        })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional `<requests> <context> <measured_steps> <retained_requests>` runs
    // one configuration, e.g. under a sampling profiler. `retained_requests`
    // finished prompts are left in the prefix cache before the measured
    // requests start, like a long-lived server.
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|arg| arg.parse().ok())
        .collect();
    let (configs, measured_steps): (Vec<(u64, usize, u64)>, u32) = match args.as_slice() {
        [r, c, n, k] => (vec![(*r as u64, *c, *k as u64)], *n as u32),
        _ => (
            [1_u64, 16]
                .into_iter()
                .flat_map(|r| {
                    [4_096_usize, 32_768]
                        .into_iter()
                        .flat_map(move |c| [0_u64, 8].into_iter().map(move |k| (r, c, k)))
                })
                .collect(),
            MEASURED_STEPS,
        ),
    };
    for (requests, context, retained) in configs {
        for trial in 0..TRIALS {
            let total_blocks = ((requests + retained) as usize
                * (context + 2 * measured_steps as usize))
                .div_ceil(BLOCK_SIZE_TOKENS as usize) as u32;
            let mut engine = EngineCore::with_runtime_components(
                KvManagerConfig::validated(CacheGroupId(0), BLOCK_SIZE_TOKENS, total_blocks),
                NoopRunner,
                DeterministicSampler,
            );
            // Populate the retained prefix cache: run each prompt to completion.
            for id in 0..retained {
                submit(&mut engine, 1_000 + id, context, 1)?;
                for _ in 0..10_000 {
                    let outcome = engine.step(context as u32 + 1, true)?;
                    if outcome.schedule_plan.execution_batch.is_none() {
                        break;
                    }
                }
            }
            for id in 0..requests {
                submit(&mut engine, id, context, u32::MAX)?;
            }
            // Drain prefill with a budget that covers the whole batch.
            let budget = (requests as usize * context) as u32;
            let mut guard = 0;
            while is_prefilling(&engine.step(budget, true)?) {
                guard += 1;
                assert!(guard < 10_000, "prefill did not drain");
            }
            for _ in 0..WARMUP_STEPS {
                black_box(engine.step(requests as u32, true)?);
            }
            let start = Instant::now();
            let mut cpu_time_us = 0_u64;
            for _ in 0..measured_steps {
                let outcome = engine.step(requests as u32, true)?;
                // The window must be pure steady-state decode of every live request.
                assert!(
                    !is_prefilling(&outcome),
                    "prefill work inside the measured window"
                );
                assert_eq!(
                    outcome.schedule_plan.selected_requests.len() as u64,
                    requests,
                    "every live request must decode each measured step"
                );
                cpu_time_us += outcome.metrics.cpu_time_us;
                black_box(&outcome);
            }
            let step_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(measured_steps);
            println!(
                "{{\"requests\":{requests},\"context\":{context},\"retained_requests\":{retained},\"trial\":{trial},\"steps\":{measured_steps},\"step_us\":{step_us:.3},\"metrics_cpu_time_us\":{:.3}}}",
                cpu_time_us as f64 / f64::from(measured_steps)
            );
        }
    }
    Ok(())
}
