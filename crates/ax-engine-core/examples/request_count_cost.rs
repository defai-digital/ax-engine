//! CPU-only comparison of snapshot counting with direct record counting.
//! This does not execute a model or measure inference throughput.

use ax_engine_core::{
    CacheGroupId, ModelId, RequestId, RequestManager, RequestSubmission, SamplingParams, SequenceNo,
};
use std::hint::black_box;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const ITERATIONS: u32 = 2_000;
    for requests in [1, 16] {
        for context in [4_096, 32_768, 131_072] {
            let mut manager = RequestManager::new(CacheGroupId(0));
            for id in 0..requests {
                manager.submit(RequestSubmission {
                    request_id: RequestId(id),
                    model_id: ModelId("qwen3".into()),
                    input_tokens: vec![7; context],
                    multimodal_inputs: Default::default(),
                    sampling_params: SamplingParams::default(),
                    max_output_tokens: 4096,
                    arrival_sequence: SequenceNo(id),
                    metadata: None,
                })?;
            }
            assert_eq!(manager.snapshots().len(), manager.records_len());
            for _ in 0..100 {
                black_box(manager.snapshots().len());
                black_box(black_box(&manager).records_len());
            }
            for trial in 0..6 {
                // Alternate order to reduce systematic phase-order effects.
                let mut timings = [0.0; 2];
                for phase in 0..2 {
                    let method = (trial + phase) % 2;
                    let start = Instant::now();
                    for _ in 0..ITERATIONS {
                        let count = if method == 0 {
                            manager.snapshots().len()
                        } else {
                            black_box(&manager).records_len()
                        };
                        black_box(count);
                    }
                    timings[method] = start.elapsed().as_secs_f64() * 1e6 / f64::from(ITERATIONS);
                }
                println!(
                    "{{\"requests\":{requests},\"context\":{context},\"trial\":{trial},\"iterations\":{ITERATIONS},\"snapshot_count_us\":{:.6},\"record_count_us\":{:.6},\"prompt_bytes_copied_before\":{},\"prompt_bytes_copied_after\":0}}",
                    timings[0],
                    timings[1],
                    requests as usize * context * size_of::<u32>()
                );
            }
        }
    }
    Ok(())
}
