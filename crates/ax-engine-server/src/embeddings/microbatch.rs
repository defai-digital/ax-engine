use std::collections::BTreeMap;
use std::env;
use std::sync::Arc;
use std::time::Duration;

use ax_engine_sdk::{EmbeddingPooling, EngineSessionError};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Instant, sleep_until};

use crate::admission::AdmissionPermit;
use crate::app_state::{
    EmbeddingBatchItem, EmbeddingBatchKey, EmbeddingBatchRequestOptions, EmbeddingBatchRunItem,
    EmbeddingMicroBatcher,
};
use crate::generation::service::NativeGenerationService;

const DEFAULT_EMBEDDING_MICROBATCH_WINDOW_MS: u64 = 2;
const DEFAULT_EMBEDDING_MICROBATCH_MAX_BATCH: usize = 32;
const DEFAULT_EMBEDDING_MICROBATCH_QUEUE_CAPACITY: usize = 1024;

impl EmbeddingMicroBatcher {
    pub(crate) fn spawn(generation_service: std::sync::Arc<NativeGenerationService>) -> Arc<Self> {
        let capacity = embedding_microbatch_queue_capacity();
        let (sender, receiver) = mpsc::channel(capacity);
        let batch_window = embedding_microbatch_window();
        let max_batch = embedding_microbatch_max_batch();
        // The worker must NOT hold a sender clone: the channel closing when
        // the last external sender drops is what lets the worker drain any
        // queued items and exit after a model hot-swap replaces LiveState.
        // A keepalive here pins the old EngineSession (and its weights)
        // forever on every swap.
        tokio::spawn(run_embedding_microbatch_worker(
            receiver,
            generation_service,
            batch_window,
            max_batch,
        ));
        Arc::new(Self { sender })
    }

    pub(crate) async fn embed(
        &self,
        input: Vec<u32>,
        pooling: EmbeddingPooling,
        normalize: bool,
        admission_permit: AdmissionPermit,
    ) -> Result<Vec<f32>, EngineSessionError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.sender
            .send(EmbeddingBatchItem {
                input,
                pooling,
                normalize,
                admission_permit,
                response_tx,
            })
            .await
            .map_err(|_| EngineSessionError::EmbeddingFailed {
                message: "embedding microbatch queue is unavailable",
            })?;

        response_rx
            .await
            .map_err(|_| EngineSessionError::EmbeddingFailed {
                message: "embedding microbatch response channel closed",
            })?
    }
}

async fn run_embedding_microbatch_worker(
    mut receiver: mpsc::Receiver<EmbeddingBatchItem>,
    generation_service: std::sync::Arc<NativeGenerationService>,
    batch_window: Duration,
    max_batch: usize,
) {
    while let Some(first) = receiver.recv().await {
        let batch =
            collect_embedding_microbatch(&mut receiver, first, batch_window, max_batch).await;
        let run_items: Vec<EmbeddingBatchRunItem> = batch
            .iter()
            .map(|item| EmbeddingBatchRunItem {
                input: item.input.clone(),
                pooling: item.pooling,
                normalize: item.normalize,
            })
            .collect();
        let responses = run_embedding_batch(generation_service.clone(), run_items).await;
        for (item, response) in batch.into_iter().zip(responses) {
            let _ = item.response_tx.send(response);
            drop(item.admission_permit);
        }
    }
}

async fn collect_embedding_microbatch(
    receiver: &mut mpsc::Receiver<EmbeddingBatchItem>,
    first: EmbeddingBatchItem,
    batch_window: Duration,
    max_batch: usize,
) -> Vec<EmbeddingBatchItem> {
    let target = max_batch.max(1);
    let mut batch = Vec::with_capacity(target);
    batch.push(first);
    if target == 1 {
        return batch;
    }

    if batch_window.is_zero() {
        while batch.len() < target {
            match receiver.try_recv() {
                Ok(item) => batch.push(item),
                Err(_) => break,
            }
        }
        return batch;
    }

    let deadline = Instant::now() + batch_window;
    while batch.len() < target {
        tokio::select! {
            maybe = receiver.recv() => {
                match maybe {
                    Some(item) => batch.push(item),
                    None => break,
                }
            }
            _ = sleep_until(deadline) => break,
        }
    }

    batch
}

async fn run_embedding_batch(
    generation_service: std::sync::Arc<NativeGenerationService>,
    items: Vec<EmbeddingBatchRunItem>,
) -> Vec<Result<Vec<f32>, EngineSessionError>> {
    let groups = collect_embedding_batch_groups(
        &items
            .iter()
            .map(|item| EmbeddingBatchRequestOptions {
                pooling: item.pooling,
                normalize: item.normalize,
            })
            .collect::<Vec<_>>(),
    );

    let item_count = items.len();
    match generation_service
        .execute(move |session| {
            let mut outputs: Vec<Option<Result<Vec<f32>, EngineSessionError>>> =
                (0..item_count).map(|_| None).collect();

            for (key, indices) in groups {
                let pooling = pooling_from_code(key.pooling_code);
                let batch_inputs: Vec<Vec<u32>> = indices
                    .iter()
                    .map(|index| items[*index].input.clone())
                    .collect();

                // Prefer the contiguous `embed_batch_flat` path: it does
                // one device-to-host read per group instead of B, so the
                // microbatcher's response dispatch loop walks a `&[f32]`
                // slice rather than allocating `B` separate `Vec<f32>`s.
                // We still need a `Vec<f32>` per request for the JSON
                // response, so slice + to_vec on the way out.
                match session.embed_batch_flat(&batch_inputs, pooling, key.normalize) {
                    Ok(matrix) if matrix.batch_size == indices.len() => {
                        for (index, sentence_idx) in
                            indices.iter().copied().zip(0..matrix.batch_size)
                        {
                            outputs[index] = Some(Ok(matrix.row(sentence_idx).to_vec()));
                        }
                    }
                    Ok(_) | Err(_) => {
                        // Fallback preserves per-request error behavior even if grouped execution fails.
                        for index in indices {
                            outputs[index] = Some(session.embed(
                                &items[index].input,
                                items[index].pooling,
                                items[index].normalize,
                            ));
                        }
                    }
                }
            }

            Ok(outputs
                .into_iter()
                .map(|value| {
                    value.unwrap_or(Err(EngineSessionError::EmbeddingFailed {
                        message: "embedding microbatch output missing",
                    }))
                })
                .collect::<Vec<_>>())
        })
        .await
    {
        Ok(outputs) => outputs,
        Err(error) => {
            tracing::error!(%error, "embedding microbatch worker failed");
            (0..item_count)
                .map(|_| {
                    Err(EngineSessionError::EmbeddingFailed {
                        message: "embedding microbatch worker failed",
                    })
                })
                .collect()
        }
    }
}

pub(crate) fn collect_embedding_batch_groups(
    options: &[EmbeddingBatchRequestOptions],
) -> Vec<(EmbeddingBatchKey, Vec<usize>)> {
    let mut grouped: BTreeMap<EmbeddingBatchKey, Vec<usize>> = BTreeMap::new();
    for (index, option) in options.iter().enumerate() {
        grouped
            .entry(EmbeddingBatchKey {
                pooling_code: pooling_code(option.pooling),
                normalize: option.normalize,
            })
            .or_default()
            .push(index);
    }
    grouped.into_iter().collect()
}

pub(crate) fn pooling_code(pooling: EmbeddingPooling) -> u8 {
    match pooling {
        EmbeddingPooling::Mean => 0,
        EmbeddingPooling::Last => 1,
        EmbeddingPooling::Cls => 2,
    }
}

fn pooling_from_code(code: u8) -> EmbeddingPooling {
    match code {
        0 => EmbeddingPooling::Mean,
        1 => EmbeddingPooling::Last,
        _ => EmbeddingPooling::Cls,
    }
}

fn embedding_microbatch_window() -> Duration {
    let millis = env::var("AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(DEFAULT_EMBEDDING_MICROBATCH_WINDOW_MS)
        .min(100);
    Duration::from_millis(millis)
}

fn embedding_microbatch_max_batch() -> usize {
    env::var("AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_EMBEDDING_MICROBATCH_MAX_BATCH)
        .clamp(1, 512)
}

fn embedding_microbatch_queue_capacity() -> usize {
    env::var("AX_ENGINE_EMBED_MICROBATCH_QUEUE_CAPACITY")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_EMBEDDING_MICROBATCH_QUEUE_CAPACITY)
        .clamp(64, 8192)
}
