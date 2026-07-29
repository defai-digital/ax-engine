//! MLX activation framing for static cross-host pipeline execution.
//!
//! The transport itself is intentionally outside this module. This boundary
//! materializes a contiguous activation, binds it to an immutable cluster
//! generation, and reconstructs it only after integrity validation.

use std::collections::BTreeMap;

use ax_engine_core::{
    ActivationDtype, ActivationFrame, ActivationFrameHeader, PIPELINE_WIRE_VERSION,
    PipelineContractError, PipelineRequestLedger, PipelineTopology, sha256_hex,
};
use mlx_sys::{MlxArray, MlxDtype, contiguous, eval};
use thiserror::Error;

use crate::kv_cache::MlxKVCache;
use crate::model::{
    ModelConfig, PipelineStageForwardError, PipelineStageInput, forward_pipeline_stage,
};
use crate::weights::PipelineStageWeights;

/// Immutable routing identity supplied by the rank runtime.
#[derive(Debug, Clone, Copy)]
pub struct ActivationRoute<'a> {
    pub topology: &'a PipelineTopology,
    pub request_id: u64,
    pub request_sequence: u64,
    pub source_rank: u16,
    pub token_offset: u64,
}

/// MLX-facing name for the transport-neutral core activation frame.
pub type ActivationPacket = ActivationFrame;

/// Materialize and frame a stage output for the immediately following rank.
pub fn encode_activation(
    hidden: &MlxArray,
    route: ActivationRoute<'_>,
) -> Result<ActivationPacket, PipelineActivationError> {
    route.topology.validate()?;
    let source = route
        .topology
        .assignment(route.source_rank)
        .ok_or(PipelineContractError::UnknownRank(route.source_rank))?;
    if source.owns_output_head {
        return Err(PipelineActivationError::OutputHeadHasNoSuccessor);
    }

    let dtype = activation_dtype(hidden.dtype())?;
    let shape_i32 = hidden.shape();
    if shape_i32.len() != 3 || shape_i32.iter().any(|dimension| *dimension <= 0) {
        return Err(PipelineActivationError::InvalidMlxShape(shape_i32));
    }
    let shape = shape_i32
        .iter()
        .map(|dimension| u32::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PipelineActivationError::InvalidMlxShape(shape_i32.clone()))?;

    // A sliced MLX view may not be row-major. The wire format is always a
    // compact row-major payload so receivers never inherit sender strides.
    let materialized = contiguous(hidden, None);
    eval(&[&materialized]);
    let byte_len = materialized.nbytes();
    let raw = materialized.data_raw();
    if raw.is_null() && byte_len != 0 {
        return Err(PipelineActivationError::UnreadableMlxArray);
    }
    let payload = if byte_len == 0 {
        Vec::new()
    } else {
        // `eval` above makes the MLX-owned buffer readable for the duration of
        // this borrow; copy it before `materialized` is dropped.
        unsafe { std::slice::from_raw_parts(raw, byte_len) }.to_vec()
    };
    let destination_rank = route
        .source_rank
        .checked_add(1)
        .ok_or(PipelineActivationError::OutputHeadHasNoSuccessor)?;
    let header = ActivationFrameHeader {
        wire_version: PIPELINE_WIRE_VERSION,
        cluster_id: route.topology.cluster_id.clone(),
        generation: route.topology.generation,
        manifest_digest: route.topology.manifest_digest.clone(),
        model_artifact_digest: route.topology.model_artifact_digest.clone(),
        request_id: route.request_id,
        request_sequence: route.request_sequence,
        source_rank: route.source_rank,
        destination_rank,
        layer_boundary: source.layers.end,
        token_offset: route.token_offset,
        token_count: shape[1],
        dtype,
        shape,
        payload_bytes: payload.len() as u64,
        payload_sha256: sha256_hex(&payload),
    };
    let packet = ActivationPacket { header, payload };
    packet.validate(route.topology)?;
    Ok(packet)
}

/// Reconstruct a receiver-owned MLX activation after all fences pass.
pub fn decode_activation(
    packet: &ActivationPacket,
    topology: &PipelineTopology,
    destination_rank: u16,
) -> Result<MlxArray, PipelineActivationError> {
    packet.validate(topology)?;
    if packet.header.destination_rank != destination_rank {
        return Err(PipelineActivationError::WrongDestination {
            expected: destination_rank,
            actual: packet.header.destination_rank,
        });
    }
    let shape = packet
        .header
        .shape
        .iter()
        .map(|dimension| i32::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PipelineActivationError::ShapeExceedsMlx)?;
    Ok(MlxArray::from_raw_data(
        packet.payload.as_ptr(),
        packet.payload.len(),
        &shape,
        mlx_dtype(packet.header.dtype),
    ))
}

/// Evaluated output of one pipeline rank step.
pub enum PipelineRankOutput {
    /// Hidden states for the immediately following rank.
    Activation(ActivationPacket),
    /// Last-position logits produced by the final rank.
    Logits(MlxArray),
}

/// Stateful stage executor with per-request KV and replay/cancellation fencing.
pub struct PipelineRankExecutor {
    topology: PipelineTopology,
    rank: u16,
    config: ModelConfig,
    weights: PipelineStageWeights,
    ledger: PipelineRequestLedger,
    caches: BTreeMap<u64, MlxKVCache>,
}

impl PipelineRankExecutor {
    pub fn new(
        topology: PipelineTopology,
        rank: u16,
        config: ModelConfig,
        weights: PipelineStageWeights,
    ) -> Result<Self, PipelineRankError> {
        topology.validate()?;
        let assignment = topology
            .assignment(rank)
            .ok_or(PipelineContractError::UnknownRank(rank))?;
        if assignment != &weights.assignment {
            return Err(PipelineRankError::AssignmentMismatch);
        }
        if topology.total_layers as usize != config.layer_count {
            return Err(PipelineRankError::ModelLayerCountMismatch {
                topology: topology.total_layers,
                model: config.layer_count,
            });
        }
        Ok(Self {
            topology,
            rank,
            config,
            weights,
            ledger: PipelineRequestLedger::default(),
            caches: BTreeMap::new(),
        })
    }

    /// Execute prefill or decode tokens on rank 0.
    pub fn execute_tokens(
        &mut self,
        request_id: u64,
        request_sequence: u64,
        token_offset: u64,
        token_ids: &[u32],
    ) -> Result<PipelineRankOutput, PipelineRankError> {
        if self.rank != 0 || !self.weights.assignment.owns_embeddings {
            return Err(PipelineRankError::TokensRequireRankZero);
        }
        let token_count =
            u32::try_from(token_ids.len()).map_err(|_| PipelineRankError::TokenCountOverflow)?;
        self.execute(
            request_id,
            request_sequence,
            token_offset,
            token_count,
            PipelineStageInput::Tokens(token_ids),
        )
    }

    /// Validate and execute an activation received from the preceding rank.
    pub fn execute_activation(
        &mut self,
        packet: &ActivationPacket,
    ) -> Result<PipelineRankOutput, PipelineRankError> {
        let hidden = decode_activation(packet, &self.topology, self.rank)?;
        let header = &packet.header;
        self.execute(
            header.request_id,
            header.request_sequence,
            header.token_offset,
            header.token_count,
            PipelineStageInput::Hidden(&hidden),
        )
    }

    fn execute(
        &mut self,
        request_id: u64,
        request_sequence: u64,
        token_offset: u64,
        token_count: u32,
        input: PipelineStageInput<'_>,
    ) -> Result<PipelineRankOutput, PipelineRankError> {
        self.ledger
            .begin_step(request_id, request_sequence, token_offset)?;
        let cache = self
            .caches
            .entry(request_id)
            .or_insert_with(|| MlxKVCache::new(self.config.layer_count));
        if cache.seq_len() as u64 != token_offset {
            return Err(PipelineContractError::UnexpectedTokenOffset {
                expected: cache.seq_len() as u64,
                actual: token_offset,
            }
            .into());
        }
        let output = forward_pipeline_stage(
            &self.config,
            &self.weights,
            input,
            cache,
            usize::try_from(token_offset).map_err(|_| PipelineRankError::TokenOffsetOverflow)?,
        )?;
        let mut eval_refs = vec![&output];
        eval_refs.extend(cache.collect_eval_refs());
        eval(&eval_refs);
        drop(eval_refs);

        let rank_output = if self.weights.assignment.owns_output_head {
            PipelineRankOutput::Logits(output)
        } else {
            PipelineRankOutput::Activation(encode_activation(
                &output,
                ActivationRoute {
                    topology: &self.topology,
                    request_id,
                    request_sequence,
                    source_rank: self.rank,
                    token_offset,
                },
            )?)
        };
        cache.advance(token_count as usize);
        self.ledger
            .commit_step(request_id, request_sequence, token_count)?;
        Ok(rank_output)
    }

    /// Drop KV state and permanently reject replay of this request ID in the
    /// current generation.
    pub fn close_request(&mut self, request_id: u64) {
        self.caches.remove(&request_id);
        self.ledger.close(request_id);
    }

    pub fn active_requests(&self) -> usize {
        self.caches.len()
    }

    /// Diagnostic used by correctness tests and rank health reporting.
    pub fn request_cache_has_layer(&self, request_id: u64, global_layer: usize) -> bool {
        self.caches
            .get(&request_id)
            .and_then(|cache| cache.logical_layer_kv(global_layer))
            .is_some()
    }
}

fn activation_dtype(dtype: MlxDtype) -> Result<ActivationDtype, PipelineActivationError> {
    match dtype {
        MlxDtype::Bfloat16 => Ok(ActivationDtype::Bfloat16),
        MlxDtype::Float16 => Ok(ActivationDtype::Float16),
        MlxDtype::Float32 => Ok(ActivationDtype::Float32),
        unsupported => Err(PipelineActivationError::UnsupportedDtype(unsupported)),
    }
}

fn mlx_dtype(dtype: ActivationDtype) -> MlxDtype {
    match dtype {
        ActivationDtype::Bfloat16 => MlxDtype::Bfloat16,
        ActivationDtype::Float16 => MlxDtype::Float16,
        ActivationDtype::Float32 => MlxDtype::Float32,
    }
}

#[derive(Debug, Error)]
pub enum PipelineActivationError {
    #[error(transparent)]
    Contract(#[from] PipelineContractError),
    #[error("rank owning the output head has no activation successor")]
    OutputHeadHasNoSuccessor,
    #[error("unsupported activation dtype {0:?}")]
    UnsupportedDtype(MlxDtype),
    #[error("activation has invalid MLX shape {0:?}")]
    InvalidMlxShape(Vec<i32>),
    #[error("evaluated MLX activation has no readable data pointer")]
    UnreadableMlxArray,
    #[error("activation shape exceeds MLX i32 dimensions")]
    ShapeExceedsMlx,
    #[error("activation addressed to rank {actual}, not receiver rank {expected}")]
    WrongDestination { expected: u16, actual: u16 },
}

#[derive(Debug, Error)]
pub enum PipelineRankError {
    #[error(transparent)]
    Contract(#[from] PipelineContractError),
    #[error(transparent)]
    Activation(#[from] PipelineActivationError),
    #[error(transparent)]
    Forward(#[from] PipelineStageForwardError),
    #[error("pipeline topology assignment does not match loaded stage weights")]
    AssignmentMismatch,
    #[error("pipeline topology has {topology} layers but model config has {model}")]
    ModelLayerCountMismatch { topology: u32, model: usize },
    #[error("token IDs may only enter the embedding-owning rank 0")]
    TokensRequireRankZero,
    #[error("token count exceeds the pipeline wire format")]
    TokenCountOverflow,
    #[error("token offset exceeds this host's address space")]
    TokenOffsetOverflow,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use ax_engine_core::{PipelineLayerRange, PipelineRankAssignment, PipelineTopology};

    fn topology() -> PipelineTopology {
        PipelineTopology {
            cluster_id: "cluster-a".into(),
            generation: 9,
            manifest_digest: "manifest-a".into(),
            model_artifact_digest: "model-a".into(),
            total_layers: 4,
            micro_batch_limit: 2,
            ranks: vec![
                PipelineRankAssignment {
                    rank: 0,
                    node_identity_digest: "node-a".into(),
                    layers: PipelineLayerRange { start: 0, end: 2 },
                    owns_embeddings: true,
                    owns_output_head: false,
                },
                PipelineRankAssignment {
                    rank: 1,
                    node_identity_digest: "node-b".into(),
                    layers: PipelineLayerRange { start: 2, end: 4 },
                    owns_embeddings: false,
                    owns_output_head: true,
                },
            ],
        }
    }

    #[test]
    fn f32_activation_round_trips_with_generation_fence() {
        let values = [0.25_f32, -1.5, 2.0, 8.25];
        let hidden = MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(&values),
            &[1, 2, 2],
            MlxDtype::Float32,
        );
        let topology = topology();
        let packet = encode_activation(
            &hidden,
            ActivationRoute {
                topology: &topology,
                request_id: 41,
                request_sequence: 1,
                source_rank: 0,
                token_offset: 7,
            },
        )
        .expect("activation should encode");
        let restored = decode_activation(&packet, &topology, 1).expect("activation should decode");
        eval(&[&restored]);
        assert_eq!(restored.shape(), vec![1, 2, 2]);
        assert_eq!(restored.data_f32(), values);
    }

    #[test]
    fn wrong_receiver_rank_is_rejected() {
        let values = [1.0_f32, 2.0];
        let hidden = MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(&values),
            &[1, 1, 2],
            MlxDtype::Float32,
        );
        let topology = topology();
        let packet = encode_activation(
            &hidden,
            ActivationRoute {
                topology: &topology,
                request_id: 42,
                request_sequence: 1,
                source_rank: 0,
                token_offset: 0,
            },
        )
        .expect("activation should encode");
        assert!(matches!(
            decode_activation(&packet, &topology, 0),
            Err(PipelineActivationError::WrongDestination {
                expected: 0,
                actual: 1
            })
        ));
    }
}
