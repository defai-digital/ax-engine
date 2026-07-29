//! Runtime-neutral contracts for static cross-host pipeline execution.
//!
//! AX Serving owns placement and admission. AX Engine owns these execution
//! contracts, transformer partitioning, activation transfer, KV state, and
//! generation fencing.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;

/// Wire protocol version for rank-to-rank activation frames.
pub const PIPELINE_WIRE_VERSION: u16 = 1;
const PIPELINE_FRAME_MAGIC: &[u8; 4] = b"AXPF";
const PIPELINE_MAX_HEADER_BYTES: usize = 64 * 1024;

/// Half-open transformer-layer range `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineLayerRange {
    pub start: u32,
    pub end: u32,
}

impl PipelineLayerRange {
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }

    pub const fn len(self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    pub const fn contains(self, layer: u32) -> bool {
        layer >= self.start && layer < self.end
    }
}

/// Immutable assignment for one pipeline rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineRankAssignment {
    pub rank: u16,
    pub node_identity_digest: String,
    pub layers: PipelineLayerRange,
    pub owns_embeddings: bool,
    pub owns_output_head: bool,
}

/// Static pipeline topology selected for one model generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineTopology {
    pub cluster_id: String,
    pub generation: u64,
    pub manifest_digest: String,
    pub model_artifact_digest: String,
    pub total_layers: u32,
    pub ranks: Vec<PipelineRankAssignment>,
}

impl PipelineTopology {
    /// Validate the complete gang before any model weights are loaded.
    pub fn validate(&self) -> Result<(), PipelineContractError> {
        validate_identity("cluster_id", &self.cluster_id)?;
        validate_identity("manifest_digest", &self.manifest_digest)?;
        validate_identity("model_artifact_digest", &self.model_artifact_digest)?;
        if self.generation == 0 {
            return Err(PipelineContractError::ZeroGeneration);
        }
        if self.total_layers == 0 {
            return Err(PipelineContractError::ZeroLayers);
        }
        if self.ranks.len() < 2 || self.ranks.len() > usize::from(u16::MAX) {
            return Err(PipelineContractError::InvalidRankCount(self.ranks.len()));
        }

        let mut seen_ranks = BTreeSet::new();
        let mut seen_nodes = BTreeSet::new();
        for (index, rank) in self.ranks.iter().enumerate() {
            let expected_rank =
                u16::try_from(index).map_err(|_| PipelineContractError::InvalidRankCount(index))?;
            if rank.rank != expected_rank || !seen_ranks.insert(rank.rank) {
                return Err(PipelineContractError::NonContiguousRanks {
                    expected: expected_rank,
                    actual: rank.rank,
                });
            }
            validate_identity("node_identity_digest", &rank.node_identity_digest)?;
            if !seen_nodes.insert(rank.node_identity_digest.as_str()) {
                return Err(PipelineContractError::DuplicateNodeIdentity);
            }
            if rank.layers.start >= rank.layers.end || rank.layers.end > self.total_layers {
                return Err(PipelineContractError::InvalidLayerRange {
                    rank: rank.rank,
                    start: rank.layers.start,
                    end: rank.layers.end,
                });
            }
            let expected_start = if index == 0 {
                0
            } else {
                self.ranks[index - 1].layers.end
            };
            if rank.layers.start != expected_start {
                return Err(PipelineContractError::LayerCoverageGap {
                    expected_start,
                    actual_start: rank.layers.start,
                });
            }
            let should_own_embeddings = index == 0;
            let should_own_output_head = index + 1 == self.ranks.len();
            if rank.owns_embeddings != should_own_embeddings
                || rank.owns_output_head != should_own_output_head
            {
                return Err(PipelineContractError::InvalidEndpointOwnership { rank: rank.rank });
            }
        }
        if self.ranks.last().map(|rank| rank.layers.end) != Some(self.total_layers) {
            return Err(PipelineContractError::IncompleteLayerCoverage);
        }
        Ok(())
    }

    pub fn assignment(&self, rank: u16) -> Option<&PipelineRankAssignment> {
        self.ranks
            .get(usize::from(rank))
            .filter(|assignment| assignment.rank == rank)
    }

    /// Fence an incoming frame against this exact immutable generation.
    pub fn validate_frame_route(
        &self,
        frame: &ActivationFrameHeader,
    ) -> Result<(), PipelineContractError> {
        frame.validate()?;
        if frame.cluster_id != self.cluster_id
            || frame.generation != self.generation
            || frame.manifest_digest != self.manifest_digest
            || frame.model_artifact_digest != self.model_artifact_digest
        {
            return Err(PipelineContractError::StaleOrForeignGeneration);
        }
        let source = self
            .assignment(frame.source_rank)
            .ok_or(PipelineContractError::UnknownRank(frame.source_rank))?;
        let destination = self
            .assignment(frame.destination_rank)
            .ok_or(PipelineContractError::UnknownRank(frame.destination_rank))?;
        if frame.destination_rank != frame.source_rank.saturating_add(1)
            || source.layers.end != destination.layers.start
            || frame.layer_boundary != source.layers.end
        {
            return Err(PipelineContractError::InvalidActivationRoute);
        }
        Ok(())
    }
}

/// Activation scalar type supported by the initial pipeline wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationDtype {
    Bfloat16,
    Float16,
    Float32,
}

impl ActivationDtype {
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bfloat16 | Self::Float16 => 2,
            Self::Float32 => 4,
        }
    }
}

/// Integrity- and generation-bound header for one activation payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationFrameHeader {
    pub wire_version: u16,
    pub cluster_id: String,
    pub generation: u64,
    pub manifest_digest: String,
    pub model_artifact_digest: String,
    pub request_id: u64,
    pub request_sequence: u64,
    pub source_rank: u16,
    pub destination_rank: u16,
    pub layer_boundary: u32,
    pub token_offset: u64,
    pub token_count: u32,
    pub dtype: ActivationDtype,
    pub shape: Vec<u32>,
    pub payload_bytes: u64,
    pub payload_sha256: String,
}

impl ActivationFrameHeader {
    pub fn validate(&self) -> Result<(), PipelineContractError> {
        if self.wire_version != PIPELINE_WIRE_VERSION {
            return Err(PipelineContractError::UnsupportedWireVersion(
                self.wire_version,
            ));
        }
        validate_identity("cluster_id", &self.cluster_id)?;
        validate_identity("manifest_digest", &self.manifest_digest)?;
        validate_identity("model_artifact_digest", &self.model_artifact_digest)?;
        validate_sha256(&self.payload_sha256)?;
        if self.generation == 0 {
            return Err(PipelineContractError::ZeroGeneration);
        }
        if self.request_id == 0 || self.request_sequence == 0 {
            return Err(PipelineContractError::ZeroRequestIdentity);
        }
        if self.destination_rank != self.source_rank.saturating_add(1) {
            return Err(PipelineContractError::InvalidActivationRoute);
        }
        if self.token_count == 0 || self.shape.len() != 3 {
            return Err(PipelineContractError::InvalidActivationShape);
        }
        if self.shape[0] != 1 || self.shape[1] != self.token_count || self.shape[2] == 0 {
            return Err(PipelineContractError::InvalidActivationShape);
        }
        let elements = self.shape.iter().try_fold(1_usize, |total, dimension| {
            total.checked_mul(*dimension as usize)
        });
        let expected_bytes = elements
            .and_then(|elements| elements.checked_mul(self.dtype.size_bytes()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(PipelineContractError::ActivationSizeOverflow)?;
        if self.payload_bytes != expected_bytes {
            return Err(PipelineContractError::ActivationSizeMismatch {
                expected: expected_bytes,
                actual: self.payload_bytes,
            });
        }
        Ok(())
    }

    pub fn verify_payload(&self, payload: &[u8]) -> Result<(), PipelineContractError> {
        self.validate()?;
        if u64::try_from(payload.len()).ok() != Some(self.payload_bytes) {
            return Err(PipelineContractError::ActivationSizeMismatch {
                expected: self.payload_bytes,
                actual: payload.len() as u64,
            });
        }
        if sha256_hex(payload) != self.payload_sha256 {
            return Err(PipelineContractError::ActivationDigestMismatch);
        }
        Ok(())
    }
}

/// Transport-neutral activation frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationFrame {
    pub header: ActivationFrameHeader,
    pub payload: Vec<u8>,
}

impl ActivationFrame {
    pub fn validate(&self, topology: &PipelineTopology) -> Result<(), PipelineContractError> {
        topology.validate_frame_route(&self.header)?;
        self.header.verify_payload(&self.payload)
    }

    /// Encode as `AXPF | header_len:u32be | header_json | payload`.
    pub fn encode(&self, topology: &PipelineTopology) -> Result<Vec<u8>, PipelineContractError> {
        self.validate(topology)?;
        let header = serde_json::to_vec(&self.header)
            .map_err(|error| PipelineContractError::HeaderSerialization(error.to_string()))?;
        if header.len() > PIPELINE_MAX_HEADER_BYTES {
            return Err(PipelineContractError::HeaderTooLarge(header.len()));
        }
        let header_len = u32::try_from(header.len())
            .map_err(|_| PipelineContractError::HeaderTooLarge(header.len()))?;
        let capacity = 8_usize
            .checked_add(header.len())
            .and_then(|size| size.checked_add(self.payload.len()))
            .ok_or(PipelineContractError::ActivationSizeOverflow)?;
        let mut frame = Vec::with_capacity(capacity);
        frame.extend_from_slice(PIPELINE_FRAME_MAGIC);
        frame.extend_from_slice(&header_len.to_be_bytes());
        frame.extend_from_slice(&header);
        frame.extend_from_slice(&self.payload);
        Ok(frame)
    }

    /// Decode with an operator-provided payload ceiling before allocation.
    pub fn decode(
        bytes: &[u8],
        topology: &PipelineTopology,
        maximum_payload_bytes: u64,
    ) -> Result<Self, PipelineContractError> {
        if bytes.len() < 8 || &bytes[..4] != PIPELINE_FRAME_MAGIC {
            return Err(PipelineContractError::MalformedActivationFrame);
        }
        let header_len = u32::from_be_bytes(
            bytes[4..8]
                .try_into()
                .map_err(|_| PipelineContractError::MalformedActivationFrame)?,
        ) as usize;
        if header_len == 0 || header_len > PIPELINE_MAX_HEADER_BYTES {
            return Err(PipelineContractError::HeaderTooLarge(header_len));
        }
        let payload_start = 8_usize
            .checked_add(header_len)
            .ok_or(PipelineContractError::MalformedActivationFrame)?;
        if payload_start > bytes.len() {
            return Err(PipelineContractError::MalformedActivationFrame);
        }
        let header = serde_json::from_slice::<ActivationFrameHeader>(&bytes[8..payload_start])
            .map_err(|error| PipelineContractError::HeaderDeserialization(error.to_string()))?;
        if header.payload_bytes > maximum_payload_bytes {
            return Err(PipelineContractError::PayloadLimitExceeded {
                limit: maximum_payload_bytes,
                actual: header.payload_bytes,
            });
        }
        let payload = bytes[payload_start..].to_vec();
        let frame = Self { header, payload };
        frame.validate(topology)?;
        Ok(frame)
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

/// Per-rank replay, ordering, cancellation, and token-offset fence.
///
/// Every rank keeps its own ledger for the active immutable generation. A
/// request ID may never be resurrected after cancellation or completion.
#[derive(Debug, Default)]
pub struct PipelineRequestLedger {
    active: BTreeMap<u64, RequestCursor>,
    closed: BTreeSet<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestCursor {
    next_sequence: u64,
    next_token_offset: u64,
}

impl PipelineRequestLedger {
    /// Admit or validate the next prefill/decode work item for a request.
    pub fn begin_step(
        &mut self,
        request_id: u64,
        request_sequence: u64,
        token_offset: u64,
    ) -> Result<(), PipelineContractError> {
        if request_id == 0 || request_sequence == 0 {
            return Err(PipelineContractError::ZeroRequestIdentity);
        }
        if self.closed.contains(&request_id) {
            return Err(PipelineContractError::ClosedRequest(request_id));
        }
        let cursor = self.active.entry(request_id).or_insert(RequestCursor {
            next_sequence: 1,
            next_token_offset: 0,
        });
        if request_sequence != cursor.next_sequence {
            return Err(PipelineContractError::UnexpectedRequestSequence {
                expected: cursor.next_sequence,
                actual: request_sequence,
            });
        }
        if token_offset != cursor.next_token_offset {
            return Err(PipelineContractError::UnexpectedTokenOffset {
                expected: cursor.next_token_offset,
                actual: token_offset,
            });
        }
        Ok(())
    }

    /// Commit a successfully evaluated work item.
    pub fn commit_step(
        &mut self,
        request_id: u64,
        request_sequence: u64,
        token_count: u32,
    ) -> Result<(), PipelineContractError> {
        if token_count == 0 {
            return Err(PipelineContractError::InvalidActivationShape);
        }
        let cursor = self
            .active
            .get_mut(&request_id)
            .ok_or(PipelineContractError::UnknownRequest(request_id))?;
        if request_sequence != cursor.next_sequence {
            return Err(PipelineContractError::UnexpectedRequestSequence {
                expected: cursor.next_sequence,
                actual: request_sequence,
            });
        }
        cursor.next_sequence = cursor
            .next_sequence
            .checked_add(1)
            .ok_or(PipelineContractError::RequestSequenceOverflow)?;
        cursor.next_token_offset = cursor
            .next_token_offset
            .checked_add(u64::from(token_count))
            .ok_or(PipelineContractError::RequestSequenceOverflow)?;
        Ok(())
    }

    /// Cancel or complete a request and permanently reject same-generation replay.
    pub fn close(&mut self, request_id: u64) {
        self.active.remove(&request_id);
        self.closed.insert(request_id);
    }

    pub fn is_active(&self, request_id: u64) -> bool {
        self.active.contains_key(&request_id)
    }
}

fn validate_identity(field: &'static str, value: &str) -> Result<(), PipelineContractError> {
    if value.is_empty() || value.len() > 256 || value.chars().any(char::is_whitespace) {
        return Err(PipelineContractError::InvalidIdentity(field));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), PipelineContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PipelineContractError::InvalidPayloadDigest);
    }
    Ok(())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PipelineContractError {
    #[error("{0} must be a non-empty whitespace-free identity of at most 256 characters")]
    InvalidIdentity(&'static str),
    #[error("pipeline generation must be greater than zero")]
    ZeroGeneration,
    #[error("pipeline model must contain at least one transformer layer")]
    ZeroLayers,
    #[error("static pipeline requires between 2 and 65535 ranks, got {0}")]
    InvalidRankCount(usize),
    #[error("ranks must be dense and ordered: expected {expected}, got {actual}")]
    NonContiguousRanks { expected: u16, actual: u16 },
    #[error("a physical node identity may occur only once in the initial pipeline topology")]
    DuplicateNodeIdentity,
    #[error("rank {rank} has invalid layer range [{start}, {end})")]
    InvalidLayerRange { rank: u16, start: u32, end: u32 },
    #[error(
        "layer coverage must be contiguous: expected start {expected_start}, got {actual_start}"
    )]
    LayerCoverageGap {
        expected_start: u32,
        actual_start: u32,
    },
    #[error("pipeline layer coverage does not end at total_layers")]
    IncompleteLayerCoverage,
    #[error("rank {rank} has invalid embedding or output-head ownership")]
    InvalidEndpointOwnership { rank: u16 },
    #[error("unsupported pipeline wire version {0}")]
    UnsupportedWireVersion(u16),
    #[error("request_id and request_sequence must be greater than zero")]
    ZeroRequestIdentity,
    #[error("activation must travel to the immediately following pipeline rank")]
    InvalidActivationRoute,
    #[error("activation shape must be [1, token_count, hidden_size]")]
    InvalidActivationShape,
    #[error("activation byte length overflow")]
    ActivationSizeOverflow,
    #[error("activation byte length mismatch: expected {expected}, got {actual}")]
    ActivationSizeMismatch { expected: u64, actual: u64 },
    #[error("activation payload digest must be lowercase SHA-256")]
    InvalidPayloadDigest,
    #[error("activation payload digest does not match its header")]
    ActivationDigestMismatch,
    #[error("frame belongs to a stale or foreign cluster generation")]
    StaleOrForeignGeneration,
    #[error("unknown pipeline rank {0}")]
    UnknownRank(u16),
    #[error("request {0} is already cancelled or complete in this generation")]
    ClosedRequest(u64),
    #[error("request sequence mismatch: expected {expected}, got {actual}")]
    UnexpectedRequestSequence { expected: u64, actual: u64 },
    #[error("token offset mismatch: expected {expected}, got {actual}")]
    UnexpectedTokenOffset { expected: u64, actual: u64 },
    #[error("request {0} has not been admitted")]
    UnknownRequest(u64),
    #[error("request sequence or token offset overflow")]
    RequestSequenceOverflow,
    #[error("activation frame is malformed or truncated")]
    MalformedActivationFrame,
    #[error("activation frame header is too large: {0} bytes")]
    HeaderTooLarge(usize),
    #[error("activation payload exceeds configured limit {limit}: {actual} bytes")]
    PayloadLimitExceeded { limit: u64, actual: u64 },
    #[error("failed to serialize activation header: {0}")]
    HeaderSerialization(String),
    #[error("failed to deserialize activation header: {0}")]
    HeaderDeserialization(String),
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn topology() -> PipelineTopology {
        PipelineTopology {
            cluster_id: "cluster-a".into(),
            generation: 7,
            manifest_digest: "manifest-sha256".into(),
            model_artifact_digest: "model-sha256".into(),
            total_layers: 4,
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

    fn frame(payload: &[u8]) -> ActivationFrameHeader {
        ActivationFrameHeader {
            wire_version: PIPELINE_WIRE_VERSION,
            cluster_id: "cluster-a".into(),
            generation: 7,
            manifest_digest: "manifest-sha256".into(),
            model_artifact_digest: "model-sha256".into(),
            request_id: 11,
            request_sequence: 1,
            source_rank: 0,
            destination_rank: 1,
            layer_boundary: 2,
            token_offset: 0,
            token_count: 2,
            dtype: ActivationDtype::Float32,
            shape: vec![1, 2, 2],
            payload_bytes: payload.len() as u64,
            payload_sha256: sha256_hex(payload),
        }
    }

    #[test]
    fn contiguous_topology_and_next_rank_frame_validate() {
        let topology = topology();
        let payload = [0_u8; 16];
        let frame = frame(&payload);
        assert_eq!(topology.validate(), Ok(()));
        assert_eq!(topology.validate_frame_route(&frame), Ok(()));
        assert_eq!(frame.verify_payload(&payload), Ok(()));
    }

    #[test]
    fn topology_rejects_layer_gap() {
        let mut topology = topology();
        topology.ranks[1].layers.start = 3;
        assert_eq!(
            topology.validate(),
            Err(PipelineContractError::LayerCoverageGap {
                expected_start: 2,
                actual_start: 3,
            })
        );
    }

    #[test]
    fn frame_rejects_stale_generation_before_execution() {
        let topology = topology();
        let payload = [0_u8; 16];
        let mut frame = frame(&payload);
        frame.generation = 6;
        assert_eq!(
            topology.validate_frame_route(&frame),
            Err(PipelineContractError::StaleOrForeignGeneration)
        );
    }

    #[test]
    fn frame_rejects_corrupted_payload() {
        let payload = [0_u8; 16];
        let frame = frame(&payload);
        let mut corrupted = payload;
        corrupted[4] = 1;
        assert_eq!(
            frame.verify_payload(&corrupted),
            Err(PipelineContractError::ActivationDigestMismatch)
        );
    }

    #[test]
    fn request_ledger_fences_replay_offset_and_resurrection() {
        let mut ledger = PipelineRequestLedger::default();
        assert_eq!(ledger.begin_step(7, 1, 0), Ok(()));
        assert_eq!(ledger.commit_step(7, 1, 4), Ok(()));
        assert_eq!(
            ledger.begin_step(7, 1, 0),
            Err(PipelineContractError::UnexpectedRequestSequence {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            ledger.begin_step(7, 2, 3),
            Err(PipelineContractError::UnexpectedTokenOffset {
                expected: 4,
                actual: 3,
            })
        );
        assert_eq!(ledger.begin_step(7, 2, 4), Ok(()));
        ledger.close(7);
        assert_eq!(
            ledger.begin_step(7, 2, 4),
            Err(PipelineContractError::ClosedRequest(7))
        );
    }

    #[test]
    fn activation_binary_frame_round_trips_and_enforces_limit() {
        let topology = topology();
        let payload = [0_u8; 16];
        let frame = ActivationFrame {
            header: frame(&payload),
            payload: payload.to_vec(),
        };
        let encoded = frame.encode(&topology).expect("frame should encode");
        assert_eq!(
            ActivationFrame::decode(&encoded, &topology, 16).expect("frame should decode"),
            frame
        );
        assert!(matches!(
            ActivationFrame::decode(&encoded, &topology, 15),
            Err(PipelineContractError::PayloadLimitExceeded {
                limit: 15,
                actual: 16
            })
        ));
    }
}
