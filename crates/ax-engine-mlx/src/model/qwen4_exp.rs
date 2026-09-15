//! Flash Next trunk and transactional request state.

#[cfg(test)]
use mlx_sys::eval;
use std::io::Read;

#[cfg(test)]
pub(crate) mod profiling;

use mlx_sys::{MlxArray, MlxDtype, add, astype, reshape};

use super::shared::qwen4_exp_attention::Qwen4ExpAttentionCache;
use super::shared::qwen4_exp_gdn::Qwen4ExpGdnState;
use super::shared::qwen4_exp_ple::Qwen4ExpPleConvState;
use super::shared::utils::{ProjectionBatchPolicy, qw_with_policy};
use crate::kv_cache::MlxKVCacheSerializeError;
use crate::qwen4_exp_ngram::NgramHistory;
use crate::qwen4_exp_qsa::QsaIndexKeyCache;
use crate::weights::qwen4_exp::{Qwen4ExpAttentionBranch, Qwen4ExpWeights};

#[derive(Clone)]
enum AttentionState {
    Gdn(Option<Qwen4ExpGdnState>),
    Qsa(Qwen4ExpAttentionCache),
}

#[derive(Clone)]
struct PleState {
    conv: Option<Qwen4ExpPleConvState>,
    history: NgramHistory,
}

#[derive(Clone)]
struct LayerState {
    attention: AttentionState,
    ple: Option<PleState>,
}

// Bounded per-layer state codec. `MlxKVCache` owns the wire header, the
// layer loop, and the actual tensor byte encoding; it hands this module one
// tensor at a time through `write_tensor` / `read_tensor` callbacks so this
// file never touches raw tensor bytes. Everything below only encodes which
// branch a layer is, whether it carries state yet, and (for PLE) the recent
// token history — never shared weights or n-gram table rows.
const ATTENTION_TAG_GDN: u8 = 0;
const ATTENTION_TAG_QSA: u8 = 1;
const TAG_ABSENT: u8 = 0;
const TAG_PRESENT: u8 = 1;
/// Generous bound on the recent-token history a PLE layer can carry;
/// real checkpoints use `ngram_size - 1` entries (order 2-4 in practice).
/// Anything past this is a corrupted or hostile payload, not a real model.
const MAX_NGRAM_HISTORY: usize = 1024;

enum SerializedAttention {
    Gdn {
        populated: Option<(MlxArray, MlxArray)>,
    },
    Qsa {
        keys: Option<MlxArray>,
        values: Option<MlxArray>,
        index_keys: Option<MlxArray>,
    },
}

struct SerializedPle {
    conv: Option<MlxArray>,
    history: Vec<u32>,
}

/// One layer's checkpoint state, decoded from the wire format but not yet
/// checked against a concrete model. [`Qwen4ExpState::from_serialized_layers`]
/// assembles a full state from these; [`Qwen4ExpState::rebind_for_model`]
/// later checks every tensor against the model that will adopt it.
pub(crate) struct Qwen4ExpSerializedLayer {
    owner: u64,
    attention: SerializedAttention,
    ple: Option<SerializedPle>,
}

fn read_u8(reader: &mut dyn Read) -> Result<u8, MlxKVCacheSerializeError> {
    let mut buf = [0u8; 1];
    reader
        .read_exact(&mut buf)
        .map_err(|_| MlxKVCacheSerializeError::UnexpectedEof)?;
    Ok(buf[0])
}

fn read_u32(reader: &mut dyn Read) -> Result<u32, MlxKVCacheSerializeError> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|_| MlxKVCacheSerializeError::UnexpectedEof)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(reader: &mut dyn Read) -> Result<u64, MlxKVCacheSerializeError> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|_| MlxKVCacheSerializeError::UnexpectedEof)?;
    Ok(u64::from_le_bytes(buf))
}

/// A request checkpoint is a cheap clone of immutable tensor handles.
/// Rejected speculative tokens are discarded by restoring the checkpoint
/// and replaying the accepted prefix; recurrent state cannot be sliced.
#[derive(Clone)]
pub(crate) struct Qwen4ExpState {
    owner: u64,
    position: usize,
    layers: Vec<LayerState>,
}

impl Qwen4ExpState {
    pub(crate) fn new(weights: &Qwen4ExpWeights, owner: u64) -> Self {
        Self {
            owner,
            position: 0,
            layers: weights
                .layers
                .iter()
                .map(|layer| LayerState {
                    attention: match &layer.attention {
                        Qwen4ExpAttentionBranch::Gdn(_) => AttentionState::Gdn(None),
                        Qwen4ExpAttentionBranch::Qsa(_) => {
                            AttentionState::Qsa(Qwen4ExpAttentionCache::empty())
                        }
                    },
                    ple: layer.ple.as_ref().map(|ple| PleState {
                        conv: None,
                        history: ple.layout.initial_history(),
                    }),
                })
                .collect(),
        }
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn arrays(&self) -> Vec<&MlxArray> {
        let mut result = Vec::new();
        for layer in &self.layers {
            match &layer.attention {
                AttentionState::Gdn(Some(state)) => {
                    result.push(&state.conv);
                    result.push(&state.recurrent);
                }
                AttentionState::Gdn(None) => {}
                AttentionState::Qsa(state) => {
                    result.extend(state.keys());
                    result.extend(state.values());
                    result.extend(state.index().keys());
                }
            }
            if let Some(conv) = layer.ple.as_ref().and_then(|s| s.conv.as_ref()) {
                result.push(conv.array());
            }
        }
        result
    }

    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.arrays()
            .iter()
            .fold(0usize, |n, array| n.saturating_add(array.nbytes()))
    }

    /// Tensor storage charged to the request: recurrent/PLE history is fixed
    /// size; main QSA KV and index keys grow with the logical prefix.
    pub(crate) fn account_usage(&self, usage: &mut crate::kv_cache::MlxKVCacheUsage) {
        for layer in &self.layers {
            let mut linear_bytes = 0u64;
            match &layer.attention {
                AttentionState::Gdn(Some(state)) => {
                    linear_bytes = (state.conv.nbytes() as u64)
                        .saturating_add(state.recurrent.nbytes() as u64);
                }
                AttentionState::Gdn(None) => {}
                AttentionState::Qsa(state) if state.keys().is_some() => {
                    let bytes = state
                        .keys()
                        .into_iter()
                        .chain(state.values())
                        .chain(state.index().keys())
                        .fold(0u64, |total, array| {
                            total.saturating_add(array.nbytes() as u64)
                        });
                    usage.full_attention_layers = usage.full_attention_layers.saturating_add(1);
                    usage.capacity_tokens = usage.capacity_tokens.saturating_add(self.position);
                    usage.logical_bytes = usage.logical_bytes.saturating_add(bytes);
                    usage.capacity_bytes = usage.capacity_bytes.saturating_add(bytes);
                }
                AttentionState::Qsa(_) => {}
            }
            if let Some(conv) = layer.ple.as_ref().and_then(|state| state.conv.as_ref()) {
                linear_bytes = linear_bytes.saturating_add(conv.array().nbytes() as u64);
            }
            if linear_bytes > 0 {
                usage.linear_state_layers = usage.linear_state_layers.saturating_add(1);
                usage.linear_state_bytes = usage.linear_state_bytes.saturating_add(linear_bytes);
            }
        }
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Encode layer `index` into `out`. `write_tensor` is the caller's
    /// tensor codec (shared with the rest of `MlxKVCache`'s wire format);
    /// this method only decides which tensors exist and calls it.
    pub(crate) fn write_layer(
        &self,
        index: usize,
        out: &mut Vec<u8>,
        write_tensor: fn(&mut Vec<u8>, &MlxArray),
    ) {
        let layer = &self.layers[index];
        out.extend_from_slice(&self.owner.to_le_bytes());
        match &layer.attention {
            AttentionState::Gdn(state) => {
                out.push(ATTENTION_TAG_GDN);
                match state {
                    Some(state) => {
                        out.push(TAG_PRESENT);
                        write_tensor(out, &state.conv);
                        write_tensor(out, &state.recurrent);
                    }
                    None => out.push(TAG_ABSENT),
                }
            }
            AttentionState::Qsa(cache) => {
                out.push(ATTENTION_TAG_QSA);
                match (cache.keys(), cache.values()) {
                    (Some(keys), Some(values)) => {
                        out.push(TAG_PRESENT);
                        write_tensor(out, keys);
                        write_tensor(out, values);
                    }
                    _ => out.push(TAG_ABSENT),
                }
                match cache.index().keys() {
                    Some(index_keys) => {
                        out.push(TAG_PRESENT);
                        write_tensor(out, index_keys);
                    }
                    None => out.push(TAG_ABSENT),
                }
            }
        }
        match &layer.ple {
            Some(state) => {
                out.push(TAG_PRESENT);
                match state.conv.as_ref() {
                    Some(conv) => {
                        out.push(TAG_PRESENT);
                        write_tensor(out, conv.array());
                    }
                    None => out.push(TAG_ABSENT),
                }
                let recent = state.history.recent();
                out.extend_from_slice(&(recent.len() as u32).to_le_bytes());
                for token in recent {
                    out.extend_from_slice(&token.to_le_bytes());
                }
            }
            None => out.push(TAG_ABSENT),
        }
    }

    /// Decode one layer written by [`Self::write_layer`]. Does not know the
    /// model this will be adopted for; [`Self::from_serialized_layers`] and
    /// [`Self::rebind_for_model`] check that afterward.
    pub(crate) fn read_layer(
        reader: &mut dyn Read,
        read_tensor: fn(&mut dyn Read) -> Result<MlxArray, MlxKVCacheSerializeError>,
    ) -> Result<Qwen4ExpSerializedLayer, MlxKVCacheSerializeError> {
        let owner = read_u64(reader)?;
        let attention = match read_u8(reader)? {
            ATTENTION_TAG_GDN => {
                let populated = match read_u8(reader)? {
                    TAG_PRESENT => Some((read_tensor(reader)?, read_tensor(reader)?)),
                    TAG_ABSENT => None,
                    other => return Err(MlxKVCacheSerializeError::UnknownLayerKind(other)),
                };
                SerializedAttention::Gdn { populated }
            }
            ATTENTION_TAG_QSA => {
                let (keys, values) = match read_u8(reader)? {
                    TAG_PRESENT => (Some(read_tensor(reader)?), Some(read_tensor(reader)?)),
                    TAG_ABSENT => (None, None),
                    other => return Err(MlxKVCacheSerializeError::UnknownLayerKind(other)),
                };
                let index_keys = match read_u8(reader)? {
                    TAG_PRESENT => Some(read_tensor(reader)?),
                    TAG_ABSENT => None,
                    other => return Err(MlxKVCacheSerializeError::UnknownLayerKind(other)),
                };
                SerializedAttention::Qsa {
                    keys,
                    values,
                    index_keys,
                }
            }
            other => return Err(MlxKVCacheSerializeError::UnknownLayerKind(other)),
        };
        let ple = match read_u8(reader)? {
            TAG_PRESENT => {
                let conv = match read_u8(reader)? {
                    TAG_PRESENT => Some(read_tensor(reader)?),
                    TAG_ABSENT => None,
                    other => return Err(MlxKVCacheSerializeError::UnknownLayerKind(other)),
                };
                let history_len = read_u32(reader)? as usize;
                if history_len > MAX_NGRAM_HISTORY {
                    return Err(MlxKVCacheSerializeError::BadShape(history_len));
                }
                let mut history = Vec::new();
                history
                    .try_reserve_exact(history_len)
                    .map_err(|_| MlxKVCacheSerializeError::UnexpectedEof)?;
                for _ in 0..history_len {
                    history.push(read_u32(reader)?);
                }
                Some(SerializedPle { conv, history })
            }
            TAG_ABSENT => None,
            other => return Err(MlxKVCacheSerializeError::UnknownLayerKind(other)),
        };
        Ok(Qwen4ExpSerializedLayer {
            owner,
            attention,
            ple,
        })
    }

    /// Assemble a state from decoded layers. Every layer must agree on the
    /// owner it was written under (defense in depth: the wire format lets a
    /// tampered payload splice layers from different owners together), and a
    /// QSA layer's main K/V and indexer key history must advance together
    /// (they always do on the forward path; anything else is a truncated or
    /// hand-edited payload).
    pub(crate) fn from_serialized_layers(
        position: usize,
        layers: Vec<Qwen4ExpSerializedLayer>,
    ) -> Result<Self, MlxKVCacheSerializeError> {
        let Some(first) = layers.first() else {
            return Err(MlxKVCacheSerializeError::EmptySnapshot);
        };
        let owner = first.owner;
        let mut built = Vec::new();
        built
            .try_reserve_exact(layers.len())
            .map_err(|_| MlxKVCacheSerializeError::UnexpectedEof)?;
        for (index, layer) in layers.into_iter().enumerate() {
            if layer.owner != owner {
                return Err(MlxKVCacheSerializeError::BadShape(index));
            }
            let attention = match layer.attention {
                SerializedAttention::Gdn { populated } => AttentionState::Gdn(
                    populated.map(|(conv, recurrent)| Qwen4ExpGdnState { conv, recurrent }),
                ),
                SerializedAttention::Qsa {
                    keys,
                    values,
                    index_keys,
                } => {
                    if keys.is_some() != index_keys.is_some() {
                        return Err(MlxKVCacheSerializeError::IncompleteLinearLayer(index));
                    }
                    if let (Some(keys), Some(index_keys)) = (&keys, &index_keys)
                        && keys.shape().get(1) != index_keys.shape().get(1)
                    {
                        return Err(MlxKVCacheSerializeError::IncompleteLinearLayer(index));
                    }
                    let index_cache = QsaIndexKeyCache::from_serialized(index_keys);
                    let cache = Qwen4ExpAttentionCache::from_serialized(keys, values, index_cache)
                        .map_err(|_| MlxKVCacheSerializeError::IncompleteLinearLayer(index))?;
                    AttentionState::Qsa(cache)
                }
            };
            let ple = layer.ple.map(|state| PleState {
                conv: state.conv.map(Qwen4ExpPleConvState::from_serialized),
                history: NgramHistory::from_recent(state.history),
            });
            built.push(LayerState { attention, ple });
        }
        let state = Self {
            owner,
            position,
            layers: built,
        };
        state
            .validate_snapshot(state.layer_count(), position)
            .map_err(MlxKVCacheSerializeError::FlashNextState)?;
        Ok(state)
    }

    /// Structural completeness check before this snapshot is trusted: the
    /// caller's own layer count and the token count it is adopting the
    /// snapshot for must match what was decoded, and if it claims any
    /// tokens at all then every layer's attention branch must actually
    /// carry state (the wire format cannot express which layers a specific
    /// model requires, so a payload can decode cleanly while still being a
    /// truncated or foreign snapshot).
    pub(crate) fn validate_snapshot(
        &self,
        expected_layers: usize,
        expected_tokens: usize,
    ) -> Result<(), String> {
        if self.layers.len() != expected_layers || self.layers.is_empty() {
            return Err("qwen4_exp snapshot layer count does not match the model".into());
        }
        if self.position != expected_tokens || self.position > i32::MAX as usize {
            return Err("qwen4_exp snapshot position does not match the prefix".into());
        }
        let populated = expected_tokens > 0;
        for (index, layer) in self.layers.iter().enumerate() {
            match &layer.attention {
                AttentionState::Gdn(state) => {
                    if state.is_some() != populated {
                        return Err(format!(
                            "qwen4_exp layer {index}: GDN presence does not match prefix"
                        ));
                    }
                    if let Some(state) = state
                        && (state.conv.ndim() != 3
                            || state.recurrent.ndim() != 4
                            || state.conv.shape()[0] != 1
                            || state.recurrent.shape()[0] != 1
                            || !matches!(
                                state.conv.dtype(),
                                MlxDtype::Float32 | MlxDtype::Bfloat16 | MlxDtype::Float16
                            )
                            || state.recurrent.dtype() != MlxDtype::Float32)
                    {
                        return Err(format!("qwen4_exp layer {index}: invalid GDN cache layout"));
                    }
                }
                AttentionState::Qsa(cache) => {
                    if cache.token_count().map_err(|e| e.to_string())? != expected_tokens
                        || cache.index().token_count().map_err(|e| e.to_string())?
                            != expected_tokens
                        || cache.keys().is_some() != populated
                        || cache.index().keys().is_some() != populated
                    {
                        return Err(format!(
                            "qwen4_exp layer {index}: QSA length does not match prefix"
                        ));
                    }
                    for array in cache
                        .keys()
                        .into_iter()
                        .chain(cache.values())
                        .chain(cache.index().keys())
                    {
                        if array.shape()[0] != 1
                            || !matches!(
                                array.dtype(),
                                MlxDtype::Float32 | MlxDtype::Bfloat16 | MlxDtype::Float16
                            )
                        {
                            return Err(format!(
                                "qwen4_exp layer {index}: invalid QSA cache batch or dtype"
                            ));
                        }
                    }
                }
            }
            if let Some(state) = &layer.ple {
                if state.conv.is_some() != populated
                    || state.history.recent().len() > MAX_NGRAM_HISTORY
                {
                    return Err(format!("qwen4_exp layer {index}: incomplete PLE state"));
                }
                if let Some(conv) = &state.conv
                    && (conv.array().ndim() != 3
                        || conv.array().shape()[0] != 1
                        || !matches!(
                            conv.array().dtype(),
                            MlxDtype::Float32 | MlxDtype::Bfloat16 | MlxDtype::Float16
                        ))
                {
                    return Err(format!("qwen4_exp layer {index}: invalid PLE cache layout"));
                }
            }
        }
        Ok(())
    }

    /// Validate every layer's branch kind, tensor shapes/dtypes, and (for
    /// PLE) recent-token history against `weights`, then adopt `owner`.
    /// The owner only changes if every check below passes, so a forward
    /// call against the *old* owner keeps failing until the caller commits
    /// to this rebind, and a forward call against the *new* owner keeps
    /// failing until it does. The caller (outer prefix-cache lookup) is
    /// responsible for key/integrity validation before calling this.
    pub(crate) fn rebind_for_model(
        &mut self,
        weights: &Qwen4ExpWeights,
        owner: u64,
    ) -> Result<(), String> {
        self.validate_snapshot(weights.layers.len(), self.position)?;
        // Read only lazy array metadata. This matches quantized embedding dtype
        // selection without evaluating weights or reading n-gram table rows.
        let dtype =
            super::embed_tokens(&[0], &weights.token_embedding, weights.layout.hidden_size())
                .dtype();
        for (index, (layer, weights)) in self.layers.iter().zip(&weights.layers).enumerate() {
            match (&layer.attention, &weights.attention) {
                (AttentionState::Gdn(state), Qwen4ExpAttentionBranch::Gdn(branch)) => {
                    if let Some(state) = state {
                        branch.validate_state(state, 1, dtype)?;
                    }
                }
                (AttentionState::Qsa(cache), Qwen4ExpAttentionBranch::Qsa(branch)) => {
                    cache.validate_against(
                        branch.config(),
                        branch.indexer().config().head_dim(),
                        dtype,
                    )?;
                }
                _ => {
                    return Err(format!(
                        "qwen4_exp layer {index}: attention branch differs from model"
                    ));
                }
            }
            match (&layer.ple, &weights.ple) {
                (Some(state), Some(bundle)) => {
                    bundle.layout.validate_history(&state.history)?;
                    if self.position == 0
                        && state.history.recent() != bundle.layout.initial_history().recent()
                    {
                        return Err(format!(
                            "qwen4_exp layer {index}: fresh PLE history differs from model"
                        ));
                    }
                    if let Some(conv) = &state.conv {
                        bundle.operator.validate_state(conv, 1)?;
                    }
                }
                (None, None) => {}
                _ => {
                    return Err(format!(
                        "qwen4_exp layer {index}: PLE presence differs from model"
                    ));
                }
            }
        }
        self.owner = owner;
        Ok(())
    }
}

pub(crate) struct Qwen4ExpOutput {
    /// Packed streams before the final mixer; needed by family MTP heads.
    pub stream_hidden: MlxArray,
    /// Final mixer result; no additional final RMSNorm is applied.
    pub hidden: MlxArray,
    pub logits: MlxArray,
    pub state: Qwen4ExpState,
}

/// Runs on staged state. The caller publishes `output.state` only after this
/// function succeeds; IO or evaluation failure leaves its checkpoint intact.
pub(crate) fn forward(
    weights: &Qwen4ExpWeights,
    tokens: &[u32],
    state: &Qwen4ExpState,
    owner: u64,
    policy: ProjectionBatchPolicy,
) -> Result<Qwen4ExpOutput, String> {
    if state.owner != owner || state.layers.len() != weights.layers.len() {
        return Err("qwen4_exp request state belongs to a different model".into());
    }
    let _end = state
        .position
        .checked_add(tokens.len())
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or("qwen4_exp context length exceeds tensor dimensions")?;
    let vocabulary = weights.token_embedding.weight.shape()[0] as u32;
    if tokens.is_empty() || tokens.iter().any(|&id| id >= vocabulary) {
        return Err("qwen4_exp requires nonempty in-vocabulary token IDs".into());
    }
    let embedded = super::embed_tokens(
        tokens,
        &weights.token_embedding,
        weights.layout.hidden_size(),
    );
    let hidden = weights
        .layout
        .expand(&embedded)
        .map_err(|e| e.to_string())?;
    forward_prepared(weights, tokens, hidden, state, owner, policy)
}

fn validate_prepared_input(
    weights: &Qwen4ExpWeights,
    tokens: &[u32],
    hidden: &MlxArray,
    state: &Qwen4ExpState,
    owner: u64,
) -> Result<(usize, u32), String> {
    if state.owner != owner || state.layers.len() != weights.layers.len() {
        return Err("qwen4_exp request state belongs to a different model".into());
    }
    let end = state
        .position
        .checked_add(tokens.len())
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or("qwen4_exp context length exceeds tensor dimensions")?;
    let vocabulary = weights.token_embedding.weight.shape()[0] as u32;
    if tokens.is_empty() || tokens.iter().any(|&id| id >= vocabulary) {
        return Err("qwen4_exp requires nonempty in-vocabulary token IDs".into());
    }
    if hidden.shape() != [1, tokens.len() as i32, weights.layout.packed_width() as i32]
        || !matches!(
            hidden.dtype(),
            MlxDtype::Float32 | MlxDtype::Float16 | MlxDtype::Bfloat16
        )
    {
        return Err("qwen4_exp prepared stream has invalid shape or dtype".into());
    }
    Ok((end, vocabulary))
}

/// The one-layer MTP head has no recurrent state after QSA. Absorbing a
/// committed token needs its K/V/index cache, not its discarded MoE/logits.
pub(crate) fn advance_prepared_qsa_cache(
    weights: &Qwen4ExpWeights,
    tokens: &[u32],
    hidden: MlxArray,
    state: &Qwen4ExpState,
    owner: u64,
    policy: ProjectionBatchPolicy,
) -> Result<Qwen4ExpState, String> {
    let (end, _) = validate_prepared_input(weights, tokens, &hidden, state, owner)?;
    let [layer] = weights.layers.as_slice() else {
        return Err("qwen4_exp cache-only advance requires exactly one QSA layer".into());
    };
    let Qwen4ExpAttentionBranch::Qsa(branch) = &layer.attention else {
        return Err("qwen4_exp cache-only advance requires QSA attention".into());
    };
    if layer.ple.is_some() || state.layers[0].ple.is_some() {
        return Err("qwen4_exp cache-only advance does not support PLE".into());
    }
    let AttentionState::Qsa(cache) = &state.layers[0].attention else {
        return Err("qwen4_exp cache-only QSA state mismatch".into());
    };
    let read = layer
        .attention_hc
        .read(&hidden, policy)
        .map_err(|e| e.to_string())?;
    let cache = branch
        .forward(read.branch_input(), cache, state.position, policy)
        .map_err(|e| e.to_string())?
        .into_next_state();
    let mut next = state.clone();
    next.layers[0].attention = AttentionState::Qsa(cache);
    mlx_sys::try_eval(&next.arrays())
        .map_err(|e| format!("qwen4_exp cache-only evaluation failed: {e}"))?;
    next.position = end;
    Ok(next)
}

/// Run AX-owned layers from an explicitly prepared packed residual. The MTP
/// candidate uses this boundary with its separately owned one-layer graph.
pub(crate) fn forward_prepared(
    weights: &Qwen4ExpWeights,
    tokens: &[u32],
    mut hidden: MlxArray,
    state: &Qwen4ExpState,
    owner: u64,
    policy: ProjectionBatchPolicy,
) -> Result<Qwen4ExpOutput, String> {
    let (end, vocabulary) = validate_prepared_input(weights, tokens, &hidden, state, owner)?;
    #[cfg(test)]
    profiling::mark("embedding", &[&hidden]);
    let mut next = state.clone();
    for (index, layer) in weights.layers.iter().enumerate() {
        #[cfg(test)]
        profiling::layer(index);
        let staged = &mut next.layers[index];
        match (&layer.ple, &mut staged.ple) {
            (Some(ple), Some(cache)) => {
                let lookup = ple.layout.plan(&cache.history, tokens)?;
                let rows = ple.table.gather(&lookup.rows)?;
                #[cfg(test)]
                profiling::mark("ple_rows", &[&rows]);
                let embeddings = reshape(
                    &astype(&rows, hidden.dtype(), None),
                    &[1, tokens.len() as i32, ple.embedding_width as i32],
                    None,
                );
                let result = ple
                    .operator
                    .forward(&embeddings, &hidden, cache.conv.as_ref(), policy)
                    .map_err(|e| e.to_string())?;
                hidden = add(&hidden, result.delta(), None);
                cache.conv = Some(result.next_state().clone());
                cache.history = lookup.next_history;
                #[cfg(test)]
                profiling::mark("ple_operator", &[&hidden, result.next_state().array()]);
            }
            (None, None) => {}
            _ => return Err(format!("qwen4_exp layer {index} PLE state mismatch")),
        }
        let read = layer
            .attention_hc
            .read(&hidden, policy)
            .map_err(|e| e.to_string())?;
        #[cfg(test)]
        profiling::mark("attention_hc_read", &[read.branch_input()]);
        let delta = match (&layer.attention, &mut staged.attention) {
            (Qwen4ExpAttentionBranch::Gdn(branch), AttentionState::Gdn(cache)) => {
                let (delta, new_state) =
                    branch.forward(read.branch_input(), cache.as_ref(), policy)?;
                #[cfg(test)]
                profiling::mark("gdn", &[&delta, &new_state.conv, &new_state.recurrent]);
                *cache = Some(new_state);
                delta
            }
            (Qwen4ExpAttentionBranch::Qsa(branch), AttentionState::Qsa(cache)) => {
                let result = branch
                    .forward(read.branch_input(), cache, state.position, policy)
                    .map_err(|e| e.to_string())?;
                let delta = result.delta().clone();
                *cache = result.into_next_state();
                #[cfg(test)]
                {
                    let mut arrays = vec![&delta];
                    arrays.extend(cache.keys());
                    arrays.extend(cache.values());
                    arrays.extend(cache.index().keys());
                    profiling::mark("qsa", &arrays);
                }
                delta
            }
            _ => return Err(format!("qwen4_exp layer {index} attention state mismatch")),
        };
        hidden = read.write(&delta).map_err(|e| e.to_string())?;
        #[cfg(test)]
        profiling::mark("attention_hc_write", &[&hidden]);
        let read = layer
            .mlp_hc
            .read(&hidden, policy)
            .map_err(|e| e.to_string())?;
        #[cfg(test)]
        profiling::mark("mlp_hc_read", &[read.branch_input()]);
        let delta = layer.moe.forward(read.branch_input(), policy)?;
        hidden = read.write(&delta).map_err(|e| e.to_string())?;
        #[cfg(test)]
        profiling::mark("mlp_hc_write", &[&hidden]);
    }
    let mixed = weights
        .mixer
        .collapse(&hidden, policy)
        .map_err(|e| e.to_string())?;
    #[cfg(test)]
    profiling::mark("final_mixer", &[&mixed]);
    let logits = astype(
        &qw_with_policy(&mixed, &weights.lm_head, policy),
        MlxDtype::Float32,
        None,
    );
    let logits = reshape(&logits, &[tokens.len() as i32, vocabulary as i32], None);
    let mut arrays = next.arrays();
    arrays.extend([&logits, &mixed, &hidden]);
    mlx_sys::try_eval(&arrays).map_err(|e| format!("qwen4_exp evaluation failed: {e}"))?;
    #[cfg(test)]
    profiling::mark("lm_head", &[&logits]);
    next.position = end;
    Ok(Qwen4ExpOutput {
        stream_hidden: hidden,
        hidden: mixed,
        logits,
        state: next,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use ax_engine_core::WeightSanitize;
    use ax_engine_core::convert::convert_hf_model_dir;
    use mlx_sys::contiguous;
    use std::path::PathBuf;

    fn flatten(value: &serde_json::Value, output: &mut Vec<f32>) {
        if let Some(values) = value.as_array() {
            for value in values {
                flatten(value, output);
            }
        } else {
            output.push(value.as_f64().unwrap() as f32);
        }
    }

    fn close(actual: &[f32], expected: &[f32], tolerance: f32) -> f32 {
        assert_eq!(actual.len(), expected.len());
        let error = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| {
                assert!(a.is_finite() && b.is_finite());
                (a - b).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(error < tolerance, "maximum logit error: {error}");
        error
    }

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    /// Test-local tensor codec standing in for `MlxKVCache`'s real one:
    /// widen to float32 (safe; no raw pointer access), record the original
    /// dtype, and narrow back on read. Exercises the bounded per-layer state
    /// codec in this file without needing the real wire format.
    fn test_write_tensor(out: &mut Vec<u8>, arr: &MlxArray) {
        let original_dtype = match arr.dtype() {
            MlxDtype::Float32 => 0u8,
            MlxDtype::Float16 => 1u8,
            MlxDtype::Bfloat16 => 2u8,
            other => panic!("unsupported test dtype {other:?}"),
        };
        let widened = contiguous(&astype(arr, MlxDtype::Float32, None), None);
        eval(&[&widened]);
        out.push(original_dtype);
        let shape = widened.shape();
        out.push(shape.len() as u8);
        for dim in &shape {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        let data = widened.data_f32();
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
        for value in data {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn test_read_tensor(reader: &mut dyn Read) -> Result<MlxArray, MlxKVCacheSerializeError> {
        let dtype = match read_u8(reader)? {
            0 => MlxDtype::Float32,
            1 => MlxDtype::Float16,
            2 => MlxDtype::Bfloat16,
            other => return Err(MlxKVCacheSerializeError::UnknownDtype(other)),
        };
        let ndim = read_u8(reader)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u32(reader)? as i32);
        }
        let len = read_u64(reader)? as usize;
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(f32::from_bits(read_u32(reader)?));
        }
        let widened = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &shape,
            MlxDtype::Float32,
        );
        Ok(astype(&widened, dtype, None))
    }

    #[test]
    fn bounded_layer_codec_round_trips_fresh_and_populated_state() {
        let owner = 42u64;
        let gdn_conv = array_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let gdn_recurrent = array_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 2]);
        let qsa_keys = astype(
            &array_f32(&[0.1, 0.2, 0.3, 0.4], &[1, 2, 1, 2]),
            MlxDtype::Bfloat16,
            None,
        );
        let qsa_values = astype(
            &array_f32(&[0.5, 0.6, 0.7, 0.8], &[1, 2, 1, 2]),
            MlxDtype::Bfloat16,
            None,
        );
        let qsa_index_keys = astype(
            &array_f32(&[1.1, 1.2, 1.3, 1.4], &[1, 2, 2]),
            MlxDtype::Bfloat16,
            None,
        );
        let ple_conv = array_f32(&[0.9, 0.8], &[1, 1, 2]);
        let ple_history = vec![3u32, 4u32];
        eval(&[
            &gdn_conv,
            &gdn_recurrent,
            &qsa_keys,
            &qsa_values,
            &qsa_index_keys,
            &ple_conv,
        ]);

        let populated = Qwen4ExpState {
            owner,
            position: 2,
            layers: vec![
                LayerState {
                    attention: AttentionState::Gdn(Some(Qwen4ExpGdnState {
                        conv: gdn_conv.clone(),
                        recurrent: gdn_recurrent.clone(),
                    })),
                    ple: None,
                },
                LayerState {
                    attention: AttentionState::Qsa(
                        Qwen4ExpAttentionCache::from_serialized(
                            Some(qsa_keys.clone()),
                            Some(qsa_values.clone()),
                            QsaIndexKeyCache::from_serialized(Some(qsa_index_keys.clone())),
                        )
                        .unwrap(),
                    ),
                    ple: Some(PleState {
                        conv: Some(Qwen4ExpPleConvState::from_serialized(ple_conv.clone())),
                        history: NgramHistory::from_recent(ple_history.clone()),
                    }),
                },
            ],
        };

        let fresh = Qwen4ExpState {
            owner,
            position: 0,
            layers: vec![
                LayerState {
                    attention: AttentionState::Gdn(None),
                    ple: None,
                },
                LayerState {
                    attention: AttentionState::Qsa(Qwen4ExpAttentionCache::empty()),
                    ple: Some(PleState {
                        conv: None,
                        history: NgramHistory::from_recent(vec![7]),
                    }),
                },
            ],
        };

        for state in [&populated, &fresh] {
            let mut layers = Vec::new();
            for index in 0..state.layer_count() {
                let mut buf = Vec::new();
                state.write_layer(index, &mut buf, test_write_tensor);
                let mut cursor = std::io::Cursor::new(buf);
                layers.push(Qwen4ExpState::read_layer(&mut cursor, test_read_tensor).unwrap());
            }
            let restored = Qwen4ExpState::from_serialized_layers(state.position, layers).unwrap();
            restored
                .validate_snapshot(state.layer_count(), state.position)
                .unwrap();
            assert_eq!(restored.owner, state.owner);
            assert_eq!(restored.position, state.position);
            assert_eq!(restored.bytes() > 0, state.bytes() > 0);
        }

        // Field-level fidelity check on the populated case: dtypes and
        // values must survive the round trip exactly, not just approximately.
        let mut layers = Vec::new();
        for index in 0..populated.layer_count() {
            let mut buf = Vec::new();
            populated.write_layer(index, &mut buf, test_write_tensor);
            let mut cursor = std::io::Cursor::new(buf);
            layers.push(Qwen4ExpState::read_layer(&mut cursor, test_read_tensor).unwrap());
        }
        let restored = Qwen4ExpState::from_serialized_layers(populated.position, layers).unwrap();
        match &restored.layers[0].attention {
            AttentionState::Gdn(Some(state)) => {
                assert_eq!(state.conv.dtype(), MlxDtype::Float32);
                eval(&[&state.conv, &state.recurrent]);
                assert_eq!(state.conv.data_f32(), gdn_conv.data_f32());
                assert_eq!(state.recurrent.data_f32(), gdn_recurrent.data_f32());
            }
            _ => panic!("expected populated GDN state"),
        }
        match &restored.layers[1].attention {
            AttentionState::Qsa(cache) => {
                assert_eq!(cache.keys().unwrap().dtype(), MlxDtype::Bfloat16);
                assert_eq!(cache.values().unwrap().dtype(), MlxDtype::Bfloat16);
                assert_eq!(cache.index().keys().unwrap().dtype(), MlxDtype::Bfloat16);
            }
            _ => panic!("expected QSA state"),
        }
        match &restored.layers[1].ple {
            Some(state) => assert_eq!(state.history.recent(), ple_history.as_slice()),
            None => panic!("expected ple state"),
        }
    }

    #[test]
    fn bounded_layer_codec_rejects_malformed_or_inconsistent_payloads() {
        // Unknown attention-branch tag.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.push(9);
        let mut cursor = std::io::Cursor::new(buf);
        assert!(matches!(
            Qwen4ExpState::read_layer(&mut cursor, test_read_tensor),
            Err(MlxKVCacheSerializeError::UnknownLayerKind(9))
        ));

        // Truncated payload: nothing after the owner.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        let mut cursor = std::io::Cursor::new(buf);
        assert!(matches!(
            Qwen4ExpState::read_layer(&mut cursor, test_read_tensor),
            Err(MlxKVCacheSerializeError::UnexpectedEof)
        ));

        // Unreasonable n-gram history length.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.push(ATTENTION_TAG_GDN);
        buf.push(TAG_ABSENT);
        buf.push(TAG_PRESENT);
        buf.push(TAG_ABSENT);
        buf.extend_from_slice(&1025u32.to_le_bytes());
        let mut cursor = std::io::Cursor::new(buf);
        assert!(matches!(
            Qwen4ExpState::read_layer(&mut cursor, test_read_tensor),
            Err(MlxKVCacheSerializeError::BadShape(1025))
        ));

        // Mismatched owners across layers.
        let layer_a = Qwen4ExpSerializedLayer {
            owner: 1,
            attention: SerializedAttention::Gdn { populated: None },
            ple: None,
        };
        let layer_b = Qwen4ExpSerializedLayer {
            owner: 2,
            attention: SerializedAttention::Gdn { populated: None },
            ple: None,
        };
        assert!(matches!(
            Qwen4ExpState::from_serialized_layers(0, vec![layer_a, layer_b]),
            Err(MlxKVCacheSerializeError::BadShape(1))
        ));

        // Incomplete QSA: main K/V present but the indexer key history absent.
        let keys = array_f32(&[0.0, 0.0], &[1, 1, 1, 2]);
        let values = array_f32(&[0.0, 0.0], &[1, 1, 1, 2]);
        let layer = Qwen4ExpSerializedLayer {
            owner: 1,
            attention: SerializedAttention::Qsa {
                keys: Some(keys),
                values: Some(values),
                index_keys: None,
            },
            ple: None,
        };
        assert!(matches!(
            Qwen4ExpState::from_serialized_layers(1, vec![layer]),
            Err(MlxKVCacheSerializeError::IncompleteLinearLayer(0))
        ));

        // Position and layer-count mismatches, and a snapshot that claims
        // tokens while its only layer is empty, are all caught by
        // `validate_snapshot` before a caller trusts the restored state.
        let empty_layer = Qwen4ExpState {
            owner: 1,
            position: 3,
            layers: vec![LayerState {
                attention: AttentionState::Gdn(None),
                ple: None,
            }],
        };
        assert!(empty_layer.validate_snapshot(1, 5).is_err());
        assert!(empty_layer.validate_snapshot(2, 3).is_err());
        assert!(empty_layer.validate_snapshot(1, 3).is_err());

        let populated_layer = Qwen4ExpState {
            owner: 1,
            position: 3,
            layers: vec![LayerState {
                attention: AttentionState::Gdn(Some(Qwen4ExpGdnState {
                    conv: array_f32(&[0.0], &[1, 1, 1]),
                    recurrent: array_f32(&[0.0], &[1, 1, 1, 1]),
                })),
                ple: None,
            }],
        };
        assert!(populated_layer.validate_snapshot(1, 3).is_ok());
    }

    /// Development smoke on the explicitly selected campaign host. Successful
    /// execution is not a serving, MTP, quality, or performance certification.
    #[test]
    #[ignore = "requires a real Flash Next pack and explicit prompt IDs"]
    fn campaign_pack_direct_smoke() {
        let root =
            PathBuf::from(std::env::var_os("AX_FLASH_NEXT_PACK_DIR").expect("pack directory"));
        let tokens: Vec<u32> = serde_json::from_str(
            &std::env::var("AX_FLASH_NEXT_PROMPT_IDS").expect("prompt IDs JSON"),
        )
        .unwrap();
        let mut manifest = convert_hf_model_dir(&root).unwrap();
        if let Ok(mode) = std::env::var("AX_FLASH_NEXT_WEIGHT_SANITIZE") {
            manifest.weight_sanitize =
                serde_json::from_value(serde_json::Value::String(mode)).unwrap();
        }
        let load_start = std::time::Instant::now();
        let weights = crate::weights::qwen4_exp::load(&root, &manifest).unwrap();
        let load_seconds = load_start.elapsed().as_secs_f64();
        for layer in &weights.layers {
            if let Some(ple) = &layer.ple {
                assert_eq!(ple.table.payload_bytes_read(), 0);
            }
        }
        eprintln!("Flash Next components loaded in {load_seconds:.3}s; table payload reads=0");
        let initial = Qwen4ExpState::new(&weights, 81);
        let start = std::time::Instant::now();
        let mut output = forward(
            &weights,
            &tokens,
            &initial,
            81,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let prefill_seconds = start.elapsed().as_secs_f64();
        let mut generated = Vec::new();
        let mut decode_seconds = Vec::new();
        for _ in 0..8 {
            let shape = output.logits.shape();
            let row = mlx_sys::slice(
                &output.logits,
                &[shape[0] - 1, 0],
                &[shape[0], shape[1]],
                &[1, 1],
                None,
            );
            let token = mlx_sys::argmax(&row, None);
            eval(&[&token]);
            let token = token.data_u32()[0];
            generated.push(token);
            let start = std::time::Instant::now();
            output = forward(
                &weights,
                &[token],
                &output.state,
                81,
                ProjectionBatchPolicy::Shared,
            )
            .unwrap();
            decode_seconds.push(start.elapsed().as_secs_f64());
            assert!(output.logits.data_f32().iter().all(|v| v.is_finite()));
            eprintln!(
                "Flash Next generated token {token}; position={}",
                output.state.position()
            );
        }
        let table_bytes: u64 = weights
            .layers
            .iter()
            .filter_map(|l| l.ple.as_ref())
            .map(|p| p.table.payload_bytes_read())
            .sum();
        let result = serde_json::json!({
            "qualification":false,"route":"dedicated_qwen4_exp_development_trunk",
            "weight_sanitize":manifest.weight_sanitize,"load_seconds":load_seconds,"prefill_seconds":prefill_seconds,
            "decode_seconds":decode_seconds,"generated_ids":generated,
            "table_payload_bytes":table_bytes,"peak_mlx_bytes":mlx_sys::get_peak_memory(),
            "request_state_bytes":output.state.bytes(),
        });
        eprintln!("{}", serde_json::to_string(&result).unwrap());
        if let Some(path) = std::env::var_os("AX_FLASH_NEXT_SMOKE_OUTPUT") {
            std::fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
        }
    }

    /// Generate inputs with scripts/flash_next_oracle.py at its pinned upstream
    /// commit, then explicitly run this test with AX_FLASH_NEXT_ORACLE_DIR.
    #[test]
    #[ignore = "requires generated official Flash Next oracle artifacts"]
    fn official_trunk_logits_chunks_and_checkpoint_replay() {
        let root =
            PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").expect("oracle directory"));
        let mut manifest = convert_hf_model_dir(&root).unwrap();
        // The oracle saves raw HF tensors. This tests the dedicated trunk
        // independently of the public runtime admission gate.
        manifest.weight_sanitize = WeightSanitize::HfToMlx;
        let weights = crate::weights::qwen4_exp::load(&root, &manifest).unwrap();
        for layer in &weights.layers {
            if let Some(ple) = &layer.ple {
                assert_eq!(ple.table.payload_bytes_read(), 0);
            }
        }
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("logits.json")).unwrap()).unwrap();
        let tokens: Vec<u32> = fixture["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let mut expected = Vec::new();
        flatten(&fixture["whole"], &mut expected);
        let policy = ProjectionBatchPolicy::Shared;
        let initial = Qwen4ExpState::new(&weights, 71);
        let whole = forward(&weights, &tokens, &initial, 71, policy).unwrap();
        let tolerance = match weights.token_embedding.weight.dtype() {
            MlxDtype::Float32 => 2e-5,
            MlxDtype::Bfloat16 => 1e-3,
            MlxDtype::Float16 => 1e-4,
            _ => panic!("oracle requires unquantized floating embeddings"),
        };
        let error = close(whole.logits.data_f32(), &expected, tolerance);
        assert_eq!(initial.position(), 0);
        assert_eq!(whole.state.position(), tokens.len());
        assert!(whole.state.bytes() > 0);
        for split in 1..tokens.len() {
            let prefix = forward(&weights, &tokens[..split], &initial, 71, policy).unwrap();
            let checkpoint = prefix.state.clone();
            let suffix = forward(&weights, &tokens[split..], &checkpoint, 71, policy).unwrap();
            let mut actual = prefix.logits.data_f32().to_vec();
            actual.extend_from_slice(suffix.logits.data_f32());
            close(&actual, &expected, tolerance);
            assert_eq!(checkpoint.position(), split);
            assert!(forward(&weights, &[u32::MAX], &checkpoint, 71, policy).is_err());
            assert!(forward(&weights, &tokens[split..], &checkpoint, 72, policy).is_err());
            // Abandon a speculative suffix and replay only one accepted token.
            let accepted =
                forward(&weights, &tokens[split..split + 1], &checkpoint, 71, policy).unwrap();
            if split + 1 < tokens.len() {
                let tail =
                    forward(&weights, &tokens[split + 1..], &accepted.state, 71, policy).unwrap();
                let mut replay = prefix.logits.data_f32().to_vec();
                replay.extend_from_slice(accepted.logits.data_f32());
                replay.extend_from_slice(tail.logits.data_f32());
                close(&replay, &expected, tolerance);
            }
        }
        let mut state = initial;
        let mut actual = Vec::new();
        for token in &tokens {
            let step = forward(&weights, &[*token], &state, 71, policy).unwrap();
            actual.extend_from_slice(step.logits.data_f32());
            state = step.state;
        }
        let decode_error = close(&actual, &expected, tolerance);
        eprintln!(
            "Flash Next official trunk: whole_max_error={error}, decode_max_error={decode_error}"
        );
    }

    /// Generate inputs with scripts/flash_next_oracle.py at its pinned upstream
    /// commit, then explicitly run this test with AX_FLASH_NEXT_ORACLE_DIR.
    /// Exercises the bounded per-layer state codec against a real Flash Next
    /// checkpoint: write_layer/read_layer, from_serialized_layers,
    /// validate_snapshot, and rebind_for_model under a new owner, then
    /// confirms the restored checkpoint continues generation identically.
    #[test]
    #[ignore = "requires generated official Flash Next oracle artifacts"]
    fn official_trunk_state_round_trips_through_bounded_serialization() {
        let root =
            PathBuf::from(std::env::var_os("AX_FLASH_NEXT_ORACLE_DIR").expect("oracle directory"));
        let mut manifest = convert_hf_model_dir(&root).unwrap();
        manifest.weight_sanitize = WeightSanitize::HfToMlx;
        let weights = crate::weights::qwen4_exp::load(&root, &manifest).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("logits.json")).unwrap()).unwrap();
        let tokens: Vec<u32> = fixture["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let policy = ProjectionBatchPolicy::Shared;
        let initial = Qwen4ExpState::new(&weights, 71);
        let whole = forward(&weights, &tokens, &initial, 71, policy).unwrap();

        let mut serialized_layers = Vec::new();
        for index in 0..weights.layers.len() {
            let mut buf = Vec::new();
            whole.state.write_layer(index, &mut buf, test_write_tensor);
            let mut cursor = std::io::Cursor::new(buf);
            serialized_layers
                .push(Qwen4ExpState::read_layer(&mut cursor, test_read_tensor).unwrap());
        }
        let mut restored =
            Qwen4ExpState::from_serialized_layers(whole.state.position(), serialized_layers)
                .unwrap();
        restored
            .validate_snapshot(weights.layers.len(), tokens.len())
            .unwrap();

        let new_owner = 900u64;
        let mut bad_position = restored.clone();
        bad_position.position += 1;
        assert!(bad_position.rebind_for_model(&weights, new_owner).is_err());
        assert_eq!(bad_position.owner, 71);
        let mut bad_dtype = restored.clone();
        if let AttentionState::Gdn(Some(state)) = &mut bad_dtype.layers[0].attention {
            let dtype = if state.conv.dtype() == MlxDtype::Float16 {
                MlxDtype::Float32
            } else {
                MlxDtype::Float16
            };
            state.conv = astype(&state.conv, dtype, None);
        } else {
            panic!("oracle must start with GDN");
        }
        assert!(bad_dtype.rebind_for_model(&weights, new_owner).is_err());
        assert_eq!(bad_dtype.owner, 71);
        let mut missing_ple = restored.clone();
        missing_ple
            .layers
            .iter_mut()
            .find_map(|layer| layer.ple.as_mut())
            .unwrap()
            .conv = None;
        assert!(missing_ple.rebind_for_model(&weights, new_owner).is_err());
        assert_eq!(missing_ple.owner, 71);
        // Before rebind, the restored checkpoint still belongs to the
        // session that wrote it; forward must reject any other owner,
        // including the one it is about to be rebound to.
        assert!(forward(&weights, &tokens[..1], &restored, new_owner, policy).is_err());
        restored.rebind_for_model(&weights, new_owner).unwrap();
        // After rebind, only the new owner may continue generation; the
        // original owner is now foreign.
        assert!(forward(&weights, &tokens[..1], &restored, 71, policy).is_err());

        let continued = forward(&weights, &tokens[..1], &whole.state, 71, policy).unwrap();
        let restored_continued =
            forward(&weights, &tokens[..1], &restored, new_owner, policy).unwrap();
        assert_eq!(
            continued.logits.data_f32(),
            restored_continued.logits.data_f32()
        );
    }
}
