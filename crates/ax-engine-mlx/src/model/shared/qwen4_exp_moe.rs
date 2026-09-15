//! Flash Next MoE delta, using AX's existing gathered expert projections.
//!
//! Expert projections come in two shapes: [`Qwen4ExpExpertWeights::Resident`]
//! (already-loaded `QuantizedWeight`s, matching the pre-streaming behavior)
//! or [`Qwen4ExpExpertWeights::Streamed`] (paged in per layer from
//! [`crate::expert_stream::ExpertLayerSource`] on every forward call). The
//! router, shared-expert, routing, and gather math are identical for both.

use mlx_sys::ops::silu;
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, concatenate, contiguous, divide, expand_dims_axes, multiply,
    reshape, sigmoid, slice, softmax, sum_axis, take, try_eval,
};

use super::mlp::top_k_by_argpartition;
use super::qwen4_exp_residual::validate_projection;
use super::utils::{ProjectionBatchPolicy, qw_gather, qw_with_policy, squeeze_switch_singleton};
use crate::expert_stream::{ExpertLayerSource, LayerExpertStack};
use crate::weights::QuantizedWeight;

/// Expert projections already resident in unified memory.
pub(crate) struct Qwen4ExpResidentExperts {
    pub gate: QuantizedWeight,
    pub up: QuantizedWeight,
    pub down: QuantizedWeight,
}

/// Either resident expert projections, or a handle that pages this layer's
/// expert stack in on demand. Streamed layers never allocate a placeholder
/// stack and are not paged at construction time.
pub(crate) enum Qwen4ExpExpertWeights {
    Resident(Box<Qwen4ExpResidentExperts>),
    Streamed(std::sync::Arc<ExpertLayerSource>),
}

pub(crate) struct Qwen4ExpMoeWeights {
    pub router: QuantizedWeight,
    pub experts: Qwen4ExpExpertWeights,
    pub shared_gate: QuantizedWeight,
    pub shared_up: QuantizedWeight,
    pub shared_down: QuantizedWeight,
    pub shared_router: QuantizedWeight,
}

pub(crate) struct Qwen4ExpMoe {
    hidden: i32,
    /// Per-expert intermediate width, kept for deferred validation of paged
    /// streamed stacks (resident stacks validate this at construction).
    intermediate: i32,
    experts: usize,
    top_k: usize,
    normalize_top_k: bool,
    weights: Qwen4ExpMoeWeights,
    selected_decode: bool,
    selected_prefill: bool,
}

/// A resolved streamed expert stack: split (or already-split) gate/up plus
/// down, validated against this MoE's geometry.
struct ResolvedStreamedExperts {
    gate: QuantizedWeight,
    up: QuantizedWeight,
    down: QuantizedWeight,
}

/// Expert projections ready to feed the gather math for one forward call.
enum ResolvedExperts<'a> {
    Resident(&'a Qwen4ExpResidentExperts),
    Streamed(Box<ResolvedStreamedExperts>),
}

impl ResolvedExperts<'_> {
    fn gate(&self) -> &QuantizedWeight {
        match self {
            Self::Resident(r) => &r.gate,
            Self::Streamed(r) => &r.gate,
        }
    }

    fn up(&self) -> &QuantizedWeight {
        match self {
            Self::Resident(r) => &r.up,
            Self::Streamed(r) => &r.up,
        }
    }

    fn down(&self) -> &QuantizedWeight {
        match self {
            Self::Resident(r) => &r.down,
            Self::Streamed(r) => &r.down,
        }
    }

    fn is_streamed(&self) -> bool {
        matches!(self, Self::Streamed(_))
    }
}

impl Qwen4ExpMoe {
    pub(crate) fn new(
        hidden: usize,
        intermediate: usize,
        shared_intermediate: usize,
        experts: usize,
        top_k: usize,
        normalize_top_k: bool,
        weights: Qwen4ExpMoeWeights,
    ) -> Result<Self, String> {
        if [hidden, intermediate, shared_intermediate, experts, top_k]
            .iter()
            .any(|&d| d == 0 || d > i32::MAX as usize)
            || top_k > experts
        {
            return Err("invalid qwen4_exp MoE dimensions".into());
        }
        let (h, i, s) = (
            hidden as i32,
            intermediate as i32,
            shared_intermediate as i32,
        );
        for (name, projection, out, input) in [
            ("MoE router", &weights.router, experts as i32, h),
            ("MoE shared gate", &weights.shared_gate, s, h),
            ("MoE shared up", &weights.shared_up, s, h),
            ("MoE shared down", &weights.shared_down, h, s),
            ("MoE shared router", &weights.shared_router, 1, h),
        ] {
            validate_projection(name, projection, out, input).map_err(|e| e.to_string())?;
        }
        // Streamed experts are validated lazily on each forward: the pack is
        // not paged in (and no placeholder is allocated) at construction.
        if let Qwen4ExpExpertWeights::Resident(resident) = &weights.experts {
            for (name, projection, out, input) in [
                ("MoE expert gate", &resident.gate, i, h),
                ("MoE expert up", &resident.up, i, h),
                ("MoE expert down", &resident.down, h, i),
            ] {
                validate_expert(name, projection, experts as i32, out, input)?;
            }
        }
        Ok(Self {
            hidden: h,
            intermediate: i,
            experts,
            top_k,
            normalize_top_k,
            weights,
            selected_decode: std::env::var_os("AX_MLX_FLASH_NEXT_SELECTED_EXPERTS")
                .is_some_and(|v| v == "1"),
            selected_prefill: std::env::var_os("AX_MLX_FLASH_NEXT_SELECTED_PREFILL")
                .is_some_and(|v| v == "1"),
        })
    }

    /// Resolve this layer's expert stack into split gate/up/down, validated
    /// against this MoE's geometry. Rejects mixed/partial gate-up layouts and
    /// any dense linear bias on a streamed expert projection.
    fn resolve_streamed_experts(
        &self,
        stack: LayerExpertStack,
    ) -> Result<ResolvedStreamedExperts, String> {
        self.resolve_streamed_experts_count(stack, self.experts)
    }

    fn resolve_streamed_experts_count(
        &self,
        stack: LayerExpertStack,
        count: usize,
    ) -> Result<ResolvedStreamedExperts, String> {
        let down = stack.down_exps.ok_or_else(|| {
            "qwen4_exp streamed MoE stack is missing the down projection".to_string()
        })?;
        if down.linear_bias.is_some() {
            return Err(
                "qwen4_exp streamed MoE down projection must not carry a dense linear bias"
                    .to_string(),
            );
        }
        let (gate, up) = match (stack.gate_up_exps_packed, stack.gate_exps, stack.up_exps) {
            (Some(packed), None, None) => {
                if packed.linear_bias.is_some() {
                    return Err(
                        "qwen4_exp streamed MoE gate_up projection must not carry a dense linear bias"
                            .to_string(),
                    );
                }
                crate::weights::qwen4_exp::split_packed_expert_gate_up(
                    &packed,
                    self.intermediate,
                    "streamed expert gate_up",
                )
                .map_err(|e| e.to_string())?
            }
            (None, Some(gate), Some(up)) => {
                if gate.linear_bias.is_some() || up.linear_bias.is_some() {
                    return Err(
                        "qwen4_exp streamed MoE gate/up projections must not carry a dense linear bias"
                            .to_string(),
                    );
                }
                (gate, up)
            }
            _ => {
                return Err(
                    "qwen4_exp streamed MoE stack has a mixed or partial gate/up expert layout"
                        .to_string(),
                );
            }
        };
        let count = count as i32;
        validate_expert(
            "streamed MoE expert gate",
            &gate,
            count,
            self.intermediate,
            self.hidden,
        )?;
        validate_expert(
            "streamed MoE expert up",
            &up,
            count,
            self.intermediate,
            self.hidden,
        )?;
        validate_expert(
            "streamed MoE expert down",
            &down,
            count,
            self.hidden,
            self.intermediate,
        )?;
        Ok(ResolvedStreamedExperts { gate, up, down })
    }

    fn resolve_experts(&self) -> Result<ResolvedExperts<'_>, String> {
        match &self.weights.experts {
            Qwen4ExpExpertWeights::Resident(resident) => Ok(ResolvedExperts::Resident(resident)),
            Qwen4ExpExpertWeights::Streamed(source) => {
                let stack = source.stack().map_err(|e| e.to_string())?;
                self.resolve_streamed_experts(stack)
                    .map(Box::new)
                    .map(ResolvedExperts::Streamed)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn enable_selected_decode_for_test(&mut self) {
        self.selected_decode = true;
    }

    #[cfg(test)]
    pub(crate) fn enable_selected_prefill_for_test(&mut self) {
        self.selected_prefill = true;
    }

    fn validate_input(&self, input: &MlxArray) -> Result<Vec<i32>, String> {
        let shape = input.shape();
        if shape.len() != 3
            || shape[0] <= 0
            || shape[1] <= 0
            || shape[2] != self.hidden
            || !matches!(
                input.dtype(),
                MlxDtype::Float32 | MlxDtype::Float16 | MlxDtype::Bfloat16
            )
        {
            return Err(format!(
                "invalid qwen4_exp MoE input {:?} {:?}",
                shape,
                input.dtype()
            ));
        }
        Ok(shape)
    }

    pub(crate) fn forward(
        &self,
        input: &MlxArray,
        policy: ProjectionBatchPolicy,
    ) -> Result<MlxArray, String> {
        let shape = self.validate_input(input)?;
        if self.selected_decode
            && shape[0] == 1
            && (shape[1] == 1 || (self.selected_prefill && policy == ProjectionBatchPolicy::Shared))
            && let Qwen4ExpExpertWeights::Streamed(source) = &self.weights.experts
            && let Some(output) = self.forward_selected(input, policy, source)?
        {
            return Ok(output);
        }
        let experts = self.resolve_experts()?;
        #[cfg(test)]
        {
            let mut arrays = Vec::new();
            for projection in [experts.gate(), experts.up(), experts.down()] {
                arrays.push(&projection.weight);
                arrays.extend(projection.scales.iter());
                arrays.extend(projection.biases.iter());
            }
            crate::model::qwen4_exp::profiling::mark("expert_resolve", &arrays);
        }
        let output = self.forward_with_experts(input, policy, &experts)?;
        // Streamed layers must release their pending expert graph references
        // before the next layer resolves a (possibly different) paged stack,
        // rather than relying only on the pager's LRU eviction.
        if experts.is_streamed() {
            try_eval(&[&output])
                .map_err(|e| format!("qwen4_exp streamed MoE output eval failed: {e}"))?;
        }
        #[cfg(test)]
        crate::model::qwen4_exp::profiling::mark("moe_compute", &[&output]);
        Ok(output)
    }

    fn forward_with_experts(
        &self,
        input: &MlxArray,
        policy: ProjectionBatchPolicy,
        experts: &ResolvedExperts<'_>,
    ) -> Result<MlxArray, String> {
        let shape = self.validate_input(input)?;
        // RowExact keeps verification projections independent of chunk width.
        if matches!(policy, ProjectionBatchPolicy::RowExact) && shape[1] > 1 {
            let mut outputs = Vec::with_capacity(shape[1] as usize);
            for token in 0..shape[1] {
                let row = slice(
                    input,
                    &[0, token, 0],
                    &[shape[0], token + 1, self.hidden],
                    &[1, 1, 1],
                    None,
                );
                outputs.push(self.forward_with_experts(&row, policy, experts)?);
            }
            return Ok(concatenate(&outputs.iter().collect::<Vec<_>>(), 1, None));
        }
        let (indices, routing) = self.route(input, policy);
        self.forward_routed(input, policy, experts, &indices, &routing)
    }

    fn route(&self, input: &MlxArray, policy: ProjectionBatchPolicy) -> (MlxArray, MlxArray) {
        let w = &self.weights;
        let logits = qw_with_policy(input, &w.router, policy);
        let probabilities = softmax(&astype(&logits, MlxDtype::Float32, None), -1, None);
        let (indices, routing) =
            top_k_by_argpartition(&probabilities, self.experts, self.top_k, false);
        let routing = if self.normalize_top_k {
            divide(&routing, &sum_axis(&routing, -1, true, None), None)
        } else {
            routing
        };
        let routing = astype(&routing, logits.dtype(), None);
        (indices, routing)
    }

    fn forward_selected(
        &self,
        input: &MlxArray,
        policy: ProjectionBatchPolicy,
        source: &ExpertLayerSource,
    ) -> Result<Option<MlxArray>, String> {
        let (indices, routing) = self.route(input, policy);
        try_eval(&[&indices, &routing])
            .map_err(|e| format!("qwen4_exp selected router eval failed: {e}"))?;
        let host_indices = contiguous(&astype(&indices, MlxDtype::Uint32, None), None);
        try_eval(&[&host_indices])
            .map_err(|e| format!("qwen4_exp selected indices eval failed: {e}"))?;
        let (selected, remapped) = compact_expert_ids(host_indices.data_u32(), self.experts)?;
        let stack = if input.shape()[1] == 1 {
            source
                .selected_stack(&selected)
                .map_err(|e| e.to_string())?
        } else {
            let Some(stack) = source
                .selected_stack_if_fits(&selected)
                .map_err(|e| e.to_string())?
            else {
                return Ok(None);
            };
            stack
        };
        let experts = ResolvedExperts::Streamed(Box::new(
            self.resolve_streamed_experts_count(stack, selected.len())?,
        ));
        let indices = astype(
            &MlxArray::from_raw_data(
                remapped.as_ptr().cast(),
                std::mem::size_of_val(remapped.as_slice()),
                &indices.shape(),
                MlxDtype::Uint32,
            ),
            indices.dtype(),
            None,
        );
        let output = self.forward_routed(input, policy, &experts, &indices, &routing)?;
        try_eval(&[&output])
            .map_err(|e| format!("qwen4_exp selected MoE output eval failed: {e}"))?;
        Ok(Some(output))
    }

    fn forward_routed(
        &self,
        input: &MlxArray,
        policy: ProjectionBatchPolicy,
        experts: &ResolvedExperts<'_>,
        indices: &MlxArray,
        routing: &MlxArray,
    ) -> Result<MlxArray, String> {
        let shape = self.validate_input(input)?;
        let w = &self.weights;
        let expanded = expand_dims_axes(input, &[-2, -3], None);
        let gate = qw_gather(&expanded, experts.gate(), indices, false);
        let up = qw_gather(&expanded, experts.up(), indices, false);
        let activated = multiply(&silu(&gate, None), &up, None);
        let down = squeeze_switch_singleton(&qw_gather(&activated, experts.down(), indices, false));
        let weighted = multiply(&down, &expand_dims_axes(routing, &[-1], None), None);
        let routed = sum_axis(&weighted, -2, false, None);
        let shared = multiply(
            &silu(&qw_with_policy(input, &w.shared_gate, policy), None),
            &qw_with_policy(input, &w.shared_up, policy),
            None,
        );
        let shared = qw_with_policy(&shared, &w.shared_down, policy);
        let shared = multiply(
            &shared,
            &sigmoid(&qw_with_policy(input, &w.shared_router, policy), None),
            None,
        );
        Ok(reshape(&add(&routed, &shared, None), &shape, None))
    }
}

fn compact_expert_ids(ids: &[u32], experts: usize) -> Result<(Vec<u64>, Vec<u32>), String> {
    if ids.is_empty() || ids.iter().any(|&id| id as usize >= experts) {
        return Err("qwen4_exp selected router IDs are empty or out of bounds".into());
    }
    let mut selected = Vec::new();
    let mut remapped = Vec::with_capacity(ids.len());
    for &id in ids {
        let id = u64::from(id);
        let position = selected
            .iter()
            .position(|&seen| seen == id)
            .unwrap_or_else(|| {
                selected.push(id);
                selected.len() - 1
            });
        remapped.push(position as u32);
    }
    Ok((selected, remapped))
}

fn validate_expert(
    name: &'static str,
    projection: &QuantizedWeight,
    count: i32,
    output: i32,
    input: i32,
) -> Result<(), String> {
    for tensor in [&projection.weight]
        .into_iter()
        .chain(projection.scales.iter())
        .chain(projection.biases.iter())
    {
        if tensor.shape().len() != 3 || tensor.shape()[0] != count {
            return Err(format!(
                "qwen4_exp {name}: expected {count} expert matrices, got {:?}",
                tensor.shape()
            ));
        }
    }
    let first = reshape(&crate::attention_mask::scalar_i32(0), &[], None);
    let mut matrix = projection.clone();
    matrix.weight = take(&projection.weight, &first, 0, None);
    matrix.scales = projection.scales.as_ref().map(|a| take(a, &first, 0, None));
    matrix.biases = projection.biases.as_ref().map(|a| take(a, &first, 0, None));
    validate_projection(name, &matrix, output, input).map_err(|e| e.to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_sys::{contiguous, eval};
    use serde_json::Value;

    #[test]
    fn compact_ids_preserve_slots_and_duplicate_contributions() {
        assert_eq!(
            compact_expert_ids(&[5, 2, 5, 2], 8).unwrap(),
            (vec![5, 2], vec![0, 1, 0, 1])
        );
        let ids = [7, 1, 7, 3, 1, 5, 7, 1];
        let (union, remapped) = compact_expert_ids(&ids, 8).unwrap();
        assert_eq!(union, vec![7, 1, 3, 5]);
        assert_eq!(remapped, vec![0, 1, 0, 2, 1, 3, 0, 1]);
        for (original, compact) in ids.iter().zip(remapped) {
            assert_eq!(u64::from(*original), union[compact as usize]);
        }
        assert!(compact_expert_ids(&[1, 8], 8).is_err());
        assert!(compact_expert_ids(&[], 8).is_err());
    }

    #[test]
    fn compact_experts_keep_original_weighted_reduction_slots() {
        let (module, _) = build_resident_module();
        let resolved = module.resolve_experts().unwrap();
        let ids = [3i32, 1, 3, 2, 1, 2, 1, 1];
        let indices = MlxArray::from_raw_data(ids.as_ptr().cast(), 32, &[1, 2, 4], MlxDtype::Int32);
        let selected = [3i32, 1, 2];
        let selected = MlxArray::from_raw_data(selected.as_ptr().cast(), 12, &[3], MlxDtype::Int32);
        let gather = |projection: &QuantizedWeight| {
            let mut compact = projection.clone();
            compact.weight = take(&projection.weight, &selected, 0, None);
            compact.scales = projection
                .scales
                .as_ref()
                .map(|a| take(a, &selected, 0, None));
            compact.biases = projection
                .biases
                .as_ref()
                .map(|a| take(a, &selected, 0, None));
            compact
        };
        let compact = ResolvedExperts::Streamed(Box::new(ResolvedStreamedExperts {
            gate: gather(resolved.gate()),
            up: gather(resolved.up()),
            down: gather(resolved.down()),
        }));
        let remap = [0i32, 1, 0, 2, 1, 2, 1, 1];
        let remap = MlxArray::from_raw_data(remap.as_ptr().cast(), 32, &[1, 2, 4], MlxDtype::Int32);
        let routing = reshape(
            &MlxArray::from_f32_slice(&[0.1, 0.25, 0.2, 0.45, 0.3, 0.4, 0.1, 0.2]),
            &[1, 2, 4],
            None,
        );
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 8.0) / 32.0).collect();
        let input = reshape(&MlxArray::from_f32_slice(&values), &[1, 2, 16], None);
        let expected = module
            .forward_routed(
                &input,
                ProjectionBatchPolicy::Shared,
                &resolved,
                &indices,
                &routing,
            )
            .unwrap();
        let actual = module
            .forward_routed(
                &input,
                ProjectionBatchPolicy::Shared,
                &compact,
                &remap,
                &routing,
            )
            .unwrap();
        eval(&[&expected, &actual]);
        assert_eq!(actual.data_f32(), expected.data_f32());
    }

    fn array(value: &Value, shape: &[i32]) -> MlxArray {
        fn flatten(value: &Value) -> Vec<f32> {
            match value {
                Value::Array(items) => items.iter().flat_map(flatten).collect(),
                _ => vec![value.as_f64().unwrap() as f32],
            }
        }
        let data = flatten(value);
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            shape,
            MlxDtype::Float32,
        )
    }

    /// The oracle fixture's geometry: hidden=16, intermediate=8, experts=4.
    /// Shared with the resident-experts test and the deferred-resolution
    /// component tests, which only need the router/shared/geometry side of
    /// construction and do not depend on which expert weights are wired in.
    fn build_resident_module() -> (Qwen4ExpMoe, Value) {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/flash_next/moe.json"))
                .unwrap();
        let weights = &fixture["weights"];
        let dense = |name: &str, shape: &[i32]| {
            QuantizedWeight::new(array(&weights[name], shape), None, None)
        };
        let packed = array(&weights["experts.gate_up_proj"], &[4, 16, 16]);
        let module = Qwen4ExpMoe::new(
            16,
            8,
            8,
            4,
            2,
            true,
            Qwen4ExpMoeWeights {
                router: dense("gate.weight", &[4, 16]),
                experts: Qwen4ExpExpertWeights::Resident(Box::new(Qwen4ExpResidentExperts {
                    gate: QuantizedWeight::new(
                        slice(&packed, &[0, 0, 0], &[4, 8, 16], &[1, 1, 1], None),
                        None,
                        None,
                    ),
                    up: QuantizedWeight::new(
                        slice(&packed, &[0, 8, 0], &[4, 16, 16], &[1, 1, 1], None),
                        None,
                        None,
                    ),
                    down: dense("experts.down_proj", &[4, 16, 8]),
                })),
                shared_gate: dense("shared_expert.gate_proj.weight", &[8, 16]),
                shared_up: dense("shared_expert.up_proj.weight", &[8, 16]),
                shared_down: dense("shared_expert.down_proj.weight", &[16, 8]),
                shared_router: dense("shared_expert_gate.weight", &[1, 16]),
            },
        )
        .unwrap();
        (module, fixture)
    }

    fn zeros(shape: &[i32]) -> MlxArray {
        let count: usize = shape.iter().map(|&d| d as usize).product();
        let data = vec![0.0f32; count];
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            shape,
            MlxDtype::Float32,
        )
    }

    #[test]
    fn packed_integer_experts_without_scales_cannot_use_dense_math() {
        let values = [0u32; 4 * 8 * 2];
        let packed = MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values.as_slice()),
            &[4, 8, 2],
            MlxDtype::Uint32,
        );
        let projection = QuantizedWeight::new(packed, None, None);
        assert!(validate_expert("missing scales", &projection, 4, 8, 16).is_err());
        // Even a forged dense-width shape must reject integer storage.
        assert!(validate_expert("integer dense", &projection, 4, 8, 2).is_err());
    }

    #[test]
    fn moe_matches_official_oracle_for_batch_and_row_exact() {
        let (module, fixture) = build_resident_module();
        let input = array(&fixture["input"], &[2, 5, 16]);
        let expected = array(&fixture["output"], &[2, 5, 16]);
        eval(&[&expected]);
        for policy in [
            ProjectionBatchPolicy::Shared,
            ProjectionBatchPolicy::RowExact,
        ] {
            let actual = contiguous(&module.forward(&input, policy).unwrap(), None);
            eval(&[&actual]);
            assert_eq!(actual.shape(), expected.shape());
            for (index, (&a, &b)) in actual
                .data_f32()
                .iter()
                .zip(expected.data_f32())
                .enumerate()
            {
                assert!((a - b).abs() < 2e-6, "MoE {policy:?}[{index}]: {a} != {b}");
            }
        }
    }

    #[test]
    fn resolve_streamed_experts_rejects_missing_down() {
        let (module, _fixture) = build_resident_module();
        let stack = LayerExpertStack {
            gate_up_exps_packed: Some(QuantizedWeight::new(zeros(&[4, 16, 16]), None, None)),
            ..Default::default()
        };
        let error = module
            .resolve_streamed_experts(stack)
            .err()
            .expect("stack without a down projection must be rejected");
        assert!(
            error.contains("down projection"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn resolve_streamed_experts_rejects_mixed_gate_up_layout() {
        let (module, _fixture) = build_resident_module();
        let stack = LayerExpertStack {
            gate_up_exps_packed: Some(QuantizedWeight::new(zeros(&[4, 16, 16]), None, None)),
            gate_exps: Some(QuantizedWeight::new(zeros(&[4, 8, 16]), None, None)),
            down_exps: Some(QuantizedWeight::new(zeros(&[4, 16, 8]), None, None)),
            ..Default::default()
        };
        let error = module
            .resolve_streamed_experts(stack)
            .err()
            .expect("packed and split gate/up together must be rejected");
        assert!(
            error.contains("mixed or partial"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn streamed_forward_propagates_missing_layer_error() {
        let manifest_json = serde_json::json!({
            "schema_version": "axquant.expert-stream.v1",
            "generated_by": "test",
            "required": true,
            "mode": "layer-stack",
            "num_experts": 4,
            "experts_per_tok": 2,
            "estimated_resident_bytes": 1,
            "estimated_full_resident_bytes": 1,
            "estimated_max_layer_expert_bytes": 1,
            "resident_roles": [],
            "streamed_roles": ["expert"],
            "tensors": [{
                "name": "model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
                "file": "experts.safetensors",
                "layer": 0,
                "proj": "gate_up",
                "expert_axis": 0,
                "num_experts": 4,
                "bits": 2,
                "group_size": 64
            }]
        });
        let manifest = crate::expert_stream::ExpertStreamManifest::parse(
            &serde_json::to_vec(&manifest_json).unwrap(),
        )
        .unwrap();
        // Layer 1 has no entry in the manifest, so paging must fail before
        // any shard is touched on disk.
        let pager = std::sync::Arc::new(crate::expert_stream::ExpertStackPager::new(
            std::sync::Arc::new(manifest),
            std::path::PathBuf::from("/nonexistent"),
            1,
        ));
        let source = std::sync::Arc::new(ExpertLayerSource::new(pager, 1));

        let fixture: Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/flash_next/moe.json"))
                .unwrap();
        let weights = &fixture["weights"];
        let dense = |name: &str, shape: &[i32]| {
            QuantizedWeight::new(array(&weights[name], shape), None, None)
        };
        let module = Qwen4ExpMoe::new(
            16,
            8,
            8,
            4,
            2,
            true,
            Qwen4ExpMoeWeights {
                router: dense("gate.weight", &[4, 16]),
                experts: Qwen4ExpExpertWeights::Streamed(source),
                shared_gate: dense("shared_expert.gate_proj.weight", &[8, 16]),
                shared_up: dense("shared_expert.up_proj.weight", &[8, 16]),
                shared_down: dense("shared_expert.down_proj.weight", &[16, 8]),
                shared_router: dense("shared_expert_gate.weight", &[1, 16]),
            },
        )
        .unwrap();

        let invalid = module
            .forward(&zeros(&[1, 1, 15]), ProjectionBatchPolicy::Shared)
            .expect_err("invalid input must fail before resolving a missing layer");
        assert!(invalid.contains("invalid qwen4_exp MoE input"));
        let input = zeros(&[1, 1, 16]);
        let error = module
            .forward(&input, ProjectionBatchPolicy::Shared)
            .expect_err("layer absent from the manifest must fail closed");
        assert!(
            error.contains("layer 1"),
            "unexpected error message: {error}"
        );
    }
}
