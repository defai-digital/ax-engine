//! Flash Next MoE delta, using AX's existing gathered expert projections.
//!
//! Expert projections come in two shapes: [`Qwen4ExpExpertWeights::Resident`]
//! (already-loaded `QuantizedWeight`s, matching the pre-streaming behavior)
//! or [`Qwen4ExpExpertWeights::Streamed`] (paged in per layer from
//! [`crate::expert_stream::ExpertLayerSource`] on every forward call). The
//! router, shared-expert, routing, and gather math are identical for both.

use mlx_sys::ops::silu;
use mlx_sys::{
    MlxArray, MlxClosure, MlxDtype, add, astype, concatenate, contiguous, divide, expand_dims_axes,
    multiply, reshape, sigmoid, slice, softmax, sum_axis, take, try_eval,
};

use super::mlp::{SwitchGatherInputs, switch_gather_inputs, top_k_by_argpartition};
use super::qwen4_exp_fused::{self, Stretch};
use super::qwen4_exp_residual::validate_projection;
use super::utils::{ProjectionBatchPolicy, qw_gather, qw_with_policy, squeeze_switch_singleton};
use crate::expert_stream::{ExpertLayerSource, LayerExpertStack};
use crate::qwen4_exp_qsa::dtype_key;
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

/// Prefill chunks gather experts in expert-id order, so each expert's weights
/// are streamed once per run of rows (the layout mlx-lm's SwitchGLU uses for
/// long prompts). Decode and short verifier windows keep the original
/// unsorted gather; outputs are restored to token order before the weighted sum.
const PREFILL_EXPERT_SORT_MIN_TOKENS: i32 = 64;

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
    /// Gather experts in expert-id order for long prefill chunks.
    prefill_expert_sort: bool,
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

fn silu_projection_dtype(input: &MlxArray) -> MlxArray {
    // Official SiLU rounds once before the separate low-precision up product.
    astype(
        &silu(&astype(input, MlxDtype::Float32, None), None),
        input.dtype(),
        None,
    )
}

fn sigmoid_projection_dtype(input: &MlxArray) -> MlxArray {
    astype(
        &sigmoid(&astype(input, MlxDtype::Float32, None), None),
        input.dtype(),
        None,
    )
}

/// Closure body of the SiLU-gated expert product.
///
/// Inputs, in order: `gate`, `up`. Nothing is captured: the body reads the
/// dtype from its inputs, so one closure serves both the routed experts
/// (`[1, rows, top_k, 1, intermediate]`) and the shared expert
/// (`[1, rows, shared_intermediate]`), and MLX compiles per input signature.
fn swiglu_closure() -> Option<MlxClosure> {
    qwen4_exp_fused::compile(|inputs| {
        let (gate, up) = (inputs.get(0), inputs.get(1));
        vec![multiply(&silu_projection_dtype(&gate), &up, None)]
    })
}

/// Dump stage names of the activation and the product of a gate/up pair.
type SwigluStages = (&'static str, &'static str);

/// The routed experts, in the switch layout or the sorted-prefill layout.
const ROUTED_SWIGLU_STAGES: SwigluStages = ("moe_gate_activation", "moe_activated");

/// The shared expert, a plain `[1, rows, s]` projection.
const SHARED_SWIGLU_STAGES: SwigluStages = ("moe_shared_activation", "moe_shared_input");

/// `silu(gate) * up`, compiled when [`Stretch::MoeSwiglu`] applies.
///
/// The compiled path does not materialise the intermediate activation, so the
/// dump only sees it when the eager fallback runs. `stages` names the dumps
/// explicitly: the sorted-prefill gather yields rank-three routed tensors, so
/// the rank cannot tell the routed experts from the shared expert.
#[cfg_attr(not(test), allow(unused_variables))]
fn swiglu(gate: &MlxArray, up: &MlxArray, rows: i32, stages: SwigluStages) -> MlxArray {
    if let Some(activated) = qwen4_exp_fused::apply(
        Stretch::MoeSwiglu,
        rows,
        [0, 0, 0, 0],
        swiglu_closure,
        &[gate, up],
    ) {
        return activated;
    }
    let gate_activation = silu_projection_dtype(gate);
    let activated = multiply(&gate_activation, up, None);
    #[cfg(test)]
    {
        crate::model::qwen4_exp::profiling::dump(stages.0, &[&gate_activation]);
        crate::model::qwen4_exp::profiling::dump(stages.1, &[&activated]);
    }
    activated
}

/// Closure body of the MoE tail.
///
/// Inputs, in order: `down` (expert outputs restored to token order),
/// `routing` (`[batch, rows, top_k]`), `shared` (the shared expert's down
/// projection) and `shared_router`. The reshape back to `[batch, rows, hidden]`
/// stays outside the stretch: it is only a view.
fn combine_closure() -> Option<MlxClosure> {
    qwen4_exp_fused::compile(|inputs| {
        let (down, routing) = (inputs.get(0), inputs.get(1));
        let (shared, shared_router) = (inputs.get(2), inputs.get(3));
        let routed = sum_axis(
            &multiply(&down, &expand_dims_axes(&routing, &[-1], None), None),
            -2,
            false,
            None,
        );
        vec![add(
            &routed,
            &multiply(&shared, &sigmoid_projection_dtype(&shared_router), None),
            None,
        )]
    })
}

/// Weighted routed sum plus the gated shared expert, before the reshape.
fn combine(
    down: &MlxArray,
    routing: &MlxArray,
    shared: &MlxArray,
    shared_router: &MlxArray,
    rows: i32,
) -> MlxArray {
    if let Some(combined) = qwen4_exp_fused::apply(
        Stretch::MoeCombine,
        rows,
        [0, 0, 0, 0],
        combine_closure,
        &[down, routing, shared, shared_router],
    ) {
        return combined;
    }
    let weighted = multiply(down, &expand_dims_axes(routing, &[-1], None), None);
    let routed = sum_axis(&weighted, -2, false, None);
    let shared_score = sigmoid_projection_dtype(shared_router);
    let shared_output = multiply(shared, &shared_score, None);
    let combined = add(&routed, &shared_output, None);
    #[cfg(test)]
    {
        crate::model::qwen4_exp::profiling::dump("moe_routed", &[&routed]);
        crate::model::qwen4_exp::profiling::dump("moe_shared_score", &[&shared_score]);
        crate::model::qwen4_exp::profiling::dump("moe_shared_output", &[&shared_output]);
    }
    combined
}

/// Closure body of the routing tail: the per-row sum normalize plus the cast
/// back to the logits dtype, with the output dtype baked in. The reduction is
/// the same op the eager path runs, kept inside the body exactly like the
/// `gdn_l2_sum` reduction.
fn moe_route_tail_closure(out_dtype: MlxDtype) -> Option<MlxClosure> {
    qwen4_exp_fused::compile(move |inputs| {
        let routing = inputs.get(0);
        vec![astype(
            &divide(&routing, &sum_axis(&routing, -1, true, None), None),
            out_dtype,
            None,
        )]
    })
}

/// The normalized routing tail: divide each row by its sum and cast back to
/// the logits dtype, compiled as one stretch when [`Stretch::MoeRouteTail`]
/// applies. `rows` is the row count of `routing`; the eager fallback is the
/// original op sequence.
fn route_tail(routing: &MlxArray, out_dtype: MlxDtype, rows: i32) -> MlxArray {
    if let Some(tail) = qwen4_exp_fused::apply(
        Stretch::MoeRouteTail,
        rows,
        [dtype_key(out_dtype), 0, 0, 0],
        || moe_route_tail_closure(out_dtype),
        &[routing],
    ) {
        return tail;
    }
    astype(
        &divide(routing, &sum_axis(routing, -1, true, None), None),
        out_dtype,
        None,
    )
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
            prefill_expert_sort: crate::fastpath::flash_next_prefill_expert_sort_enabled(),
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
        self.forward_with_verifier_policy(input, policy, policy)
    }

    pub(crate) fn forward_with_verifier_policy(
        &self,
        input: &MlxArray,
        policy: ProjectionBatchPolicy,
        verifier_policy: ProjectionBatchPolicy,
    ) -> Result<MlxArray, String> {
        let shape = self.validate_input(input)?;
        if self.selected_decode
            && shape[0] == 1
            && (shape[1] == 1 || (self.selected_prefill && policy == ProjectionBatchPolicy::Shared))
            && let Qwen4ExpExpertWeights::Streamed(source) = &self.weights.experts
            && let Some(output) = self.forward_selected(input, policy, verifier_policy, source)?
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
        let output = self.forward_with_experts(input, policy, verifier_policy, &experts)?;
        // Streamed layers must release their pending expert graph references
        // before the next layer resolves a (possibly different) paged stack,
        // rather than relying only on the pager's LRU eviction.
        if experts.is_streamed() {
            try_eval(&[&output])
                .map_err(|e| format!("qwen4_exp streamed MoE output eval failed: {e}"))?;
            #[cfg(test)]
            crate::model::qwen4_exp::profiling::note_eval();
        }
        #[cfg(test)]
        crate::model::qwen4_exp::profiling::mark("moe_compute", &[&output]);
        Ok(output)
    }

    fn forward_with_experts(
        &self,
        input: &MlxArray,
        policy: ProjectionBatchPolicy,
        verifier_policy: ProjectionBatchPolicy,
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
                outputs.push(self.forward_with_experts(&row, policy, verifier_policy, experts)?);
            }
            return Ok(concatenate(&outputs.iter().collect::<Vec<_>>(), 1, None));
        }
        let (indices, routing) = self.route(input, policy);
        #[cfg(test)]
        crate::model::qwen4_exp::profiling::mark("moe_routing", &[&indices, &routing]);
        self.forward_routed(input, policy, verifier_policy, experts, &indices, &routing)
    }

    fn route(&self, input: &MlxArray, policy: ProjectionBatchPolicy) -> (MlxArray, MlxArray) {
        let w = &self.weights;
        let logits = qw_with_policy(input, &w.router, policy);
        let probabilities = softmax(&astype(&logits, MlxDtype::Float32, None), -1, None);
        let (indices, routing) =
            top_k_by_argpartition(&probabilities, self.experts, self.top_k, false);
        let routing = if self.normalize_top_k {
            route_tail(&routing, logits.dtype(), routing.shape()[1])
        } else {
            astype(&routing, logits.dtype(), None)
        };
        #[cfg(test)]
        {
            crate::model::qwen4_exp::profiling::dump("moe_router_logits", &[&logits]);
            crate::model::qwen4_exp::profiling::dump("moe_indices", &[&indices]);
            crate::model::qwen4_exp::profiling::dump("moe_routing", &[&routing]);
        }
        (indices, routing)
    }

    fn forward_selected(
        &self,
        input: &MlxArray,
        policy: ProjectionBatchPolicy,
        verifier_policy: ProjectionBatchPolicy,
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
        let output =
            self.forward_routed(input, policy, verifier_policy, &experts, &indices, &routing)?;
        try_eval(&[&output])
            .map_err(|e| format!("qwen4_exp selected MoE output eval failed: {e}"))?;
        Ok(Some(output))
    }

    fn forward_routed(
        &self,
        input: &MlxArray,
        policy: ProjectionBatchPolicy,
        verifier_policy: ProjectionBatchPolicy,
        experts: &ResolvedExperts<'_>,
        indices: &MlxArray,
        routing: &MlxArray,
    ) -> Result<MlxArray, String> {
        let shape = self.validate_input(input)?;
        let w = &self.weights;
        let expanded = expand_dims_axes(input, &[-2, -3], None);
        let gathered = if self.prefill_expert_sort
            && shape[1] >= PREFILL_EXPERT_SORT_MIN_TOKENS
            && policy == ProjectionBatchPolicy::Shared
        {
            switch_gather_inputs(&expanded, indices)
        } else {
            SwitchGatherInputs {
                x: expanded,
                indices: indices.clone(),
                sorted_indices: false,
                inv_order: None,
                original_indices_shape: indices.shape(),
            }
        };
        let sorted = gathered.sorted_indices;
        let rows = shape[1];
        let gate = qw_gather(&gathered.x, experts.gate(), &gathered.indices, sorted);
        let up = qw_gather(&gathered.x, experts.up(), &gathered.indices, sorted);
        let activated = swiglu(&gate, &up, rows, ROUTED_SWIGLU_STAGES);
        let down_gathered = qw_gather(&activated, experts.down(), &gathered.indices, sorted);
        let down = if sorted {
            gathered.unsort(down_gathered)
        } else {
            squeeze_switch_singleton(&down_gathered)
        };
        let shared_gate_policy =
            if w.shared_gate.mlx_quantization_mode() == mlx_sys::MlxQuantizationMode::Mxfp4 {
                verifier_policy
            } else {
                policy
            };
        let shared_gate = qw_with_policy(input, &w.shared_gate, shared_gate_policy);
        let shared_up_policy =
            if w.shared_up.mlx_quantization_mode() == mlx_sys::MlxQuantizationMode::Mxfp4 {
                verifier_policy
            } else {
                policy
            };
        let shared_up = qw_with_policy(input, &w.shared_up, shared_up_policy);
        let shared_input = swiglu(&shared_gate, &shared_up, rows, SHARED_SWIGLU_STAGES);
        let shared_down_policy =
            if w.shared_down.mlx_quantization_mode() == mlx_sys::MlxQuantizationMode::Mxfp4 {
                verifier_policy
            } else {
                policy
            };
        let shared = qw_with_policy(&shared_input, &w.shared_down, shared_down_policy);
        let shared_router = qw_with_policy(input, &w.shared_router, policy);
        // The reshape back to the input shape is a view, so it stays outside the
        // tail stretch.
        let output = reshape(
            &combine(&down, routing, &shared, &shared_router, rows),
            &shape,
            None,
        );
        // Only intermediates that exist on the path taken: the fused stretches
        // do not materialise the activations and the weighted sum inputs, and a
        // dump of a graph the run never built would be a different op sequence.
        #[cfg(test)]
        {
            for (stage, array) in [
                ("moe_gate", &gate),
                ("moe_up", &up),
                ("moe_down", &down),
                ("moe_shared_gate", &shared_gate),
                ("moe_shared_up", &shared_up),
                ("moe_shared_down", &shared),
                ("moe_shared_router", &shared_router),
                ("moe_delta", &output),
            ] {
                crate::model::qwen4_exp::profiling::dump(stage, &[array]);
            }
        }
        Ok(output)
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

    fn activation_fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/flash_next/moe_bf16_activations.json"
        ))
        .unwrap()
    }

    #[test]
    fn moe_bf16_silu_product_matches_official_rounding() {
        let fixture = activation_fixture();
        let length = fixture["gate"].as_array().unwrap().len() as i32;
        let gate = astype(
            &array(&fixture["gate"], &[1, length]),
            MlxDtype::Bfloat16,
            None,
        );
        let up = astype(
            &array(&fixture["up"], &[1, length]),
            MlxDtype::Bfloat16,
            None,
        );
        let actual = astype(
            &multiply(&silu_projection_dtype(&gate), &up, None),
            MlxDtype::Float32,
            None,
        );
        let expected = array(&fixture["activated"], &[1, length]);
        eval(&[&actual, &expected]);
        assert_eq!(
            actual.data_f32(),
            expected.data_f32(),
            "expert/shared SiLU then multiply"
        );
    }

    #[test]
    fn moe_bf16_shared_score_matches_official_rounding() {
        let fixture = activation_fixture();
        let length = fixture["gate"].as_array().unwrap().len() as i32;
        let gate = astype(
            &array(&fixture["gate"], &[1, length]),
            MlxDtype::Bfloat16,
            None,
        );
        let actual = astype(&sigmoid_projection_dtype(&gate), MlxDtype::Float32, None);
        let expected = array(&fixture["score"], &[1, length]);
        eval(&[&actual, &expected]);
        assert_eq!(
            actual.data_f32(),
            expected.data_f32(),
            "shared expert sigmoid"
        );
    }

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
                ProjectionBatchPolicy::Shared,
                &compact,
                &remap,
                &routing,
            )
            .unwrap();
        eval(&[&expected, &actual]);
        assert_eq!(actual.data_f32(), expected.data_f32());
    }

    #[test]
    fn prefill_expert_sort_matches_the_unsorted_gather() {
        let tokens = 70i32;
        let values: Vec<f32> = (0..tokens * 16)
            .map(|i| (((i * 37) % 101) as f32 - 50.0) / 40.0)
            .collect();
        let input = reshape(&MlxArray::from_f32_slice(&values), &[1, tokens, 16], None);
        let (mut module, _) = build_resident_module();
        module.prefill_expert_sort = false;
        let unsorted = module
            .forward(&input, ProjectionBatchPolicy::Shared)
            .unwrap();
        module.prefill_expert_sort = true;
        let sorted = module
            .forward(&input, ProjectionBatchPolicy::Shared)
            .unwrap();
        eval(&[&unsorted, &sorted]);
        assert_eq!(sorted.shape(), unsorted.shape());
        let (a, b) = (sorted.data_f32(), unsorted.data_f32());
        let max_abs = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 1.0e-5, "sorted prefill gather drifted: {max_abs}");
    }

    #[test]
    fn short_windows_and_row_exact_verification_never_sort() {
        // 8 rows stay below the prefill threshold: the sort flag must not
        // change a single bit, so decode and short verifier windows keep
        // their original arithmetic.
        let tokens = 8i32;
        let values: Vec<f32> = (0..tokens * 16)
            .map(|i| ((i % 29) as f32 - 14.0) / 20.0)
            .collect();
        let input = reshape(&MlxArray::from_f32_slice(&values), &[1, tokens, 16], None);
        let (mut module, _) = build_resident_module();
        module.prefill_expert_sort = false;
        let off = module
            .forward(&input, ProjectionBatchPolicy::Shared)
            .unwrap();
        module.prefill_expert_sort = true;
        let on = module
            .forward(&input, ProjectionBatchPolicy::Shared)
            .unwrap();
        eval(&[&off, &on]);
        assert_eq!(on.data_f32(), off.data_f32());
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

    /// MXFP4 expert stack (scales-only 4/32, classified MXFP4) with
    /// deterministic contents, for the window row-sharing test.
    fn mxfp4_stack(experts: i32, out: i32, input: i32, phase: usize) -> QuantizedWeight {
        let count = (experts * out * input) as usize;
        let data: Vec<f32> = (0..count)
            .map(|i| ((((i + phase) % 887) as f32) - 443.0) * 0.0012)
            .collect();
        let dense = MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[experts, out, input],
            MlxDtype::Float32,
        );
        let quantized = mlx_sys::quantize(
            &dense,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
        assert!(quantized.len() >= 2, "MXFP4 quantize returns weight+scales");
        eval(&[&quantized[0], &quantized[1]]);
        QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
            weight_t_view: Default::default(),
        }
    }

    fn bf16_dense(out: i32, input: i32, phase: usize) -> QuantizedWeight {
        let count = (out * input) as usize;
        let data: Vec<f32> = (0..count)
            .map(|i| ((((i + phase) % 677) as f32) - 338.0) * 0.001)
            .collect();
        QuantizedWeight::new(
            astype(
                &MlxArray::from_raw_data(
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data.as_slice()),
                    &[out, input],
                    MlxDtype::Float32,
                ),
                MlxDtype::Bfloat16,
                None,
            ),
            None,
            None,
        )
    }

    /// The block MTP verifier retains batched MoE output, so a window whose
    /// rows share an expert (M_e = 2..=4 in the grouped MXFP4 GEMM) must match
    /// the singleton per-row arithmetic bit for bit. Identical rows force the
    /// maximal collision: every row selects the same experts.
    #[test]
    fn mxfp4_moe_window_rows_sharing_experts_match_singleton_rows() {
        let module = Qwen4ExpMoe::new(
            64,
            32,
            32,
            8,
            2,
            true,
            Qwen4ExpMoeWeights {
                router: bf16_dense(8, 64, 3),
                experts: Qwen4ExpExpertWeights::Resident(Box::new(Qwen4ExpResidentExperts {
                    gate: mxfp4_stack(8, 32, 64, 11),
                    up: mxfp4_stack(8, 32, 64, 23),
                    down: mxfp4_stack(8, 64, 32, 37),
                })),
                shared_gate: bf16_dense(32, 64, 41),
                shared_up: bf16_dense(32, 64, 43),
                shared_down: bf16_dense(64, 32, 47),
                shared_router: bf16_dense(1, 64, 53),
            },
        )
        .unwrap();
        let row = |phase: usize, scale: f32| {
            let data: Vec<f32> = (0..64)
                .map(|i| ((((i + phase) % 61) as f32) - 30.0) * scale)
                .collect();
            astype(
                &MlxArray::from_raw_data(
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data.as_slice()),
                    &[1, 1, 64],
                    MlxDtype::Float32,
                ),
                MlxDtype::Bfloat16,
                None,
            )
        };
        let (a, b, c, d) = (row(0, 0.03), row(7, 0.02), row(13, -0.025), row(29, 0.04));
        for (label, rows) in [
            (
                "identical",
                vec![a.clone(), a.clone(), a.clone(), a.clone()],
            ),
            ("abab", vec![a.clone(), b.clone(), a.clone(), b.clone()]),
            ("distinct", vec![a.clone(), b.clone(), c.clone(), d.clone()]),
            ("three", vec![a.clone(), b.clone(), c.clone()]),
            ("two", vec![a.clone(), b.clone()]),
        ] {
            let refs: Vec<&MlxArray> = rows.iter().collect();
            let window = concatenate(&refs, 1, None);
            let batched = module
                .forward(&window, ProjectionBatchPolicy::Shared)
                .unwrap();
            for (s, single_input) in rows.iter().enumerate() {
                let single = module
                    .forward(single_input, ProjectionBatchPolicy::Shared)
                    .unwrap();
                let row_out = slice(
                    &batched,
                    &[0, s as i32, 0],
                    &[1, s as i32 + 1, 64],
                    &[1, 1, 1],
                    None,
                );
                let row_out = astype(&contiguous(&row_out, None), MlxDtype::Float32, None);
                let single = astype(&single, MlxDtype::Float32, None);
                eval(&[&row_out, &single]);
                assert_eq!(
                    row_out.data_f32(),
                    single.data_f32(),
                    "MoE window row {s} ({label}) diverged from the singleton row"
                );
            }
        }
    }

    /// Deterministic xorshift64 PRNG, so the adversarial differential inputs
    /// below are reproducible without pulling in an external crate (same
    /// construction as the GDN differential tests).
    struct XorShift64(u64);

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform in `+-1`.
        fn unit(&mut self) -> f32 {
            let unit = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
            unit * 2.0 - 1.0
        }
    }

    /// Adversarial f32 buffer: at PRNG-chosen positions (~1% each) one of the
    /// special classes - quiet NaNs with pseudo-random payloads, +-inf, +-0.0,
    /// subnormal and min-normal values (1e-45, 1.2e-38), +-1, large finite
    /// (3e38), sigmoid/SiLU saturation values (+-30, +-90) and bf16/f16
    /// rounding boundaries (1.00390625, 65504.0, 65520.0) - otherwise scaled
    /// noise.
    fn adversarial(rng: &mut XorShift64, len: usize, magnitude: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                match rng.next_u64() % 100 {
                    0 => f32::from_bits(0x7fc0_0000 | (rng.next_u64() as u32 & 0x003f_ffff)),
                    1 => sign * f32::INFINITY,
                    2 => sign * 0.0,
                    3 => sign * 1.0e-45,
                    4 => sign * 1.2e-38,
                    5 => sign,
                    6 => sign * 3.0e38,
                    7 => sign * 30.0,
                    8 => sign * 90.0,
                    9 => match rng.next_u64() % 3 {
                        0 => sign * (1.0 + 1.0 / 256.0),
                        1 => sign * 65504.0,
                        _ => sign * 65520.0,
                    },
                    _ => rng.unit() * magnitude,
                }
            })
            .collect()
    }

    /// `shape` adversarial buffer of `dtype`, built through an f32 widening.
    fn adversarial_array(
        rng: &mut XorShift64,
        shape: &[i32],
        magnitude: f32,
        dtype: MlxDtype,
    ) -> MlxArray {
        let len: usize = shape.iter().map(|&dim| dim as usize).product();
        let data = adversarial(rng, len, magnitude);
        astype(
            &MlxArray::from_raw_data(
                data.as_ptr().cast(),
                std::mem::size_of_val(data.as_slice()),
                shape,
                MlxDtype::Float32,
            ),
            dtype,
            None,
        )
    }

    /// Widen to f32 (lossless for f32/bf16/f16) and compare raw bits, NaN
    /// payloads included: the fused stretches must match the eager ops bit
    /// for bit.
    fn widened_bits(array: &MlxArray) -> Vec<u32> {
        let wide = contiguous(&astype(array, MlxDtype::Float32, None), None);
        eval(&[&wide]);
        wide.data_f32()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// `[1, rows, 10]` routing weights for the combine differential test:
    /// non-negative values summing to ~1 per row, or an adversarial variant
    /// with zeros and NaN (infinities are mapped to zero - the router never
    /// emits them and inf*0 would only add noise to both paths alike).
    fn routing_weights(
        rng: &mut XorShift64,
        rows: i32,
        dtype: MlxDtype,
        adversarial_routing: bool,
    ) -> MlxArray {
        let len = (rows * 10) as usize;
        let data: Vec<f32> = if adversarial_routing {
            adversarial(rng, len, 1.0)
                .into_iter()
                .map(|value| if value.is_infinite() { 0.0 } else { value })
                .collect()
        } else {
            let mut data = Vec::with_capacity(len);
            for _ in 0..rows {
                let weights: Vec<f32> = (0..10).map(|_| 0.05 + 0.1 * rng.unit().abs()).collect();
                let total: f32 = weights.iter().sum();
                data.extend(weights.iter().map(|weight| weight / total));
            }
            data
        };
        astype(
            &MlxArray::from_raw_data(
                data.as_ptr().cast(),
                std::mem::size_of_val(data.as_slice()),
                &[1, rows, 10],
                MlxDtype::Float32,
            ),
            dtype,
            None,
        )
    }

    /// The `MoeSwiglu` stretch on both call shapes: routed experts
    /// (`[1, rows, top_k, 1, intermediate]`) and the shared expert
    /// (`[1, rows, shared_intermediate]`). The forced-fused result, the
    /// forced-eager result and the original inline ops must agree bit for bit
    /// on adversarial inputs for every dtype, row count and seed, and one
    /// shared closure must serve both shapes without stale specialisation.
    #[test]
    fn moe_swiglu_matches_eager_on_adversarial_inputs() {
        for (dtype_index, dtype) in [MlxDtype::Float32, MlxDtype::Bfloat16, MlxDtype::Float16]
            .into_iter()
            .enumerate()
        {
            for rows in 1..=8i32 {
                for seed in 0..7u64 {
                    let mut rng = XorShift64::new(
                        dtype_index as u64 * 1_000_003 + seed * 613 + rows as u64 * 29,
                    );
                    let routed = [1, rows, 10, 1, 640];
                    let shared = [1, rows, 640];
                    let routed_gate = adversarial_array(&mut rng, &routed, 4.0, dtype);
                    let routed_up = adversarial_array(&mut rng, &routed, 4.0, dtype);
                    let shared_gate = adversarial_array(&mut rng, &shared, 4.0, dtype);
                    let shared_up = adversarial_array(&mut rng, &shared, 4.0, dtype);

                    // Forced eager and the original inline ops are the same op
                    // sequence; both anchors must agree before the fused run.
                    let eager_routed = {
                        let _eager = qwen4_exp_fused::force_for_test(Some(false));
                        swiglu(&routed_gate, &routed_up, rows, ROUTED_SWIGLU_STAGES)
                    };
                    let eager_shared = {
                        let _eager = qwen4_exp_fused::force_for_test(Some(false));
                        swiglu(&shared_gate, &shared_up, rows, SHARED_SWIGLU_STAGES)
                    };
                    let inline_routed =
                        multiply(&silu_projection_dtype(&routed_gate), &routed_up, None);
                    let inline_shared =
                        multiply(&silu_projection_dtype(&shared_gate), &shared_up, None);
                    assert_eq!(
                        widened_bits(&eager_routed),
                        widened_bits(&inline_routed),
                        "eager routed swiglu drifted from the inline ops \
                         ({dtype:?}, rows {rows}, seed {seed})"
                    );
                    assert_eq!(
                        widened_bits(&eager_shared),
                        widened_bits(&inline_shared),
                        "eager shared swiglu drifted from the inline ops \
                         ({dtype:?}, rows {rows}, seed {seed})"
                    );

                    qwen4_exp_fused::reset_registry_for_test();
                    let before = qwen4_exp_fused::applied_for_test(Stretch::MoeSwiglu);
                    let (fused_routed, fused_shared) = {
                        let _fused = qwen4_exp_fused::force_for_test(Some(true));
                        (
                            swiglu(&routed_gate, &routed_up, rows, ROUTED_SWIGLU_STAGES),
                            swiglu(&shared_gate, &shared_up, rows, SHARED_SWIGLU_STAGES),
                        )
                    };
                    assert!(
                        qwen4_exp_fused::applied_for_test(Stretch::MoeSwiglu) - before >= 2,
                        "each of the two fused swiglu calls must count as an apply \
                         ({dtype:?}, rows {rows}, seed {seed})"
                    );
                    assert_eq!(
                        widened_bits(&fused_routed),
                        widened_bits(&inline_routed),
                        "fused routed swiglu diverged ({dtype:?}, rows {rows}, seed {seed})"
                    );
                    assert_eq!(
                        widened_bits(&fused_shared),
                        widened_bits(&inline_shared),
                        "fused shared swiglu diverged ({dtype:?}, rows {rows}, seed {seed})"
                    );
                }
            }

            // One compiled closure serves both shapes: routed, then shared,
            // then routed again with fresh inputs must each match the eager
            // product - no stale specialisation to the first traced shape.
            qwen4_exp_fused::reset_registry_for_test();
            let mut rng = XorShift64::new(9_521 + dtype_index as u64);
            let rows = 3i32;
            let routed = [1, rows, 10, 1, 640];
            let shared = [1, rows, 640];
            let first_gate = adversarial_array(&mut rng, &routed, 4.0, dtype);
            let first_up = adversarial_array(&mut rng, &routed, 4.0, dtype);
            let shared_gate = adversarial_array(&mut rng, &shared, 4.0, dtype);
            let shared_up = adversarial_array(&mut rng, &shared, 4.0, dtype);
            let again_gate = adversarial_array(&mut rng, &routed, 4.0, dtype);
            let again_up = adversarial_array(&mut rng, &routed, 4.0, dtype);
            let (fused_first, fused_shared, fused_again) = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                (
                    swiglu(&first_gate, &first_up, rows, ROUTED_SWIGLU_STAGES),
                    swiglu(&shared_gate, &shared_up, rows, SHARED_SWIGLU_STAGES),
                    swiglu(&again_gate, &again_up, rows, ROUTED_SWIGLU_STAGES),
                )
            };
            assert_eq!(
                widened_bits(&fused_first),
                widened_bits(&multiply(
                    &silu_projection_dtype(&first_gate),
                    &first_up,
                    None
                )),
                "routed shape before the shared shape"
            );
            assert_eq!(
                widened_bits(&fused_shared),
                widened_bits(&multiply(
                    &silu_projection_dtype(&shared_gate),
                    &shared_up,
                    None
                )),
                "shared shape after the routed shape"
            );
            assert_eq!(
                widened_bits(&fused_again),
                widened_bits(&multiply(
                    &silu_projection_dtype(&again_gate),
                    &again_up,
                    None
                )),
                "routed shape again after the shared shape"
            );
        }
    }

    /// The `MoeCombine` stretch - weighted routed sum plus the gated shared
    /// expert - against forced eager and the original inline ops on
    /// adversarial inputs, through the reshape back to `[1, rows, hidden]`.
    #[test]
    fn moe_combine_matches_eager_on_adversarial_inputs() {
        for (dtype_index, dtype) in [MlxDtype::Float32, MlxDtype::Bfloat16, MlxDtype::Float16]
            .into_iter()
            .enumerate()
        {
            for rows in 1..=8i32 {
                for routing_adversarial in [false, true] {
                    let mut rng = XorShift64::new(
                        dtype_index as u64 * 1_000_003
                            + rows as u64 * 29
                            + u64::from(routing_adversarial) * 17
                            + 3,
                    );
                    let down = adversarial_array(&mut rng, &[1, rows, 10, 2560], 2.0, dtype);
                    let shared = adversarial_array(&mut rng, &[1, rows, 2560], 2.0, dtype);
                    let shared_router = adversarial_array(&mut rng, &[1, rows, 1], 8.0, dtype);
                    let routing = routing_weights(&mut rng, rows, dtype, routing_adversarial);

                    // Original inline ops, before any fusion is forced.
                    let inline = {
                        let weighted =
                            multiply(&down, &expand_dims_axes(&routing, &[-1], None), None);
                        let routed = sum_axis(&weighted, -2, false, None);
                        let shared_score = sigmoid_projection_dtype(&shared_router);
                        let shared_output = multiply(&shared, &shared_score, None);
                        add(&routed, &shared_output, None)
                    };
                    let eager = {
                        let _eager = qwen4_exp_fused::force_for_test(Some(false));
                        combine(&down, &routing, &shared, &shared_router, rows)
                    };
                    let inline_bits = widened_bits(&inline);
                    assert_eq!(
                        widened_bits(&eager),
                        inline_bits,
                        "eager combine drifted from the inline ops \
                         ({dtype:?}, rows {rows}, adversarial routing {routing_adversarial})"
                    );

                    qwen4_exp_fused::reset_registry_for_test();
                    let before = qwen4_exp_fused::applied_for_test(Stretch::MoeCombine);
                    let (fused, fused_view) = {
                        let _fused = qwen4_exp_fused::force_for_test(Some(true));
                        let fused = combine(&down, &routing, &shared, &shared_router, rows);
                        let view = reshape(&fused, &[1, rows, 2560], None);
                        (fused, view)
                    };
                    assert!(
                        qwen4_exp_fused::applied_for_test(Stretch::MoeCombine) > before,
                        "MoeCombine never applied \
                         ({dtype:?}, rows {rows}, adversarial routing {routing_adversarial})"
                    );
                    assert_eq!(
                        widened_bits(&fused),
                        inline_bits,
                        "fused combine diverged \
                         ({dtype:?}, rows {rows}, adversarial routing {routing_adversarial})"
                    );
                    // The reshape stays outside the stretch; both paths must
                    // agree through it too.
                    let inline_view = reshape(&inline, &[1, rows, 2560], None);
                    assert_eq!(
                        widened_bits(&fused_view),
                        widened_bits(&inline_view),
                        "post-reshape combine diverged \
                         ({dtype:?}, rows {rows}, adversarial routing {routing_adversarial})"
                    );
                }
            }
        }
    }

    /// End-to-end: the public MoE forward must be bit identical under every
    /// fusion mode, the counters must prove both stretches ran (routed +
    /// shared swiglu, one combine per processed row group), rows past the
    /// fuse cap must stay eager without tombstoning anything, and an injected
    /// apply failure must tombstone exactly the failing stretch while the
    /// output stays on the eager bits.
    #[test]
    fn moe_forward_fused_matches_eager_end_to_end() {
        let (module, _) = build_resident_module();
        let input = |rows: i32| {
            let values: Vec<f32> = (0..rows * 16)
                .map(|i| (((i * 29) % 53) as f32 - 26.0) / 9.0)
                .collect();
            reshape(&MlxArray::from_f32_slice(&values), &[1, rows, 16], None)
        };

        for rows in 1..=4i32 {
            for policy in [
                ProjectionBatchPolicy::Shared,
                ProjectionBatchPolicy::RowExact,
            ] {
                let eager_bits = {
                    let _eager = qwen4_exp_fused::force_for_test(Some(false));
                    widened_bits(&module.forward(&input(rows), policy).unwrap())
                };
                qwen4_exp_fused::reset_registry_for_test();
                let swiglu_before = qwen4_exp_fused::applied_for_test(Stretch::MoeSwiglu);
                let combine_before = qwen4_exp_fused::applied_for_test(Stretch::MoeCombine);
                let fused = {
                    let _fused = qwen4_exp_fused::force_for_test(Some(true));
                    module.forward(&input(rows), policy).unwrap()
                };
                assert_eq!(
                    widened_bits(&fused),
                    eager_bits,
                    "{policy:?} rows {rows}: fused MoE forward diverged"
                );
                // RowExact splits the window into per-row groups; each group
                // pays two swiglu applies (routed + shared) and one combine.
                let groups = if policy == ProjectionBatchPolicy::RowExact {
                    rows as u64
                } else {
                    1
                };
                let swiglu_floor = 2 * groups;
                assert!(
                    qwen4_exp_fused::applied_for_test(Stretch::MoeSwiglu) - swiglu_before
                        >= swiglu_floor,
                    "{policy:?} rows {rows}: expected >= {swiglu_floor} MoeSwiglu applies"
                );
                assert!(
                    qwen4_exp_fused::applied_for_test(Stretch::MoeCombine) - combine_before
                        >= groups,
                    "{policy:?} rows {rows}: expected >= {groups} MoeCombine applies"
                );
            }
        }

        // Rows past the fuse cap must not fuse at all, and must not leave a
        // tombstone behind either.
        let tall = input(9);
        let tall_bits = {
            let _eager = qwen4_exp_fused::force_for_test(Some(false));
            widened_bits(
                &module
                    .forward(&tall, ProjectionBatchPolicy::Shared)
                    .unwrap(),
            )
        };
        qwen4_exp_fused::reset_registry_for_test();
        let swiglu_before = qwen4_exp_fused::applied_for_test(Stretch::MoeSwiglu);
        let combine_before = qwen4_exp_fused::applied_for_test(Stretch::MoeCombine);
        let fused_tall = {
            let _fused = qwen4_exp_fused::force_for_test(Some(true));
            module
                .forward(&tall, ProjectionBatchPolicy::Shared)
                .unwrap()
        };
        assert_eq!(widened_bits(&fused_tall), tall_bits, "9 rows diverged");
        assert_eq!(
            qwen4_exp_fused::applied_for_test(Stretch::MoeSwiglu),
            swiglu_before,
            "9 rows must stay above the fuse cap (MoeSwiglu)"
        );
        assert_eq!(
            qwen4_exp_fused::applied_for_test(Stretch::MoeCombine),
            combine_before,
            "9 rows must stay above the fuse cap (MoeCombine)"
        );
        assert!(!qwen4_exp_fused::tombstoned_for_test(Stretch::MoeSwiglu));
        assert!(!qwen4_exp_fused::tombstoned_for_test(Stretch::MoeCombine));

        // An injected apply failure tombstones exactly the failing stretch,
        // while the forward output stays on the eager bits and the other
        // stretch keeps applying.
        let eager_rows2 = {
            let _eager = qwen4_exp_fused::force_for_test(Some(false));
            widened_bits(
                &module
                    .forward(&input(2), ProjectionBatchPolicy::Shared)
                    .unwrap(),
            )
        };
        for stretch in [Stretch::MoeSwiglu, Stretch::MoeCombine] {
            let other = if stretch == Stretch::MoeSwiglu {
                Stretch::MoeCombine
            } else {
                Stretch::MoeSwiglu
            };
            qwen4_exp_fused::reset_registry_for_test();
            let other_before = qwen4_exp_fused::applied_for_test(other);
            let injected = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                qwen4_exp_fused::fail_next_apply_for_test(stretch);
                module
                    .forward(&input(2), ProjectionBatchPolicy::Shared)
                    .unwrap()
            };
            assert_eq!(
                widened_bits(&injected),
                eager_rows2,
                "{}: forward with an injected failure diverged",
                stretch.name()
            );
            assert!(
                qwen4_exp_fused::tombstoned_for_test(stretch),
                "{}: the injected failure must tombstone",
                stretch.name()
            );
            assert!(
                !qwen4_exp_fused::tombstoned_for_test(other),
                "{}: the other stretch must not tombstone",
                stretch.name()
            );
            assert!(
                qwen4_exp_fused::applied_for_test(other) > other_before,
                "{}: the other stretch must keep applying",
                stretch.name()
            );
            // The tombstone holds on the next step, still on the eager bits.
            let after = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                module
                    .forward(&input(2), ProjectionBatchPolicy::Shared)
                    .unwrap()
            };
            assert_eq!(
                widened_bits(&after),
                eager_rows2,
                "{}: post-tombstone forward diverged",
                stretch.name()
            );
            assert!(
                !qwen4_exp_fused::tombstoned_for_test(other),
                "{}: the other stretch must survive the next step",
                stretch.name()
            );
        }
    }

    /// The `MoeRouteTail` stretch - the per-row sum normalize plus the cast
    /// back to the logits dtype - against forced eager and the original
    /// inline ops on adversarial routing, including all-zero rows (the sum is
    /// 0, so the divide is NaN on both paths) and rows whose f32 sum
    /// overflows to infinity.
    #[test]
    fn moe_route_tail_matches_eager_on_adversarial_inputs() {
        let inline_tail = |routing: &MlxArray, out_dtype: MlxDtype| {
            astype(
                &divide(routing, &sum_axis(routing, -1, true, None), None),
                out_dtype,
                None,
            )
        };
        for (dtype_index, out_dtype) in [MlxDtype::Float32, MlxDtype::Bfloat16, MlxDtype::Float16]
            .into_iter()
            .enumerate()
        {
            for rows in 1..=8i32 {
                for seed in 0..5u64 {
                    let mut rng = XorShift64::new(
                        dtype_index as u64 * 1_000_003 + rows as u64 * 29 + seed * 613 + 7,
                    );
                    let routing =
                        adversarial_array(&mut rng, &[1, rows, 10], 3.0, MlxDtype::Float32);
                    let inline_bits = widened_bits(&inline_tail(&routing, out_dtype));
                    let eager = {
                        let _eager = qwen4_exp_fused::force_for_test(Some(false));
                        route_tail(&routing, out_dtype, rows)
                    };
                    assert_eq!(
                        widened_bits(&eager),
                        inline_bits,
                        "eager route tail drifted from the inline ops \
                         ({out_dtype:?}, rows {rows}, seed {seed})"
                    );

                    qwen4_exp_fused::reset_registry_for_test();
                    let before = qwen4_exp_fused::applied_for_test(Stretch::MoeRouteTail);
                    let fused = {
                        let _fused = qwen4_exp_fused::force_for_test(Some(true));
                        route_tail(&routing, out_dtype, rows)
                    };
                    assert!(
                        qwen4_exp_fused::applied_for_test(Stretch::MoeRouteTail) > before,
                        "MoeRouteTail never applied \
                         ({out_dtype:?}, rows {rows}, seed {seed})"
                    );
                    assert_eq!(
                        widened_bits(&fused),
                        inline_bits,
                        "fused route tail diverged ({out_dtype:?}, rows {rows}, seed {seed})"
                    );
                }

                // All-zero rows: the per-row sum is 0, the divide is 0/0, and
                // both paths must cast the same NaN pattern.
                let zeros = zeros(&[1, rows, 10]);
                let inline_bits = widened_bits(&inline_tail(&zeros, out_dtype));
                qwen4_exp_fused::reset_registry_for_test();
                let fused = {
                    let _fused = qwen4_exp_fused::force_for_test(Some(true));
                    route_tail(&zeros, out_dtype, rows)
                };
                assert_eq!(
                    widened_bits(&fused),
                    inline_bits,
                    "all-zero routing rows diverged ({out_dtype:?}, rows {rows})"
                );

                // Rows whose f32 sum overflows to infinity divide to zero on
                // both paths.
                let data = vec![3.0e38f32; (rows * 10) as usize];
                let huge = MlxArray::from_raw_data(
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data.as_slice()),
                    &[1, rows, 10],
                    MlxDtype::Float32,
                );
                let inline_bits = widened_bits(&inline_tail(&huge, out_dtype));
                qwen4_exp_fused::reset_registry_for_test();
                let fused = {
                    let _fused = qwen4_exp_fused::force_for_test(Some(true));
                    route_tail(&huge, out_dtype, rows)
                };
                assert_eq!(
                    widened_bits(&fused),
                    inline_bits,
                    "overflowing-sum routing rows diverged ({out_dtype:?}, rows {rows})"
                );
            }
        }
        qwen4_exp_fused::reset_registry_for_test();

        // The registry is never reset across dtypes: the key must select one
        // closure per output dtype, and a revisited dtype must reuse its
        // entry without stale specialisation.
        for (dtype_index, out_dtype) in [
            MlxDtype::Float32,
            MlxDtype::Bfloat16,
            MlxDtype::Float16,
            MlxDtype::Bfloat16,
        ]
        .into_iter()
        .enumerate()
        {
            let mut rng = XorShift64::new(4_242 + dtype_index as u64);
            let routing = adversarial_array(&mut rng, &[1, 2, 10], 2.0, MlxDtype::Float32);
            let inline_bits = widened_bits(&inline_tail(&routing, out_dtype));
            let eager = {
                let _eager = qwen4_exp_fused::force_for_test(Some(false));
                route_tail(&routing, out_dtype, 2)
            };
            assert_eq!(
                widened_bits(&eager),
                inline_bits,
                "{out_dtype:?}: eager drifted from the inline ops"
            );
            let before = qwen4_exp_fused::applied_for_test(Stretch::MoeRouteTail);
            let fused = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                route_tail(&routing, out_dtype, 2)
            };
            assert!(
                qwen4_exp_fused::applied_for_test(Stretch::MoeRouteTail) > before,
                "{out_dtype:?}: the fused path must run after a dtype switch"
            );
            assert_eq!(
                widened_bits(&fused),
                inline_bits,
                "{out_dtype:?} must match the inline ops after a dtype switch"
            );
        }
        qwen4_exp_fused::reset_registry_for_test();

        // Rows past the fuse cap never apply the stretch and still get the
        // eager result.
        for rows in [9i32, 16] {
            let mut rng = XorShift64::new(777 + rows as u64);
            let routing = adversarial_array(&mut rng, &[1, rows, 10], 2.0, MlxDtype::Float32);
            let eager = {
                let _eager = qwen4_exp_fused::force_for_test(Some(false));
                route_tail(&routing, MlxDtype::Bfloat16, rows)
            };
            let before = qwen4_exp_fused::applied_for_test(Stretch::MoeRouteTail);
            let fused = {
                let _fused = qwen4_exp_fused::force_for_test(Some(true));
                route_tail(&routing, MlxDtype::Bfloat16, rows)
            };
            assert_eq!(
                qwen4_exp_fused::applied_for_test(Stretch::MoeRouteTail),
                before,
                "rows {rows} must stay above the fuse cap (MoeRouteTail)"
            );
            assert_eq!(
                widened_bits(&fused),
                widened_bits(&eager),
                "rows {rows} must equal the eager result"
            );
        }
        qwen4_exp_fused::reset_registry_for_test();

        // An injected apply failure tombstones the stretch, falls back to the
        // eager ops bit for bit, and the tombstone holds on the next call.
        let mut rng = XorShift64::new(99);
        let routing = adversarial_array(&mut rng, &[1, 1, 10], 2.0, MlxDtype::Float32);
        let eager = {
            let _eager = qwen4_exp_fused::force_for_test(Some(false));
            route_tail(&routing, MlxDtype::Bfloat16, 1)
        };
        let eager_bits = widened_bits(&eager);
        qwen4_exp_fused::reset_registry_for_test();
        let injected = {
            let _fused = qwen4_exp_fused::force_for_test(Some(true));
            qwen4_exp_fused::fail_next_apply_for_test(Stretch::MoeRouteTail);
            route_tail(&routing, MlxDtype::Bfloat16, 1)
        };
        assert_eq!(
            widened_bits(&injected),
            eager_bits,
            "the result with an injected failure diverged"
        );
        assert!(
            qwen4_exp_fused::tombstoned_for_test(Stretch::MoeRouteTail),
            "the injected failure must tombstone"
        );
        let after = {
            let _fused = qwen4_exp_fused::force_for_test(Some(true));
            route_tail(&routing, MlxDtype::Bfloat16, 1)
        };
        assert_eq!(
            widened_bits(&after),
            eager_bits,
            "the post-tombstone result diverged"
        );
        assert!(
            qwen4_exp_fused::tombstoned_for_test(Stretch::MoeRouteTail),
            "the tombstone must hold on the next call"
        );
        qwen4_exp_fused::reset_registry_for_test();
    }
}
