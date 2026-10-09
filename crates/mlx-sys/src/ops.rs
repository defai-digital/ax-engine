use crate::array::{MlxArray, MlxDtype, null_ffi_array};
use crate::error::{ensure_error_handler, last_error_message, panic_on_status};
use crate::ffi;
use crate::stream::{MlxStream, default_gpu_raw};

unsafe extern "C" {
    fn ax_mlx_compiled_geglu_approx_activation(
        res: *mut ffi::mlx_array,
        gate: ffi::mlx_array,
        x: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_gelu_approx_mul(
        res: *mut ffi::mlx_array,
        gate: ffi::mlx_array,
        x: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_silu_mul(
        res: *mut ffi::mlx_array,
        gate: ffi::mlx_array,
        x: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_gelu_approx_mul_matmul(
        res: *mut ffi::mlx_array,
        gate: ffi::mlx_array,
        x: ffi::mlx_array,
        weight: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_gelu_approx_mul_quantized_matmul(
        res: *mut ffi::mlx_array,
        gate: ffi::mlx_array,
        x: ffi::mlx_array,
        weight: ffi::mlx_array,
        scales: ffi::mlx_array,
        biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_silu_mul_quantized_matmul(
        res: *mut ffi::mlx_array,
        gate: ffi::mlx_array,
        x: ffi::mlx_array,
        weight: ffi::mlx_array,
        scales: ffi::mlx_array,
        biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_gelu_approx_quantized_ffn(
        res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_up_weight: ffi::mlx_array,
        gate_up_scales: ffi::mlx_array,
        gate_up_biases: ffi::mlx_array,
        down_weight: ffi::mlx_array,
        down_scales: ffi::mlx_array,
        down_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    /// mlxcel-style compiled split gate/up/gelu/down affine qmm (gs64/4bit).
    fn ax_mlx_compiled_gelu_approx_split_mlp(
        res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_weight: ffi::mlx_array,
        gate_scales: ffi::mlx_array,
        gate_biases: ffi::mlx_array,
        up_weight: ffi::mlx_array,
        up_scales: ffi::mlx_array,
        up_biases: ffi::mlx_array,
        down_weight: ffi::mlx_array,
        down_scales: ffi::mlx_array,
        down_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    /// Profile-backed: shape-specific compile of multi-token gate+up qmm only.
    fn ax_mlx_dual_qmm_swiglu(
        res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_weight: ffi::mlx_array,
        gate_scales: ffi::mlx_array,
        gate_biases: ffi::mlx_array,
        up_weight: ffi::mlx_array,
        up_scales: ffi::mlx_array,
        up_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_dual_qmm_geglu(
        res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_weight: ffi::mlx_array,
        gate_scales: ffi::mlx_array,
        gate_biases: ffi::mlx_array,
        up_weight: ffi::mlx_array,
        up_scales: ffi::mlx_array,
        up_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_dual_affine_qmm_forced(
        gate_res: *mut ffi::mlx_array,
        up_res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_weight: ffi::mlx_array,
        gate_scales: ffi::mlx_array,
        gate_biases: ffi::mlx_array,
        up_weight: ffi::mlx_array,
        up_scales: ffi::mlx_array,
        up_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_dual_stream_affine_qmm(
        gate_res: *mut ffi::mlx_array,
        up_res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_weight: ffi::mlx_array,
        gate_scales: ffi::mlx_array,
        gate_biases: ffi::mlx_array,
        up_weight: ffi::mlx_array,
        up_scales: ffi::mlx_array,
        up_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_dual_affine_qmm(
        gate_res: *mut ffi::mlx_array,
        up_res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_weight: ffi::mlx_array,
        gate_scales: ffi::mlx_array,
        gate_biases: ffi::mlx_array,
        up_weight: ffi::mlx_array,
        up_scales: ffi::mlx_array,
        up_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_compiled_dual_gate_up_qmm(
        gate_res: *mut ffi::mlx_array,
        up_res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_weight: ffi::mlx_array,
        gate_scales: ffi::mlx_array,
        gate_biases: ffi::mlx_array,
        up_weight: ffi::mlx_array,
        up_scales: ffi::mlx_array,
        up_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_compiled_dual_gate_up_qmm_forced(
        gate_res: *mut ffi::mlx_array,
        up_res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        gate_weight: ffi::mlx_array,
        gate_scales: ffi::mlx_array,
        gate_biases: ffi::mlx_array,
        up_weight: ffi::mlx_array,
        up_scales: ffi::mlx_array,
        up_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_qk_norm_rope_bhsd_from_proj(
        res: *mut ffi::mlx_array,
        proj: ffi::mlx_array,
        norm: ffi::mlx_array,
        n_heads: libc::c_int,
        head_dim: libc::c_int,
        eps: libc::c_float,
        rope_dims: libc::c_int,
        traditional: libc::c_int,
        has_base: libc::c_int,
        base: libc::c_float,
        offset: libc::c_int,
        freqs: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_gemma4_post_attn_ffn_block(
        res: *mut ffi::mlx_array,
        hidden: ffi::mlx_array,
        attn_out: ffi::mlx_array,
        ffn_norm: ffi::mlx_array,
        ffn_post_norm: ffi::mlx_array,
        layer_scalar: ffi::mlx_array,
        gate_up_weight: ffi::mlx_array,
        gate_up_scales: ffi::mlx_array,
        gate_up_biases: ffi::mlx_array,
        down_weight: ffi::mlx_array,
        down_scales: ffi::mlx_array,
        down_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        eps: libc::c_float,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_add_rms_norm_pair(
        residual_res: *mut ffi::mlx_array,
        normed_res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        y: ffi::mlx_array,
        norm_weight: ffi::mlx_array,
        eps: libc::c_float,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_rms_norm_silu_mul_normed(
        res: *mut ffi::mlx_array,
        hidden: ffi::mlx_array,
        gate: ffi::mlx_array,
        norm_weight: ffi::mlx_array,
        eps: libc::c_float,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_quantized_matmul_rms_norm(
        res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        weight: ffi::mlx_array,
        scales: ffi::mlx_array,
        biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        norm_weight: ffi::mlx_array,
        eps: libc::c_float,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_rms_norm_quantized_matmul(
        res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        norm_weight: ffi::mlx_array,
        eps: libc::c_float,
        weight: ffi::mlx_array,
        scales: ffi::mlx_array,
        biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_fused_causal_prefill_attention(
        out: *mut ffi::mlx_array,
        k_out: *mut ffi::mlx_array,
        v_out: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        attn_norm: ffi::mlx_array,
        eps: libc::c_float,
        qkv_weight: ffi::mlx_array,
        qkv_scales: ffi::mlx_array,
        qkv_biases: ffi::mlx_array,
        q_norm: ffi::mlx_array,
        k_norm: ffi::mlx_array,
        qk_eps: libc::c_float,
        v_norm_no_scale: bool,
        num_heads: libc::c_int,
        num_kv_heads: libc::c_int,
        head_dim: libc::c_int,
        rope_dims: libc::c_int,
        rope_base: libc::c_float,
        rope_freqs: ffi::mlx_array,
        scale: libc::c_float,
        o_weight: ffi::mlx_array,
        o_scales: ffi::mlx_array,
        o_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        post_norm: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_fused_causal_prefill_attention_split(
        out: *mut ffi::mlx_array,
        k_out: *mut ffi::mlx_array,
        v_out: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        attn_norm: ffi::mlx_array,
        eps: libc::c_float,
        q_weight: ffi::mlx_array,
        q_scales: ffi::mlx_array,
        q_biases: ffi::mlx_array,
        k_weight: ffi::mlx_array,
        k_scales: ffi::mlx_array,
        k_biases: ffi::mlx_array,
        v_weight: ffi::mlx_array,
        v_scales: ffi::mlx_array,
        v_biases: ffi::mlx_array,
        q_norm: ffi::mlx_array,
        k_norm: ffi::mlx_array,
        qk_eps: libc::c_float,
        v_norm_no_scale: bool,
        value_from_key: bool,
        num_heads: libc::c_int,
        num_kv_heads: libc::c_int,
        head_dim: libc::c_int,
        rope_dims: libc::c_int,
        rope_base: libc::c_float,
        rope_freqs: ffi::mlx_array,
        scale: libc::c_float,
        o_weight: ffi::mlx_array,
        o_scales: ffi::mlx_array,
        o_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        post_norm: ffi::mlx_array,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_fused_qkv_rope_split(
        q_out: *mut ffi::mlx_array,
        k_out: *mut ffi::mlx_array,
        v_out: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        attn_norm: ffi::mlx_array,
        eps: libc::c_float,
        q_weight: ffi::mlx_array,
        q_scales: ffi::mlx_array,
        q_biases: ffi::mlx_array,
        k_weight: ffi::mlx_array,
        k_scales: ffi::mlx_array,
        k_biases: ffi::mlx_array,
        v_weight: ffi::mlx_array,
        v_scales: ffi::mlx_array,
        v_biases: ffi::mlx_array,
        q_norm: ffi::mlx_array,
        k_norm: ffi::mlx_array,
        qk_eps: libc::c_float,
        v_norm_no_scale: bool,
        value_from_key: bool,
        num_heads: libc::c_int,
        num_kv_heads: libc::c_int,
        head_dim: libc::c_int,
        rope_dims: libc::c_int,
        rope_base: libc::c_float,
        rope_freqs: ffi::mlx_array,
        rope_offset: libc::c_int,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_fused_sdpa_oproj(
        out: *mut ffi::mlx_array,
        q: ffi::mlx_array,
        k: ffi::mlx_array,
        v: ffi::mlx_array,
        scale: libc::c_float,
        mask: ffi::mlx_array,
        o_weight: ffi::mlx_array,
        o_scales: ffi::mlx_array,
        o_biases: ffi::mlx_array,
        group_size: libc::c_int,
        bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_qwen_linear_attention_inputs_packed(
        qkv_res: *mut ffi::mlx_array,
        z_res: *mut ffi::mlx_array,
        a_res: *mut ffi::mlx_array,
        b_res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        qkvz_weight: ffi::mlx_array,
        qkvz_scales: ffi::mlx_array,
        qkvz_biases: ffi::mlx_array,
        ba_weight: ffi::mlx_array,
        ba_scales: ffi::mlx_array,
        ba_biases: ffi::mlx_array,
        num_key_heads: libc::c_int,
        num_value_heads: libc::c_int,
        key_head_dim: libc::c_int,
        value_head_dim: libc::c_int,
        group_size: libc::c_int,
        bits: libc::c_int,
        ba_group_size: libc::c_int,
        ba_bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_qwen_linear_attention_inputs_packed_compiled(
        qkv_res: *mut ffi::mlx_array,
        z_res: *mut ffi::mlx_array,
        a_res: *mut ffi::mlx_array,
        b_res: *mut ffi::mlx_array,
        x: ffi::mlx_array,
        qkvz_weight: ffi::mlx_array,
        qkvz_scales: ffi::mlx_array,
        qkvz_biases: ffi::mlx_array,
        ba_weight: ffi::mlx_array,
        ba_scales: ffi::mlx_array,
        ba_biases: ffi::mlx_array,
        num_key_heads: libc::c_int,
        num_value_heads: libc::c_int,
        key_head_dim: libc::c_int,
        value_head_dim: libc::c_int,
        group_size: libc::c_int,
        bits: libc::c_int,
        ba_group_size: libc::c_int,
        ba_bits: libc::c_int,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_qwen_linear_attention_post_input(
        q_res: *mut ffi::mlx_array,
        k_res: *mut ffi::mlx_array,
        v_res: *mut ffi::mlx_array,
        new_conv_state_res: *mut ffi::mlx_array,
        qkv: ffi::mlx_array,
        conv_weight: ffi::mlx_array,
        cached_conv_state: ffi::mlx_array,
        num_key_heads: libc::c_int,
        key_head_dim: libc::c_int,
        num_value_heads: libc::c_int,
        value_head_dim: libc::c_int,
        conv_kernel_dim: libc::c_int,
        q_scale: libc::c_float,
        k_scale: libc::c_float,
        rms_norm_eps: libc::c_float,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;

    fn ax_mlx_qwen_linear_attention_post_input_compiled(
        q_res: *mut ffi::mlx_array,
        k_res: *mut ffi::mlx_array,
        v_res: *mut ffi::mlx_array,
        new_conv_state_res: *mut ffi::mlx_array,
        qkv: ffi::mlx_array,
        conv_weight: ffi::mlx_array,
        cached_conv_state: ffi::mlx_array,
        num_key_heads: libc::c_int,
        key_head_dim: libc::c_int,
        num_value_heads: libc::c_int,
        value_head_dim: libc::c_int,
        conv_kernel_dim: libc::c_int,
        q_scale: libc::c_float,
        k_scale: libc::c_float,
        rms_norm_eps: libc::c_float,
        stream: ffi::mlx_stream,
    ) -> libc::c_int;
}

fn optional_int(value: Option<i32>, default: i32) -> ffi::mlx_optional_int_ {
    ffi::mlx_optional_int_ {
        has_value: value.is_some(),
        value: value.unwrap_or(default),
    }
}

fn optional_dtype(value: Option<MlxDtype>, default: MlxDtype) -> ffi::mlx_optional_dtype_ {
    ffi::mlx_optional_dtype_ {
        has_value: value.is_some(),
        value: value.unwrap_or(default).to_ffi(),
    }
}

macro_rules! unary_op {
    ($name:ident, $ffi_fn:ident) => {
        pub fn $name(a: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
            $crate::op_count::bump();
            unsafe {
                let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
                let mut res = MlxArray::empty();
                ensure_error_handler();
                let rc = ffi::$ffi_fn(&mut res.inner, a.inner, stream);
                panic_on_status(stringify!($ffi_fn), rc);
                res
            }
        }
    };
}

macro_rules! binary_op {
    ($name:ident, $ffi_fn:ident) => {
        pub fn $name(a: &MlxArray, b: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
            $crate::op_count::bump();
            unsafe {
                let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
                let mut res = MlxArray::empty();
                ensure_error_handler();
                let rc = ffi::$ffi_fn(&mut res.inner, a.inner, b.inner, stream);
                panic_on_status(stringify!($ffi_fn), rc);
                res
            }
        }
    };
}

macro_rules! checked_ffi {
    ($operation:literal, $call:expr) => {{
        ensure_error_handler();
        let rc = $call;
        panic_on_status($operation, rc);
    }};
}

binary_op!(add, mlx_add);
binary_op!(subtract, mlx_subtract);
binary_op!(divide, mlx_divide);
binary_op!(multiply, mlx_multiply);
binary_op!(matmul, mlx_matmul);
binary_op!(greater_equal, mlx_greater_equal);
binary_op!(less, mlx_less);
binary_op!(less_equal, mlx_less_equal);
binary_op!(logical_and, mlx_logical_and);
binary_op!(maximum, mlx_maximum);
binary_op!(minimum, mlx_minimum);
binary_op!(power, mlx_power);
binary_op!(equal, mlx_equal);
binary_op!(not_equal, mlx_not_equal);

unary_op!(sigmoid, mlx_sigmoid);
unary_op!(tanh, mlx_tanh);
unary_op!(erf, mlx_erf);
unary_op!(exp, mlx_exp);
unary_op!(log, mlx_log);
unary_op!(log1p, mlx_log1p);
unary_op!(negative, mlx_negative);

/// silu(x) = x * sigmoid(x)
pub fn silu(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    let sig = sigmoid(x, s);
    multiply(x, &sig, s)
}

/// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))  — exact GELU used for GEGLU activations.
pub fn gelu(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    let dtype = x.dtype();
    let mk_scalar = |v: f32| cached_scalar(v, dtype);
    let inv_sqrt2 = mk_scalar(std::f32::consts::FRAC_1_SQRT_2);
    let scaled = multiply(x, &inv_sqrt2, s);
    let erf_val = erf(&scaled, s);
    let one_plus_erf = add(&erf_val, &mk_scalar(1.0), s);
    let half_x = multiply(x, &mk_scalar(0.5), s);
    multiply(&half_x, &one_plus_erf, s)
}

/// gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// Matches mlx-lm's `nn.gelu_approx`. Used by Gemma4 per-layer input gate.
pub fn gelu_approx(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    let dtype = x.dtype();
    let mk_scalar = |v: f32| cached_scalar(v, dtype);
    // sqrt(2/π)
    let sqrt_2_over_pi: f32 = 0.797_884_6;
    let coeff: f32 = 0.044_715;
    let x2 = multiply(x, x, s);
    let x3 = multiply(&x2, x, s);
    let cx3 = multiply(&mk_scalar(coeff), &x3, s);
    let inner = add(x, &cx3, s);
    let t = tanh(&multiply(&mk_scalar(sqrt_2_over_pi), &inner, s), s);
    let one_plus_t = add(&mk_scalar(1.0), &t, s);
    multiply(&multiply(&mk_scalar(0.5), x, s), &one_plus_t, s)
}

/// Compute `gelu_approx(gate) * x` through AX's direct MLX C++ shim.
///
/// This keeps the exact mlx-lm Gemma-family math but collapses the Rust ->
/// C++ FFI boundary for the scalar-heavy activation chain to one call.
/// If the direct shim reports an error, fall back to the portable wrapper
/// composition rather than surfacing a hard runtime failure.
pub fn gelu_approx_mul(gate: &MlxArray, x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_gelu_approx_mul(&mut res.inner, gate.inner, x.inner, stream);
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();
    multiply(&gelu_approx(gate, s), x, s)
}

/// mlxcel `compiled_geglu_approx_activation` residual: process-static
/// `mx::compile(shapeless=true)` over `gelu_tanh_approx(gate) * x`.
///
/// Gated by `AX_MLX_COMPILED_GEGLU_ACTIVATION=1` in C++ (fail-closed when
/// unset). Returns `None` when the env kill-switch is off or the compiled
/// path errors so callers can fall back to Metal / imperative GEGLU.
pub fn compiled_geglu_approx_activation(
    gate: &MlxArray,
    x: &MlxArray,
    s: Option<&MlxStream>,
) -> Option<MlxArray> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc =
            ax_mlx_compiled_geglu_approx_activation(&mut res.inner, gate.inner, x.inner, stream);
        if rc == 0 {
            crate::op_count::bump();
            return Some(res);
        }
    }
    crate::error::clear_stale_error();
    None
}

/// OpenAI GPT-OSS / llama.cpp `SWIGLU_OAI` expert activation.
///
/// Matches mlx-lm `gpt_oss.swiglu` and ggml `ggml_swiglu_oai`:
///   gate' = min(gate, limit)
///   up'   = clamp(up, -limit, limit)
///   out   = gate' * sigmoid(alpha * gate') * (up' + 1)
/// with alpha=1.702, limit=7.0.
///
/// Argument order matches mlx-lm SwiGLU(up, gate) call sites in SwitchGLU
/// (`activation(x_up, x_gate)` → `swiglu(x_linear=up, x_glu=gate)`).
pub fn swiglu_oai(up: &MlxArray, gate: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    swiglu_oai_with_params(up, gate, 1.702, 7.0, s)
}

/// Parameterized OpenAI MoE SwiGLU (see [`swiglu_oai`]).
pub(crate) fn swiglu_oai_with_params(
    up: &MlxArray,
    gate: &MlxArray,
    alpha: f32,
    limit: f32,
    s: Option<&MlxStream>,
) -> MlxArray {
    let dtype = gate.dtype();
    let limit_arr = cached_scalar(limit, dtype);
    let neg_limit = cached_scalar(-limit, dtype);
    let alpha_arr = cached_scalar(alpha, dtype);
    let one = cached_scalar(1.0, dtype);

    // gate' = min(gate, limit)  — mlx-lm only upper-clips the glu branch
    let gate_c = minimum(gate, &limit_arr, s);
    // up' = clamp(up, -limit, limit)
    let up_c = clip(up, &neg_limit, &limit_arr, s);

    let glu_scaled = multiply(&gate_c, &alpha_arr, s);
    let sig = sigmoid(&glu_scaled, s);
    let out_glu = multiply(&gate_c, &sig, s);
    let up_p1 = add(&up_c, &one, s);
    multiply(&out_glu, &up_p1, s)
}

/// Compute `silu(gate) * x` through AX's direct MLX C++ shim.
///
/// This preserves Qwen-family SwiGLU math while collapsing the portable
/// `sigmoid + multiply + multiply` wrapper chain behind one FFI call. If the
/// direct shim reports an error, fall back to the portable wrapper composition.
pub fn silu_mul(gate: &MlxArray, x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_silu_mul(&mut res.inner, gate.inner, x.inner, stream);
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();
    multiply(&silu(gate, s), x, s)
}

/// Compute `matmul(gelu_approx(gate) * x, weight)` through AX's direct MLX C++ shim.
///
/// This is a microbenchmark/probe surface for the direct-MLX PRD. It collapses
/// the activation and the following dense projection behind one Rust FFI call.
/// Runtime model code should keep using the portable path until a real-shape
/// artifact proves this candidate clears the promotion gate.
pub fn gelu_approx_mul_matmul(
    gate: &MlxArray,
    x: &MlxArray,
    weight: &MlxArray,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_gelu_approx_mul_matmul(
            &mut res.inner,
            gate.inner,
            x.inner,
            weight.inner,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();
    matmul(&multiply(&gelu_approx(gate, s), x, s), weight, s)
}

/// Compute `quantized_matmul(gelu_approx(gate) * x, weight, ...)`.
/// Affine is the only quantization mode with a group-bias channel; the
/// block-float modes are fully determined by `(group_size, bits)`. Mirrors
/// the shim-side `infer_qmm_mode` so the Rust fallbacks of the fused qmm
/// helpers agree with their C++ entries on scales-only weights.
fn infer_fused_qmm_mode(has_biases: bool, group_size: i32, bits: i32) -> MlxQuantizationMode {
    if has_biases {
        MlxQuantizationMode::Affine
    } else if bits == 4 && group_size == 32 {
        MlxQuantizationMode::Mxfp4
    } else if bits == 8 && group_size == 32 {
        MlxQuantizationMode::Mxfp8
    } else if bits == 4 && group_size == 16 {
        MlxQuantizationMode::Nvfp4
    } else {
        MlxQuantizationMode::Affine
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gelu_approx_mul_quantized_matmul(
    gate: &MlxArray,
    x: &MlxArray,
    weight: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_gelu_approx_mul_quantized_matmul(
            &mut res.inner,
            gate.inner,
            x.inner,
            weight.inner,
            scales.inner,
            biases,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();
    let hidden = gelu_approx_mul(gate, x, s);
    quantized_matmul_with_mode(
        &hidden,
        weight,
        scales,
        biases,
        true,
        Some(group_size),
        Some(bits),
        infer_fused_qmm_mode(biases.is_some(), group_size, bits),
        s,
    )
}

/// Compute `quantized_matmul(silu(gate) * x, weight, ...)` in one C++ call.
/// Qwen SwiGLU analog of [`gelu_approx_mul_quantized_matmul`].
#[allow(clippy::too_many_arguments)]
pub fn silu_mul_quantized_matmul(
    gate: &MlxArray,
    x: &MlxArray,
    weight: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<MlxArray> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_silu_mul_quantized_matmul(
            &mut res.inner,
            gate.inner,
            x.inner,
            weight.inner,
            scales.inner,
            biases,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some(res);
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Compute a packed dense GEGLU FFN block through AX's direct MLX C++ shim.
///
/// The expression is:
///
/// ```text
/// gate_up = quantized_matmul(x, gate_up_weight, ...)
/// gate, up = split(gate_up, 2, axis=-1)
/// hidden = gelu_approx(gate) * up
/// out = quantized_matmul(hidden, down_weight, ...)
/// ```
///
/// Dual affine qmm + GEGLU product in one C++ call (no `mx::compile`):
/// `gelu_approx(qmm(x,gate)) * qmm(x,up)`.
///
/// mlxcel multi-token bits=8 residual: two `UnifiedLinear::forward` +
/// `compiled_geglu_approx_activation`. Collapses three host round-trips into
/// one for pure prefill gate_up residual (~3.3s). Fail-closed: returns `None`
/// on shim error so callers keep portable split + Metal GEGLU.
#[allow(clippy::too_many_arguments)]
pub fn dual_qmm_geglu(
    x: &MlxArray,
    gate_weight: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_weight: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<MlxArray> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_dual_qmm_geglu(
            &mut res.inner,
            x.inner,
            gate_weight.inner,
            gate_scales.inner,
            gate_biases.inner,
            up_weight.inner,
            up_scales.inner,
            up_biases.inner,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some(res);
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Dual affine qmm + SwiGLU in one C++ call: `silu(qmm(x,gate)) * qmm(x,up)`.
/// Qwen analog of [`dual_qmm_geglu`]. No `mx::compile`, no down fuse, no
/// dual-stream.
#[allow(clippy::too_many_arguments)]
pub fn dual_qmm_swiglu(
    x: &MlxArray,
    gate_weight: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_weight: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<MlxArray> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_dual_qmm_swiglu(
            &mut res.inner,
            x.inner,
            gate_weight.inner,
            gate_scales.inner,
            gate_biases.inner,
            up_weight.inner,
            up_scales.inner,
            up_biases.inner,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some(res);
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Dual affine qmm only (no compile, no GEGLU) — one C++ call returns
/// `(gate, up)` so Metal GEGLU stays on the production path.
///
/// Engages when `AX_MLX_DUAL_AFFINE_QMM=1` **or** `AX_MLX_DUAL_STREAM_GATE_UP=1`
/// (default both OFF). Dual-stream issues gate/up on two process-static GPU
/// streams for potential M5 Max concurrency. Fail-closed → `None`.
#[allow(clippy::too_many_arguments)]
pub fn dual_affine_qmm(
    x: &MlxArray,
    gate_weight: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_weight: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut gate = MlxArray::empty();
        let mut up = MlxArray::empty();
        let rc = ax_mlx_dual_affine_qmm(
            &mut gate.inner,
            &mut up.inner,
            x.inner,
            gate_weight.inner,
            gate_scales.inner,
            gate_biases.inner,
            up_weight.inner,
            up_scales.inner,
            up_biases.inner,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((gate, up));
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Issue gate/up affine qmm on two process-static GPU streams. No env gate:
/// the Rust family/seq predicate is the only switch.
#[allow(clippy::too_many_arguments)]
pub fn dual_stream_affine_qmm(
    x: &MlxArray,
    gate_weight: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_weight: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut gate = MlxArray::empty();
        let mut up = MlxArray::empty();
        let rc = ax_mlx_dual_stream_affine_qmm(
            &mut gate.inner,
            &mut up.inner,
            x.inner,
            gate_weight.inner,
            gate_scales.inner,
            gate_biases.inner,
            up_weight.inner,
            up_scales.inner,
            up_biases.inner,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((gate, up));
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Like [`dual_affine_qmm`] but skips the C++ env gate and dual-stream.
/// Qwen split-prefill uses this so the Rust family/seq flag is the only switch.
#[allow(clippy::too_many_arguments)]
pub fn dual_affine_qmm_forced(
    x: &MlxArray,
    gate_weight: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_weight: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut gate = MlxArray::empty();
        let mut up = MlxArray::empty();
        let rc = ax_mlx_dual_affine_qmm_forced(
            &mut gate.inner,
            &mut up.inner,
            x.inner,
            gate_weight.inner,
            gate_scales.inner,
            gate_biases.inner,
            up_weight.inner,
            up_scales.inner,
            up_biases.inner,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((gate, up));
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Shape-specific compile of multi-token **gate + up** affine qmm only
/// (profile residual: gate_up dominates pure Gemma prefill). Leaves gelu/down
/// outside the compile window. **Default OFF** after mbp-m5 pure wall ~+2.1%;
/// enable with `AX_MLX_COMPILED_DUAL_GATE_UP=1`.
#[allow(clippy::too_many_arguments)]
pub fn compiled_dual_gate_up_qmm(
    x: &MlxArray,
    gate_weight: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_weight: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut gate = MlxArray::empty();
        let mut up = MlxArray::empty();
        let rc = ax_mlx_compiled_dual_gate_up_qmm(
            &mut gate.inner,
            &mut up.inner,
            x.inner,
            gate_weight.inner,
            gate_scales.inner,
            gate_biases.inner,
            up_weight.inner,
            up_scales.inner,
            up_biases.inner,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((gate, up));
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Same compile as [`compiled_dual_gate_up_qmm`] but skips the Gemma
/// `AX_MLX_COMPILED_DUAL_GATE_UP` kill-switch. Qwen split prefill uses this
/// so two 4-bit affine qmms share one shape-specific `mx::compile` without
/// flipping Gemma (default-OFF after the 13.8k +2.1% wall reject).
#[allow(clippy::too_many_arguments)]
pub fn compiled_dual_gate_up_qmm_forced(
    x: &MlxArray,
    gate_weight: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_weight: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut gate = MlxArray::empty();
        let mut up = MlxArray::empty();
        let rc = ax_mlx_compiled_dual_gate_up_qmm_forced(
            &mut gate.inner,
            &mut up.inner,
            x.inner,
            gate_weight.inner,
            gate_scales.inner,
            gate_biases.inner,
            up_weight.inner,
            up_scales.inner,
            up_biases.inner,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((gate, up));
        }
    }
    crate::error::clear_stale_error();
    None
}

/// mlxcel `compiled_gelu_approx_mlp_forward` for **split** gate/up/down affine
/// projections. Compiles the full qmm+gelu_approx+qmm chain:
/// - gs64/bits=4 and single-token decode: `shapeless=true` (mlxcel #680)
/// - AXQ 4-bit (`group_size != 64`) seq=128: opt-in shape-specific compile
///   via `AX_MLX_COMPILED_QGELU_AXQ_P128=1` (default OFF after wash)
/// - multi-token non-4bit (flip Gemma MLP bits=8): opt-in shape-specific
///   compile via `AX_MLX_COMPILED_QGELU_PREFILL_SHAPED=1` (mlxcel #705 pattern;
///   default OFF after mbp-m5 pure wall measured ~+2% regression)
///
/// Returns `None` when the quant layout is unsupported, the kill-switch
/// `AX_MLX_COMPILED_QGELU_MLP=0` is set, or an opt-in shape-specific path
/// is left unset. Fail-closed → portable split path.
#[allow(clippy::too_many_arguments)]
pub fn compiled_gelu_approx_split_mlp(
    x: &MlxArray,
    gate_weight: &MlxArray,
    gate_scales: &MlxArray,
    gate_biases: &MlxArray,
    up_weight: &MlxArray,
    up_scales: &MlxArray,
    up_biases: &MlxArray,
    down_weight: &MlxArray,
    down_scales: &MlxArray,
    down_biases: &MlxArray,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<MlxArray> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_compiled_gelu_approx_split_mlp(
            &mut res.inner,
            x.inner,
            gate_weight.inner,
            gate_scales.inner,
            gate_biases.inner,
            up_weight.inner,
            up_scales.inner,
            up_biases.inner,
            down_weight.inner,
            down_scales.inner,
            down_biases.inner,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some(res);
        }
    }
    crate::error::clear_stale_error();
    None
}

/// This remains a probe surface until a real-shape artifact clears the PRD's
/// promotion gate. Production model code should continue using the portable
/// route unless a later commit adds explicit routing and kill switches.
#[allow(clippy::too_many_arguments)]
pub fn gelu_approx_quantized_ffn(
    x: &MlxArray,
    gate_up_weight: &MlxArray,
    gate_up_scales: &MlxArray,
    gate_up_biases: Option<&MlxArray>,
    down_weight: &MlxArray,
    down_scales: &MlxArray,
    down_biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let gate_up_biases = gate_up_biases
            .map(|biases| biases.inner)
            .unwrap_or_else(null_ffi_array);
        let down_biases = down_biases
            .map(|biases| biases.inner)
            .unwrap_or_else(null_ffi_array);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_gelu_approx_quantized_ffn(
            &mut res.inner,
            x.inner,
            gate_up_weight.inner,
            gate_up_scales.inner,
            gate_up_biases,
            down_weight.inner,
            down_scales.inner,
            down_biases,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();
    let gate_up = quantized_matmul(
        x,
        gate_up_weight,
        gate_up_scales,
        gate_up_biases,
        true,
        Some(group_size),
        Some(bits),
        s,
    );
    let packed_dim = gate_up
        .shape()
        .last()
        .copied()
        .expect("gate/up projection must have a last dimension");
    let half = packed_dim / 2;
    let gate = slice_last_dim(&gate_up, 0, half, s);
    let up = slice_last_dim(&gate_up, half, packed_dim, s);
    let hidden = gelu_approx_mul(&gate, &up, s);
    quantized_matmul(
        &hidden,
        down_weight,
        down_scales,
        down_biases,
        true,
        Some(group_size),
        Some(bits),
        s,
    )
}

/// Probe-only direct C++ shim for:
///
/// ```text
/// as_strided([B, S, H * D] -> [B, H, S, D])
/// rms_norm(..., norm)
/// rope(...)
/// ```
///
/// This is intentionally not used by production model code yet. It gives the
/// direct-MLX PRD a narrow measurement surface for the QK-norm + RoPE region
/// before any routing or kill-switch work is considered.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_bhsd_from_proj(
    proj: &MlxArray,
    norm: Option<&MlxArray>,
    n_heads: i32,
    head_dim: i32,
    eps: f32,
    rope_dims: i32,
    traditional: bool,
    base: Option<f32>,
    offset: i32,
    freqs: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_qk_norm_rope_bhsd_from_proj(
            &mut res.inner,
            proj.inner,
            norm.map(|n| n.inner).unwrap_or_else(null_ffi_array),
            n_heads,
            head_dim,
            eps,
            rope_dims,
            i32::from(traditional),
            i32::from(base.is_some()),
            base.unwrap_or(1.0),
            offset,
            freqs.map(|f| f.inner).unwrap_or_else(null_ffi_array),
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();

    let shape = proj.shape();
    let batch = shape.first().copied().unwrap_or(1);
    let seq = shape.get(1).copied().unwrap_or(1);
    let width = i64::from(n_heads) * i64::from(head_dim);
    let bhsd = as_strided(
        proj,
        &[batch, n_heads, seq, head_dim],
        &[i64::from(seq) * width, i64::from(head_dim), width, 1],
        0,
        s,
    );
    let normed = crate::fast::rms_norm(&bhsd, norm, eps, s);
    crate::fast::rope(&normed, rope_dims, traditional, base, 1.0, offset, freqs, s)
}

/// Probe-only direct C++ shim for Gemma4's dense post-attention FFN block.
///
/// The expression mirrors the non-MoE Gemma4 post-attention region after the
/// attention output projection has already been post-normalized:
///
/// ```text
/// residual = hidden + attn_out
/// normed = rms_norm(residual, ffn_norm)
/// ffn = quantized_down(geglu(quantized_gate_up(normed)))
/// ffn = rms_norm(ffn, ffn_post_norm)      # optional
/// out = residual + ffn
/// out = out * layer_scalar                # optional
/// ```
///
/// This is intentionally a probe surface. It validates a larger graph boundary
/// than the no-go standalone FFN shim before production routing is considered.
#[allow(clippy::too_many_arguments)]
pub fn gemma4_post_attn_ffn_block(
    hidden: &MlxArray,
    attn_out: &MlxArray,
    ffn_norm: &MlxArray,
    ffn_post_norm: Option<&MlxArray>,
    layer_scalar: Option<&MlxArray>,
    gate_up_weight: &MlxArray,
    gate_up_scales: &MlxArray,
    gate_up_biases: Option<&MlxArray>,
    down_weight: &MlxArray,
    down_scales: &MlxArray,
    down_biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    eps: f32,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_gemma4_post_attn_ffn_block(
            &mut res.inner,
            hidden.inner,
            attn_out.inner,
            ffn_norm.inner,
            ffn_post_norm
                .map(|norm| norm.inner)
                .unwrap_or_else(null_ffi_array),
            layer_scalar
                .map(|scalar| scalar.inner)
                .unwrap_or_else(null_ffi_array),
            gate_up_weight.inner,
            gate_up_scales.inner,
            gate_up_biases
                .map(|biases| biases.inner)
                .unwrap_or_else(null_ffi_array),
            down_weight.inner,
            down_scales.inner,
            down_biases
                .map(|biases| biases.inner)
                .unwrap_or_else(null_ffi_array),
            group_size,
            bits,
            eps,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();

    let residual = add(hidden, attn_out, s);
    let normed = crate::fast::rms_norm(&residual, Some(ffn_norm), eps, s);
    let gate_up = quantized_matmul(
        &normed,
        gate_up_weight,
        gate_up_scales,
        gate_up_biases,
        true,
        Some(group_size),
        Some(bits),
        s,
    );
    let packed_dim = gate_up
        .shape()
        .last()
        .copied()
        .expect("gate/up projection must have a last dimension");
    let half = packed_dim / 2;
    let gate = slice_last_dim(&gate_up, 0, half, s);
    let up = slice_last_dim(&gate_up, half, packed_dim, s);
    let ffn_hidden = gelu_approx_mul(&gate, &up, s);
    let mut ffn_out = quantized_matmul(
        &ffn_hidden,
        down_weight,
        down_scales,
        down_biases,
        true,
        Some(group_size),
        Some(bits),
        s,
    );
    if let Some(norm) = ffn_post_norm {
        ffn_out = crate::fast::rms_norm(&ffn_out, Some(norm), eps, s);
    }
    let out = add(&residual, &ffn_out, s);
    if let Some(scalar) = layer_scalar {
        multiply(&out, scalar, s)
    } else {
        out
    }
}

/// Direct C++ shim for Qwen linear-attention packed input projection.
///
/// This collapses the packed QKVZ/BA projection plus reshape/slice/concat
/// staging into one Rust FFI call. It returns `None` on unsupported shapes or
/// shim failure so model code can keep the portable MLX composition as the
/// fail-closed fallback.
#[allow(clippy::too_many_arguments)]
pub fn qwen_linear_attention_inputs_packed(
    x: &MlxArray,
    qkvz_weight: &MlxArray,
    qkvz_scales: Option<&MlxArray>,
    qkvz_biases: Option<&MlxArray>,
    ba_weight: &MlxArray,
    ba_scales: Option<&MlxArray>,
    ba_biases: Option<&MlxArray>,
    num_key_heads: i32,
    num_value_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
    group_size: i32,
    bits: i32,
    ba_group_size: i32,
    ba_bits: i32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut qkv = MlxArray::empty();
        let mut z = MlxArray::empty();
        let mut a = MlxArray::empty();
        let mut b = MlxArray::empty();
        let rc = ax_mlx_qwen_linear_attention_inputs_packed(
            &mut qkv.inner,
            &mut z.inner,
            &mut a.inner,
            &mut b.inner,
            x.inner,
            qkvz_weight.inner,
            qkvz_scales
                .map(|scales| scales.inner)
                .unwrap_or_else(null_ffi_array),
            qkvz_biases
                .map(|biases| biases.inner)
                .unwrap_or_else(null_ffi_array),
            ba_weight.inner,
            ba_scales
                .map(|scales| scales.inner)
                .unwrap_or_else(null_ffi_array),
            ba_biases
                .map(|biases| biases.inner)
                .unwrap_or_else(null_ffi_array),
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            group_size,
            bits,
            ba_group_size,
            ba_bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((qkv, z, a, b));
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Shape-specific `mx::compile` of [`qwen_linear_attention_inputs_packed`].
///
/// Requires scales on both QKVZ and BA, with group biases either on both
/// (affine) or on neither (scales-only block-float; the shim infers the mode
/// from `group_size`/`bits`). Returns `None` on unsupported shapes, mixed
/// bias contracts, or compile failure so the caller can keep the imperative
/// packed path.
#[allow(clippy::too_many_arguments)]
pub fn qwen_linear_attention_inputs_packed_compiled(
    x: &MlxArray,
    qkvz_weight: &MlxArray,
    qkvz_scales: &MlxArray,
    qkvz_biases: Option<&MlxArray>,
    ba_weight: &MlxArray,
    ba_scales: &MlxArray,
    ba_biases: Option<&MlxArray>,
    num_key_heads: i32,
    num_value_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
    group_size: i32,
    bits: i32,
    ba_group_size: i32,
    ba_bits: i32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut qkv = MlxArray::empty();
        let mut z = MlxArray::empty();
        let mut a = MlxArray::empty();
        let mut b = MlxArray::empty();
        let rc = ax_mlx_qwen_linear_attention_inputs_packed_compiled(
            &mut qkv.inner,
            &mut z.inner,
            &mut a.inner,
            &mut b.inner,
            x.inner,
            qkvz_weight.inner,
            qkvz_scales.inner,
            qkvz_biases
                .map(|biases| biases.inner)
                .unwrap_or_else(null_ffi_array),
            ba_weight.inner,
            ba_scales.inner,
            ba_biases
                .map(|biases| biases.inner)
                .unwrap_or_else(null_ffi_array),
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            group_size,
            bits,
            ba_group_size,
            ba_bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((qkv, z, a, b));
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Direct C++ shim for the Qwen linear-attention post-input block: depthwise
/// `conv1d` (with cached state carry), SiLU, last-dim split, head-major reshape,
/// per-head RMSNorm on q and k, and scale-by-precomputed-constants.
///
/// This is everything between `qwen_linear_attention_inputs_packed` and the
/// `qwen35_gated_delta_v3` custom Metal kernel, fused into one Rust→C++ FFI
/// round-trip. The per-decode-token FFI dispatch count for a Qwen 3.6 27B
/// linear-attention layer drops from ~14 to 1 — the savings are bounded by the
/// AX-vs-mlx-python marshalling delta (~250ns/op), not by GPU work.
///
/// Returns `(q, k, v, new_conv_state)` on success, or `None` if the C++ side
/// rejects the shapes (the caller must keep the portable composition as a
/// fail-closed fallback).
#[allow(clippy::too_many_arguments)]
pub fn qwen_linear_attention_post_input(
    qkv: &MlxArray,
    conv_weight: &MlxArray,
    cached_conv_state: Option<&MlxArray>,
    num_key_heads: i32,
    key_head_dim: i32,
    num_value_heads: i32,
    value_head_dim: i32,
    conv_kernel_dim: i32,
    q_scale: f32,
    k_scale: f32,
    rms_norm_eps: f32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut q = MlxArray::empty();
        let mut k = MlxArray::empty();
        let mut v = MlxArray::empty();
        let mut new_conv_state = MlxArray::empty();
        let rc = ax_mlx_qwen_linear_attention_post_input(
            &mut q.inner,
            &mut k.inner,
            &mut v.inner,
            &mut new_conv_state.inner,
            qkv.inner,
            conv_weight.inner,
            cached_conv_state
                .map(|state| state.inner)
                .unwrap_or_else(null_ffi_array),
            num_key_heads,
            key_head_dim,
            num_value_heads,
            value_head_dim,
            conv_kernel_dim,
            q_scale,
            k_scale,
            rms_norm_eps,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((q, k, v, new_conv_state));
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Shape-specific `mx::compile` of [`qwen_linear_attention_post_input`].
///
/// Requires an explicit conv-state tensor (chunk 1 passes zeros). Returns
/// `None` on unsupported shapes or compile failure.
#[allow(clippy::too_many_arguments)]
pub fn qwen_linear_attention_post_input_compiled(
    qkv: &MlxArray,
    conv_weight: &MlxArray,
    cached_conv_state: &MlxArray,
    num_key_heads: i32,
    key_head_dim: i32,
    num_value_heads: i32,
    value_head_dim: i32,
    conv_kernel_dim: i32,
    q_scale: f32,
    k_scale: f32,
    rms_norm_eps: f32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut q = MlxArray::empty();
        let mut k = MlxArray::empty();
        let mut v = MlxArray::empty();
        let mut new_conv_state = MlxArray::empty();
        let rc = ax_mlx_qwen_linear_attention_post_input_compiled(
            &mut q.inner,
            &mut k.inner,
            &mut v.inner,
            &mut new_conv_state.inner,
            qkv.inner,
            conv_weight.inner,
            cached_conv_state.inner,
            num_key_heads,
            key_head_dim,
            num_value_heads,
            value_head_dim,
            conv_kernel_dim,
            q_scale,
            k_scale,
            rms_norm_eps,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((q, k, v, new_conv_state));
        }
    }
    crate::error::clear_stale_error();
    None
}

/// Fast LayerNorm over the last dimension.
///
/// This is the same MLX primitive used by `nn.LayerNorm`; unlike `rms_norm`,
/// it subtracts the mean and applies both affine weight and bias.
pub fn layer_norm(
    x: &MlxArray,
    weight: &MlxArray,
    bias: &MlxArray,
    eps: f32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_fast_layer_norm",
            ffi::mlx_fast_layer_norm(
                &mut res.inner,
                x.inner,
                weight.inner,
                bias.inner,
                eps,
                stream,
            )
        );
        res
    }
}

/// Compute `(add(x, y), rms_norm(add(x, y), norm_weight, eps))` in one C++ call.
///
/// Both outputs are usually needed immediately: the residual sum for the
/// downstream FFN residual add, and the normed output as the FFN matmul
/// input. Returning them together saves one MLX graph node per call site
/// versus the two-step composition.
///
/// Falls back to `add + rms_norm` on shim error.
pub fn add_rms_norm_pair(
    x: &MlxArray,
    y: &MlxArray,
    norm_weight: &MlxArray,
    eps: f32,
    s: Option<&MlxStream>,
) -> (MlxArray, MlxArray) {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut residual = MlxArray::empty();
        let mut normed = MlxArray::empty();
        let rc = ax_mlx_add_rms_norm_pair(
            &mut residual.inner,
            &mut normed.inner,
            x.inner,
            y.inner,
            norm_weight.inner,
            eps,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            crate::op_count::bump();
            return (residual, normed);
        }
    }
    crate::error::clear_stale_error();
    let residual = add(x, y, s);
    let normed = crate::fast::rms_norm(&residual, Some(norm_weight), eps, s);
    (residual, normed)
}

/// `astype(silu_mul(astype(gate, f32), astype(rms_norm(hidden), f32)), hidden.dtype)`.
///
/// One C++ FFI that builds the same five MLX ops as the exact portable
/// linear-attention RMS+SiLU gate. Not `mx::compile` and not Metal.
pub fn rms_norm_silu_mul_normed(
    hidden: &MlxArray,
    gate: &MlxArray,
    norm_weight: &MlxArray,
    eps: f32,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_rms_norm_silu_mul_normed(
            &mut res.inner,
            hidden.inner,
            gate.inner,
            norm_weight.inner,
            eps,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();
    let normed = crate::fast::rms_norm(hidden, Some(norm_weight), eps, s);
    let gated = silu_mul(
        &astype(gate, MlxDtype::Float32, s),
        &astype(&normed, MlxDtype::Float32, s),
        s,
    );
    astype(&gated, hidden.dtype(), s)
}

/// Compute `rms_norm(quantized_matmul(x, weight, ...), norm_weight, eps)` in one C++ call.
///
/// Fuses a quantized down-projection with the following RMSNorm (post-FFN
/// norm pattern in Gemma-family dense layers). Saves one MLX graph node per
/// call site.
///
/// Falls back to `quantized_matmul + rms_norm` on shim error.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_rms_norm(
    x: &MlxArray,
    weight: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    norm_weight: &MlxArray,
    eps: f32,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_quantized_matmul_rms_norm(
            &mut res.inner,
            x.inner,
            weight.inner,
            scales.inner,
            biases_raw,
            group_size,
            bits,
            norm_weight.inner,
            eps,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();
    let projected = quantized_matmul_with_mode(
        x,
        weight,
        scales,
        biases,
        true,
        Some(group_size),
        Some(bits),
        infer_fused_qmm_mode(biases.is_some(), group_size, bits),
        s,
    );
    crate::fast::rms_norm(&projected, Some(norm_weight), eps, s)
}

/// Compute `quantized_matmul(rms_norm(x, norm_weight, eps), weight, …)` in one
/// C++ call (attn input-norm + packed QKV / linear pattern).
///
/// Falls back to `rms_norm + quantized_matmul` on shim error.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_quantized_matmul(
    x: &MlxArray,
    norm_weight: &MlxArray,
    eps: f32,
    weight: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let mut res = MlxArray::empty();
        let rc = ax_mlx_rms_norm_quantized_matmul(
            &mut res.inner,
            x.inner,
            norm_weight.inner,
            eps,
            weight.inner,
            scales.inner,
            biases_raw,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return res;
        }
    }
    crate::error::clear_stale_error();
    let normed = crate::fast::rms_norm(x, Some(norm_weight), eps, s);
    quantized_matmul_with_mode(
        &normed,
        weight,
        scales,
        biases,
        true,
        Some(group_size),
        Some(bits),
        infer_fused_qmm_mode(biases.is_some(), group_size, bits),
        s,
    )
}

fn fused_debug_error(entry: &str) {
    if std::env::var_os("AX_MLX_PREFILL_TIME_DEBUG").is_some()
        && let Some(msg) = crate::error::take_last_error()
    {
        eprintln!("AX_PREFILL_TIME_DEBUG fused shim error [{entry}]: {msg}");
        return;
    }
    crate::error::clear_stale_error();
}

/// One-call fused offset-0 prefill attention (mlxcel
/// `fused_causal_prefill_attention` residual): rms_norm -> packed-QKV qmm ->
/// split/BHSD -> optional per-head QK rms_norm -> rope(offset 0) -> maskless
/// "causal" SDPA -> o-proj qmm. Returns `(attn_out, k, v)` with K/V roped in
/// BHSD for the caller's cache append. Returns `None` when the shim call
/// fails so callers keep their portable per-op path as fallback.
#[allow(clippy::too_many_arguments)]
pub fn fused_causal_prefill_attention(
    x: &MlxArray,
    attn_norm: &MlxArray,
    eps: f32,
    qkv_weight: &MlxArray,
    qkv_scales: &MlxArray,
    qkv_biases: Option<&MlxArray>,
    q_norm: Option<&MlxArray>,
    k_norm: Option<&MlxArray>,
    qk_eps: f32,
    v_norm_no_scale: bool,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_dims: i32,
    rope_base: f32,
    rope_freqs: Option<&MlxArray>,
    scale: f32,
    o_weight: &MlxArray,
    o_scales: &MlxArray,
    o_biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    post_norm: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut out = MlxArray::empty();
        let mut k_out = MlxArray::empty();
        let mut v_out = MlxArray::empty();
        let rc = ax_mlx_fused_causal_prefill_attention(
            &mut out.inner,
            &mut k_out.inner,
            &mut v_out.inner,
            x.inner,
            attn_norm.inner,
            eps,
            qkv_weight.inner,
            qkv_scales.inner,
            qkv_biases.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            q_norm.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            k_norm.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            qk_eps,
            v_norm_no_scale,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            rope_base,
            rope_freqs.map(|f| f.inner).unwrap_or_else(null_ffi_array),
            scale,
            o_weight.inner,
            o_scales.inner,
            o_biases.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            group_size,
            bits,
            post_norm.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((out, k_out, v_out));
        }
    }
    fused_debug_error("fused_causal_prefill_attention");
    None
}

/// Split-projection variant of [`fused_causal_prefill_attention`] for
/// checkpoints that ship separate Q/K/V quantized weights.
#[allow(clippy::too_many_arguments)]
pub fn fused_causal_prefill_attention_split(
    x: &MlxArray,
    attn_norm: &MlxArray,
    eps: f32,
    q_w: (&MlxArray, &MlxArray, Option<&MlxArray>),
    k_w: (&MlxArray, &MlxArray, Option<&MlxArray>),
    v_w: (&MlxArray, &MlxArray, Option<&MlxArray>),
    q_norm: Option<&MlxArray>,
    k_norm: Option<&MlxArray>,
    qk_eps: f32,
    v_norm_no_scale: bool,
    value_from_key: bool,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_dims: i32,
    rope_base: f32,
    rope_freqs: Option<&MlxArray>,
    scale: f32,
    o_weight: &MlxArray,
    o_scales: &MlxArray,
    o_biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    post_norm: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut out = MlxArray::empty();
        let mut k_out = MlxArray::empty();
        let mut v_out = MlxArray::empty();
        let rc = ax_mlx_fused_causal_prefill_attention_split(
            &mut out.inner,
            &mut k_out.inner,
            &mut v_out.inner,
            x.inner,
            attn_norm.inner,
            eps,
            q_w.0.inner,
            q_w.1.inner,
            q_w.2.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            k_w.0.inner,
            k_w.1.inner,
            k_w.2.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            v_w.0.inner,
            v_w.1.inner,
            v_w.2.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            q_norm.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            k_norm.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            qk_eps,
            v_norm_no_scale,
            value_from_key,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            rope_base,
            rope_freqs.map(|f| f.inner).unwrap_or_else(null_ffi_array),
            scale,
            o_weight.inner,
            o_scales.inner,
            o_biases.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            group_size,
            bits,
            post_norm.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((out, k_out, v_out));
        }
    }
    fused_debug_error("fused_causal_prefill_attention_split");
    None
}

/// Stage 1 of the offset-chunk fused prefill pair: rms_norm -> split QKV
/// qmm -> BHSD -> optional per-head QK/V norms -> rope at `rope_offset`.
/// Returns roped `(q, k, v)` so the caller can run its normal cache append
/// between this and [`fused_sdpa_oproj`]. `None` on shim failure.
#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_rope_split(
    x: &MlxArray,
    attn_norm: &MlxArray,
    eps: f32,
    q_w: (&MlxArray, &MlxArray, Option<&MlxArray>),
    k_w: (&MlxArray, &MlxArray, Option<&MlxArray>),
    v_w: (&MlxArray, &MlxArray, Option<&MlxArray>),
    q_norm: Option<&MlxArray>,
    k_norm: Option<&MlxArray>,
    qk_eps: f32,
    v_norm_no_scale: bool,
    value_from_key: bool,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rope_dims: i32,
    rope_base: f32,
    rope_freqs: Option<&MlxArray>,
    rope_offset: i32,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<(MlxArray, MlxArray, MlxArray)> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut q_out = MlxArray::empty();
        let mut k_out = MlxArray::empty();
        let mut v_out = MlxArray::empty();
        let rc = ax_mlx_fused_qkv_rope_split(
            &mut q_out.inner,
            &mut k_out.inner,
            &mut v_out.inner,
            x.inner,
            attn_norm.inner,
            eps,
            q_w.0.inner,
            q_w.1.inner,
            q_w.2.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            k_w.0.inner,
            k_w.1.inner,
            k_w.2.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            v_w.0.inner,
            v_w.1.inner,
            v_w.2.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            q_norm.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            k_norm.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            qk_eps,
            v_norm_no_scale,
            value_from_key,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            rope_base,
            rope_freqs.map(|f| f.inner).unwrap_or_else(null_ffi_array),
            rope_offset,
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some((q_out, k_out, v_out));
        }
    }
    fused_debug_error("fused_qkv_rope_split");
    None
}

/// Stage 2 of the offset-chunk fused prefill pair: bottom-right-aligned
/// "causal" fast SDPA over the full cached K/V plus the o-proj qmm.
/// Inputs are BHSD; K/V may be longer than Q (chunked prefill history).
/// `None` on shim failure.
#[allow(clippy::too_many_arguments)]
pub fn fused_sdpa_oproj(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
    o_weight: &MlxArray,
    o_scales: &MlxArray,
    o_biases: Option<&MlxArray>,
    group_size: i32,
    bits: i32,
    s: Option<&MlxStream>,
) -> Option<MlxArray> {
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut out = MlxArray::empty();
        let rc = ax_mlx_fused_sdpa_oproj(
            &mut out.inner,
            q.inner,
            k.inner,
            v.inner,
            scale,
            mask.map(|m| m.inner).unwrap_or_else(null_ffi_array),
            o_weight.inner,
            o_scales.inner,
            o_biases.map(|b| b.inner).unwrap_or_else(null_ffi_array),
            group_size,
            bits,
            stream,
        );
        if rc == 0 {
            crate::op_count::bump();
            return Some(out);
        }
    }
    fused_debug_error("fused_sdpa_oproj");
    None
}

pub fn astype(a: &MlxArray, dtype: MlxDtype, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_astype",
            ffi::mlx_astype(&mut res.inner, a.inner, dtype.to_ffi(), stream)
        );
        res
    }
}

/// Reinterpret the bytes of `a` as `dtype` without converting values.
///
/// Unlike [`astype`] which converts element values, `view` reinterprets the
/// underlying memory. For example, viewing a u8 array of length 4N as u32
/// produces an array of length N where each element packs 4 original bytes.
pub fn view(a: &MlxArray, dtype: MlxDtype, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_view",
            ffi::mlx_view(&mut res.inner, a.inner, dtype.to_ffi(), stream)
        );
        res
    }
}

pub fn arange(
    start: f64,
    stop: f64,
    step: f64,
    dtype: MlxDtype,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_arange",
            ffi::mlx_arange(&mut res.inner, start, stop, step, dtype.to_ffi(), stream)
        );
        res
    }
}

pub fn reshape(a: &MlxArray, shape: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_reshape",
            ffi::mlx_reshape(&mut res.inner, a.inner, shape.as_ptr(), shape.len(), stream)
        );
        res
    }
}

pub fn broadcast_to(a: &MlxArray, shape: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_broadcast_to",
            ffi::mlx_broadcast_to(&mut res.inner, a.inner, shape.as_ptr(), shape.len(), stream)
        );
        res
    }
}

pub fn transpose(a: &MlxArray, axes: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_transpose_axes",
            ffi::mlx_transpose_axes(&mut res.inner, a.inner, axes.as_ptr(), axes.len(), stream)
        );
        res
    }
}

pub fn expand_dims(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_expand_dims",
            ffi::mlx_expand_dims(&mut res.inner, a.inner, axis, stream)
        );
        res
    }
}

pub fn expand_dims_axes(a: &MlxArray, axes: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_expand_dims_axes",
            ffi::mlx_expand_dims_axes(&mut res.inner, a.inner, axes.as_ptr(), axes.len(), stream)
        );
        res
    }
}

pub fn softmax(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_softmax_axis",
            ffi::mlx_softmax_axis(&mut res.inner, a.inner, axis, false, stream)
        );
        res
    }
}

pub fn softmax_precise(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_softmax_axis",
            ffi::mlx_softmax_axis(&mut res.inner, a.inner, axis, true, stream)
        );
        res
    }
}

pub fn concatenate(arrays: &[&MlxArray], axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let vec = ffi::mlx_vector_array_new();
        for arr in arrays {
            checked_ffi!(
                "mlx_vector_array_append_value",
                ffi::mlx_vector_array_append_value(vec, arr.inner)
            );
        }
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_concatenate_axis",
            ffi::mlx_concatenate_axis(&mut res.inner, vec, axis, stream)
        );
        ffi::mlx_vector_array_free(vec);
        res
    }
}

pub fn stack(arrays: &[&MlxArray], axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let vec = ffi::mlx_vector_array_new();
        for arr in arrays {
            checked_ffi!(
                "mlx_vector_array_append_value",
                ffi::mlx_vector_array_append_value(vec, arr.inner)
            );
        }
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_stack_axis",
            ffi::mlx_stack_axis(&mut res.inner, vec, axis, stream)
        );
        ffi::mlx_vector_array_free(vec);
        res
    }
}

/// Gather rows by integer indices along axis 0.
pub fn take(a: &MlxArray, indices: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_take_axis",
            ffi::mlx_take_axis(&mut res.inner, a.inner, indices.inner, axis, stream)
        );
        res
    }
}

/// Argmax over the last axis.
/// Repeat elements of an array along an axis (repeat-interleave semantics).
///
/// `axis=1` on `[1, n_kv_heads, seq, head_dim]` with `repeats=4` produces
/// `[1, n_heads, seq, head_dim]` suitable for GQA head expansion.
pub fn repeat_axis(a: &MlxArray, repeats: i32, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_repeat_axis",
            ffi::mlx_repeat_axis(&mut res.inner, a.inner, repeats, axis, stream)
        );
        res
    }
}

/// Slice array along every axis.
///
/// `start`, `stop`, and `strides` must each have length equal to `a.ndim()`.
pub fn slice(
    a: &MlxArray,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_slice",
            ffi::mlx_slice(
                &mut res.inner,
                a.inner,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                stream,
            )
        );
        res
    }
}

/// Reinterpret a contiguous array with a new shape and explicit strides.
///
/// Creates a non-contiguous view of the underlying buffer — no data is copied.
/// The caller must ensure the strides describe a valid layout of the source data.
///
/// Typical use: replace reshape + transpose (2 graph nodes) with a single
/// `as_strided` view (1 graph node).
pub fn as_strided(
    a: &MlxArray,
    shape: &[i32],
    strides: &[i64],
    offset: usize,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_as_strided",
            ffi::mlx_as_strided(
                &mut res.inner,
                a.inner,
                shape.as_ptr(),
                shape.len(),
                strides.as_ptr(),
                strides.len(),
                offset,
                stream,
            )
        );
        res
    }
}

/// Split `a` into `num_splits` equal parts along `axis`.
///
/// Returns `num_splits` views of the original data (no copy when the split is
/// uniform).  `a.shape()[axis]` must be divisible by `num_splits`.
pub fn split(a: &MlxArray, num_splits: i32, axis: i32, s: Option<&MlxStream>) -> Vec<MlxArray> {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut out_vec = ffi::mlx_vector_array_new();
        checked_ffi!(
            "mlx_split",
            ffi::mlx_split(&mut out_vec, a.inner, num_splits, axis, stream)
        );
        ensure_error_handler();
        let n = ffi::mlx_vector_array_size(out_vec);
        if n == usize::MAX {
            ffi::mlx_vector_array_free(out_vec);
            panic!("{}", last_error_message("mlx_vector_array_size"));
        }
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut arr = MlxArray::empty();
            checked_ffi!(
                "mlx_vector_array_get",
                ffi::mlx_vector_array_get(&mut arr.inner, out_vec, i)
            );
            result.push(arr);
        }
        ffi::mlx_vector_array_free(out_vec);
        result
    }
}

/// Slice the last dimension of `a` from index `start` to `end` (exclusive).
pub fn slice_last_dim(a: &MlxArray, start: i32, end: i32, s: Option<&MlxStream>) -> MlxArray {
    let ndim = a.ndim();
    assert!(
        ndim > 0,
        "slice_last_dim requires at least 1-dimensional array"
    );
    let shape = a.shape();
    let mut start_vec = vec![0i32; ndim];
    let mut stop_vec = shape.clone();
    let strides_vec = vec![1i32; ndim];
    start_vec[ndim - 1] = start;
    stop_vec[ndim - 1] = end;
    slice(a, &start_vec, &stop_vec, &strides_vec, s)
}

/// Allocate a zero-filled array with the given shape and dtype.
pub fn zeros(shape: &[i32], dtype: MlxDtype, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_zeros",
            ffi::mlx_zeros(
                &mut res.inner,
                shape.as_ptr(),
                shape.len(),
                dtype.to_ffi(),
                stream,
            )
        );
        res
    }
}

/// Return a copy of `src` with `update` written at `src[start:stop:strides]`.
///
/// Mirrors Python `mx.array[start:stop:strides] = update`.  Unlike concatenate,
/// this avoids re-copying existing data — only the `update` region is written.
pub fn slice_update(
    src: &MlxArray,
    update: &MlxArray,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_slice_update",
            ffi::mlx_slice_update(
                &mut res.inner,
                src.inner,
                update.inner,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                stream,
            )
        );
        res
    }
}

/// Slice `a` at array-valued starting indices while keeping a static output
/// shape. `start[i]` applies to `axes[i]`; every other axis starts at zero.
///
/// This is the graph-safe counterpart of [`slice`] for compiled state machines:
/// the start position can change between calls without changing leaf shapes.
pub fn slice_dynamic(
    a: &MlxArray,
    start: &MlxArray,
    axes: &[i32],
    slice_size: &[i32],
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_slice_dynamic",
            ffi::mlx_slice_dynamic(
                &mut res.inner,
                a.inner,
                start.inner,
                axes.as_ptr(),
                axes.len(),
                slice_size.as_ptr(),
                slice_size.len(),
                stream,
            )
        );
        res
    }
}

/// Return `src` with `update` written at array-valued starting indices.
/// `start[i]` applies to `axes[i]`; the update shape determines the extent.
pub fn slice_update_dynamic(
    src: &MlxArray,
    update: &MlxArray,
    start: &MlxArray,
    axes: &[i32],
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_slice_update_dynamic",
            ffi::mlx_slice_update_dynamic(
                &mut res.inner,
                src.inner,
                update.inner,
                start.inner,
                axes.as_ptr(),
                axes.len(),
                stream,
            )
        );
        res
    }
}

pub fn contiguous(a: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_contiguous",
            ffi::mlx_contiguous(&mut res.inner, a.inner, false, stream)
        );
        res
    }
}

pub fn clip(a: &MlxArray, min: &MlxArray, max: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_clip",
            ffi::mlx_clip(&mut res.inner, a.inner, min.inner, max.inner, stream)
        );
        res
    }
}

pub fn where_cond(
    condition: &MlxArray,
    x: &MlxArray,
    y: &MlxArray,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_where",
            ffi::mlx_where(&mut res.inner, condition.inner, x.inner, y.inner, stream)
        );
        res
    }
}

pub fn conv1d(
    input: &MlxArray,
    weight: &MlxArray,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_conv1d",
            ffi::mlx_conv1d(
                &mut res.inner,
                input.inner,
                weight.inner,
                stride,
                padding,
                dilation,
                groups,
                stream,
            )
        );
        res
    }
}

/// 2D convolution. Input is NHWC, weight is OHWI (MLX layout).
///
/// Scalar `stride` / `padding` / `dilation` are applied to both H and W.
pub fn conv2d(
    input: &MlxArray,
    weight: &MlxArray,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_conv2d",
            ffi::mlx_conv2d(
                &mut res.inner,
                input.inner,
                weight.inner,
                stride,
                padding,
                dilation,
                groups,
                stream,
            )
        );
        res
    }
}

pub fn argmax(a: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let ndim = a.ndim();
        assert!(ndim > 0, "argmax requires at least 1-dimensional array");
        let axis = ndim as i32 - 1;
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_argmax_axis",
            ffi::mlx_argmax_axis(&mut res.inner, a.inner, axis, false, stream)
        );
        res
    }
}

pub fn argsort_axis(a: &MlxArray, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_argsort_axis",
            ffi::mlx_argsort_axis(&mut res.inner, a.inner, axis, stream)
        );
        res
    }
}

/// Argpartition along `axis`. Returns indices such that the element at position
/// `kth` would be in its sorted position. Negative kth counts from the end.
pub fn argpartition_axis(a: &MlxArray, kth: i32, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_argpartition_axis",
            ffi::mlx_argpartition_axis(&mut res.inner, a.inner, kth, axis, stream)
        );
        res
    }
}

pub fn take_along_axis(
    a: &MlxArray,
    indices: &MlxArray,
    axis: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_take_along_axis",
            ffi::mlx_take_along_axis(&mut res.inner, a.inner, indices.inner, axis, stream)
        );
        res
    }
}

pub fn put_along_axis(
    a: &MlxArray,
    indices: &MlxArray,
    values: &MlxArray,
    axis: i32,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_put_along_axis",
            ffi::mlx_put_along_axis(
                &mut res.inner,
                a.inner,
                indices.inner,
                values.inner,
                axis,
                stream,
            )
        );
        res
    }
}

pub fn sum_axis(a: &MlxArray, axis: i32, keepdims: bool, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_sum_axis",
            ffi::mlx_sum_axis(&mut res.inner, a.inner, axis, keepdims, stream)
        );
        res
    }
}

/// Cumulative sum along `axis`.
///
/// `reverse` computes the cumsum in reverse order; `inclusive` includes the
/// current element in the running sum (standard inclusive prefix sum when
/// both are default/false/true respectively).
pub fn cumsum(
    a: &MlxArray,
    axis: i32,
    reverse: bool,
    inclusive: bool,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_cumsum",
            ffi::mlx_cumsum(&mut res.inner, a.inner, axis, reverse, inclusive, stream)
        );
        res
    }
}

/// Batched matmul selecting rows of `b` via `rhs_indices`.
///
/// Computes `a[lhs_i] @ b[rhs_i]` for each index pair. `lhs_indices` is
/// always null here (use all rows of `a`). `b` shape: `[N, K, L]`.
pub fn gather_mm(
    a: &MlxArray,
    b: &MlxArray,
    rhs_indices: &MlxArray,
    sorted_indices: bool,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let null_arr = null_ffi_array();
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_gather_mm",
            ffi::mlx_gather_mm(
                &mut res.inner,
                a.inner,
                b.inner,
                null_arr,
                rhs_indices.inner,
                sorted_indices,
                stream,
            )
        );
        res
    }
}

/// Quantized batched matmul selecting experts via `rhs_indices` (affine mode).
///
/// Equivalent to `gather_mm` on the dequantized weight. With `transpose=true`,
/// computes `x @ w[rhs_i].T` for each selected expert.
///
/// For MXFP4 / other modes use [`gather_qmm_with_mode`].
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &MlxArray,
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    rhs_indices: &MlxArray,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    sorted_indices: bool,
    s: Option<&MlxStream>,
) -> MlxArray {
    gather_qmm_with_mode(
        x,
        w,
        scales,
        biases,
        rhs_indices,
        transpose,
        group_size,
        bits,
        MlxQuantizationMode::Affine,
        sorted_indices,
        s,
    )
}

/// Quantized gather-matmul with an explicit quantization [`MlxQuantizationMode`].
///
/// GPT-OSS MoE experts use `mode = Mxfp4` with `group_size = 32`, `bits = 4`,
/// and `biases = None` so expert weights can stay packed in memory instead of
/// being dequantized to BF16 at load time.
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm_with_mode(
    x: &MlxArray,
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    rhs_indices: &MlxArray,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: MlxQuantizationMode,
    sorted_indices: bool,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let null_arr = null_ffi_array();
        let gs = ffi::mlx_optional_int_ {
            has_value: group_size.is_some(),
            value: group_size.unwrap_or(mode.default_group_size()),
        };
        let bs = ffi::mlx_optional_int_ {
            has_value: bits.is_some(),
            value: bits.unwrap_or(mode.default_bits()),
        };
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_gather_qmm",
            ffi::mlx_gather_qmm(
                &mut res.inner,
                x.inner,
                w.inner,
                scales.inner,
                biases_raw,
                null_arr,
                rhs_indices.inner,
                transpose,
                gs,
                bs,
                mode.as_ptr(),
                sorted_indices,
                stream,
            )
        );
        res
    }
}

/// Dequantize a packed-int4 weight tensor to floating point.
///
/// `w` must be the packed uint32 tensor from MLX quantized format.
/// `group_size` and `bits` default to 64 and 4 when `None`.
pub fn dequantize(
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: Option<i32>,
    bits: Option<i32>,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let gs = ffi::mlx_optional_int_ {
            has_value: group_size.is_some(),
            value: group_size.unwrap_or(64),
        };
        let bs = ffi::mlx_optional_int_ {
            has_value: bits.is_some(),
            value: bits.unwrap_or(4),
        };
        let no_dtype = ffi::mlx_optional_dtype_ {
            has_value: false,
            value: ffi::mlx_dtype_::MLX_FLOAT32,
        };
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_dequantize",
            ffi::mlx_dequantize(
                &mut res.inner,
                w.inner,
                scales.inner,
                biases_raw,
                gs,
                bs,
                c"affine".as_ptr(),
                null_ffi_array(),
                no_dtype,
                stream,
            )
        );
        res
    }
}

/// MLX quantization modes supported by `quantize` / `dequantize_with_mode`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxQuantizationMode {
    Affine,
    Mxfp4,
    Mxfp8,
    Nvfp4,
}

impl MlxQuantizationMode {
    fn as_ptr(self) -> *const std::ffi::c_char {
        match self {
            Self::Affine => c"affine".as_ptr(),
            Self::Mxfp4 => c"mxfp4".as_ptr(),
            Self::Mxfp8 => c"mxfp8".as_ptr(),
            Self::Nvfp4 => c"nvfp4".as_ptr(),
        }
    }

    fn default_group_size(self) -> i32 {
        match self {
            Self::Affine => 64,
            Self::Nvfp4 => 16,
            Self::Mxfp4 | Self::Mxfp8 => 32,
        }
    }

    fn default_bits(self) -> i32 {
        match self {
            Self::Mxfp8 => 8,
            Self::Affine | Self::Mxfp4 | Self::Nvfp4 => 4,
        }
    }
}

/// Quantize a floating-point weight matrix along its last axis.
///
/// Affine mode returns `[packed_weight, scales, biases]`; FP modes return
/// `[packed_weight, scales]`, matching MLX's `mx.quantize` contract.
pub fn quantize(
    w: &MlxArray,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: MlxQuantizationMode,
    global_scale: Option<&MlxArray>,
    s: Option<&MlxStream>,
) -> Vec<MlxArray> {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let global_scale = global_scale
            .map(|scale| scale.inner)
            .unwrap_or_else(null_ffi_array);
        let gs = optional_int(group_size, mode.default_group_size());
        let bs = optional_int(bits, mode.default_bits());
        let mut raw = ffi::mlx_vector_array {
            ctx: std::ptr::null_mut(),
        };
        checked_ffi!(
            "mlx_quantize",
            ffi::mlx_quantize(
                &mut raw,
                w.inner,
                gs,
                bs,
                mode.as_ptr(),
                global_scale,
                stream,
            )
        );
        let len = ffi::mlx_vector_array_size(raw);
        if len == usize::MAX {
            ffi::mlx_vector_array_free(raw);
            panic!(
                "{}",
                crate::error::last_error_message("mlx_vector_array_size")
            );
        }
        let mut result = Vec::with_capacity(len);
        for idx in 0..len {
            let mut arr = null_ffi_array();
            checked_ffi!(
                "mlx_vector_array_get",
                ffi::mlx_vector_array_get(&mut arr, raw, idx)
            );
            result.push(MlxArray::from_raw(arr));
        }
        ffi::mlx_vector_array_free(raw);
        result
    }
}

/// Dequantize a matrix produced by `quantize`.
///
/// This is the mode-aware counterpart to `dequantize`, which preserves the
/// legacy affine-only wrapper used by existing model-loading code.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_with_mode(
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: MlxQuantizationMode,
    global_scale: Option<&MlxArray>,
    dtype: Option<MlxDtype>,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let global_scale = global_scale
            .map(|scale| scale.inner)
            .unwrap_or_else(null_ffi_array);
        let gs = optional_int(group_size, mode.default_group_size());
        let bs = optional_int(bits, mode.default_bits());
        let dtype = optional_dtype(dtype, MlxDtype::Float32);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_dequantize",
            ffi::mlx_dequantize(
                &mut res.inner,
                w.inner,
                scales.inner,
                biases_raw,
                gs,
                bs,
                mode.as_ptr(),
                global_scale,
                dtype,
                stream,
            )
        );
        res
    }
}

/// Convert a floating-point array to MLX's E4M3 float8 representation.
pub fn to_fp8(x: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_to_fp8",
            ffi::mlx_to_fp8(&mut res.inner, x.inner, stream)
        );
        res
    }
}

/// Convert an MLX E4M3 float8 array back to a floating-point dtype.
pub fn from_fp8(x: &MlxArray, dtype: MlxDtype, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_from_fp8",
            ffi::mlx_from_fp8(&mut res.inner, x.inner, dtype.to_ffi(), stream)
        );
        res
    }
}

/// Quantized matmul: x @ dequantize(w, scales, biases) with affine mode.
///
/// `group_size` and `bits` use the MLX defaults (64 and 4) when `None`.
/// Prefer [`quantized_matmul_with_mode`] for mxfp4/mxfp8/nvfp4 weights.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul(
    x: &MlxArray,
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    s: Option<&MlxStream>,
) -> MlxArray {
    quantized_matmul_with_mode(
        x,
        w,
        scales,
        biases,
        transpose,
        group_size,
        bits,
        MlxQuantizationMode::Affine,
        s,
    )
}

/// Mode-aware quantized matmul (affine / mxfp4 / mxfp8 / nvfp4).
///
/// MXFP8 language towers (e.g. Unlimited-OCR) pack dense linears without
/// affine group biases; passing mode correctly avoids MLX requiring biases.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_with_mode(
    x: &MlxArray,
    w: &MlxArray,
    scales: &MlxArray,
    biases: Option<&MlxArray>,
    transpose: bool,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: MlxQuantizationMode,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);

        let gs = ffi::mlx_optional_int_ {
            has_value: group_size.is_some(),
            value: group_size.unwrap_or(mode.default_group_size()),
        };
        let bs = ffi::mlx_optional_int_ {
            has_value: bits.is_some(),
            value: bits.unwrap_or(mode.default_bits()),
        };

        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_quantized_matmul",
            ffi::mlx_quantized_matmul(
                &mut res.inner,
                x.inner,
                w.inner,
                scales.inner,
                biases_raw,
                transpose,
                gs,
                bs,
                mode.as_ptr(),
                stream,
            )
        );
        res
    }
}

/// Return a cached scalar `MlxArray` of `value` materialised in `dtype`.
///
/// Functions like `gelu` and `gelu_approx` previously built four short-lived
/// scalar arrays per call (each call to `from_raw_data` + `astype`), which
/// allocated 8 `MlxArray` wrappers per activation invocation. On Gemma 4
/// E2B decode that contributed ~500 transient MlxArray instances per
/// forward pass (~30% of the AX-vs-mlx_lm gap surfaced in
/// `gemma4_e2b_eval_barrier_audit.v1`). Caching scalars by
/// `(value_bits, dtype)` collapses those allocations to one per unique
/// (constant, dtype) pair across the entire process lifetime.
pub fn cached_scalar(value: f32, dtype: MlxDtype) -> MlxArray {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<(u32, MlxDtype), MlxArray>>> = OnceLock::new();
    let key = (value.to_bits(), dtype);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // Graceful degradation on mutex poison: fall through to uncached compute.
    // Under `panic = "abort"` a poisoned mutex would otherwise crash the process.
    let Some(mut map) = cache.lock().ok() else {
        return make_scalar_array(value, dtype);
    };
    if let Some(arr) = map.get(&key) {
        return arr.clone();
    }
    let out = make_scalar_array(value, dtype);
    map.insert(key, out.clone());
    out
}

fn make_scalar_array(value: f32, dtype: MlxDtype) -> MlxArray {
    let f32_arr = MlxArray::from_raw_data(
        &value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    if dtype == MlxDtype::Float32 {
        f32_arr
    } else {
        astype(&f32_arr, dtype, None)
    }
}

/// Top-k values along the last axis.
pub fn topk(a: &MlxArray, k: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_topk",
            ffi::mlx_topk(&mut res.inner, a.inner, k, stream)
        );
        res
    }
}

/// Top-k values along `axis`.
pub fn topk_axis(a: &MlxArray, k: i32, axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_topk_axis",
            ffi::mlx_topk_axis(&mut res.inner, a.inner, k, axis, stream)
        );
        res
    }
}

/// Flatten axes `start_axis..=end_axis` into a single axis.
pub fn flatten(a: &MlxArray, start_axis: i32, end_axis: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_flatten",
            ffi::mlx_flatten(&mut res.inner, a.inner, start_axis, end_axis, stream)
        );
        res
    }
}

/// Repeat the array `repeats` times along a new leading axis.
///
/// Unlike `repeat_axis` (interleave semantics), this tiles the entire array.
pub fn repeat(a: &MlxArray, repeats: i32, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_repeat",
            ffi::mlx_repeat(&mut res.inner, a.inner, repeats, stream)
        );
        res
    }
}

unary_op!(cos, mlx_cos);
unary_op!(sin, mlx_sin);
unary_op!(floor, mlx_floor);
unary_op!(stop_gradient, mlx_stop_gradient);
binary_op!(outer, mlx_outer);

/// Pad array along specified axes with constant value.
pub fn pad(
    a: &MlxArray,
    axes: &[i32],
    low_pad: &[i32],
    high_pad: &[i32],
    pad_value: &MlxArray,
    s: Option<&MlxStream>,
) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_pad",
            ffi::mlx_pad(
                &mut res.inner,
                a.inner,
                axes.as_ptr(),
                axes.len(),
                low_pad.as_ptr(),
                low_pad.len(),
                high_pad.as_ptr(),
                high_pad.len(),
                pad_value.inner,
                c"constant".as_ptr(),
                stream,
            )
        );
        res
    }
}

/// Unflatten axis into a new shape (inverse of `flatten`).
pub fn unflatten(a: &MlxArray, axis: i32, shape: &[i32], s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_unflatten",
            ffi::mlx_unflatten(
                &mut res.inner,
                a.inner,
                axis,
                shape.as_ptr(),
                shape.len(),
                stream,
            )
        );
        res
    }
}

/// GPU-side categorical sampling from logits.
///
/// Returns a `[1]` shaped `u32` array containing the sampled token index.
/// Equivalent to `mx.random.categorical(logits * (1/temperature), axis=-1)`.
/// The caller must scale logits by `1/temperature` before calling this
/// (or pass unscaled logits for temperature=1.0).
///
/// Uses MLX's internal RNG state (`key=null`). Not reproducible across runs.
pub fn random_categorical(logits: &MlxArray, s: Option<&MlxStream>) -> MlxArray {
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let mut res = MlxArray::empty();
        checked_ffi!(
            "mlx_random_categorical",
            ffi::mlx_random_categorical(&mut res.inner, logits.inner, -1, null_ffi_array(), stream)
        );
        res
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod gather_qmm_mxfp4_tests {
    use super::*;
    use crate::eval;

    #[test]
    fn gather_qmm_mxfp4_matches_dequant_gather_mm() {
        // Small expert tensor: E=4, out=8, in=64 (group_size=32 mxfp4).
        let e = 4i32;
        let out = 8i32;
        let inn = 64i32;
        let data: Vec<f32> = (0..(e * out * inn) as usize)
            .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
            .collect();
        let w = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
            &[e, out, inn],
            MlxDtype::Float32,
        );
        let w = astype(&w, MlxDtype::Bfloat16, None);
        let parts = quantize(
            &w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
        assert_eq!(parts.len(), 2, "mxfp4 quant returns [packed, scales]");
        let packed = &parts[0];
        let scales = &parts[1];

        let x_data = vec![0.02f32; 64];
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            x_data.len() * std::mem::size_of::<f32>(),
            &[1, 1, 1, 1, inn],
            MlxDtype::Float32,
        );
        let x = astype(&x, MlxDtype::Bfloat16, None);
        let idx_data = [0u32, 1, 2, 3];
        let indices = MlxArray::from_raw_data(
            idx_data.as_ptr() as *const u8,
            idx_data.len() * std::mem::size_of::<u32>(),
            &[1, 1, 4],
            MlxDtype::Uint32,
        );

        let dequant = dequantize_with_mode(
            packed,
            scales,
            None,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
            Some(MlxDtype::Bfloat16),
            None,
        );
        // [E, out, in] -> [E, in, out] for dense gather_mm.
        let ndim = dequant.ndim() as i32;
        let mut axes: Vec<i32> = (0..ndim).collect();
        let last = axes.len() - 1;
        axes.swap(last - 1, last);
        let wt = transpose(&dequant, &axes, None);
        let y_d = gather_mm(&x, &wt, &indices, false, None);
        let df = astype(&y_d, MlxDtype::Float32, None);
        eval(&[&df]);
        let dv = df.data_f32();
        // MLX 0.32.3 inserts global_scale before sorted_indices. Exercise
        // both dispatch modes with no global scale through the same C ABI.
        for sorted_indices in [false, true] {
            let y_q = gather_qmm_with_mode(
                &x,
                packed,
                scales,
                None,
                &indices,
                true,
                Some(32),
                Some(4),
                MlxQuantizationMode::Mxfp4,
                sorted_indices,
                None,
            );
            assert_eq!(y_q.shape(), y_d.shape());
            let qf = astype(&y_q, MlxDtype::Float32, None);
            eval(&[&qf]);
            let qv = qf.data_f32();
            assert_eq!(qv.len(), dv.len());
            let max_abs = qv
                .iter()
                .zip(dv.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs == 0.0,
                "mxfp4 gather_qmm must match dequant+gather_mm bit-exactly for BF16 path, sorted_indices={sorted_indices}, max_abs={max_abs}"
            );
        }
    }
}

#[cfg(test)]
mod swiglu_oai_tests {
    use super::*;
    use crate::eval;

    #[test]
    fn swiglu_oai_matches_portable_formula() {
        // Reference: mlx-lm gpt_oss.swiglu / ggml_swiglu_oai
        let up_data = [-10.0f32, -1.0, 0.0, 1.0, 3.5, 9.0];
        let gate_data = [-2.0f32, -0.5, 0.0, 0.5, 7.0, 20.0];
        let up = MlxArray::from_raw_data(
            up_data.as_ptr() as *const u8,
            up_data.len() * 4,
            &[1, 6],
            MlxDtype::Float32,
        );
        let gate = MlxArray::from_raw_data(
            gate_data.as_ptr() as *const u8,
            gate_data.len() * 4,
            &[1, 6],
            MlxDtype::Float32,
        );
        let out = swiglu_oai(&up, &gate, None);
        eval(&[&out]);
        let got = out.data_f32();

        let alpha = 1.702f32;
        let limit = 7.0f32;
        for i in 0..6 {
            let mut g = gate_data[i];
            if g > limit {
                g = limit;
            }
            let mut u = up_data[i];
            if u > limit {
                u = limit;
            }
            if u < -limit {
                u = -limit;
            }
            let sig = 1.0 / (1.0 + (-alpha * g).exp());
            let expected = (g * sig) * (u + 1.0);
            let err = (got[i] - expected).abs();
            assert!(
                err < 1e-5,
                "i={i} got={} expected={} err={err}",
                got[i],
                expected
            );
        }
    }
}
