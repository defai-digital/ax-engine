use crate::array::{MlxArray, MlxDtype, null_ffi_array};
use crate::error::{ensure_error_handler, last_error_message, panic_on_status};
use crate::ffi;
use crate::stream::{MlxStream, default_gpu_raw};

unsafe extern "C" {
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
    quantized_matmul(
        &hidden,
        weight,
        scales,
        biases,
        true,
        Some(group_size),
        Some(bits),
        s,
    )
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
    let projected = quantized_matmul(
        x,
        weight,
        scales,
        biases,
        true,
        Some(group_size),
        Some(bits),
        s,
    );
    crate::fast::rms_norm(&projected, Some(norm_weight), eps, s)
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

/// Quantized batched matmul selecting experts via `rhs_indices`.
///
/// Equivalent to `gather_mm` on the dequantized weight. With `transpose=true`,
/// computes `x @ w[rhs_i].T` for each selected expert.
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
    crate::op_count::bump();
    unsafe {
        let stream = s.map(|s| s.inner).unwrap_or_else(default_gpu_raw);
        let biases_raw = biases.map(|b| b.inner).unwrap_or_else(null_ffi_array);
        let null_arr = null_ffi_array();
        let gs = ffi::mlx_optional_int_ {
            has_value: group_size.is_some(),
            value: group_size.unwrap_or(64),
        };
        let bs = ffi::mlx_optional_int_ {
            has_value: bits.is_some(),
            value: bits.unwrap_or(4),
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
                c"affine".as_ptr(),
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

/// Quantized matmul: x @ dequantize(w, scales, biases).
///
/// `group_size` and `bits` use the MLX defaults (64 and 4) when `None`.
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
                c"affine".as_ptr(),
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
mod tests {
    use super::*;
    use crate::transforms::{eval, eval_first_u32};

    #[test]
    fn conv1d_depthwise_reports_reference_shape() {
        let input = zeros(&[1, 6, 2], MlxDtype::Float32, None);
        let weight = zeros(&[2, 3, 1], MlxDtype::Float32, None);
        let out = conv1d(&input, &weight, 1, 0, 1, 2, None);

        assert_eq!(out.shape(), vec![1, 4, 2]);
    }

    #[test]
    fn stack_and_where_preserve_expected_shapes() {
        let a = zeros(&[2, 3], MlxDtype::Float32, None);
        let b = zeros(&[2, 3], MlxDtype::Float32, None);
        let condition = zeros(&[2, 3], MlxDtype::Bool, None);

        assert_eq!(stack(&[&a, &b], 1, None).shape(), vec![2, 2, 3]);
        assert_eq!(where_cond(&condition, &a, &b, None).shape(), vec![2, 3]);
    }

    #[test]
    fn scalar_math_wrappers_keep_input_shape() {
        let a = zeros(&[2, 3], MlxDtype::Float32, None);
        let min_value = 0.0_f32;
        let max_value = 1.0_f32;
        let min = MlxArray::from_raw_data(
            &min_value as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1],
            MlxDtype::Float32,
        );
        let max = MlxArray::from_raw_data(
            &max_value as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1],
            MlxDtype::Float32,
        );

        assert_eq!(
            log1p(&exp(&negative(&a, None), None), None).shape(),
            vec![2, 3]
        );
        assert_eq!(clip(&a, &min, &max, None).shape(), vec![2, 3]);
    }

    #[test]
    fn gelu_approx_mul_direct_matches_portable_composition() {
        let gate_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.125).collect();
        let x_data: Vec<f32> = (0..24).map(|i| ((i as f32) + 1.0) * 0.03125).collect();
        let gate = MlxArray::from_raw_data(
            gate_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_data.as_slice()),
            &[2, 3, 4],
            MlxDtype::Float32,
        );
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[2, 3, 4],
            MlxDtype::Float32,
        );

        let prev = crate::op_count::op_count_snapshot();
        let direct = gelu_approx_mul(&gate, &x, None);
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ activation shim should count as one Rust FFI dispatch"
        );
        let portable = multiply(&gelu_approx(&gate, None), &x, None);
        let direct_f32 = astype(&direct, MlxDtype::Float32, None);
        let portable_f32 = astype(&portable, MlxDtype::Float32, None);
        eval(&[&direct_f32, &portable_f32]);

        assert_eq!(direct_f32.shape(), vec![2, 3, 4]);
        assert_eq!(
            direct_f32.data_f32().to_vec(),
            portable_f32.data_f32().to_vec(),
            "direct C++ activation shim must preserve mlx-lm GEGLU math"
        );
    }

    #[test]
    fn silu_mul_direct_matches_portable_composition() {
        let gate_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.125).collect();
        let x_data: Vec<f32> = (0..24).map(|i| ((i as f32) + 1.0) * 0.03125).collect();
        let gate = MlxArray::from_raw_data(
            gate_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_data.as_slice()),
            &[2, 3, 4],
            MlxDtype::Float32,
        );
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[2, 3, 4],
            MlxDtype::Float32,
        );

        let prev = crate::op_count::op_count_snapshot();
        let direct = silu_mul(&gate, &x, None);
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ SwiGLU activation shim should count as one Rust FFI dispatch"
        );
        let portable = multiply(&silu(&gate, None), &x, None);
        let direct_f32 = astype(&direct, MlxDtype::Float32, None);
        let portable_f32 = astype(&portable, MlxDtype::Float32, None);
        eval(&[&direct_f32, &portable_f32]);

        assert_eq!(direct_f32.shape(), vec![2, 3, 4]);
        assert_eq!(
            direct_f32.data_f32().to_vec(),
            portable_f32.data_f32().to_vec(),
            "direct C++ SwiGLU activation shim must preserve silu(gate) * x math"
        );
    }

    #[test]
    fn gelu_approx_mul_matmul_direct_matches_portable_composition() {
        let gate_data: Vec<f32> = (0..6).map(|i| ((i as f32) - 3.0) * 0.25).collect();
        let x_data: Vec<f32> = (0..6).map(|i| ((i as f32) + 1.0) * 0.125).collect();
        let weight_data: Vec<f32> = (0..12).map(|i| ((i as f32) - 6.0) * 0.0625).collect();
        let gate = MlxArray::from_raw_data(
            gate_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_data.as_slice()),
            &[2, 3],
            MlxDtype::Float32,
        );
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[2, 3],
            MlxDtype::Float32,
        );
        let weight = MlxArray::from_raw_data(
            weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(weight_data.as_slice()),
            &[3, 4],
            MlxDtype::Float32,
        );

        let prev = crate::op_count::op_count_snapshot();
        let direct = gelu_approx_mul_matmul(&gate, &x, &weight, None);
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ activation+matmul shim should count as one Rust FFI dispatch"
        );
        let portable = matmul(
            &multiply(&gelu_approx(&gate, None), &x, None),
            &weight,
            None,
        );
        eval(&[&direct, &portable]);

        assert_eq!(direct.shape(), vec![2, 4]);
        assert_eq!(
            direct.data_f32().to_vec(),
            portable.data_f32().to_vec(),
            "direct C++ activation+matmul shim must preserve portable math"
        );
    }

    #[test]
    fn qk_norm_rope_direct_matches_portable_composition() {
        let seq = 3_i32;
        let n_heads = 2_i32;
        let head_dim = 4_i32;
        let width = n_heads * head_dim;
        let proj_data: Vec<f32> = (0..(seq * width))
            .map(|i| ((i as f32) - 11.0) * 0.03125)
            .collect();
        let norm_data: Vec<f32> = (0..head_dim).map(|i| 0.75 + (i as f32) * 0.125).collect();
        let proj = MlxArray::from_raw_data(
            proj_data.as_ptr() as *const u8,
            std::mem::size_of_val(proj_data.as_slice()),
            &[1, seq, width],
            MlxDtype::Float32,
        );
        let norm = MlxArray::from_raw_data(
            norm_data.as_ptr() as *const u8,
            std::mem::size_of_val(norm_data.as_slice()),
            &[head_dim],
            MlxDtype::Float32,
        );

        let prev = crate::op_count::op_count_snapshot();
        let direct = qk_norm_rope_bhsd_from_proj(
            &proj,
            Some(&norm),
            n_heads,
            head_dim,
            1.0e-6,
            head_dim,
            false,
            Some(10_000.0),
            2,
            None,
            None,
        );
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ QK-norm+RoPE probe should count as one Rust FFI dispatch"
        );

        let bhsd = as_strided(
            &proj,
            &[1, n_heads, seq, head_dim],
            &[
                i64::from(seq * width),
                i64::from(head_dim),
                i64::from(width),
                1,
            ],
            0,
            None,
        );
        let normed = crate::fast::rms_norm(&bhsd, Some(&norm), 1.0e-6, None);
        let reference =
            crate::fast::rope(&normed, head_dim, false, Some(10_000.0), 1.0, 2, None, None);
        let direct = contiguous(&direct, None);
        let reference = contiguous(&reference, None);
        eval(&[&direct, &reference]);

        assert_eq!(direct.shape(), vec![1, n_heads, seq, head_dim]);
        assert_eq!(reference.shape(), direct.shape());
        assert_close_f32(direct.data_f32(), reference.data_f32(), 1.0e-6);
    }

    #[test]
    fn qwen_linear_attention_inputs_packed_direct_matches_portable_composition() {
        let seq = 2_i32;
        let hidden = 32_i32;
        let num_key_heads = 2_i32;
        let num_value_heads = 4_i32;
        let key_head_dim = 3_i32;
        let value_head_dim = 2_i32;
        let value_heads_per_key = num_value_heads / num_key_heads;
        let value_dim_per_key = value_heads_per_key * value_head_dim;
        let qkvz_per_key = key_head_dim * 2 + value_dim_per_key * 2;
        let qkvz_out = num_key_heads * qkvz_per_key;
        let ba_out = num_key_heads * value_heads_per_key * 2;

        let x_data: Vec<f32> = (0..(seq * hidden))
            .map(|i| ((i as f32) - 31.0) * 0.03125)
            .collect();
        let qkvz_weight_data: Vec<f32> = (0..(qkvz_out * hidden))
            .map(|i| ((i as f32) - 448.0) * 0.0005)
            .collect();
        let ba_weight_data: Vec<f32> = (0..(ba_out * hidden))
            .map(|i| ((i as f32) - 128.0) * 0.001)
            .collect();
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[1, seq, hidden],
            MlxDtype::Float32,
        );
        let qkvz_weight = MlxArray::from_raw_data(
            qkvz_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(qkvz_weight_data.as_slice()),
            &[qkvz_out, hidden],
            MlxDtype::Float32,
        );
        let ba_weight = MlxArray::from_raw_data(
            ba_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(ba_weight_data.as_slice()),
            &[ba_out, hidden],
            MlxDtype::Float32,
        );
        let qkvz_q = quantize(
            &qkvz_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let ba_q = quantize(
            &ba_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(qkvz_q.len(), 3);
        assert_eq!(ba_q.len(), 3);

        let prev = crate::op_count::op_count_snapshot();
        let (direct_qkv, direct_z, direct_a, direct_b) = qwen_linear_attention_inputs_packed(
            &x,
            &qkvz_q[0],
            Some(&qkvz_q[1]),
            Some(&qkvz_q[2]),
            &ba_q[0],
            Some(&ba_q[1]),
            Some(&ba_q[2]),
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            32,
            4,
            None,
        )
        .expect("direct packed linear-attention shim should accept qwen-compatible shapes");
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ linear-attention input shim should count as one Rust FFI dispatch"
        );

        let mixed_qkvz = quantized_matmul(
            &x,
            &qkvz_q[0],
            &qkvz_q[1],
            Some(&qkvz_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let mixed_qkvz = reshape(&mixed_qkvz, &[1, seq, num_key_heads, qkvz_per_key], None);
        let q = slice_last_dim(&mixed_qkvz, 0, key_head_dim, None);
        let k = slice_last_dim(&mixed_qkvz, key_head_dim, key_head_dim * 2, None);
        let v = slice_last_dim(
            &mixed_qkvz,
            key_head_dim * 2,
            key_head_dim * 2 + value_dim_per_key,
            None,
        );
        let z = slice_last_dim(
            &mixed_qkvz,
            key_head_dim * 2 + value_dim_per_key,
            qkvz_per_key,
            None,
        );
        let portable_qkv = concatenate(
            &[
                &reshape(&q, &[1, seq, num_key_heads * key_head_dim], None),
                &reshape(&k, &[1, seq, num_key_heads * key_head_dim], None),
                &reshape(&v, &[1, seq, num_value_heads * value_head_dim], None),
            ],
            2,
            None,
        );
        let portable_z = reshape(&z, &[1, seq, num_value_heads, value_head_dim], None);

        let mixed_ba = quantized_matmul(
            &x,
            &ba_q[0],
            &ba_q[1],
            Some(&ba_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let ba = reshape(
            &mixed_ba,
            &[1, seq, num_key_heads, value_heads_per_key * 2],
            None,
        );
        let portable_b = reshape(
            &slice_last_dim(&ba, 0, value_heads_per_key, None),
            &[1, seq, num_value_heads],
            None,
        );
        let portable_a = reshape(
            &slice_last_dim(&ba, value_heads_per_key, value_heads_per_key * 2, None),
            &[1, seq, num_value_heads],
            None,
        );

        let direct_qkv = contiguous(&direct_qkv, None);
        let direct_z = contiguous(&direct_z, None);
        let direct_a = contiguous(&direct_a, None);
        let direct_b = contiguous(&direct_b, None);
        let portable_qkv = contiguous(&portable_qkv, None);
        let portable_z = contiguous(&portable_z, None);
        let portable_a = contiguous(&portable_a, None);
        let portable_b = contiguous(&portable_b, None);
        eval(&[
            &direct_qkv,
            &direct_z,
            &direct_a,
            &direct_b,
            &portable_qkv,
            &portable_z,
            &portable_a,
            &portable_b,
        ]);

        assert_eq!(
            direct_qkv.shape(),
            vec![
                1,
                seq,
                num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim
            ]
        );
        assert_eq!(
            direct_z.shape(),
            vec![1, seq, num_value_heads, value_head_dim]
        );
        assert_eq!(direct_a.shape(), vec![1, seq, num_value_heads]);
        assert_eq!(direct_b.shape(), vec![1, seq, num_value_heads]);
        assert_close_f32(direct_qkv.data_f32(), portable_qkv.data_f32(), 1.0e-6);
        assert_close_f32(direct_z.data_f32(), portable_z.data_f32(), 1.0e-6);
        assert_close_f32(direct_a.data_f32(), portable_a.data_f32(), 1.0e-6);
        assert_close_f32(direct_b.data_f32(), portable_b.data_f32(), 1.0e-6);
    }

    #[test]
    fn qwen_linear_attention_post_input_direct_matches_portable_composition() {
        let batch = 1_i32;
        let seq = 2_i32;
        let num_key_heads = 2_i32;
        let key_head_dim = 4_i32;
        let num_value_heads = 4_i32;
        let value_head_dim = 3_i32;
        let conv_kernel_dim = 4_i32;
        let tail_len = conv_kernel_dim - 1;
        let key_dim = num_key_heads * key_head_dim;
        let value_dim = num_value_heads * value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let (q_scale, k_scale) = (1.0_f32 / (key_head_dim as f32).sqrt(), 0.5_f32);
        let eps = 1.0e-6_f32;

        let qkv_data: Vec<f32> = (0..(batch * seq * conv_dim))
            .map(|i| ((i as f32) - 16.0) * 0.0625)
            .collect();
        let conv_weight_data: Vec<f32> = (0..(conv_dim * conv_kernel_dim))
            .map(|i| ((i as f32) - 32.0) * 0.015625)
            .collect();
        let cached_state_data: Vec<f32> = (0..(batch * tail_len * conv_dim))
            .map(|i| ((i as f32) - 8.0) * 0.03125)
            .collect();
        let qkv = MlxArray::from_raw_data(
            qkv_data.as_ptr() as *const u8,
            std::mem::size_of_val(qkv_data.as_slice()),
            &[batch, seq, conv_dim],
            MlxDtype::Float32,
        );
        // conv1d weight layout is [out_channels, kernel_w, in_channels_per_group].
        // Depthwise (`groups = conv_dim`) → in_channels_per_group = 1.
        let conv_weight = MlxArray::from_raw_data(
            conv_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(conv_weight_data.as_slice()),
            &[conv_dim, conv_kernel_dim, 1],
            MlxDtype::Float32,
        );
        let cached_state = MlxArray::from_raw_data(
            cached_state_data.as_ptr() as *const u8,
            std::mem::size_of_val(cached_state_data.as_slice()),
            &[batch, tail_len, conv_dim],
            MlxDtype::Float32,
        );

        let prev = crate::op_count::op_count_snapshot();
        let (direct_q, direct_k, direct_v, direct_state) = qwen_linear_attention_post_input(
            &qkv,
            &conv_weight,
            Some(&cached_state),
            num_key_heads,
            key_head_dim,
            num_value_heads,
            value_head_dim,
            conv_kernel_dim,
            q_scale,
            k_scale,
            eps,
            None,
        )
        .expect("post-input shim should accept qwen-compatible shapes");
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ post-input shim should count as one Rust FFI dispatch"
        );

        // Portable composition: matches `linear_attention_conv1d` +
        // `split_linear_attention_qkv` + `normalize_linear_attention_qk` in
        // `crates/ax-engine-mlx/src/linear_attention_ops.rs`.
        let conv_input = concatenate(&[&cached_state, &qkv], 1, None);
        let total = conv_input.shape()[1];
        let portable_state = slice(
            &conv_input,
            &[0, total - tail_len, 0],
            &[batch, total, conv_dim],
            &[1, 1, 1],
            None,
        );
        let conv_out = conv1d(&conv_input, &conv_weight, 1, 0, 1, conv_dim, None);
        let silued = multiply(&conv_out, &sigmoid(&conv_out, None), None);
        let q_flat = slice_last_dim(&silued, 0, key_dim, None);
        let k_flat = slice_last_dim(&silued, key_dim, 2 * key_dim, None);
        let v_flat = slice_last_dim(&silued, 2 * key_dim, 2 * key_dim + value_dim, None);
        let q_heads = reshape(&q_flat, &[batch, seq, num_key_heads, key_head_dim], None);
        let k_heads = reshape(&k_flat, &[batch, seq, num_key_heads, key_head_dim], None);
        let v_heads = reshape(
            &v_flat,
            &[batch, seq, num_value_heads, value_head_dim],
            None,
        );
        let q_normed = crate::fast::rms_norm(&q_heads, None, eps, None);
        let k_normed = crate::fast::rms_norm(&k_heads, None, eps, None);
        let q_scale_arr = MlxArray::from_raw_data(
            &q_scale as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1],
            MlxDtype::Float32,
        );
        let k_scale_arr = MlxArray::from_raw_data(
            &k_scale as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1],
            MlxDtype::Float32,
        );
        let portable_q = multiply(&q_normed, &q_scale_arr, None);
        let portable_k = multiply(&k_normed, &k_scale_arr, None);
        let portable_v = v_heads;

        eval(&[
            &direct_q,
            &direct_k,
            &direct_v,
            &direct_state,
            &portable_q,
            &portable_k,
            &portable_v,
            &portable_state,
        ]);

        assert_eq!(
            direct_q.shape(),
            vec![batch, seq, num_key_heads, key_head_dim]
        );
        assert_eq!(
            direct_k.shape(),
            vec![batch, seq, num_key_heads, key_head_dim]
        );
        assert_eq!(
            direct_v.shape(),
            vec![batch, seq, num_value_heads, value_head_dim]
        );
        assert_eq!(direct_state.shape(), vec![batch, tail_len, conv_dim]);
        assert_close_f32(direct_q.data_f32(), portable_q.data_f32(), 1.0e-6);
        assert_close_f32(direct_k.data_f32(), portable_k.data_f32(), 1.0e-6);
        assert_close_f32(direct_v.data_f32(), portable_v.data_f32(), 1.0e-6);
        assert_close_f32(direct_state.data_f32(), portable_state.data_f32(), 1.0e-6);
    }

    #[test]
    fn qwen_linear_attention_post_input_direct_handles_empty_cached_state() {
        // First decode step in a fresh sequence: cached_conv_state is None,
        // shim must materialise zeros internally and still return the right
        // tail slice.
        let batch = 1_i32;
        let seq = 1_i32;
        let num_key_heads = 1_i32;
        let key_head_dim = 32_i32; // gated_delta kernel requires Dk % 32 == 0
        let num_value_heads = 1_i32;
        let value_head_dim = 4_i32;
        let conv_kernel_dim = 4_i32;
        let key_dim = num_key_heads * key_head_dim;
        let value_dim = num_value_heads * value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;

        let qkv_data: Vec<f32> = (0..(batch * seq * conv_dim))
            .map(|i| i as f32 * 0.01)
            .collect();
        let conv_weight_data: Vec<f32> = (0..(conv_dim * conv_kernel_dim))
            .map(|i| ((i as f32) - 16.0) * 0.0078125)
            .collect();
        let qkv = MlxArray::from_raw_data(
            qkv_data.as_ptr() as *const u8,
            std::mem::size_of_val(qkv_data.as_slice()),
            &[batch, seq, conv_dim],
            MlxDtype::Float32,
        );
        let conv_weight = MlxArray::from_raw_data(
            conv_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(conv_weight_data.as_slice()),
            &[conv_dim, conv_kernel_dim, 1],
            MlxDtype::Float32,
        );

        let result = qwen_linear_attention_post_input(
            &qkv,
            &conv_weight,
            None,
            num_key_heads,
            key_head_dim,
            num_value_heads,
            value_head_dim,
            conv_kernel_dim,
            1.0,
            1.0,
            1.0e-6,
            None,
        );
        assert!(
            result.is_some(),
            "shim must accept None cached_conv_state by materialising zeros internally"
        );
        let (_, _, _, new_state) = result.unwrap();
        assert_eq!(
            new_state.shape(),
            vec![batch, conv_kernel_dim - 1, conv_dim]
        );
    }

    #[test]
    fn gelu_approx_mul_quantized_matmul_direct_matches_portable_composition() {
        let gate_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.03125).collect();
        let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) + 1.0) * 0.015625).collect();
        let weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.0005).collect();
        let gate = MlxArray::from_raw_data(
            gate_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_data.as_slice()),
            &[2, 32],
            MlxDtype::Float32,
        );
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[2, 32],
            MlxDtype::Float32,
        );
        let weight = MlxArray::from_raw_data(
            weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(weight_data.as_slice()),
            &[32, 32],
            MlxDtype::Float32,
        );
        let quantized = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(quantized.len(), 3);

        let prev = crate::op_count::op_count_snapshot();
        let direct = gelu_approx_mul_quantized_matmul(
            &gate,
            &x,
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            32,
            4,
            None,
        );
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ activation+quantized-matmul shim should count as one Rust FFI dispatch"
        );

        let hidden = gelu_approx_mul(&gate, &x, None);
        let portable = quantized_matmul(
            &hidden,
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        eval(&[&direct, &portable]);

        assert_eq!(direct.shape(), vec![2, 32]);
        assert_eq!(
            direct.data_f32().to_vec(),
            portable.data_f32().to_vec(),
            "direct C++ activation+quantized-matmul shim must preserve portable math"
        );
    }

    #[test]
    fn gelu_approx_quantized_ffn_direct_matches_portable_composition() {
        let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.03125).collect();
        let gate_up_weight_data: Vec<f32> =
            (0..2048).map(|i| ((i as f32) - 1024.0) * 0.0005).collect();
        let down_weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.0005).collect();
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[2, 32],
            MlxDtype::Float32,
        );
        let gate_up_weight = MlxArray::from_raw_data(
            gate_up_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_up_weight_data.as_slice()),
            &[64, 32],
            MlxDtype::Float32,
        );
        let down_weight = MlxArray::from_raw_data(
            down_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(down_weight_data.as_slice()),
            &[32, 32],
            MlxDtype::Float32,
        );
        let gate_up_q = quantize(
            &gate_up_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down_q = quantize(
            &down_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(gate_up_q.len(), 3);
        assert_eq!(down_q.len(), 3);

        let prev = crate::op_count::op_count_snapshot();
        let direct = gelu_approx_quantized_ffn(
            &x,
            &gate_up_q[0],
            &gate_up_q[1],
            Some(&gate_up_q[2]),
            &down_q[0],
            &down_q[1],
            Some(&down_q[2]),
            32,
            4,
            None,
        );
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ quantized FFN shim should count as one Rust FFI dispatch"
        );

        let gate_up = quantized_matmul(
            &x,
            &gate_up_q[0],
            &gate_up_q[1],
            Some(&gate_up_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let gate = slice_last_dim(&gate_up, 0, 32, None);
        let up = slice_last_dim(&gate_up, 32, 64, None);
        let hidden = gelu_approx_mul(&gate, &up, None);
        let portable = quantized_matmul(
            &hidden,
            &down_q[0],
            &down_q[1],
            Some(&down_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        eval(&[&direct, &portable]);

        assert_eq!(direct.shape(), vec![2, 32]);
        assert_eq!(
            direct.data_f32().to_vec(),
            portable.data_f32().to_vec(),
            "direct C++ quantized FFN shim must preserve portable math"
        );
    }

    #[test]
    fn gemma4_post_attn_ffn_block_direct_matches_portable_composition() {
        let hidden_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.03125).collect();
        let attn_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 18.0) * 0.015625).collect();
        let norm_data: Vec<f32> = (0..32).map(|i| 0.75 + (i as f32) * 0.0078125).collect();
        let post_norm_data: Vec<f32> = (0..32).map(|i| 0.875 + (i as f32) * 0.00390625).collect();
        let layer_scalar_data = vec![0.9375_f32];
        let gate_up_weight_data: Vec<f32> =
            (0..2048).map(|i| ((i as f32) - 1024.0) * 0.0005).collect();
        let down_weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.0005).collect();
        let hidden = MlxArray::from_raw_data(
            hidden_data.as_ptr() as *const u8,
            std::mem::size_of_val(hidden_data.as_slice()),
            &[1, 2, 32],
            MlxDtype::Float32,
        );
        let attn = MlxArray::from_raw_data(
            attn_data.as_ptr() as *const u8,
            std::mem::size_of_val(attn_data.as_slice()),
            &[1, 2, 32],
            MlxDtype::Float32,
        );
        let norm = MlxArray::from_raw_data(
            norm_data.as_ptr() as *const u8,
            std::mem::size_of_val(norm_data.as_slice()),
            &[32],
            MlxDtype::Float32,
        );
        let post_norm = MlxArray::from_raw_data(
            post_norm_data.as_ptr() as *const u8,
            std::mem::size_of_val(post_norm_data.as_slice()),
            &[32],
            MlxDtype::Float32,
        );
        let layer_scalar = MlxArray::from_raw_data(
            layer_scalar_data.as_ptr() as *const u8,
            std::mem::size_of_val(layer_scalar_data.as_slice()),
            &[1],
            MlxDtype::Float32,
        );
        let gate_up_weight = MlxArray::from_raw_data(
            gate_up_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(gate_up_weight_data.as_slice()),
            &[64, 32],
            MlxDtype::Float32,
        );
        let down_weight = MlxArray::from_raw_data(
            down_weight_data.as_ptr() as *const u8,
            std::mem::size_of_val(down_weight_data.as_slice()),
            &[32, 32],
            MlxDtype::Float32,
        );
        let gate_up_q = quantize(
            &gate_up_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down_q = quantize(
            &down_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );

        let prev = crate::op_count::op_count_snapshot();
        let direct = gemma4_post_attn_ffn_block(
            &hidden,
            &attn,
            &norm,
            Some(&post_norm),
            Some(&layer_scalar),
            &gate_up_q[0],
            &gate_up_q[1],
            Some(&gate_up_q[2]),
            &down_q[0],
            &down_q[1],
            Some(&down_q[2]),
            32,
            4,
            1.0e-6,
            None,
        );
        assert_eq!(
            crate::op_count::op_count_take(prev),
            1,
            "direct C++ Gemma4 post-attention FFN block should count as one Rust FFI dispatch"
        );

        let residual = add(&hidden, &attn, None);
        let normed = crate::fast::rms_norm(&residual, Some(&norm), 1.0e-6, None);
        let gate_up = quantized_matmul(
            &normed,
            &gate_up_q[0],
            &gate_up_q[1],
            Some(&gate_up_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let gate = slice_last_dim(&gate_up, 0, 32, None);
        let up = slice_last_dim(&gate_up, 32, 64, None);
        let ffn_hidden = gelu_approx_mul(&gate, &up, None);
        let ffn = quantized_matmul(
            &ffn_hidden,
            &down_q[0],
            &down_q[1],
            Some(&down_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let ffn = crate::fast::rms_norm(&ffn, Some(&post_norm), 1.0e-6, None);
        let portable = multiply(&add(&residual, &ffn, None), &layer_scalar, None);
        eval(&[&direct, &portable]);

        assert_eq!(direct.shape(), vec![1, 2, 32]);
        assert_close_f32(direct.data_f32(), portable.data_f32(), 1.0e-6);
    }

    #[test]
    fn quantize_affine_round_trip_reports_expected_shapes() {
        let values = (0..128)
            .map(|i| (i as f32 - 64.0) / 64.0)
            .collect::<Vec<_>>();
        let w = MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values.as_slice()),
            &[2, 64],
            MlxDtype::Float32,
        );

        let quantized = quantize(
            &w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );

        assert_eq!(quantized.len(), 3);
        assert_eq!(quantized[0].shape(), vec![2, 8]);
        assert_eq!(quantized[0].dtype(), MlxDtype::Uint32);
        assert_eq!(quantized[1].shape(), vec![2, 2]);
        assert_eq!(quantized[1].dtype(), MlxDtype::Float32);
        assert_eq!(quantized[2].shape(), vec![2, 2]);
        assert_eq!(quantized[2].dtype(), MlxDtype::Float32);

        let restored = dequantize_with_mode(
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            Some(MlxDtype::Float32),
            None,
        );
        eval(&[&restored]);

        assert_eq!(restored.shape(), vec![2, 64]);
        assert_eq!(restored.dtype(), MlxDtype::Float32);
    }

    #[test]
    fn fp8_conversion_wrappers_preserve_shape_contract() {
        let values = [0.0_f32, 1.0, -2.0, 4.0];
        let x = MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(&values),
            &[2, 2],
            MlxDtype::Float32,
        );

        let fp8 = to_fp8(&x, None);
        eval(&[&fp8]);
        assert_eq!(fp8.shape(), vec![2, 2]);
        assert_eq!(fp8.dtype(), MlxDtype::Uint8);

        let restored = from_fp8(&fp8, MlxDtype::Float32, None);
        eval(&[&restored]);
        assert_eq!(restored.shape(), vec![2, 2]);
        assert_eq!(restored.dtype(), MlxDtype::Float32);
    }

    fn assert_close_f32(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            assert!(
                delta <= tolerance,
                "mismatch at {idx}: actual={a}, expected={e}, delta={delta}, tolerance={tolerance}"
            );
        }
    }

    // ── Task 2: Shape / structural ops ──────────────────────────────

    #[test]
    fn as_strided_preserves_data_under_valid_strides() {
        // Create a [2,3] array and reshape via as_strided with identity strides.
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let a = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(&data[..]),
            &[2, 3],
            MlxDtype::Float32,
        );
        // Reshape [2,3] -> [3,2] with row-major strides [2,1].
        let b = as_strided(&a, &[3, 2], &[2, 1], 0, None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![3, 2]);
        assert_eq!(b.data_f32().len(), 6);
    }

    #[test]
    fn broadcast_to_expands_dims() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let b = broadcast_to(&a, &[4, 3], None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![4, 3]);
    }

    #[test]
    fn expand_dims_inserts_singleton() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let b = expand_dims(&a, 0, None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![1, 3]);
    }

    #[test]
    fn expand_dims_axes_inserts_multiple() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let b = expand_dims_axes(&a, &[0, 2], None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![1, 3, 1]);
    }

    #[test]
    fn flatten_collapses_dims() {
        let a = zeros(&[2, 3, 4], MlxDtype::Float32, None);
        let b = flatten(&a, 0, -1, None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![24]);
    }

    #[test]
    fn pad_extends_with_value() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let pad_val = MlxArray::from_f32(0.0);
        let b = pad(&a, &[0], &[2], &[1], &pad_val, None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![6]);
        assert_eq!(b.data_f32(), &[0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn repeat_along_axis() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let b = repeat(&a, 2, None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![6]);
        assert_eq!(b.data_f32(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn repeat_axis_explicit() {
        let a = zeros(&[2, 3], MlxDtype::Float32, None);
        let b = repeat_axis(&a, 4, 1, None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![2, 12]);
    }

    #[test]
    fn unflatten_splits_one_dim() {
        let a = zeros(&[24], MlxDtype::Float32, None);
        let b = unflatten(&a, 0, &[2, 3, 4], None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![2, 3, 4]);
    }

    #[test]
    fn transpose_with_axes() {
        let a = zeros(&[2, 3, 4], MlxDtype::Float32, None);
        let b = transpose(&a, &[2, 0, 1], None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![4, 2, 3]);
    }

    #[test]
    fn slice_update_replaces_region() {
        let src = MlxArray::from_f32_slice(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let upd = MlxArray::from_f32_slice(&[99.0, 98.0]);
        let b = slice_update(&src, &upd, &[1], &[3], &[1], None);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![5]);
        assert_eq!(b.data_f32(), &[0.0, 99.0, 98.0, 3.0, 4.0]);
    }

    #[test]
    fn contiguous_makes_non_contiguous_contiguous() {
        // transpose produces a non-contiguous view
        let a = zeros(&[2, 3], MlxDtype::Float32, None);
        let t = transpose(&a, &[1, 0], None);
        let c = contiguous(&t, None);
        eval(&[&c]);
        assert_eq!(c.shape(), vec![3, 2]);
    }

    #[test]
    fn view_reinterprets_dtype() {
        let data: Vec<u32> = vec![0x3F800000, 0x40000000]; // 1.0f32, 2.0f32 bit patterns
        let a = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(&data[..]),
            &[2],
            MlxDtype::Uint32,
        );
        let b = view(&a, MlxDtype::Float32, None);
        eval(&[&b]);
        assert_eq!(b.dtype(), MlxDtype::Float32);
        assert_eq!(b.data_f32(), &[1.0, 2.0]);
    }

    // ── Task 3: Reduction / sort ops ───────────────────────────────

    #[test]
    fn argmax_returns_correct_index() {
        let a = MlxArray::from_f32_slice(&[0.1, 0.5, 0.3, 0.9]);
        let idx = argmax(&a, None);
        assert_eq!(eval_first_u32(&idx), 3);
    }

    #[test]
    fn argpartition_splits_at_kth() {
        let a = MlxArray::from_f32_slice(&[3.0, 1.0, 4.0, 1.5, 2.0]);
        let idx = argpartition_axis(&a, 2, 0, None);
        let idx_f = astype(&idx, MlxDtype::Float32, None);
        eval(&[&idx_f]);
        assert_eq!(idx.shape(), vec![5]);
    }

    #[test]
    fn argsort_produces_sorted_indices() {
        let a = MlxArray::from_f32_slice(&[3.0, 1.0, 2.0]);
        let idx = argsort_axis(&a, 0, None);
        let idx_f = astype(&idx, MlxDtype::Float32, None);
        eval(&[&idx_f]);
        assert_eq!(idx_f.data_f32(), &[1.0, 2.0, 0.0]);
    }

    #[test]
    fn cumsum_inclusive_and_exclusive() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0]);
        let inc = cumsum(&a, 0, false, true, None);
        let exc = cumsum(&a, 0, false, false, None);
        eval(&[&inc, &exc]);
        assert_eq!(inc.data_f32(), &[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(exc.data_f32(), &[0.0, 1.0, 3.0, 6.0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let s = softmax(&a, 0, None);
        let total = sum_axis(&s, 0, false, None);
        eval(&[&total]);
        assert_close_f32(total.data_f32(), &[1.0], 1e-5);
    }

    #[test]
    fn sum_along_axis() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let a = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(&data[..]),
            &[2, 3],
            MlxDtype::Float32,
        );
        let s0 = sum_axis(&a, 0, false, None);
        let s1 = sum_axis(&a, 1, false, None);
        eval(&[&s0, &s1]);
        assert_eq!(s0.data_f32(), &[3.0, 5.0, 7.0]); // sum over rows
        assert_eq!(s1.data_f32(), &[3.0, 12.0]); // sum over cols
    }

    #[test]
    fn topk_returns_top_k_values_and_indices() {
        let a = MlxArray::from_f32_slice(&[1.0, 5.0, 3.0, 4.0, 2.0]);
        let result = topk(&a, 3, None);
        eval(&[&result]);
        assert_eq!(result.shape(), vec![3]);
        // MLX topk returns values in ascending order
        assert_eq!(result.data_f32(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn topk_axis_works_on_non_last() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let a = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(&data[..]),
            &[3, 2],
            MlxDtype::Float32,
        );
        let result = topk_axis(&a, 2, 0, None);
        eval(&[&result]);
        assert_eq!(result.shape(), vec![2, 2]);
    }

    #[test]
    fn take_along_axis_gathers() {
        let a = MlxArray::from_f32_slice(&[10.0, 20.0, 30.0, 40.0]);
        let indices = MlxArray::from_raw_data(
            &[2u32, 0u32] as *const u32 as *const u8,
            std::mem::size_of::<u32>() * 2,
            &[2],
            MlxDtype::Uint32,
        );
        let result = take_along_axis(&a, &indices, 0, None);
        eval(&[&result]);
        assert_eq!(result.data_f32(), &[30.0, 10.0]);
    }

    #[test]
    fn take_axis_gathers_single_axis() {
        let a = MlxArray::from_f32_slice(&[10.0, 20.0, 30.0, 40.0]);
        let indices = MlxArray::from_raw_data(
            &[3u32, 1u32] as *const u32 as *const u8,
            std::mem::size_of::<u32>() * 2,
            &[2],
            MlxDtype::Uint32,
        );
        let result = take(&a, &indices, 0, None);
        eval(&[&result]);
        assert_eq!(result.data_f32(), &[40.0, 20.0]);
    }

    #[test]
    fn put_along_axis_scatters() {
        let a = MlxArray::from_f32_slice(&[0.0, 0.0, 0.0, 0.0]);
        let indices = MlxArray::from_raw_data(
            &[1u32, 3u32] as *const u32 as *const u8,
            std::mem::size_of::<u32>() * 2,
            &[2],
            MlxDtype::Uint32,
        );
        let values = MlxArray::from_f32_slice(&[99.0, 88.0]);
        let result = put_along_axis(&a, &indices, &values, 0, None);
        eval(&[&result]);
        assert_eq!(result.data_f32(), &[0.0, 99.0, 0.0, 88.0]);
    }

    // ── Task 4: Creation + binary/unary ops ──────────────────────

    #[test]
    fn zeros_creates_all_zero_array() {
        let a = zeros(&[3, 4], MlxDtype::Float32, None);
        eval(&[&a]);
        assert_eq!(a.shape(), vec![3, 4]);
        assert!(a.data_f32().iter().all(|&v| v == 0.0));

        let b = zeros(&[5], MlxDtype::Int32, None);
        let bf = astype(&b, MlxDtype::Float32, None);
        eval(&[&bf]);
        assert!(bf.data_f32().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn arange_produces_sequence() {
        let a = arange(0.0, 5.0, 1.0, MlxDtype::Float32, None);
        eval(&[&a]);
        assert_eq!(a.data_f32(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn random_categorical_produces_valid_indices() {
        let logits = MlxArray::from_f32_slice(&[0.0, 10.0, 0.0, 0.0]);
        let token = random_categorical(&logits, None);
        assert_eq!(eval_first_u32(&token), 1); // heavily biased toward index 1
    }

    #[test]
    fn binary_ops_shape_and_value_correctness() {
        let a = MlxArray::from_f32_slice(&[4.0, 6.0, 8.0]);
        let b = MlxArray::from_f32_slice(&[2.0, 3.0, 4.0]);

        let d = divide(&a, &b, None);
        eval(&[&d]);
        assert_eq!(d.data_f32(), &[2.0, 2.0, 2.0]);

        let s = subtract(&a, &b, None);
        eval(&[&s]);
        assert_eq!(s.data_f32(), &[2.0, 3.0, 4.0]);

        let p = power(&b, &MlxArray::from_f32(2.0), None);
        eval(&[&p]);
        assert_eq!(p.data_f32(), &[4.0, 9.0, 16.0]);

        let mx = maximum(&a, &b, None);
        eval(&[&mx]);
        assert_eq!(mx.data_f32(), &[4.0, 6.0, 8.0]);

        let mn = minimum(&a, &b, None);
        eval(&[&mn]);
        assert_eq!(mn.data_f32(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn comparison_ops_return_bool() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let b = MlxArray::from_f32_slice(&[2.0, 2.0, 1.0]);

        let lt = less(&a, &b, None);
        eval(&[&lt]);
        assert_eq!(lt.dtype(), MlxDtype::Bool);
        let lt_f = astype(&lt, MlxDtype::Float32, None);
        eval(&[&lt_f]);
        assert_eq!(lt_f.data_f32(), &[1.0, 0.0, 0.0]);

        let eq = equal(&a, &b, None);
        let eq_f = astype(&eq, MlxDtype::Float32, None);
        eval(&[&eq_f]);
        assert_eq!(eq_f.data_f32(), &[0.0, 1.0, 0.0]);

        let ne = not_equal(&a, &b, None);
        let ne_f = astype(&ne, MlxDtype::Float32, None);
        eval(&[&ne_f]);
        assert_eq!(ne_f.data_f32(), &[1.0, 0.0, 1.0]);

        let ge = greater_equal(&a, &b, None);
        let ge_f = astype(&ge, MlxDtype::Float32, None);
        eval(&[&ge_f]);
        assert_eq!(ge_f.data_f32(), &[0.0, 1.0, 1.0]);

        let le = less_equal(&a, &b, None);
        let le_f = astype(&le, MlxDtype::Float32, None);
        eval(&[&le_f]);
        assert_eq!(le_f.data_f32(), &[1.0, 1.0, 0.0]);
    }

    #[test]
    fn logical_and_on_bool_arrays() {
        let t: Vec<u8> = vec![1, 1, 0, 0];
        let f: Vec<u8> = vec![1, 0, 1, 0];
        let a = MlxArray::from_raw_data(t.as_ptr(), 4, &[4], MlxDtype::Bool);
        let b = MlxArray::from_raw_data(f.as_ptr(), 4, &[4], MlxDtype::Bool);
        let r = logical_and(&a, &b, None);
        let rf = astype(&r, MlxDtype::Float32, None);
        eval(&[&rf]);
        assert_eq!(rf.data_f32(), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn outer_product_shape_and_values() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let b = MlxArray::from_f32_slice(&[10.0, 20.0]);
        let r = outer(&a, &b, None);
        eval(&[&r]);
        assert_eq!(r.shape(), vec![3, 2]);
        assert_eq!(r.data_f32(), &[10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    #[test]
    fn unary_ops_correctness() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);

        let n = negative(&a, None);
        eval(&[&n]);
        assert_eq!(n.data_f32(), &[-1.0, -2.0, -3.0]);

        let e = exp(&MlxArray::from_f32(0.0), None);
        eval(&[&e]);
        assert_eq!(e.data_f32(), &[1.0]);

        let l = log(&MlxArray::from_f32(1.0), None);
        eval(&[&l]);
        assert_eq!(l.data_f32(), &[0.0]);

        let lp = log1p(&MlxArray::from_f32(0.0), None);
        eval(&[&lp]);
        assert_eq!(lp.data_f32(), &[0.0]);

        let c = cos(&MlxArray::from_f32(0.0), None);
        eval(&[&c]);
        assert_close_f32(c.data_f32(), &[1.0], 1e-6);

        let s = sin(&MlxArray::from_f32(0.0), None);
        eval(&[&s]);
        assert_close_f32(s.data_f32(), &[0.0], 1e-6);

        let fl = floor(&MlxArray::from_f32(2.7), None);
        eval(&[&fl]);
        assert_eq!(fl.data_f32(), &[2.0]);
    }

    #[test]
    fn clip_clamps_values() {
        let a = MlxArray::from_f32_slice(&[-5.0, 0.5, 1.5, 5.0]);
        let lo = MlxArray::from_f32(0.0);
        let hi = MlxArray::from_f32(1.0);
        let r = clip(&a, &lo, &hi, None);
        eval(&[&r]);
        assert_eq!(r.data_f32(), &[0.0, 0.5, 1.0, 1.0]);
    }

    // ── Task 9: Integration smoke test ───────────────────────────

    #[test]
    fn mini_transformer_forward_pass_chains_many_ops() {
        // Simulate a mini transformer step:
        // input -> reshape -> matmul -> softmax -> topk -> take_along_axis
        let batch: i32 = 2;
        let seq: i32 = 4;
        let dim: i32 = 8;
        let vocab: i32 = 16;

        // Input embeddings [batch, seq, dim]
        let input_data: Vec<f32> = (0..batch * seq * dim).map(|i| (i as f32) * 0.01).collect();
        let input = MlxArray::from_raw_data(
            input_data.as_ptr() as *const u8,
            std::mem::size_of_val(&input_data[..]),
            &[batch, seq, dim],
            MlxDtype::Float32,
        );

        // Weight matrix [dim, vocab]
        let w_data: Vec<f32> = (0..dim * vocab)
            .map(|i| ((i as f32) - 64.0) * 0.01)
            .collect();
        let weight = MlxArray::from_raw_data(
            w_data.as_ptr() as *const u8,
            std::mem::size_of_val(&w_data[..]),
            &[dim, vocab],
            MlxDtype::Float32,
        );

        // flatten batch*seq for matmul -> [batch*seq, dim]
        let flat = reshape(&input, &[batch * seq, dim], None);
        // logits = flat @ weight -> [batch*seq, vocab]
        let logits = matmul(&flat, &weight, None);
        // softmax over vocab dim
        let probs = softmax(&logits, -1, None);
        // topk=3 from logits along last axis
        let top_vals = topk_axis(&logits, 3, -1, None);
        // argmax for greedy decode
        let token = argmax(&logits, None);

        eval(&[&probs, &top_vals, &token]);

        assert_eq!(probs.shape(), vec![batch * seq, vocab]);
        assert_eq!(top_vals.shape(), vec![batch * seq, 3]);

        // Verify softmax sums to ~1 for each row
        let row_sum = sum_axis(&probs, -1, false, None);
        eval(&[&row_sum]);
        for &s in row_sum.data_f32() {
            assert_close_f32(&[s], &[1.0], 1e-4);
        }
    }
}
