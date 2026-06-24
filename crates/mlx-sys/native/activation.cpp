#include "ax_shim_internal.h"

#include <tuple>

#include "mlx/fast.h"

namespace {

mx::array gelu_approx_mul_impl(
    const mx::array& gate,
    const mx::array& x,
    mx::StreamOrDevice stream) {
  auto dtype = gate.dtype();
  auto half = mx::array(0.5f, dtype);
  auto one = mx::array(1.0f, dtype);
  auto sqrt_2_over_pi = mx::array(0.7978846f, dtype);
  auto coeff = mx::array(0.044715f, dtype);

  auto gate2 = mx::multiply(gate, gate, stream);
  auto gate3 = mx::multiply(gate2, gate, stream);
  auto cubic = mx::multiply(coeff, gate3, stream);
  auto inner = mx::add(gate, cubic, stream);
  auto t = mx::tanh(mx::multiply(sqrt_2_over_pi, inner, stream), stream);
  auto activated = mx::multiply(
      mx::multiply(half, gate, stream), mx::add(one, t, stream), stream);
  return mx::multiply(activated, x, stream);
}

mx::array silu_mul_impl(
    const mx::array& gate,
    const mx::array& x,
    mx::StreamOrDevice stream) {
  auto activated = mx::multiply(gate, mx::sigmoid(gate, stream), stream);
  return mx::multiply(activated, x, stream);
}

mx::array quantized_matmul_affine_impl(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    std::optional<mx::array> biases,
    int group_size,
    int bits,
    mx::StreamOrDevice stream) {
  return mx::quantized_matmul(
      x,
      weight,
      scales,
      std::move(biases),
      true,
      std::make_optional<int>(group_size),
      std::make_optional<int>(bits),
      "affine",
      stream);
}

mx::array projection_affine_or_dense_impl(
    const mx::array& x,
    const mx::array& weight,
    std::optional<mx::array> scales,
    std::optional<mx::array> biases,
    int group_size,
    int bits,
    mx::StreamOrDevice stream) {
  if (scales.has_value()) {
    return quantized_matmul_affine_impl(
        x, weight, *scales, std::move(biases), group_size, bits, stream);
  }
  auto transposed_weight = mx::transpose(weight, {1, 0}, stream);
  return mx::matmul(x, transposed_weight, stream);
}

mx::array slice_last_dim_impl(
    const mx::array& x,
    int start,
    int stop,
    mx::StreamOrDevice stream) {
  if (x.ndim() == 0) {
    throw std::runtime_error("slice_last_dim expects at least one dimension");
  }
  mx::Shape starts(x.ndim(), 0);
  mx::Shape stops = x.shape();
  starts[x.ndim() - 1] = start;
  stops[x.ndim() - 1] = stop;
  return mx::slice(x, std::move(starts), std::move(stops), stream);
}

std::tuple<mx::array, mx::array, mx::array, mx::array>
qwen_linear_attention_inputs_packed_impl(
    const mx::array& x,
    const mx::array& qkvz_weight,
    std::optional<mx::array> qkvz_scales,
    std::optional<mx::array> qkvz_biases,
    const mx::array& ba_weight,
    std::optional<mx::array> ba_scales,
    std::optional<mx::array> ba_biases,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    int group_size,
    int bits,
    mx::StreamOrDevice stream) {
  if (num_key_heads <= 0 || num_value_heads <= 0 ||
      key_head_dim <= 0 || value_head_dim <= 0) {
    throw std::runtime_error("linear attention packed inputs require positive dimensions");
  }
  if (num_value_heads % num_key_heads != 0) {
    throw std::runtime_error("num_value_heads must be divisible by num_key_heads");
  }

  auto mixed_qkvz = projection_affine_or_dense_impl(
      x,
      qkvz_weight,
      std::move(qkvz_scales),
      std::move(qkvz_biases),
      group_size,
      bits,
      stream);
  if (mixed_qkvz.ndim() != 3) {
    throw std::runtime_error("packed qkvz projection must produce [B, S, C]");
  }

  const int batch = mixed_qkvz.shape(0);
  const int seq = mixed_qkvz.shape(1);
  const int value_heads_per_key = num_value_heads / num_key_heads;
  const int value_dim_per_key = value_heads_per_key * value_head_dim;
  const int qkvz_per_key = key_head_dim * 2 + value_dim_per_key * 2;
  if (mixed_qkvz.shape(2) != num_key_heads * qkvz_per_key) {
    throw std::runtime_error("packed qkvz width does not match linear attention config");
  }

  mixed_qkvz = mx::reshape(
      mixed_qkvz,
      {batch, seq, num_key_heads, qkvz_per_key},
      stream);
  auto q = slice_last_dim_impl(mixed_qkvz, 0, key_head_dim, stream);
  auto k = slice_last_dim_impl(mixed_qkvz, key_head_dim, key_head_dim * 2, stream);
  auto v = slice_last_dim_impl(
      mixed_qkvz,
      key_head_dim * 2,
      key_head_dim * 2 + value_dim_per_key,
      stream);
  auto z = slice_last_dim_impl(
      mixed_qkvz,
      key_head_dim * 2 + value_dim_per_key,
      qkvz_per_key,
      stream);

  const int key_dim = num_key_heads * key_head_dim;
  const int value_dim = num_value_heads * value_head_dim;
  auto qkv = mx::concatenate(
      {
          mx::reshape(q, {batch, seq, key_dim}, stream),
          mx::reshape(k, {batch, seq, key_dim}, stream),
          mx::reshape(v, {batch, seq, value_dim}, stream),
      },
      2,
      stream);
  z = mx::reshape(z, {batch, seq, num_value_heads, value_head_dim}, stream);

  auto mixed_ba = projection_affine_or_dense_impl(
      x,
      ba_weight,
      std::move(ba_scales),
      std::move(ba_biases),
      group_size,
      bits,
      stream);
  if (mixed_ba.ndim() != 3) {
    throw std::runtime_error("packed ba projection must produce [B, S, C]");
  }
  if (mixed_ba.shape(0) != batch || mixed_ba.shape(1) != seq) {
    throw std::runtime_error("packed ba projection shape must match qkvz batch/sequence");
  }
  if (mixed_ba.shape(2) != num_key_heads * value_heads_per_key * 2) {
    throw std::runtime_error("packed ba width does not match linear attention config");
  }

  auto ba = mx::reshape(
      mixed_ba,
      {batch, seq, num_key_heads, value_heads_per_key * 2},
      stream);
  auto b = mx::reshape(
      slice_last_dim_impl(ba, 0, value_heads_per_key, stream),
      {batch, seq, num_value_heads},
      stream);
  auto a = mx::reshape(
      slice_last_dim_impl(ba, value_heads_per_key, value_heads_per_key * 2, stream),
      {batch, seq, num_value_heads},
      stream);

  return {qkv, z, a, b};
}

// Qwen linear-attention post-input block: from the concatenated qkv produced
// by `qwen_linear_attention_inputs_packed` (or the portable composition),
// run depthwise conv1d with cached state carry, SiLU, last-dim split into
// (q_flat, k_flat, v_flat), reshape into head-major layout, then RMSNorm + scale
// on q and k. This is everything between input staging and the
// `qwen35_gated_delta_v3` custom Metal kernel, packed into one Rust→C++ FFI
// round-trip so the per-decode-token op-count drops from ~14 mlx-c dispatches
// per linear-attention layer to 1.
std::tuple<mx::array, mx::array, mx::array, mx::array>
qwen_linear_attention_post_input_impl(
    const mx::array& qkv,
    const mx::array& conv_weight,
    std::optional<mx::array> cached_conv_state,
    int num_key_heads,
    int key_head_dim,
    int num_value_heads,
    int value_head_dim,
    int conv_kernel_dim,
    float q_scale,
    float k_scale,
    float rms_norm_eps,
    mx::StreamOrDevice stream) {
  if (qkv.ndim() != 3) {
    throw std::runtime_error("linear attention post-input expects [B, T, conv_dim] qkv");
  }
  if (conv_weight.ndim() != 3) {
    throw std::runtime_error("linear attention post-input expects 3-D conv1d weight");
  }
  if (num_key_heads <= 0 || num_value_heads <= 0 || key_head_dim <= 0 ||
      value_head_dim <= 0 || conv_kernel_dim <= 0) {
    throw std::runtime_error("linear attention post-input requires positive dimensions");
  }
  if (num_value_heads % num_key_heads != 0) {
    throw std::runtime_error("num_value_heads must be divisible by num_key_heads");
  }

  const int batch = qkv.shape(0);
  const int seq = qkv.shape(1);
  const int conv_dim = qkv.shape(2);
  const int key_dim = num_key_heads * key_head_dim;
  const int value_dim = num_value_heads * value_head_dim;
  if (conv_dim != 2 * key_dim + value_dim) {
    throw std::runtime_error(
        "linear attention post-input qkv last dim must equal 2*key_dim + value_dim");
  }
  const int tail_len = conv_kernel_dim - 1;
  if (tail_len < 0) {
    throw std::runtime_error("conv_kernel_dim must be >= 1");
  }

  // 1. Build (or reuse) the conv-state carry. mlx_lm initialises state as
  //    `mx::zeros([B, tail_len, conv_dim], dtype)` matching the qkv dtype.
  mx::array conv_state = cached_conv_state.has_value()
      ? std::move(cached_conv_state.value())
      : mx::zeros({batch, tail_len, conv_dim}, qkv.dtype(), stream);
  if (conv_state.ndim() != 3 || conv_state.shape(0) != batch ||
      conv_state.shape(1) != tail_len || conv_state.shape(2) != conv_dim) {
    throw std::runtime_error("cached conv state shape mismatch");
  }

  // 2. Concatenate prior state with current step's qkv along the time axis,
  //    then slice the tail to feed forward as the next call's state.
  auto conv_input = mx::concatenate({conv_state, qkv}, 1, stream);
  const int total = conv_input.shape(1);
  auto new_conv_state = mx::slice(
      conv_input,
      mx::Shape{0, total - tail_len, 0},
      mx::Shape{batch, total, conv_dim},
      mx::Shape{1, 1, 1},
      stream);

  // 3. Depthwise conv1d (groups = conv_dim) and SiLU. The Rust composition
  //    uses padding=0; the prepended state carry compensates for the kernel
  //    receptive field so no zero padding is required.
  auto conv_out = mx::conv1d(conv_input, conv_weight, 1, 0, 1, conv_dim, stream);
  conv_out = mx::multiply(conv_out, mx::sigmoid(conv_out, stream), stream);

  // 4. Split [B, T, conv_dim] into (q_flat, k_flat, v_flat) along last axis.
  auto q_flat = slice_last_dim_impl(conv_out, 0, key_dim, stream);
  auto k_flat = slice_last_dim_impl(conv_out, key_dim, 2 * key_dim, stream);
  auto v_flat = slice_last_dim_impl(conv_out, 2 * key_dim, 2 * key_dim + value_dim, stream);

  // 5. Reshape into head-major layout matching `split_linear_attention_qkv`.
  auto q = mx::reshape(q_flat, {batch, seq, num_key_heads, key_head_dim}, stream);
  auto k = mx::reshape(k_flat, {batch, seq, num_key_heads, key_head_dim}, stream);
  auto v = mx::reshape(v_flat, {batch, seq, num_value_heads, value_head_dim}, stream);

  // 6. Per-head RMSNorm on q and k (no learned weight, just normalisation).
  //    Matches `normalize_linear_attention_qk` which calls `rms_norm(q, None, eps)`.
  std::optional<mx::array> no_weight = std::nullopt;
  auto q_normed = mx::fast::rms_norm(q, no_weight, rms_norm_eps, stream);
  auto k_normed = mx::fast::rms_norm(k, no_weight, rms_norm_eps, stream);

  // 7. Scale by precomputed (q_scale, k_scale) constants. These are derived
  //    from `linear_attention_qk_scale(key_head_dim)` in Rust; passing them in
  //    avoids replicating that derivation here.
  auto q_scale_arr = mx::array(q_scale, q.dtype());
  auto k_scale_arr = mx::array(k_scale, k.dtype());
  q = mx::multiply(q_normed, q_scale_arr, stream);
  k = mx::multiply(k_normed, k_scale_arr, stream);

  return {q, k, v, new_conv_state};
}

mx::array qk_norm_rope_bhsd_from_proj_impl(
    const mx::array& proj,
    std::optional<mx::array> norm,
    int n_heads,
    int head_dim,
    float eps,
    int rope_dims,
    bool traditional,
    bool has_base,
    float base,
    int offset,
    std::optional<mx::array> freqs,
    mx::StreamOrDevice stream) {
  if (proj.ndim() != 3) {
    throw std::runtime_error("qk_norm_rope expects [B, S, H * D] projection");
  }
  if (n_heads <= 0 || head_dim <= 0) {
    throw std::runtime_error("qk_norm_rope expects positive head dimensions");
  }
  auto batch = proj.shape(0);
  auto seq = proj.shape(1);
  auto width = proj.shape(2);
  if (width != n_heads * head_dim) {
    throw std::runtime_error("qk_norm_rope projection width does not match heads * head_dim");
  }

  mx::Shape bhsd_shape{batch, n_heads, seq, head_dim};
  mx::Strides bhsd_strides{
      static_cast<int64_t>(seq) * n_heads * head_dim,
      head_dim,
      static_cast<int64_t>(n_heads) * head_dim,
      1};
  auto bhsd = mx::as_strided(proj, bhsd_shape, bhsd_strides, 0, stream);
  auto normed = mx::fast::rms_norm(bhsd, norm, eps, stream);
  std::optional<float> base_opt = has_base ? std::make_optional(base) : std::nullopt;
  return mx::fast::rope(
      normed,
      rope_dims,
      traditional,
      base_opt,
      1.0f,
      offset,
      freqs,
      stream);
}

mx::array gemma4_post_attn_ffn_block_impl(
    const mx::array& hidden,
    const mx::array& attn_out,
    const mx::array& ffn_norm,
    std::optional<mx::array> ffn_post_norm,
    std::optional<mx::array> layer_scalar,
    const mx::array& gate_up_weight,
    const mx::array& gate_up_scales,
    std::optional<mx::array> gate_up_biases,
    const mx::array& down_weight,
    const mx::array& down_scales,
    std::optional<mx::array> down_biases,
    int group_size,
    int bits,
    float eps,
    mx::StreamOrDevice stream) {
  auto residual = mx::add(hidden, attn_out, stream);
  auto normed = mx::fast::rms_norm(residual, ffn_norm, eps, stream);
  auto gate_up = quantized_matmul_affine_impl(
      normed,
      gate_up_weight,
      gate_up_scales,
      std::move(gate_up_biases),
      group_size,
      bits,
      stream);
  auto parts = mx::split(gate_up, 2, -1, stream);
  if (parts.size() != 2) {
    throw std::runtime_error("expected gate_up split to produce two arrays");
  }
  auto ffn_hidden = gelu_approx_mul_impl(parts[0], parts[1], stream);
  auto ffn_out = quantized_matmul_affine_impl(
      ffn_hidden,
      down_weight,
      down_scales,
      std::move(down_biases),
      group_size,
      bits,
      stream);
  if (ffn_post_norm.has_value()) {
    ffn_out = mx::fast::rms_norm(ffn_out, std::move(ffn_post_norm), eps, stream);
  }
  auto out = mx::add(residual, ffn_out, stream);
  if (layer_scalar.has_value()) {
    out = mx::multiply(out, *layer_scalar, stream);
  }
  return out;
}

std::pair<mx::array, mx::array> add_rms_norm_pair_impl(
    const mx::array& x,
    const mx::array& y,
    const mx::array& norm_weight,
    float eps,
    mx::StreamOrDevice stream) {
  auto residual = mx::add(x, y, stream);
  auto normed = mx::fast::rms_norm(residual, norm_weight, eps, stream);
  return {residual, normed};
}

mx::array quantized_matmul_rms_norm_impl(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    std::optional<mx::array> biases,
    int group_size,
    int bits,
    const mx::array& norm_weight,
    float eps,
    mx::StreamOrDevice stream) {
  auto projected = quantized_matmul_affine_impl(
      x,
      weight,
      scales,
      std::move(biases),
      group_size,
      bits,
      stream);
  return mx::fast::rms_norm(projected, norm_weight, eps, stream);
}

} // namespace

extern "C" int ax_mlx_gelu_approx_mul(
    mlx_array* res,
    const mlx_array gate,
    const mlx_array x,
    const mlx_stream stream) {
  AX_TRY {
    aset(
        res,
        gelu_approx_mul_impl(aref(gate), aref(x), sd(stream)));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_silu_mul(
    mlx_array* res,
    const mlx_array gate,
    const mlx_array x,
    const mlx_stream stream) {
  AX_TRY {
    aset(
        res,
        silu_mul_impl(aref(gate), aref(x), sd(stream)));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_gelu_approx_mul_matmul(
    mlx_array* res,
    const mlx_array gate,
    const mlx_array x,
    const mlx_array weight,
    const mlx_stream stream) {
  AX_TRY {
    auto s = sd(stream);
    auto hidden = gelu_approx_mul_impl(aref(gate), aref(x), s);
    aset(res, mx::matmul(hidden, aref(weight), s));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_gelu_approx_mul_quantized_matmul(
    mlx_array* res,
    const mlx_array gate,
    const mlx_array x,
    const mlx_array weight,
    const mlx_array scales,
    const mlx_array biases,
    int group_size,
    int bits,
    const mlx_stream stream) {
  AX_TRY {
    auto s = sd(stream);
    auto hidden = gelu_approx_mul_impl(aref(gate), aref(x), s);
    aset(
        res,
        quantized_matmul_affine_impl(
            hidden,
            aref(weight),
            aref(scales),
            opt_arr(biases),
            group_size,
            bits,
            s));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_add_rms_norm_pair(
    mlx_array* residual_res,
    mlx_array* normed_res,
    const mlx_array x,
    const mlx_array y,
    const mlx_array norm_weight,
    float eps,
    const mlx_stream stream) {
  AX_TRY {
    auto [residual, normed] = add_rms_norm_pair_impl(
        aref(x),
        aref(y),
        aref(norm_weight),
        eps,
        sd(stream));
    aset(residual_res, std::move(residual));
    aset(normed_res, std::move(normed));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_quantized_matmul_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight,
    const mlx_array scales,
    const mlx_array biases,
    int group_size,
    int bits,
    const mlx_array norm_weight,
    float eps,
    const mlx_stream stream) {
  AX_TRY {
    aset(
        res,
        quantized_matmul_rms_norm_impl(
            aref(x),
            aref(weight),
            aref(scales),
            opt_arr(biases),
            group_size,
            bits,
            aref(norm_weight),
            eps,
            sd(stream)));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_gelu_approx_quantized_ffn(
    mlx_array* res,
    const mlx_array x,
    const mlx_array gate_up_weight,
    const mlx_array gate_up_scales,
    const mlx_array gate_up_biases,
    const mlx_array down_weight,
    const mlx_array down_scales,
    const mlx_array down_biases,
    int group_size,
    int bits,
    const mlx_stream stream) {
  AX_TRY {
    auto s = sd(stream);
    auto gate_up = quantized_matmul_affine_impl(
        aref(x),
        aref(gate_up_weight),
        aref(gate_up_scales),
        opt_arr(gate_up_biases),
        group_size,
        bits,
        s);
    auto parts = mx::split(gate_up, 2, -1, s);
    if (parts.size() != 2) {
      throw std::runtime_error("expected gate_up split to produce two arrays");
    }
    auto hidden = gelu_approx_mul_impl(parts[0], parts[1], s);
    aset(
        res,
        quantized_matmul_affine_impl(
            hidden,
            aref(down_weight),
            aref(down_scales),
            opt_arr(down_biases),
            group_size,
            bits,
            s));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_qwen_linear_attention_inputs_packed(
    mlx_array* qkv_res,
    mlx_array* z_res,
    mlx_array* a_res,
    mlx_array* b_res,
    const mlx_array x,
    const mlx_array qkvz_weight,
    const mlx_array qkvz_scales,
    const mlx_array qkvz_biases,
    const mlx_array ba_weight,
    const mlx_array ba_scales,
    const mlx_array ba_biases,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    int group_size,
    int bits,
    const mlx_stream stream) {
  AX_TRY {
    auto [qkv, z, a, b] = qwen_linear_attention_inputs_packed_impl(
        aref(x),
        aref(qkvz_weight),
        opt_arr(qkvz_scales),
        opt_arr(qkvz_biases),
        aref(ba_weight),
        opt_arr(ba_scales),
        opt_arr(ba_biases),
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        group_size,
        bits,
        sd(stream));
    aset(qkv_res, std::move(qkv));
    aset(z_res, std::move(z));
    aset(a_res, std::move(a));
    aset(b_res, std::move(b));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_qwen_linear_attention_post_input(
    mlx_array* q_res,
    mlx_array* k_res,
    mlx_array* v_res,
    mlx_array* new_conv_state_res,
    const mlx_array qkv,
    const mlx_array conv_weight,
    const mlx_array cached_conv_state,
    int num_key_heads,
    int key_head_dim,
    int num_value_heads,
    int value_head_dim,
    int conv_kernel_dim,
    float q_scale,
    float k_scale,
    float rms_norm_eps,
    const mlx_stream stream) {
  AX_TRY {
    auto [q, k, v, new_conv_state] = qwen_linear_attention_post_input_impl(
        aref(qkv),
        aref(conv_weight),
        opt_arr(cached_conv_state),
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
        conv_kernel_dim,
        q_scale,
        k_scale,
        rms_norm_eps,
        sd(stream));
    aset(q_res, std::move(q));
    aset(k_res, std::move(k));
    aset(v_res, std::move(v));
    aset(new_conv_state_res, std::move(new_conv_state));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_qk_norm_rope_bhsd_from_proj(
    mlx_array* res,
    const mlx_array proj,
    const mlx_array norm,
    int n_heads,
    int head_dim,
    float eps,
    int rope_dims,
    int traditional,
    int has_base,
    float base,
    int offset,
    const mlx_array freqs,
    const mlx_stream stream) {
  AX_TRY {
    aset(
        res,
        qk_norm_rope_bhsd_from_proj_impl(
            aref(proj),
            opt_arr(norm),
            n_heads,
            head_dim,
            eps,
            rope_dims,
            traditional != 0,
            has_base != 0,
            base,
            offset,
            opt_arr(freqs),
            sd(stream)));
    return 0;
  } AX_CATCH
}

extern "C" int ax_mlx_gemma4_post_attn_ffn_block(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array attn_out,
    const mlx_array ffn_norm,
    const mlx_array ffn_post_norm,
    const mlx_array layer_scalar,
    const mlx_array gate_up_weight,
    const mlx_array gate_up_scales,
    const mlx_array gate_up_biases,
    const mlx_array down_weight,
    const mlx_array down_scales,
    const mlx_array down_biases,
    int group_size,
    int bits,
    float eps,
    const mlx_stream stream) {
  AX_TRY {
    aset(
        res,
        gemma4_post_attn_ffn_block_impl(
            aref(hidden),
            aref(attn_out),
            aref(ffn_norm),
            opt_arr(ffn_post_norm),
            opt_arr(layer_scalar),
            aref(gate_up_weight),
            aref(gate_up_scales),
            opt_arr(gate_up_biases),
            aref(down_weight),
            aref(down_scales),
            opt_arr(down_biases),
            group_size,
            bits,
            eps,
            sd(stream)));
    return 0;
  } AX_CATCH
}
