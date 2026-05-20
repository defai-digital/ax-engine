#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace {
namespace mx = mlx::core;

const mx::array& array_ref(mlx_array arr) {
  if (!arr.ctx) {
    throw std::runtime_error("expected a non-empty mlx_array");
  }
  // mlx_array is a public C handle whose ctx points at the owning MLX C++
  // array. Keeping this shim in mlx-sys confines that ABI assumption to one
  // internal boundary instead of spreading it through runtime code.
  return *static_cast<mx::array*>(arr.ctx);
}

mx::StreamOrDevice stream_or_default(mlx_stream stream) {
  if (!stream.ctx) {
    return {};
  }
  return *static_cast<mx::Stream*>(stream.ctx);
}

std::optional<mx::array> optional_array(mlx_array arr) {
  if (!arr.ctx) {
    return std::nullopt;
  }
  return array_ref(arr);
}

void set_array(mlx_array* dst, mx::array&& value) {
  if (dst == nullptr) {
    throw std::runtime_error("expected a non-null mlx_array output");
  }
  if (dst->ctx) {
    *static_cast<mx::array*>(dst->ctx) = std::move(value);
  } else {
    dst->ctx = new mx::array(std::move(value));
  }
}

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
} // namespace

extern "C" int ax_mlx_gelu_approx_mul(
    mlx_array* res,
    const mlx_array gate,
    const mlx_array x,
    const mlx_stream stream) {
  try {
    set_array(
        res,
        gelu_approx_mul_impl(array_ref(gate), array_ref(x), stream_or_default(stream)));
    return 0;
  } catch (...) {
    return 1;
  }
}

extern "C" int ax_mlx_gelu_approx_mul_matmul(
    mlx_array* res,
    const mlx_array gate,
    const mlx_array x,
    const mlx_array weight,
    const mlx_stream stream) {
  try {
    auto s = stream_or_default(stream);
    auto hidden = gelu_approx_mul_impl(array_ref(gate), array_ref(x), s);
    set_array(res, mx::matmul(hidden, array_ref(weight), s));
    return 0;
  } catch (...) {
    return 1;
  }
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
  try {
    auto s = stream_or_default(stream);
    auto gate_up = quantized_matmul_affine_impl(
        array_ref(x),
        array_ref(gate_up_weight),
        array_ref(gate_up_scales),
        optional_array(gate_up_biases),
        group_size,
        bits,
        s);
    auto parts = mx::split(gate_up, 2, -1, s);
    if (parts.size() != 2) {
      throw std::runtime_error("expected gate_up split to produce two arrays");
    }
    auto hidden = gelu_approx_mul_impl(parts[0], parts[1], s);
    set_array(
        res,
        quantized_matmul_affine_impl(
            hidden,
            array_ref(down_weight),
            array_ref(down_scales),
            optional_array(down_biases),
            group_size,
            bits,
            s));
    return 0;
  } catch (...) {
    return 1;
  }
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
  try {
    auto [qkv, z, a, b] = qwen_linear_attention_inputs_packed_impl(
        array_ref(x),
        array_ref(qkvz_weight),
        optional_array(qkvz_scales),
        optional_array(qkvz_biases),
        array_ref(ba_weight),
        optional_array(ba_scales),
        optional_array(ba_biases),
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        group_size,
        bits,
        stream_or_default(stream));
    set_array(qkv_res, std::move(qkv));
    set_array(z_res, std::move(z));
    set_array(a_res, std::move(a));
    set_array(b_res, std::move(b));
    return 0;
  } catch (...) {
    return 1;
  }
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
  try {
    set_array(
        res,
        qk_norm_rope_bhsd_from_proj_impl(
            array_ref(proj),
            optional_array(norm),
            n_heads,
            head_dim,
            eps,
            rope_dims,
            traditional != 0,
            has_base != 0,
            base,
            offset,
            optional_array(freqs),
            stream_or_default(stream)));
    return 0;
  } catch (...) {
    return 1;
  }
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
  try {
    set_array(
        res,
        gemma4_post_attn_ffn_block_impl(
            array_ref(hidden),
            array_ref(attn_out),
            array_ref(ffn_norm),
            optional_array(ffn_post_norm),
            optional_array(layer_scalar),
            array_ref(gate_up_weight),
            array_ref(gate_up_scales),
            optional_array(gate_up_biases),
            array_ref(down_weight),
            array_ref(down_scales),
            optional_array(down_biases),
            group_size,
            bits,
            eps,
            stream_or_default(stream)));
    return 0;
  } catch (...) {
    return 1;
  }
}
