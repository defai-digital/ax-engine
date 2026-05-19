#include <exception>
#include <stdexcept>
#include <utility>

#include "mlx/c/array.h"
#include "mlx/c/stream.h"
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
