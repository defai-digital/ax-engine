// Syntax-only probe compiled by build.rs. It is not linked into mlx-sys.
//
// The call is well-formed only when fast::scaled_dot_product_attention takes
// a positional int window_size after scale. MLX 0.32.4 takes the mask string
// in that position, so this translation unit fails to compile and the
// windowed wrapper stays out of the build.

#include <optional>
#include <string>

#include "mlx/array.h"
#include "mlx/fast.h"

template <typename Function>
auto ax_mlx_call_sdpa_window(
    Function function,
    const mlx::core::array& queries,
    const mlx::core::array& keys,
    const mlx::core::array& values,
    float scale,
    int window_size) -> decltype(function(
    queries,
    keys,
    values,
    scale,
    window_size,
    std::string(),
    std::optional<mlx::core::array>(),
    std::optional<mlx::core::array>(),
    false,
    mlx::core::StreamOrDevice())) {
  return function(
      queries,
      keys,
      values,
      scale,
      window_size,
      std::string(),
      std::optional<mlx::core::array>(),
      std::optional<mlx::core::array>(),
      false,
      mlx::core::StreamOrDevice());
}

void ax_mlx_sdpa_window_probe(
    const mlx::core::array& queries,
    const mlx::core::array& keys,
    const mlx::core::array& values) {
  (void)ax_mlx_call_sdpa_window(
      mlx::core::fast::scaled_dot_product_attention, queries, keys, values, 1.0f, -1);
}
