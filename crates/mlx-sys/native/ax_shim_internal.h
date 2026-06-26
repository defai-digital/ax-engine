/* ax_shim_internal.h — shared C++ helpers for ax_shim.cpp and activation.cpp.
 *
 * Both translation units operate on the same mlx_array / mlx_stream handle
 * layout, so the casting helpers and error-handling macros live here to
 * avoid duplication and keep null-check logic consistent.
 *
 * This header is internal to mlx-sys/native/ and must NOT be installed or
 * exposed to downstream consumers.
 */
#ifndef AX_SHIM_INTERNAL_H
#define AX_SHIM_INTERNAL_H

#include "ax_shim.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

/* ================================================================
 * Error propagation — defined in ax_shim.cpp, used by both TUs.
 * ================================================================ */

/// Store the current exception's message via the thread-local error handler.
/// Must be called inside a `catch` block so `std::current_exception()` is valid.
void ax_set_current_error();

/* ================================================================
 * Handle-casting helpers
 *
 * Every helper that dereferences a handle performs a null check and
 * throws on a null ctx, so a programming error surfaces as a caught
 * exception instead of undefined behaviour.
 * ================================================================ */

inline mx::array& aref(mlx_array a) {
  if (!a.ctx) throw std::runtime_error("expected a non-empty mlx_array");
  return *static_cast<mx::array*>(a.ctx);
}

inline mx::Stream& sref(mlx_stream s) {
  if (!s.ctx) throw std::runtime_error("expected a non-empty mlx_stream");
  return *static_cast<mx::Stream*>(s.ctx);
}

inline mx::StreamOrDevice sd(mlx_stream s) {
  return s.ctx ? mx::StreamOrDevice(*static_cast<mx::Stream*>(s.ctx))
               : mx::StreamOrDevice{};
}

inline void aset(mlx_array* dst, mx::array&& v) {
  if (!dst) throw std::runtime_error("expected a non-null mlx_array output");
  if (dst->ctx) *static_cast<mx::array*>(dst->ctx) = std::move(v);
  else dst->ctx = new mx::array(std::move(v));
}

inline void aset(mlx_array* dst, const mx::array& v) {
  if (!dst) throw std::runtime_error("expected a non-null mlx_array output");
  if (dst->ctx) *static_cast<mx::array*>(dst->ctx) = v;
  else dst->ctx = new mx::array(v);
}

inline std::vector<mx::array>& varef(mlx_vector_array v) {
  if (!v.ctx) throw std::runtime_error("expected a non-empty mlx_vector_array");
  return *static_cast<std::vector<mx::array>*>(v.ctx);
}

inline void vaset(mlx_vector_array* dst, std::vector<mx::array>&& v) {
  if (!dst) throw std::runtime_error("expected a non-null mlx_vector_array output");
  if (dst->ctx) *static_cast<std::vector<mx::array>*>(dst->ctx) = std::move(v);
  else dst->ctx = new std::vector<mx::array>(std::move(v));
}

inline std::vector<std::string>& vsref(mlx_vector_string v) {
  if (!v.ctx) throw std::runtime_error("expected a non-empty mlx_vector_string");
  return *static_cast<std::vector<std::string>*>(v.ctx);
}

inline mx::Device& dref(mlx_device d) {
  if (!d.ctx) throw std::runtime_error("expected a non-empty mlx_device");
  return *static_cast<mx::Device*>(d.ctx);
}

inline std::optional<mx::array> opt_arr(mlx_array a) {
  return a.ctx ? std::make_optional(aref(a)) : std::nullopt;
}

/* ================================================================
 * Error-handling macros — identical for both translation units.
 * ================================================================ */

#define AX_TRY try
#define AX_CATCH \
  catch (const std::exception&) { ax_set_current_error(); return 1; } \
  catch (...) { ax_set_current_error(); return 1; }
#define AX_CATCH_NULL \
  catch (const std::exception&) { ax_set_current_error(); return {nullptr}; } \
  catch (...) { ax_set_current_error(); return {nullptr}; }

#endif /* AX_SHIM_INTERNAL_H */
