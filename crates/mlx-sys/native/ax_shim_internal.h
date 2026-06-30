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

#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
 * Type-tagged handles
 *
 * Every handle's ctx points to an ax_tagged<T> wrapper that stores a
 * magic number before the wrapped object.  Accessor helpers verify the
 * tag at runtime, turning wrong-handle-type bugs (passing a stream
 * where an array is expected) into caught exceptions instead of
 * silent type-punning UB.  The C ABI struct layout ({ void* ctx; })
 * is unchanged — only the internal allocation format differs.
 * ================================================================ */

// Forward declaration for error reporting in typed_delete.
void ax_set_error(const char* msg);

enum ax_magic : uint32_t {
  AX_MAGIC_ARRAY       = 0xA11A0001,
  AX_MAGIC_STREAM      = 0xA11A0002,
  AX_MAGIC_VEC_ARRAY   = 0xA11A0003,
  AX_MAGIC_VEC_STRING  = 0xA11A0004,
  AX_MAGIC_DEVICE      = 0xA11A0005,
  AX_MAGIC_DEVICE_INFO = 0xA11A0006,
  AX_MAGIC_CLOSURE     = 0xA11A0007,
  AX_MAGIC_MAP_SA      = 0xA11A0008,
  AX_MAGIC_MAP_SS      = 0xA11A0009,
  AX_MAGIC_METAL_KERN  = 0xA11A000A,
  AX_MAGIC_METAL_CFG   = 0xA11A000B,
  AX_MAGIC_ITER_SA     = 0xA11A000C,
};

/// Wraps a C++ object with a leading magic-number tag.  The trivial
/// destructor is safe because typed_delete calls obj.~T() explicitly
/// before freeing the memory.
template<typename T>
struct ax_tagged {
  uint32_t magic;
  T obj;
  ~ax_tagged() = default;
};

/// Unwrap a handle, verifying the magic tag matches the expected type.
/// Returns a pointer to the wrapped object, or nullptr for a null handle.
template<uint32_t Magic, typename T>
inline T* unwrap(void* ctx) {
  if (!ctx) return nullptr;
  auto* h = static_cast<ax_tagged<T>*>(ctx);
  if (h->magic != Magic) {
    throw std::runtime_error("handle type mismatch: wrong magic number");
  }
  return &h->obj;
}

/// Same as unwrap but returns a reference, throwing on null.
template<uint32_t Magic, typename T>
inline T& unwrap_ref(void* ctx, const char* type_name) {
  T* p = unwrap<Magic, T>(ctx);
  if (!p) throw std::runtime_error(
      std::string("expected a non-empty ") + type_name);
  return *p;
}

/// Allocate a new tagged handle wrapping the given object.
template<uint32_t Magic, typename T, typename... Args>
inline void* make_handle(Args&&... args) {
  void* mem = ::operator new(sizeof(ax_tagged<T>));
  auto* h = static_cast<ax_tagged<T>*>(mem);
  h->magic = Magic;
  std::construct_at(&h->obj, std::forward<Args>(args)...);
  return h;
}

/// Free a tagged handle, verifying the magic tag before destructing.
/// Reports (but does not throw on) tag mismatches — free functions
/// should not throw across the C ABI.
template<uint32_t Magic, typename T>
inline void typed_delete(void* ctx) {
  if (!ctx) return;
  auto* h = static_cast<ax_tagged<T>*>(ctx);
  if (h->magic != Magic) {
    ax_set_error("handle type mismatch in free: wrong magic number");
    return;
  }
  h->obj.~T();
  ::operator delete(h);
}

/* ================================================================
 * Handle-casting helpers (now type-tagged)
 * ================================================================ */

inline mx::array& aref(mlx_array a) {
  return unwrap_ref<AX_MAGIC_ARRAY, mx::array>(a.ctx, "mlx_array");
}

inline mx::Stream& sref(mlx_stream s) {
  return unwrap_ref<AX_MAGIC_STREAM, mx::Stream>(s.ctx, "mlx_stream");
}

inline mx::StreamOrDevice sd(mlx_stream s) {
  auto* p = unwrap<AX_MAGIC_STREAM, mx::Stream>(s.ctx);
  return p ? mx::StreamOrDevice(*p) : mx::StreamOrDevice{};
}

inline void aset(mlx_array* dst, mx::array&& v) {
  if (!dst) throw std::runtime_error("expected a non-null mlx_array output");
  auto* p = unwrap<AX_MAGIC_ARRAY, mx::array>(dst->ctx);
  if (p) *p = std::move(v);
  else dst->ctx = make_handle<AX_MAGIC_ARRAY, mx::array>(std::move(v));
}

inline void aset(mlx_array* dst, const mx::array& v) {
  if (!dst) throw std::runtime_error("expected a non-null mlx_array output");
  auto* p = unwrap<AX_MAGIC_ARRAY, mx::array>(dst->ctx);
  if (p) *p = v;
  else dst->ctx = make_handle<AX_MAGIC_ARRAY, mx::array>(v);
}

inline std::vector<mx::array>& varef(mlx_vector_array v) {
  return unwrap_ref<AX_MAGIC_VEC_ARRAY, std::vector<mx::array>>(
      v.ctx, "mlx_vector_array");
}

inline void vaset(mlx_vector_array* dst, std::vector<mx::array>&& v) {
  if (!dst) throw std::runtime_error(
      "expected a non-null mlx_vector_array output");
  auto* p = unwrap<AX_MAGIC_VEC_ARRAY, std::vector<mx::array>>(dst->ctx);
  if (p) *p = std::move(v);
  else dst->ctx = make_handle<AX_MAGIC_VEC_ARRAY, std::vector<mx::array>>(
      std::move(v));
}

inline std::vector<std::string>& vsref(mlx_vector_string v) {
  return unwrap_ref<AX_MAGIC_VEC_STRING, std::vector<std::string>>(
      v.ctx, "mlx_vector_string");
}

inline mx::Device& dref(mlx_device d) {
  return unwrap_ref<AX_MAGIC_DEVICE, mx::Device>(d.ctx, "mlx_device");
}

inline std::optional<mx::array> opt_arr(mlx_array a) {
  return a.ctx ? std::make_optional(aref(a)) : std::nullopt;
}

/* ================================================================
 * Small helpers
 * ================================================================ */

/// Construct a std::string from a possibly-null C pointer.  Passing a null
/// const char* to std::string is undefined behaviour; this helper returns an
/// empty string instead, turning a potential crash into a caught exception
/// from the downstream MLX API.
inline std::string safe_str(const char* s) {
  return s ? std::string(s) : std::string();
}

/// Build an mx::Shape from a raw int pointer + count.  For shapes with ≤ 8
/// dimensions (the overwhelming majority of LLM ops: scalars, vectors,
/// matrices, [B,S,H,D] tensors, grouped-attention [B,G,H,S,D]) a stack-
/// allocated std::array avoids a heap round-trip on every reshape /
/// transpose / slice.
inline mx::Shape make_shape(const int* p, size_t n) {
  if (n == 0 || !p) {
    return mx::Shape{};
  }
  if (n <= 8) {
    std::array<int, 8> buf{};
    std::memcpy(buf.data(), p, n * sizeof(int));
    return mx::Shape(buf.data(), buf.data() + n);
  }
  return mx::Shape(p, p + n);
}

/// Build a std::vector<int> from a raw pointer + count, using a small-buffer
/// optimisation for ≤ 8 elements.  Avoids a heap alloc for the common
/// transpose / expand-dims axes that are 1–4 elements.
inline std::vector<int> make_small_vec(const int* p, size_t n) {
  if (n == 0 || !p) {
    return std::vector<int>{};
  }
  if (n <= 8) {
    std::array<int, 8> buf{};
    std::memcpy(buf.data(), p, n * sizeof(int));
    return std::vector<int>(buf.data(), buf.data() + n);
  }
  return std::vector<int>(p, p + n);
}

/// Build an mx::Strides (SmallVector<int64_t>) from a raw int64 pointer +
/// count.  Mirrors make_shape's null guard: constructing a SmallVector from
/// a null iterator is undefined behaviour, so we return an empty strides
/// vector instead, letting the downstream MLX API surface a caught exception.
inline mx::Strides make_strides(const int64_t* p, size_t n) {
  if (n == 0 || !p) {
    return mx::Strides{};
  }
  return mx::Strides(p, p + n);
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
/* Sentinel for size_t-returning accessors (ndim, nbytes, vector size). */
#define AX_CATCH_SIZE_MAX \
  catch (const std::exception&) { ax_set_current_error(); return SIZE_MAX; } \
  catch (...) { ax_set_current_error(); return SIZE_MAX; }

#endif /* AX_SHIM_INTERNAL_H */
