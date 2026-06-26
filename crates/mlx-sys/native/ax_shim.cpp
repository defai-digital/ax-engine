/* ax_shim.cpp — C ABI shim calling mlx::core directly (no mlx-c middleman).
 * Process-global error handler replaces mlx-c's global handler.
 */
#include "ax_shim.h"
#include "ax_shim_internal.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <variant>
#include <vector>

#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"
#include "mlx/compile.h"

/* ================================================================
 * Process-global error handler (replaces mlx-c's global handler)
 * ================================================================ */
static thread_local std::string ax_tls_error;
static std::mutex ax_error_handler_mutex;
static mlx_error_handler_func ax_error_handler = nullptr;
static std::shared_ptr<void> ax_error_handler_data = nullptr;

static void ax_set_error(const char* msg) {
  ax_tls_error = msg ? msg : "";
  mlx_error_handler_func handler = nullptr;
  std::shared_ptr<void> data = nullptr;
  {
    std::lock_guard<std::mutex> lock(ax_error_handler_mutex);
    handler = ax_error_handler;
    data = ax_error_handler_data;
  }
  if (handler) handler(ax_tls_error.c_str(), data.get());
}

void ax_set_current_error() {
  // Use std::current_exception() to safely check whether an exception is
  // active before rethrowing.  A bare `throw;` outside a catch block calls
  // std::terminate — this guard turns a future programming error into a
  // recoverable error message instead of a hard crash.
  auto ex = std::current_exception();
  if (!ex) { ax_set_error("unknown error (no active exception)"); return; }
  try { std::rethrow_exception(ex); }
  catch (const std::exception& e) { ax_set_error(e.what()); }
  catch (...) { ax_set_error("unknown MLX exception"); }
}

extern "C" void mlx_set_error_handler(
    mlx_error_handler_func handler, void* data, void (*dtor)(void*)) {
  std::lock_guard<std::mutex> lock(ax_error_handler_mutex);
  ax_error_handler = handler;
  if (dtor) {
    ax_error_handler_data = std::shared_ptr<void>(data, dtor);
  } else {
    ax_error_handler_data = nullptr;
  }
}

/* ================================================================
 * Helper inline functions
 * ================================================================ */
namespace {

inline mx::Dtype to_dtype(mlx_dtype d) {
  static mx::Dtype m[] = {
    mx::bool_, mx::uint8, mx::uint16, mx::uint32, mx::uint64,
    mx::int8, mx::int16, mx::int32, mx::int64,
    mx::float16, mx::float32, mx::float64, mx::bfloat16, mx::complex64};
  int idx = (int)d;
  if (idx < 0 || idx >= (int)(sizeof(m) / sizeof(m[0]))) {
    throw std::runtime_error("invalid mlx_dtype value");
  }
  return m[idx];
}

inline mx::array make_array(const void* data, mx::Shape shape, mx::Dtype dtype) {
  switch (dtype.val()) {
    case mx::Dtype::Val::bool_:     return mx::array(static_cast<const bool*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::uint8:     return mx::array(static_cast<const uint8_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::uint16:    return mx::array(static_cast<const uint16_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::uint32:    return mx::array(static_cast<const uint32_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::uint64:    return mx::array(static_cast<const uint64_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::int8:      return mx::array(static_cast<const int8_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::int16:     return mx::array(static_cast<const int16_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::int32:     return mx::array(static_cast<const int32_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::int64:     return mx::array(static_cast<const int64_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::float16:   return mx::array(static_cast<const mx::float16_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::float32:   return mx::array(static_cast<const float*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::float64:   return mx::array(static_cast<const double*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::bfloat16:  return mx::array(static_cast<const mx::bfloat16_t*>(data), std::move(shape), dtype);
    case mx::Dtype::Val::complex64: return mx::array(static_cast<const mx::complex64_t*>(data), std::move(shape), dtype);
    default: throw std::runtime_error("unknown dtype");
  }
}

using device_info_t = std::unordered_map<std::string, std::variant<std::string, size_t>>;
inline device_info_t& diref(mlx_device_info d) {
  if (!d.ctx) throw std::runtime_error("expected a non-empty mlx_device_info");
  return *static_cast<device_info_t*>(d.ctx);
}

inline std::optional<int> opt_int(mlx_optional_int o) {
  return o.has_value ? std::make_optional(o.value) : std::nullopt;
}
inline std::optional<mx::Dtype> opt_dtype(mlx_optional_dtype o) {
  return o.has_value ? std::make_optional(to_dtype(o.value)) : std::nullopt;
}

using closure_fn = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;
inline closure_fn& cref(mlx_closure c) {
  if (!c.ctx) throw std::runtime_error("expected a non-empty mlx_closure");
  return *static_cast<closure_fn*>(c.ctx);
}

struct metal_cfg_cpp {
  std::vector<mx::Shape> out_shapes;
  std::vector<mx::Dtype> out_dtypes;
  std::tuple<int,int,int> grid;
  std::tuple<int,int,int> thread_group;
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> template_args;
  std::optional<float> init_value;
  bool verbose = false;
};
inline metal_cfg_cpp& mcfgref(mlx_fast_metal_kernel_config c) {
  if (!c.ctx) throw std::runtime_error("expected a non-empty mlx_fast_metal_kernel_config");
  return *static_cast<metal_cfg_cpp*>(c.ctx);
}
struct metal_kern_cpp { mx::fast::CustomKernelFunction fn; metal_kern_cpp(mx::fast::CustomKernelFunction f) : fn(f) {} };
inline metal_kern_cpp& mkref(mlx_fast_metal_kernel k) {
  if (!k.ctx) throw std::runtime_error("expected a non-empty mlx_fast_metal_kernel");
  return *static_cast<metal_kern_cpp*>(k.ctx);
}

} // anonymous namespace

/* Convenience macros for mechanical op wrappers.
 * AX_TRY / AX_CATCH / AX_CATCH_NULL are defined in ax_shim_internal.h. */
#define UNARY(name, cpp) \
  extern "C" int name(mlx_array* r, const mlx_array a, const mlx_stream s) { \
    AX_TRY { aset(r, mx::cpp(aref(a), sd(s))); return 0; } AX_CATCH }
#define BINARY(name, cpp) \
  extern "C" int name(mlx_array* r, const mlx_array a, const mlx_array b, const mlx_stream s) { \
    AX_TRY { aset(r, mx::cpp(aref(a), aref(b), sd(s))); return 0; } AX_CATCH }

/* ================================================================
 * Array lifecycle
 * ================================================================ */
extern "C" mlx_array mlx_array_new(void) { return mlx_array{nullptr}; }
extern "C" int mlx_array_free(mlx_array a) { if (a.ctx) delete static_cast<mx::array*>(a.ctx); return 0; }

extern "C" mlx_array mlx_array_new_data(
    const void* data, const int* shape, int dim, mlx_dtype dtype) {
  AX_TRY { return mlx_array{new mx::array(make_array(data, make_shape(shape, dim), to_dtype(dtype)))};
  } AX_CATCH_NULL
}

extern "C" mlx_array mlx_array_new_data_managed_payload(
    void* data, const int* shape, int dim, mlx_dtype dtype,
    void* payload, void (*dtor)(void*)) {
  AX_TRY {
    auto cpp_dtype = to_dtype(dtype);
    auto cpp_shape = mx::Shape(shape, shape + dim);
    // The deleter receives the caller's lifetime `payload` handle, NOT the
    // `data` pointer. This is intentional: `data` may be an interior pointer
    // into `payload`'s allocation (e.g. an mmap offset into an Arc<Mmap>).
    // The shared_ptr's void* argument is unused; the captured payload is the
    // resource the caller asked us to release.
    auto cpp_deleter = [dtor, payload](void*) { dtor(payload); };
    return mlx_array{new mx::array(data, std::move(cpp_shape), cpp_dtype, cpp_deleter)};
  } AX_CATCH_NULL
}

extern "C" int mlx_array_set(mlx_array* dst, const mlx_array src) {
  AX_TRY { aset(dst, aref(src)); return 0; } AX_CATCH
}

extern "C" size_t mlx_array_ndim(const mlx_array a) { return aref(a).ndim(); }
extern "C" const int* mlx_array_shape(const mlx_array a) { return aref(a).shape().data(); }
extern "C" mlx_dtype mlx_array_dtype(const mlx_array a) {
  static mlx_dtype m[] = { MLX_BOOL, MLX_UINT8, MLX_UINT16, MLX_UINT32, MLX_UINT64,
    MLX_INT8, MLX_INT16, MLX_INT32, MLX_INT64, MLX_FLOAT16, MLX_FLOAT32,
    MLX_FLOAT64, MLX_BFLOAT16, MLX_COMPLEX64 };
  return m[(int)aref(a).dtype().val()];
}
extern "C" size_t mlx_array_nbytes(const mlx_array a) { return aref(a).nbytes(); }
extern "C" const float* mlx_array_data_float32(const mlx_array a) { return aref(a).data<float>(); }
extern "C" const uint8_t* mlx_array_data_uint8(const mlx_array a) { return aref(a).data<uint8_t>(); }
extern "C" const uint32_t* mlx_array_data_uint32(const mlx_array a) { return aref(a).data<uint32_t>(); }

/* ================================================================
 * Stream
 * ================================================================ */
extern "C" mlx_stream mlx_stream_new_device(mlx_device dev) {
  AX_TRY { return mlx_stream{new mx::Stream(mx::new_stream(dref(dev)))}; } AX_CATCH_NULL
}
extern "C" int mlx_stream_free(mlx_stream s) { if (s.ctx) delete static_cast<mx::Stream*>(s.ctx); return 0; }
extern "C" bool mlx_stream_equal(mlx_stream a, mlx_stream b) {
  if (!a.ctx && !b.ctx) return true;
  if (!a.ctx || !b.ctx) return false;
  return sref(a) == sref(b);
}
extern "C" int mlx_set_default_stream(mlx_stream s) {
  AX_TRY { mx::set_default_stream(s.ctx ? sref(s) : mx::default_stream(mx::default_device())); return 0; } AX_CATCH
}
extern "C" mlx_stream mlx_default_cpu_stream_new(void) {
  AX_TRY { return mlx_stream{new mx::Stream(mx::default_stream(mx::Device::cpu))}; } AX_CATCH_NULL
}
extern "C" mlx_stream mlx_default_gpu_stream_new(void) {
  AX_TRY { return mlx_stream{new mx::Stream(mx::default_stream(mx::Device::gpu))}; } AX_CATCH_NULL
}

/* ================================================================
 * Device
 * ================================================================ */
extern "C" mlx_device mlx_device_new_type(mlx_device_type t, int idx) {
  AX_TRY {
    auto dt = (t == MLX_CPU) ? mx::Device::DeviceType::cpu : mx::Device::DeviceType::gpu;
    return mlx_device{new mx::Device(dt, idx)};
  } AX_CATCH_NULL
}
extern "C" int mlx_device_free(mlx_device d) { if (d.ctx) delete static_cast<mx::Device*>(d.ctx); return 0; }

extern "C" mlx_device_info mlx_device_info_new(void) { return mlx_device_info{nullptr}; }
extern "C" int mlx_device_info_get(mlx_device_info* info, mlx_device dev) {
  AX_TRY {
    auto& d = dref(dev);
    const auto& props = mx::device_info(d);
    device_info_t m;
    for (auto& [k,v] : props) {
      std::visit([&](const auto& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, std::string>)
          m[k] = val;
        else if constexpr (std::is_same_v<T, size_t>)
          m[k] = val;
        // Ignore unknown variant alternatives silently — future MLX
        // versions may add bool/int alternatives that device_info_get_size
        // does not need.
      }, v);
    }
    if (info->ctx) *static_cast<device_info_t*>(info->ctx) = std::move(m);
    else info->ctx = new device_info_t(std::move(m));
    return 0;
  } AX_CATCH
}
extern "C" int mlx_device_info_free(mlx_device_info i) { if (i.ctx) delete static_cast<device_info_t*>(i.ctx); return 0; }
extern "C" int mlx_device_info_get_size(size_t* val, mlx_device_info info, const char* key) {
  AX_TRY {
    auto& m = diref(info);
    auto it = m.find(key);
    if (it == m.end()) return 2;
    if (auto* p = std::get_if<size_t>(&it->second)) { *val = *p; return 0; }
    return 2;
  } AX_CATCH
}

/* ================================================================
 * Ops — shape / transform
 * ================================================================ */
extern "C" int mlx_astype(mlx_array* r, const mlx_array a, mlx_dtype dt, const mlx_stream s) {
  AX_TRY { aset(r, mx::astype(aref(a), to_dtype(dt), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_view(mlx_array* r, const mlx_array a, mlx_dtype dt, const mlx_stream s) {
  AX_TRY { aset(r, mx::view(aref(a), to_dtype(dt), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_reshape(mlx_array* r, const mlx_array a, const int* sh, size_t n, const mlx_stream s) {
  AX_TRY { aset(r, mx::reshape(aref(a), make_shape(sh, n), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_transpose_axes(mlx_array* r, const mlx_array a, const int* ax, size_t n, const mlx_stream s) {
  AX_TRY { aset(r, mx::transpose(aref(a), std::vector<int>(ax, ax+n), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_expand_dims(mlx_array* r, const mlx_array a, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::expand_dims(aref(a), ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_expand_dims_axes(mlx_array* r, const mlx_array a, const int* ax, size_t n, const mlx_stream s) {
  AX_TRY { aset(r, mx::expand_dims(aref(a), std::vector<int>(ax, ax+n), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_broadcast_to(mlx_array* r, const mlx_array a, const int* sh, size_t n, const mlx_stream s) {
  AX_TRY { aset(r, mx::broadcast_to(aref(a), make_shape(sh, n), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_flatten(mlx_array* r, const mlx_array a, int sa, int ea, const mlx_stream s) {
  AX_TRY { aset(r, mx::flatten(aref(a), sa, ea, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_unflatten(mlx_array* r, const mlx_array a, int ax, const int* sh, size_t n, const mlx_stream s) {
  AX_TRY { aset(r, mx::unflatten(aref(a), ax, make_shape(sh, n), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_as_strided(mlx_array* r, const mlx_array a, const int* sh, size_t sn,
    const int64_t* st, size_t stn, size_t off, const mlx_stream s) {
  AX_TRY { aset(r, mx::as_strided(aref(a), make_shape(sh, sn), mx::Strides(st, st+stn), off, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_concatenate_axis(mlx_array* r, const mlx_vector_array arrs, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::concatenate(varef(arrs), ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_split(mlx_vector_array* r, const mlx_array a, int n, int ax, const mlx_stream s) {
  AX_TRY { vaset(r, mx::split(aref(a), n, ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_stack_axis(mlx_array* r, const mlx_vector_array arrs, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::stack(varef(arrs), ax, sd(s))); return 0; } AX_CATCH }
/* NOTE: slice/slice_update strides use Shape (int), not Strides (int64_t) */
extern "C" int mlx_slice(mlx_array* r, const mlx_array a, const int* st, size_t sn,
    const int* sp, size_t spn, const int* str, size_t strn, const mlx_stream s) {
  AX_TRY { aset(r, mx::slice(aref(a), make_shape(st,sn), make_shape(sp,spn), make_shape(str,strn), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_slice_update(mlx_array* r, const mlx_array src, const mlx_array upd,
    const int* st, size_t sn, const int* sp, size_t spn, const int* str, size_t strn, const mlx_stream s) {
  AX_TRY { aset(r, mx::slice_update(aref(src), aref(upd), make_shape(st,sn), make_shape(sp,spn), make_shape(str,strn), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_repeat(mlx_array* r, const mlx_array a, int rep, const mlx_stream s) {
  AX_TRY { aset(r, mx::repeat(aref(a), rep, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_repeat_axis(mlx_array* r, const mlx_array a, int rep, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::repeat(aref(a), rep, ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_pad(mlx_array* r, const mlx_array a, const int* axes, size_t an,
    const int* lo, size_t lon, const int* hi, size_t hin,
    const mlx_array pv, const char* mode, const mlx_stream s) {
  AX_TRY {
    auto m = safe_str(mode);
    aset(r, mx::pad(aref(a), std::vector<int>(axes,axes+an),
      make_shape(lo,lon), make_shape(hi,hin),
      aref(pv), m, sd(s)));
    return 0;
  } AX_CATCH
}

/* Unary ops */
UNARY(mlx_exp, exp) UNARY(mlx_log, log) UNARY(mlx_log1p, log1p)
UNARY(mlx_tanh, tanh) UNARY(mlx_sigmoid, sigmoid) UNARY(mlx_erf, erf)
UNARY(mlx_negative, negative) UNARY(mlx_floor, floor)
UNARY(mlx_cos, cos) UNARY(mlx_sin, sin)
UNARY(mlx_stop_gradient, stop_gradient)

extern "C" int mlx_contiguous(mlx_array* r, const mlx_array a, bool col, const mlx_stream s) {
  AX_TRY { aset(r, mx::contiguous(aref(a), col, sd(s))); return 0; } AX_CATCH }

/* Binary ops */
BINARY(mlx_add, add) BINARY(mlx_subtract, subtract) BINARY(mlx_multiply, multiply)
BINARY(mlx_divide, divide) BINARY(mlx_power, power)
BINARY(mlx_maximum, maximum) BINARY(mlx_minimum, minimum)
BINARY(mlx_equal, equal) BINARY(mlx_not_equal, not_equal)
BINARY(mlx_greater_equal, greater_equal) BINARY(mlx_less, less) BINARY(mlx_less_equal, less_equal)
/* where is ternary: condition, x, y */
extern "C" int mlx_where(mlx_array* r, const mlx_array cond, const mlx_array x, const mlx_array y, const mlx_stream s) {
  AX_TRY { aset(r, mx::where(aref(cond), aref(x), aref(y), sd(s))); return 0; } AX_CATCH }
BINARY(mlx_outer, outer) BINARY(mlx_logical_and, logical_and)

/* Reduction / indexing */
extern "C" int mlx_sum_axis(mlx_array* r, const mlx_array a, int ax, bool kd, const mlx_stream s) {
  AX_TRY { aset(r, mx::sum(aref(a), ax, kd, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_argmax_axis(mlx_array* r, const mlx_array a, int ax, bool kd, const mlx_stream s) {
  AX_TRY { aset(r, mx::argmax(aref(a), ax, kd, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_argpartition_axis(mlx_array* r, const mlx_array a, int kth, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::argpartition(aref(a), kth, ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_argsort_axis(mlx_array* r, const mlx_array a, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::argsort(aref(a), ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_topk(mlx_array* r, const mlx_array a, int k, const mlx_stream s) {
  AX_TRY { aset(r, mx::topk(aref(a), k, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_topk_axis(mlx_array* r, const mlx_array a, int k, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::topk(aref(a), k, ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_take_axis(mlx_array* r, const mlx_array a, const mlx_array idx, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::take(aref(a), aref(idx), ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_take_along_axis(mlx_array* r, const mlx_array a, const mlx_array idx, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::take_along_axis(aref(a), aref(idx), ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_put_along_axis(mlx_array* r, const mlx_array a, const mlx_array idx, const mlx_array v, int ax, const mlx_stream s) {
  AX_TRY { aset(r, mx::put_along_axis(aref(a), aref(idx), aref(v), ax, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_cumsum(mlx_array* r, const mlx_array a, int ax, bool rev, bool inc, const mlx_stream s) {
  AX_TRY { aset(r, mx::cumsum(aref(a), ax, rev, inc, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_softmax_axis(mlx_array* r, const mlx_array a, int ax, bool precise, const mlx_stream s) {
  AX_TRY { aset(r, mx::softmax(aref(a), ax, precise, sd(s))); return 0; } AX_CATCH }

/* Creation */
extern "C" int mlx_zeros(mlx_array* r, const int* sh, size_t n, mlx_dtype dt, const mlx_stream s) {
  AX_TRY { aset(r, mx::zeros(make_shape(sh, n), to_dtype(dt), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_arange(mlx_array* r, double start, double stop, double step, mlx_dtype dt, const mlx_stream s) {
  AX_TRY { aset(r, mx::arange(start, stop, step, to_dtype(dt), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_random_categorical(mlx_array* r, const mlx_array logits, int ax, const mlx_array key, const mlx_stream s) {
  AX_TRY { aset(r, mx::random::categorical(aref(logits), ax, opt_arr(key), sd(s))); return 0; } AX_CATCH }

/* Linear algebra */
extern "C" int mlx_matmul(mlx_array* r, const mlx_array a, const mlx_array b, const mlx_stream s) {
  AX_TRY { aset(r, mx::matmul(aref(a), aref(b), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_conv1d(mlx_array* r, const mlx_array inp, const mlx_array w, int st, int pad, int dil, int grp, const mlx_stream s) {
  AX_TRY { aset(r, mx::conv1d(aref(inp), aref(w), st, pad, dil, grp, sd(s))); return 0; } AX_CATCH }

/* Clip */
extern "C" int mlx_clip(mlx_array* r, const mlx_array a, const mlx_array mn, const mlx_array mx_, const mlx_stream s) {
  AX_TRY { aset(r, mx::clip(aref(a), opt_arr(mn), opt_arr(mx_), sd(s))); return 0; } AX_CATCH }

/* ================================================================
 * Quantized ops
 * ================================================================ */
extern "C" int mlx_quantize(mlx_vector_array* r, const mlx_array w, mlx_optional_int gs, mlx_optional_int bits,
    const char* mode, const mlx_array gscale, const mlx_stream s) {
  AX_TRY { vaset(r, mx::quantize(aref(w), opt_int(gs), opt_int(bits), safe_str(mode), opt_arr(gscale), sd(s))); return 0; } AX_CATCH }

extern "C" int mlx_dequantize(mlx_array* r, const mlx_array w, const mlx_array scales, const mlx_array biases,
    mlx_optional_int gs, mlx_optional_int bits, const char* mode, const mlx_array gscale, mlx_optional_dtype dt, const mlx_stream s) {
  AX_TRY { aset(r, mx::dequantize(aref(w), aref(scales), opt_arr(biases), opt_int(gs), opt_int(bits), safe_str(mode), opt_arr(gscale), opt_dtype(dt), sd(s))); return 0; } AX_CATCH }

extern "C" int mlx_quantized_matmul(mlx_array* r, const mlx_array x, const mlx_array w, const mlx_array scales,
    const mlx_array biases, bool tr, mlx_optional_int gs, mlx_optional_int bits, const char* mode, const mlx_stream s) {
  AX_TRY { aset(r, mx::quantized_matmul(aref(x), aref(w), aref(scales), opt_arr(biases), tr, opt_int(gs), opt_int(bits), safe_str(mode), sd(s))); return 0; } AX_CATCH }

extern "C" int mlx_gather_mm(mlx_array* r, const mlx_array a, const mlx_array b,
    const mlx_array li, const mlx_array ri, bool sorted, const mlx_stream s) {
  AX_TRY { aset(r, mx::gather_mm(aref(a), aref(b), opt_arr(li), opt_arr(ri), sorted, sd(s))); return 0; } AX_CATCH }

extern "C" int mlx_gather_qmm(mlx_array* r, const mlx_array x, const mlx_array w, const mlx_array scales,
    const mlx_array biases, const mlx_array li, const mlx_array ri, bool tr,
    mlx_optional_int gs, mlx_optional_int bits, const char* mode, bool sorted, const mlx_stream s) {
  AX_TRY { aset(r, mx::gather_qmm(aref(x), aref(w), aref(scales), opt_arr(biases), opt_arr(li), opt_arr(ri), tr, opt_int(gs), opt_int(bits), safe_str(mode), sorted, sd(s))); return 0; } AX_CATCH }

extern "C" int mlx_to_fp8(mlx_array* r, const mlx_array x, const mlx_stream s) {
  AX_TRY { aset(r, mx::to_fp8(aref(x), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_from_fp8(mlx_array* r, const mlx_array x, mlx_dtype dt, const mlx_stream s) {
  AX_TRY { aset(r, mx::from_fp8(aref(x), to_dtype(dt), sd(s))); return 0; } AX_CATCH }

/* ================================================================
 * Fast kernels
 * ================================================================ */
extern "C" int mlx_fast_rms_norm(mlx_array* r, const mlx_array x, const mlx_array w, float eps, const mlx_stream s) {
  AX_TRY { aset(r, mx::fast::rms_norm(aref(x), opt_arr(w), eps, sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_fast_rope(mlx_array* r, const mlx_array x, int dims, bool trad,
    mlx_optional_float base, float scale, int offset, const mlx_array freqs, const mlx_stream s) {
  AX_TRY {
    auto b = base.has_value ? std::make_optional<float>(base.value) : std::nullopt;
    aset(r, mx::fast::rope(aref(x), dims, trad, b, scale, offset, opt_arr(freqs), sd(s)));
    return 0;
  } AX_CATCH
}
extern "C" int mlx_fast_scaled_dot_product_attention(mlx_array* r,
    const mlx_array q, const mlx_array k, const mlx_array v,
    float scale, const char* mask_mode, const mlx_array mask_arr, const mlx_array sinks, const mlx_stream s) {
  AX_TRY { aset(r, mx::fast::scaled_dot_product_attention(aref(q), aref(k), aref(v), scale, safe_str(mask_mode), opt_arr(mask_arr), opt_arr(sinks), sd(s))); return 0; } AX_CATCH }
extern "C" int mlx_fast_layer_norm(mlx_array* r, const mlx_array x, const mlx_array w, const mlx_array b, float eps, const mlx_stream s) {
  AX_TRY { aset(r, mx::fast::layer_norm(aref(x), opt_arr(w), opt_arr(b), eps, sd(s))); return 0; } AX_CATCH }

/* ================================================================
 * Metal kernel dispatch
 * ================================================================ */
extern "C" mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void) {
  AX_TRY { return mlx_fast_metal_kernel_config{new metal_cfg_cpp()}; } AX_CATCH_NULL
}
extern "C" void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config c) {
  if (c.ctx) delete static_cast<metal_cfg_cpp*>(c.ctx);
}
extern "C" int mlx_fast_metal_kernel_config_add_output_arg(mlx_fast_metal_kernel_config c, const int* sh, size_t n, mlx_dtype dt) {
  AX_TRY { auto& cfg = mcfgref(c); cfg.out_shapes.push_back(mx::Shape(sh,sh+n)); cfg.out_dtypes.push_back(to_dtype(dt)); return 0; } AX_CATCH }
extern "C" int mlx_fast_metal_kernel_config_set_grid(mlx_fast_metal_kernel_config c, int g1, int g2, int g3) {
  AX_TRY { mcfgref(c).grid = {g1,g2,g3}; return 0; } AX_CATCH }
extern "C" int mlx_fast_metal_kernel_config_set_thread_group(mlx_fast_metal_kernel_config c, int t1, int t2, int t3) {
  AX_TRY { mcfgref(c).thread_group = {t1,t2,t3}; return 0; } AX_CATCH }
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_dtype(mlx_fast_metal_kernel_config c, const char* name, mlx_dtype dt) {
  AX_TRY { mcfgref(c).template_args.push_back({safe_str(name), to_dtype(dt)}); return 0; } AX_CATCH }
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_int(mlx_fast_metal_kernel_config c, const char* name, int v) {
  AX_TRY { mcfgref(c).template_args.push_back({safe_str(name), v}); return 0; } AX_CATCH }
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_bool(mlx_fast_metal_kernel_config c, const char* name, bool v) {
  AX_TRY { mcfgref(c).template_args.push_back({safe_str(name), v}); return 0; } AX_CATCH }

extern "C" mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name, const mlx_vector_string inp, const mlx_vector_string out,
    const char* source, const char* header, bool erc, bool atomic) {
  AX_TRY {
    return mlx_fast_metal_kernel{new metal_kern_cpp(
      mx::fast::metal_kernel(name, vsref(inp), vsref(out), source, header, erc, atomic))};
  } AX_CATCH_NULL
}
extern "C" void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel k) {
  if (k.ctx) delete static_cast<metal_kern_cpp*>(k.ctx);
}
extern "C" int mlx_fast_metal_kernel_apply(mlx_vector_array* out, mlx_fast_metal_kernel k,
    const mlx_vector_array inp, const mlx_fast_metal_kernel_config cfg, const mlx_stream s) {
  AX_TRY {
    auto& c = mcfgref(cfg);
    vaset(out, mkref(k).fn(
      varef(inp), c.out_shapes, c.out_dtypes, c.grid, c.thread_group,
      c.template_args, c.init_value, c.verbose, sd(s)));
    return 0;
  } AX_CATCH
}

/* ================================================================
 * I/O + containers
 * ================================================================ */
extern "C" int mlx_load_safetensors(mlx_map_string_to_array* r0, mlx_map_string_to_string* r1,
    const char* file, const mlx_stream s) {
  AX_TRY {
    auto result = mx::load_safetensors(safe_str(file), sd(s));
    auto& tensors = result.first;
    auto& metadata = result.second;
    if (r0->ctx) *static_cast<std::unordered_map<std::string, mx::array>*>(r0->ctx) = std::move(tensors);
    else r0->ctx = new std::unordered_map<std::string, mx::array>(std::move(tensors));
    if (r1->ctx) *static_cast<std::unordered_map<std::string, std::string>*>(r1->ctx) = std::move(metadata);
    else r1->ctx = new std::unordered_map<std::string, std::string>(std::move(metadata));
    return 0;
  } AX_CATCH
}

using str_arr_map = std::unordered_map<std::string, mx::array>;
using str_str_map = std::unordered_map<std::string, std::string>;

extern "C" mlx_map_string_to_array mlx_map_string_to_array_new(void) { return {new str_arr_map()}; }
extern "C" int mlx_map_string_to_array_free(mlx_map_string_to_array m) { if (m.ctx) delete static_cast<str_arr_map*>(m.ctx); return 0; }

extern "C" mlx_map_string_to_array_iterator mlx_map_string_to_array_iterator_new(mlx_map_string_to_array m) {
  auto* it = new str_arr_map::iterator(static_cast<str_arr_map*>(m.ctx)->begin());
  return mlx_map_string_to_array_iterator{it, m.ctx};
}
extern "C" int mlx_map_string_to_array_iterator_free(mlx_map_string_to_array_iterator it) {
  if (it.ctx) delete static_cast<str_arr_map::iterator*>(it.ctx); return 0;
}
/* Returns 0 when an entry is written, 1 at end-of-iteration, and 2 when an
 * MLX exception is caught while materializing the value. Distinguishing the
 * error case (2) from normal termination (1) lets the Rust caller surface a
 * genuine load failure instead of silently treating it as a short map. The
 * AX_TRY/CATCH also prevents a throwing `aset` from unwinding across the C ABI. */
extern "C" int mlx_map_string_to_array_iterator_next(
    const char** key, mlx_array* val, mlx_map_string_to_array_iterator it) {
  // Captured before the risky materialization so the catch can name the tensor
  // that failed to load (the most useful diagnostic for a blocked model).
  const char* current_key = nullptr;
  AX_TRY {
    auto& mit = *static_cast<str_arr_map::iterator*>(it.ctx);
    auto& m = *static_cast<str_arr_map*>(it.map_ctx);
    if (mit == m.end()) return 1;
    current_key = mit->first.c_str();
    *key = current_key;
    aset(val, mit->second);
    ++mit;
    return 0;
  } catch (const std::exception& e) {
    if (current_key) ax_set_error((std::string("tensor '") + current_key + "': " + e.what()).c_str());
    else ax_set_error(e.what());
    return 2;
  } catch (...) {
    if (current_key) ax_set_error((std::string("tensor '") + current_key + "' raised an unknown MLX exception").c_str());
    else ax_set_error("unknown MLX exception");
    return 2;
  }
}

extern "C" mlx_map_string_to_string mlx_map_string_to_string_new(void) { return {new str_str_map()}; }
extern "C" int mlx_map_string_to_string_free(mlx_map_string_to_string m) { if (m.ctx) delete static_cast<str_str_map*>(m.ctx); return 0; }

extern "C" mlx_vector_array mlx_vector_array_new(void) { return {new std::vector<mx::array>()}; }
extern "C" int mlx_vector_array_free(mlx_vector_array v) { if (v.ctx) delete static_cast<std::vector<mx::array>*>(v.ctx); return 0; }
extern "C" int mlx_vector_array_append_value(mlx_vector_array v, const mlx_array a) {
  AX_TRY { varef(v).push_back(aref(a)); return 0; } AX_CATCH }
extern "C" size_t mlx_vector_array_size(mlx_vector_array v) { return varef(v).size(); }
extern "C" int mlx_vector_array_get(mlx_array* r, const mlx_vector_array v, size_t idx) {
  AX_TRY { aset(r, varef(v).at(idx)); return 0; } AX_CATCH }

extern "C" mlx_vector_string mlx_vector_string_new(void) { return {new std::vector<std::string>()}; }
extern "C" int mlx_vector_string_free(mlx_vector_string v) { if (v.ctx) delete static_cast<std::vector<std::string>*>(v.ctx); return 0; }
extern "C" int mlx_vector_string_append_value(mlx_vector_string v, const char* val) {
  AX_TRY { vsref(v).push_back(safe_str(val)); return 0; } AX_CATCH }

/* ================================================================
 * Closure + compile
 * ================================================================ */
extern "C" mlx_closure mlx_closure_new_func_payload(
    int (*fun)(mlx_vector_array*, const mlx_vector_array, void*),
    void* payload, void (*dtor)(void*)) {
  AX_TRY {
    auto payload_holder = dtor
      ? std::shared_ptr<void>(payload, dtor)
      : std::shared_ptr<void>(payload, [](void*) {});
    auto fn = [fun, payload_holder](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
      // Ownership contract with the Rust trampoline (closure_trampoline in
      // mlx-sys/src/closure.rs): it *borrows* the input vector (from_borrowed,
      // never frees it) and *overwrites* `out.ctx` with a freshly heap-allocated
      // output vector that it does not own either (into_raw). So this side owns
      // both allocations and must free them on every exit path. Note the real
      // outputs live at `out.ctx` after the call, not in any vector we passed in
      // — earlier versions read/moved the wrong (empty) vector and/or leaked the
      // output vector, pinning every output mx::array handle in unified memory.
      auto in_vec = std::make_unique<std::vector<mx::array>>(inputs);
      mlx_vector_array in{in_vec.get()};
      mlx_vector_array out{nullptr};
      int rc = fun(&out, in, payload_holder.get());
      // Adopt the callback's output vector so it is freed even on the error
      // paths below.
      std::unique_ptr<std::vector<mx::array>> out_vec(
          static_cast<std::vector<mx::array>*>(out.ctx));
      if (rc != 0) throw std::runtime_error("closure callback failed");
      if (!out_vec) throw std::runtime_error("closure callback produced no output vector");
      return std::move(*out_vec);
    };
    return mlx_closure{new closure_fn(std::move(fn))};
  } AX_CATCH_NULL
}

extern "C" int mlx_closure_apply(mlx_vector_array* r, mlx_closure cls, const mlx_vector_array input) {
  AX_TRY { vaset(r, cref(cls)(varef(input))); return 0; } AX_CATCH }
extern "C" int mlx_closure_free(mlx_closure c) { if (c.ctx) delete static_cast<closure_fn*>(c.ctx); return 0; }

extern "C" int mlx_compile(mlx_closure* r, const mlx_closure fun, bool shapeless) {
  AX_TRY {
    auto compiled = mx::compile(cref(fun), shapeless);
    if (r->ctx) *static_cast<closure_fn*>(r->ctx) = std::move(compiled);
    else r->ctx = new closure_fn(std::move(compiled));
    return 0;
  } AX_CATCH
}

/* ================================================================
 * Memory management
 * ================================================================ */
extern "C" int mlx_clear_cache(void) {
  AX_TRY { mx::clear_cache(); return 0; } AX_CATCH }
extern "C" int mlx_set_wired_limit(size_t* r, size_t limit) {
  AX_TRY { *r = mx::set_wired_limit(limit); return 0; } AX_CATCH }
extern "C" int mlx_get_active_memory(size_t* r) {
  AX_TRY { *r = mx::get_active_memory(); return 0; } AX_CATCH }
extern "C" int mlx_set_memory_limit(size_t* r, size_t limit) {
  AX_TRY { *r = mx::set_memory_limit(limit); return 0; } AX_CATCH }
extern "C" int mlx_set_cache_limit(size_t* r, size_t limit) {
  AX_TRY { *r = mx::set_cache_limit(limit); return 0; } AX_CATCH }
extern "C" int mlx_get_peak_memory(size_t* r) {
  AX_TRY { *r = mx::get_peak_memory(); return 0; } AX_CATCH }
extern "C" int mlx_reset_peak_memory(void) {
  AX_TRY { mx::reset_peak_memory(); return 0; } AX_CATCH }
extern "C" int mlx_get_cache_memory(size_t* r) {
  AX_TRY { *r = mx::get_cache_memory(); return 0; } AX_CATCH }

/* ================================================================
 * Transforms
 * ================================================================ */
extern "C" int mlx_eval(const mlx_vector_array outputs) {
  AX_TRY { mx::eval(varef(outputs)); return 0; } AX_CATCH }
extern "C" int mlx_async_eval(const mlx_vector_array outputs) {
  AX_TRY { mx::async_eval(varef(outputs)); return 0; } AX_CATCH }
extern "C" int mlx_enable_compile(void) {
  AX_TRY { mx::enable_compile(); return 0; } AX_CATCH }
