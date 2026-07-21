/* ax_shim.h — drop-in C ABI replacement for mlx-c.
 *
 * Every type and function declared here matches the mlx-c public API exactly
 * so that existing bindgen output (ffi::mlx_*) requires zero changes.
 * The implementation in ax_shim.cpp calls mlx::core directly, bypassing the
 * mlx-c wrapper layer.
 */
#ifndef AX_SHIM_H
#define AX_SHIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Handle types — all are { void* ctx; } pointing at heap C++ objects
 * ================================================================ */

typedef struct mlx_array_ { void* ctx; } mlx_array;
typedef struct mlx_stream_ { void* ctx; } mlx_stream;
typedef struct mlx_vector_array_ { void* ctx; } mlx_vector_array;
typedef struct mlx_vector_string_ { void* ctx; } mlx_vector_string;
typedef struct mlx_device_ { void* ctx; } mlx_device;
typedef struct mlx_device_info_ { void* ctx; } mlx_device_info;
typedef struct mlx_closure_ { void* ctx; } mlx_closure;
typedef struct mlx_map_string_to_array_ { void* ctx; } mlx_map_string_to_array;
typedef struct mlx_map_string_to_string_ { void* ctx; } mlx_map_string_to_string;
typedef struct mlx_fast_metal_kernel_ { void* ctx; } mlx_fast_metal_kernel;
typedef struct mlx_fast_metal_kernel_config_ { void* ctx; } mlx_fast_metal_kernel_config;

typedef struct mlx_map_string_to_array_iterator_ {
  void* ctx;
  void* map_ctx;
} mlx_map_string_to_array_iterator;

/* ================================================================
 * Enums
 * ================================================================ */

typedef enum mlx_dtype_ {
  MLX_BOOL,
  MLX_UINT8,
  MLX_UINT16,
  MLX_UINT32,
  MLX_UINT64,
  MLX_INT8,
  MLX_INT16,
  MLX_INT32,
  MLX_INT64,
  MLX_FLOAT16,
  MLX_FLOAT32,
  MLX_FLOAT64,
  MLX_BFLOAT16,
  MLX_COMPLEX64,
} mlx_dtype;

typedef enum mlx_device_type_ { MLX_CPU, MLX_GPU } mlx_device_type;

typedef enum mlx_compile_mode_ {
  MLX_COMPILE_MODE_DISABLED,
  MLX_COMPILE_MODE_NO_SIMPLIFY,
  MLX_COMPILE_MODE_NO_FUSE,
  MLX_COMPILE_MODE_ENABLED
} mlx_compile_mode;

/* ================================================================
 * Optional structs
 * ================================================================ */

typedef struct mlx_optional_int_ {
  int value;
  bool has_value;
} mlx_optional_int;

typedef struct mlx_optional_float_ {
  float value;
  bool has_value;
} mlx_optional_float;

typedef struct mlx_optional_dtype_ {
  mlx_dtype value;
  bool has_value;
} mlx_optional_dtype;

/* ================================================================
 * Build sanity
 * ================================================================ */

/* Compare the MLX version the shim was compiled against (MLX_VERSION_* from
 * mlx/version.h) with the version reported by the loaded libmlx at runtime.
 * The shim binds to the mlx::core C++ ABI, so version skew — e.g. a Homebrew
 * `mlx` upgrade without a shim rebuild — can fail as silent memory
 * corruption. Returns 0 on match; on mismatch stores a diagnostic in the
 * error slot (and invokes the error handler) and returns 1. */
int ax_shim_check_mlx_version(void);

/* Return the version reported by the loaded libmlx. The pointer is owned by
 * MLX and remains valid for the process lifetime. */
const char* ax_shim_mlx_version(void);

/* ================================================================
 * Error handling
 * ================================================================ */

typedef void (*mlx_error_handler_func)(const char* msg, void* data);
void mlx_set_error_handler(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*));

/* ================================================================
 * Array lifecycle
 * ================================================================ */

mlx_array mlx_array_new(void);
int mlx_array_free(mlx_array arr);
mlx_array mlx_array_new_data(
    const void* data, const int* shape, int dim, mlx_dtype dtype);
mlx_array mlx_array_new_data_managed_payload(
    void* data, const int* shape, int dim, mlx_dtype dtype,
    void* payload, void (*dtor)(void*));
int mlx_array_set(mlx_array* arr, const mlx_array src);
size_t mlx_array_ndim(const mlx_array arr);
const int* mlx_array_shape(const mlx_array arr);
mlx_dtype mlx_array_dtype(const mlx_array arr);
size_t mlx_array_nbytes(const mlx_array arr);
/* 1 = evaluated (data readable), 0 = not yet evaluated, -1 = invalid handle
 * (error recorded). Data accessors below fail with a recorded error instead
 * of crashing when the array is unevaluated; this predicate lets callers
 * assert the precondition explicitly. */
int ax_shim_array_is_evaled(const mlx_array arr);
const float* mlx_array_data_float32(const mlx_array arr);
const uint8_t* mlx_array_data_uint8(const mlx_array arr);
const uint32_t* mlx_array_data_uint32(const mlx_array arr);

/* ================================================================
 * Stream
 * ================================================================ */

mlx_stream mlx_stream_new_device(mlx_device dev);
int mlx_stream_free(mlx_stream stream);
bool mlx_stream_equal(mlx_stream lhs, mlx_stream rhs);
int mlx_set_default_stream(mlx_stream stream);
mlx_stream mlx_default_cpu_stream_new(void);
mlx_stream mlx_default_gpu_stream_new(void);

/* ================================================================
 * Device
 * ================================================================ */

mlx_device mlx_device_new_type(mlx_device_type type, int index);
int mlx_device_free(mlx_device dev);
mlx_device_info mlx_device_info_new(void);
int mlx_device_info_get(mlx_device_info* info, mlx_device dev);
int mlx_device_info_free(mlx_device_info info);
int mlx_device_info_get_size(
    size_t* value, mlx_device_info info, const char* key);
int mlx_device_info_get_string(
    const char** value, mlx_device_info info, const char* key);

/* ================================================================
 * Ops — arithmetic, shape, math, reduction, creation
 * ================================================================ */

int mlx_astype(mlx_array* res, const mlx_array a, mlx_dtype dtype, const mlx_stream s);
int mlx_view(mlx_array* res, const mlx_array a, mlx_dtype dtype, const mlx_stream s);
int mlx_reshape(mlx_array* res, const mlx_array a, const int* shape, size_t shape_num, const mlx_stream s);
int mlx_transpose_axes(mlx_array* res, const mlx_array a, const int* axes, size_t axes_num, const mlx_stream s);
int mlx_expand_dims(mlx_array* res, const mlx_array a, int axis, const mlx_stream s);
int mlx_expand_dims_axes(mlx_array* res, const mlx_array a, const int* axes, size_t axes_num, const mlx_stream s);
int mlx_broadcast_to(mlx_array* res, const mlx_array a, const int* shape, size_t shape_num, const mlx_stream s);
int mlx_flatten(mlx_array* res, const mlx_array a, int start_axis, int end_axis, const mlx_stream s);
int mlx_unflatten(mlx_array* res, const mlx_array a, int axis, const int* shape, size_t shape_num, const mlx_stream s);
int mlx_as_strided(mlx_array* res, const mlx_array a, const int* shape, size_t shape_num, const int64_t* strides, size_t strides_num, size_t offset, const mlx_stream s);
int mlx_concatenate_axis(mlx_array* res, const mlx_vector_array arrays, int axis, const mlx_stream s);
int mlx_split(mlx_vector_array* res, const mlx_array a, int num_splits, int axis, const mlx_stream s);
int mlx_stack_axis(mlx_array* res, const mlx_vector_array arrays, int axis, const mlx_stream s);
int mlx_slice(mlx_array* res, const mlx_array a, const int* start, size_t start_num, const int* stop, size_t stop_num, const int* strides, size_t strides_num, const mlx_stream s);
int mlx_slice_update(mlx_array* res, const mlx_array src, const mlx_array update, const int* start, size_t start_num, const int* stop, size_t stop_num, const int* strides, size_t strides_num, const mlx_stream s);
int mlx_repeat(mlx_array* res, const mlx_array arr, int repeats, const mlx_stream s);
int mlx_repeat_axis(mlx_array* res, const mlx_array arr, int repeats, int axis, const mlx_stream s);
int mlx_pad(mlx_array* res, const mlx_array a, const int* axes, size_t axes_num, const int* low_pad_size, size_t low_pad_size_num, const int* high_pad_size, size_t high_pad_size_num, const mlx_array pad_value, const char* mode, const mlx_stream s);

/* Unary ops */
int mlx_exp(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_log(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_log1p(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_tanh(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_sigmoid(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_erf(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_negative(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_floor(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_cos(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_sin(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_stop_gradient(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_contiguous(mlx_array* res, const mlx_array a, bool allow_col_major, const mlx_stream s);

/* Binary ops */
int mlx_add(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_subtract(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_multiply(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_divide(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_power(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_maximum(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_minimum(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_equal(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_not_equal(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_greater_equal(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_less(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_less_equal(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_where(mlx_array* res, const mlx_array condition, const mlx_array x, const mlx_array y, const mlx_stream s);
int mlx_outer(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_logical_and(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);

/* Reduction / indexing */
int mlx_sum_axis(mlx_array* res, const mlx_array a, int axis, bool keepdims, const mlx_stream s);
int mlx_argmax_axis(mlx_array* res, const mlx_array a, int axis, bool keepdims, const mlx_stream s);
int mlx_argpartition_axis(mlx_array* res, const mlx_array a, int kth, int axis, const mlx_stream s);
int mlx_argsort_axis(mlx_array* res, const mlx_array a, int axis, const mlx_stream s);
int mlx_topk(mlx_array* res, const mlx_array a, int k, const mlx_stream s);
int mlx_topk_axis(mlx_array* res, const mlx_array a, int k, int axis, const mlx_stream s);
int mlx_take_axis(mlx_array* res, const mlx_array a, const mlx_array indices, int axis, const mlx_stream s);
int mlx_take_along_axis(mlx_array* res, const mlx_array a, const mlx_array indices, int axis, const mlx_stream s);
int mlx_put_along_axis(mlx_array* res, const mlx_array a, const mlx_array indices, const mlx_array values, int axis, const mlx_stream s);
int mlx_cumsum(mlx_array* res, const mlx_array a, int axis, bool reverse, bool inclusive, const mlx_stream s);
int mlx_softmax_axis(mlx_array* res, const mlx_array a, int axis, bool precise, const mlx_stream s);

/* Creation */
int mlx_zeros(mlx_array* res, const int* shape, size_t shape_num, mlx_dtype dtype, const mlx_stream s);
int mlx_arange(mlx_array* res, double start, double stop, double step, mlx_dtype dtype, const mlx_stream s);
int mlx_random_categorical(mlx_array* res, const mlx_array logits, int axis, const mlx_array key, const mlx_stream s);

/* Linear algebra */
int mlx_matmul(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
int mlx_conv1d(mlx_array* res, const mlx_array input, const mlx_array weight, int stride, int padding, int dilation, int groups, const mlx_stream s);
/* input NHWC, weight OHWI; scalar stride/padding/dilation applied to H and W */
int mlx_conv2d(mlx_array* res, const mlx_array input, const mlx_array weight, int stride, int padding, int dilation, int groups, const mlx_stream s);

/* Clip */
int mlx_clip(mlx_array* res, const mlx_array a, const mlx_array a_min, const mlx_array a_max, const mlx_stream s);

/* ================================================================
 * Quantized ops
 * ================================================================ */

int mlx_quantize(mlx_vector_array* res, const mlx_array w, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, const mlx_array global_scale, const mlx_stream s);
int mlx_dequantize(mlx_array* res, const mlx_array w, const mlx_array scales, const mlx_array biases, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, const mlx_array global_scale, mlx_optional_dtype dtype, const mlx_stream s);
int mlx_quantized_matmul(mlx_array* res, const mlx_array x, const mlx_array w, const mlx_array scales, const mlx_array biases, bool transpose, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, const mlx_stream s);
int mlx_gather_mm(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_array lhs_indices, const mlx_array rhs_indices, bool sorted_indices, const mlx_stream s);
int mlx_gather_qmm(mlx_array* res, const mlx_array x, const mlx_array w, const mlx_array scales, const mlx_array biases, const mlx_array lhs_indices, const mlx_array rhs_indices, bool transpose, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, bool sorted_indices, const mlx_stream s);
int mlx_to_fp8(mlx_array* res, const mlx_array x, const mlx_stream s);
int mlx_from_fp8(mlx_array* res, const mlx_array x, mlx_dtype dtype, const mlx_stream s);

/* ================================================================
 * Fast kernels
 * ================================================================ */

int mlx_fast_rms_norm(mlx_array* res, const mlx_array x, const mlx_array weight, float eps, const mlx_stream s);
int mlx_fast_rope(mlx_array* res, const mlx_array x, int dims, bool traditional, mlx_optional_float base, float scale, int offset, const mlx_array freqs, const mlx_stream s);
int mlx_fast_rope_dynamic(mlx_array* res, const mlx_array x, int dims, bool traditional, mlx_optional_float base, float scale, const mlx_array offset, const mlx_array freqs, const mlx_stream s);
int mlx_fast_scaled_dot_product_attention(mlx_array* res, const mlx_array queries, const mlx_array keys, const mlx_array values, float scale, const char* mask_mode, const mlx_array mask_arr, const mlx_array sinks, const mlx_stream s);
int mlx_fast_layer_norm(mlx_array* res, const mlx_array x, const mlx_array weight, const mlx_array bias, float eps, const mlx_stream s);

/* ================================================================
 * Metal kernel dispatch
 * ================================================================ */

mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void);
void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config cls);
int mlx_fast_metal_kernel_config_add_output_arg(mlx_fast_metal_kernel_config cls, const int* shape, size_t size, mlx_dtype dtype);
int mlx_fast_metal_kernel_config_set_grid(mlx_fast_metal_kernel_config cls, int grid1, int grid2, int grid3);
int mlx_fast_metal_kernel_config_set_thread_group(mlx_fast_metal_kernel_config cls, int thread1, int thread2, int thread3);
int mlx_fast_metal_kernel_config_add_template_arg_dtype(mlx_fast_metal_kernel_config cls, const char* name, mlx_dtype dtype);
int mlx_fast_metal_kernel_config_add_template_arg_int(mlx_fast_metal_kernel_config cls, const char* name, int value);
int mlx_fast_metal_kernel_config_add_template_arg_bool(mlx_fast_metal_kernel_config cls, const char* name, bool value);

mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name, const mlx_vector_string input_names,
    const mlx_vector_string output_names, const char* source,
    const char* header, bool ensure_row_contiguous, bool atomic_outputs);
void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls);
int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs, mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs, const mlx_fast_metal_kernel_config config,
    const mlx_stream stream);

/* ================================================================
 * I/O + containers
 * ================================================================ */

int mlx_load_safetensors(
    mlx_map_string_to_array* res_0, mlx_map_string_to_string* res_1,
    const char* file, const mlx_stream s);

mlx_map_string_to_array mlx_map_string_to_array_new(void);
int mlx_map_string_to_array_free(mlx_map_string_to_array map);
mlx_map_string_to_array_iterator mlx_map_string_to_array_iterator_new(mlx_map_string_to_array map);
int mlx_map_string_to_array_iterator_free(mlx_map_string_to_array_iterator it);
/* 0 = entry written, 1 = end of iteration, 2 = MLX error (see error handler). */
int mlx_map_string_to_array_iterator_next(
    const char** key, mlx_array* value, mlx_map_string_to_array_iterator it);

mlx_map_string_to_string mlx_map_string_to_string_new(void);
int mlx_map_string_to_string_free(mlx_map_string_to_string map);

mlx_vector_array mlx_vector_array_new(void);
int mlx_vector_array_free(mlx_vector_array vec);
int mlx_vector_array_append_value(mlx_vector_array vec, const mlx_array val);
size_t mlx_vector_array_size(mlx_vector_array vec);
int mlx_vector_array_get(mlx_array* res, const mlx_vector_array vec, size_t idx);

mlx_vector_string mlx_vector_string_new(void);
int mlx_vector_string_free(mlx_vector_string vec);
int mlx_vector_string_append_value(mlx_vector_string vec, const char* val);

/* ================================================================
 * Closure + compile
 * ================================================================ */

mlx_closure mlx_closure_new_func_payload(
    int (*fun)(mlx_vector_array*, const mlx_vector_array, void*),
    void* payload, void (*dtor)(void*));
int mlx_closure_apply(mlx_vector_array* res, mlx_closure cls, const mlx_vector_array input);
int mlx_closure_free(mlx_closure cls);
int mlx_compile(mlx_closure* res, const mlx_closure fun, bool shapeless);

/* ================================================================
 * Memory management
 * ================================================================ */

int mlx_clear_cache(void);
int mlx_set_wired_limit(size_t* res, size_t limit);
int mlx_get_active_memory(size_t* res);
int mlx_set_memory_limit(size_t* res, size_t limit);
int mlx_set_cache_limit(size_t* res, size_t limit);
int mlx_get_peak_memory(size_t* res);
int mlx_reset_peak_memory(void);
int mlx_get_cache_memory(size_t* res);

/* ================================================================
 * Transforms
 * ================================================================ */

int mlx_eval(const mlx_vector_array outputs);
int mlx_async_eval(const mlx_vector_array outputs);
int mlx_enable_compile(void);

#ifdef __cplusplus
}
#endif

#endif /* AX_SHIM_H */
