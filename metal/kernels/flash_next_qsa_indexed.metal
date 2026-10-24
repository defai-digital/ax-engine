// Indexed sparse SDPA for Flash Next (`qwen4_exp`) QSA prefill.
//
// This file holds an MLX "fast kernel" body, not a stand-alone `.metal`
// translation unit: `crates/ax-engine-mlx` registers it with
// `mlx_sys::MlxMetalKernel`, which wraps the body in a kernel function whose
// parameter list is the buffer names declared by the host binding. It is kept
// under `metal/kernels/` so the device code lives with the other native AX
// kernels rather than inside a Rust string literal; the Rust side only ever
// touches it through `crates/mlx-sys`.
//
// Buffers (host order): q, k_sealed, k_tail, v_sealed, v_tail, indices,
// lengths, scales -> out.
//
//   q          [Queries, Heads, HeadDim]    query rows, token-major
//   k_sealed   [SealedRows, KvHeads, HeadDim]
//   k_tail     [TailRows,   KvHeads, HeadDim]
//   v_sealed   as k_sealed
//   v_tail     as k_tail
//   indices    [Queries, MaxSelected] int32, absolute token rows, -1 padded
//   lengths    [Queries] int32, exact selected count per query
//   scales     [1] float32, softmax scale
//   out        [Queries, Heads, HeadDim]
//
// One threadgroup handles one (query, kv head) pair, so a selected K/V row is
// staged into threadgroup memory once and read by every query head that shares
// that kv head (native GQA: query head `h` reads kv head `h / Group`). The
// kernel walks only the query's own selected rows in selection order, keeps a
// float32 online-softmax accumulator, and never materialises a gathered
// [queries x selected] K/V tile.
//
// Numerical contract: softmax over the selected values in selection order,
// fp32 accumulation, output rounded once to the query dtype. This is the same
// set of values the gathered path attends to, but the reduction order and the
// online rescaling differ, so it is not bit-identical to MLX's fused SDPA.

constexpr int ValuesPerLane = HeadDim / 32;
constexpr int ThreadgroupSize = Group * 32;

// The threadgroup is exactly `Group * 32` threads wide and the grid x axis
// has the same width, so the grid and threadgroup x positions coincide; the
// grid position is the one MLX fast kernels document for both.
const uint lane = thread_position_in_grid.x & 31u;
const uint simd = thread_position_in_grid.x >> 5u;
const uint tid = thread_position_in_grid.x;
const uint query = thread_position_in_grid.y;
const uint kv_head = thread_position_in_grid.z;
const uint head = kv_head * Group + simd;

threadgroup float staged_k[HeadDim];
threadgroup float staged_v[HeadDim];

const float scale = static_cast<float>(scales[0]);
const int length = lengths[query];
const int index_base = static_cast<int>(query) * MaxSelected;

// `q_reg` carries the softmax scale so the inner loop is one FMA per dimension.
// A power-of-two scale (HeadDim = 256) is exact in every supported dtype.
float q_reg[ValuesPerLane];
float acc[ValuesPerLane];
for (int i = 0; i < ValuesPerLane; ++i) {
  const int dim = static_cast<int>(lane) + i * 32;
  q_reg[i] = static_cast<float>(q[(query * Heads + head) * HeadDim + dim]) * scale;
  acc[i] = 0.0f;
}

float running_max = -INFINITY;
float running_sum = 0.0f;

for (int j = 0; j < length; ++j) {
  const int token = indices[index_base + j];
  if (token < 0) {
    continue;
  }
  const bool in_sealed = token < SealedRows;
  const int row = in_sealed ? token : token - SealedRows;
  const int row_base = row * (KvHeads * HeadDim) + static_cast<int>(kv_head) * HeadDim;
  for (uint dim = tid; dim < static_cast<uint>(HeadDim); dim += ThreadgroupSize) {
    staged_k[dim] = in_sealed ? static_cast<float>(k_sealed[row_base + static_cast<int>(dim)])
                              : static_cast<float>(k_tail[row_base + static_cast<int>(dim)]);
    staged_v[dim] = in_sealed ? static_cast<float>(v_sealed[row_base + static_cast<int>(dim)])
                              : static_cast<float>(v_tail[row_base + static_cast<int>(dim)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float dot = 0.0f;
  for (int i = 0; i < ValuesPerLane; ++i) {
    dot += q_reg[i] * staged_k[static_cast<int>(lane) + i * 32];
  }
  const float score = simd_sum(dot);

  // Online softmax: rescale the accumulator by exp(old_max - new_max). The
  // first step has old_max = -inf, so the correction is exactly zero and the
  // accumulator starts from this row.
  const float new_max = max(running_max, score);
  const float correction = exp(running_max - new_max);
  const float weight = exp(score - new_max);
  running_sum = running_sum * correction + weight;
  for (int i = 0; i < ValuesPerLane; ++i) {
    acc[i] = acc[i] * correction + weight * staged_v[static_cast<int>(lane) + i * 32];
  }
  running_max = new_max;

  // Every lane must finish reading the staged rows before the next token
  // overwrites them.
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A query that kept nothing contributes zeros; the host never dispatches one,
// but the guard keeps a malformed index row from producing NaN.
const float normaliser = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
for (int i = 0; i < ValuesPerLane; ++i) {
  const int dim = static_cast<int>(lane) + i * 32;
  out[(query * Heads + head) * HeadDim + dim] = static_cast<T>(acc[i] * normaliser);
}
