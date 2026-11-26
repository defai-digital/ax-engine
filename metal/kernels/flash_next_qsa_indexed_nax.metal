// Indexed sparse SDPA for Flash Next (`qwen4_exp`) QSA prefill on the GPU
// tensor units (Neural Accelerator hosts, M5 and later).
//
// Like `flash_next_qsa_indexed.metal` this is an MLX "fast kernel" body with
// the same buffers, so the two share their host binding. Its header is
// `crates/ax-engine-mlx/src/model/shared/verify_qmm_mx_nax.metal`, which
// supplies `ax_nax::frag_coord()`.
//
// Buffers (host order): q, k_sealed, k_tail, v_sealed, v_tail, indices,
// lengths, scales, params -> out.
//
//   q          [Queries, Heads, HeadDim]    query rows, token-major
//   k_sealed   [SealedRows, KvHeads, HeadDim]
//   k_tail     [TailRows,   KvHeads, HeadDim]
//   v_sealed   as k_sealed
//   v_tail     as k_tail
//   indices    [Queries, MaxSelected] int32, absolute token rows, -1 padded
//   lengths    [Queries] int32, selected count per query
//   scales     [1] float32, softmax scale
//   params     [3] int32: SealedRows (at least 1), TotalRows, MaxSelected
//   out        [Queries, Heads, HeadDim]
//
// Template constants: `T` (query dtype: half or bfloat, which the tensor
// units take exactly; they would round a float operand), `HeadDim` (256),
// `Heads`, `KvHeads`, `Group` (at most 16).
//
// One threadgroup of two simdgroups handles one (query, kv head) pair. The
// `Group` query heads that share the kv head are the rows of a 16-row
// tensor-unit tile, and each simdgroup owns one half of the head dimension.
// The threadgroup walks the query's own selected rows 32 at a time:
//
//   1. each simdgroup computes Q K^T over its 128 dimensions with four
//      16x32x32 products, and the two halves meet in threadgroup memory;
//   2. both run the same float32 online softmax over the 32 scores;
//   3. each simdgroup accumulates P V over its 128 output dimensions.
//
// Nothing is gathered: K and V rows are read in place, 16 bytes at a time,
// and the only per-query state is the 16x128 float32 output of a simdgroup.
// The cost of a query depends on its selection length, not on the history.
//
// Lane layout: a lane holds rows `fm` and `fm + 8`, columns `fn .. fn + 3`,
// of every 16x16 fragment. Head dimensions are assigned to fragment columns
// so that one lane's elements are contiguous in memory:
//
//   - Q and K, product g of 4: the lane's eight elements of a row are
//     dimensions `32 g + 2 fn + 0..7`, the first four in the first fragment
//     along the product's inner dimension;
//   - V and the output: the lane's sixteen elements of a key are dimensions
//     `16 fm + 0..15`; element `4 g + 2 nf + r` is row `fm + 8 r` of output
//     fragment `nf` in product g.
//
// Q and K use the same assignment, so only the order of the float32 sum over
// the head dimension changes. A probability enters P V as two float16
// pieces, `hi = half(p)` and `lo = half(p - hi)`, along the inner dimension
// against the same values: their sum is `p` to about 2^-22, where a single
// half or bfloat piece would round it to 11 or 8 bits.
//
// Numerical contract: softmax over the query's selected rows, float32 scores
// and accumulation, output rounded once to the query dtype. This is the same
// set of values the gathered path and the masked dense call attend to; the
// reduction order and the online rescaling differ, so the result is not
// bit-identical to either. A query's output does not depend on the other
// queries of the call or on the length of the history.

static_assert(HeadDim == 256, "the lane layout covers 256-wide heads");
static_assert(Group >= 1 && Group <= 16, "one tile holds at most 16 query heads");

const uint query = thread_position_in_grid.y;
const uint kv_head = thread_position_in_grid.z;
const ushort half_index = simdgroup_index_in_threadgroup;
const ushort lane = thread_index_in_simdgroup;
const short2 coord = ax_nax::frag_coord();
const short fm = coord.y;
const short fn = coord.x;

// The two simdgroups exchange their partial scores here. Buffers alternate
// by tile, so one barrier per tile is enough: a simdgroup overwrites a buffer
// two tiles later, after the barrier in between.
threadgroup float partial[2][2 * 32 * 16];

const int sealed_rows = params[0];
const int total_rows = params[1];
const int max_selected = params[2];
const int selected = clamp(lengths[query], 0, max_selected);
const int index_base = static_cast<int>(query) * max_selected;
const float scale = scales[0];
const bool head_ok[2] = {fm < Group, fm + 8 < Group};
const int dim_base = static_cast<int>(half_index) * 128;
const int value_dim = dim_base + 16 * fm;

const device T* q_row[2];
_Pragma("clang loop unroll(full)")
for (short r = 0; r < 2; r++) {
  // A lane whose row is past the group points at the group's first head and
  // loads zeros instead.
  const int head = static_cast<int>(kv_head) * Group + (head_ok[r] ? fm + 8 * r : 0);
  q_row[r] = q + (static_cast<int64_t>(query) * Heads + head) * HeadDim + dim_base + 2 * fn;
}

constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 32, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 32, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
// A cooperative tensor holds eight elements per 16x16 fragment and lane:
// element `4 r + j` is row `fm + 8 r`, column `fn + j`. The left operand's
// fragments run along the inner dimension, the destination's along its 32
// columns, and the right operand's (32 rows by 32 inner) are ordered
// `row fragment + 2 * inner fragment`.
auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
using qa_t = metal::remove_addrspace_t<decltype(qa)>;
using kb_t = metal::remove_addrspace_t<decltype(kb)>;
auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
auto vb = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
using pa_t = metal::remove_addrspace_t<decltype(pa)>;
using vb_t = metal::remove_addrspace_t<decltype(vb)>;
// Output of product g over this half of the head dimension.
auto o0 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
auto o1 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
auto o2 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
auto o3 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
_Pragma("clang loop unroll(full)")
for (short i = 0; i < 16; i++) {
  o0[i] = 0.0f;
  o1[i] = 0.0f;
  o2[i] = 0.0f;
  o3[i] = 0.0f;
}
float running_max[2] = {-INFINITY, -INFINITY};
float running_sum[2] = {0.0f, 0.0f};

const int tiles = (selected + 31) / 32;
for (int tile = 0; tile < tiles; tile++) {
  const int slot0 = tile * 32;

  // `-1` is the padding marker, and the index tile is built on the device and
  // is not validated by the host: a slot past the query's length, or one
  // that names a row outside `[0, total_rows)`, reads row 0 of the sealed
  // rows, which always exists, and gets zero weight. A finite row 0 cannot
  // change a bit of the output that way. A non-finite value in it would
  // (0 * inf), as a non-finite value in any row does through a masked dense
  // call; the row pointers stay a function of the tile alone because every
  // way of standing in another row measured 10 to 45% slower.
  //
  // K rows this lane loads: keys fm, fm + 8, 16 + fm and 24 + fm of the tile.
  const device T* k_row[4];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 4; i++) {
    const int slot = slot0 + 16 * (i / 2) + fm + 8 * (i % 2);
    const int token = slot < selected ? indices[index_base + slot] : -1;
    const bool ok = token >= 0 && token < total_rows;
    const bool in_sealed = !ok || token < sealed_rows;
    const int row = !ok ? 0 : (in_sealed ? token : token - sealed_rows);
    k_row[i] = (in_sealed ? k_sealed : k_tail)
        + (static_cast<int64_t>(row) * KvHeads + kv_head) * HeadDim + dim_base + 2 * fn;
  }
  // Score columns this lane holds, and the V rows it loads for them: keys
  // fn .. fn + 3 and 16 + fn .. 16 + fn + 3.
  const device T* v_row[8];
  bool key_ok[8];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; i++) {
    const int slot = slot0 + 16 * (i / 4) + fn + (i % 4);
    const int token = slot < selected ? indices[index_base + slot] : -1;
    key_ok[i] = token >= 0 && token < total_rows;
    const bool in_sealed = !key_ok[i] || token < sealed_rows;
    const int row = !key_ok[i] ? 0 : (in_sealed ? token : token - sealed_rows);
    v_row[i] = (in_sealed ? v_sealed : v_tail)
        + (static_cast<int64_t>(row) * KvHeads + kv_head) * HeadDim + value_dim;
  }

  // 1. Partial scores over this simdgroup's half of the head dimension.
  auto s = qk_op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 16; i++) {
    s[i] = 0.0f;
  }
  _Pragma("clang loop unroll(full)")
  for (short g = 0; g < 4; g++) {
    _Pragma("clang loop unroll(full)")
    for (short r = 0; r < 2; r++) {
      const vec<T, 8> x =
          head_ok[r] ? *(const device vec<T, 8>*)(q_row[r] + 32 * g) : vec<T, 8>(0);
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; j++) {
        qa[4 * r + j] = x[j];
        qa[8 + 4 * r + j] = x[4 + j];
      }
    }
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 4; i++) {
      const vec<T, 8> x = *(const device vec<T, 8>*)(k_row[i] + 32 * g);
      const short nf = i / 2;
      const short r = i % 2;
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; j++) {
        kb[8 * nf + 4 * r + j] = x[j];
        kb[8 * (nf + 2) + 4 * r + j] = x[4 + j];
      }
    }
    qk_op.run(qa, kb, s);
  }

  threadgroup float* mine = partial[tile & 1] + (static_cast<int>(half_index) * 32 + lane) * 16;
  const threadgroup float* theirs =
      partial[tile & 1] + (static_cast<int>(1 - half_index) * 32 + lane) * 16;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 16; i++) {
    mine[i] = s[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Element `8 nf + 4 r + j` is head `fm + 8 r`, key `16 nf + fn + j`. The
  // sum is commutative, so both simdgroups hold identical scores.
  float score[16];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 16; i++) {
    score[i] = (s[i] + theirs[i]) * scale;
  }

  // 2. Online softmax. The four lanes that share `fm` hold one head's 32
  // scores between them. The accumulators are rescaled only when a head's
  // maximum moved; the first tile's correction is exp(-inf) = 0 on zeros.
  float p[16];
  float correction[2];
  bool rescale = false;
  _Pragma("clang loop unroll(full)")
  for (short r = 0; r < 2; r++) {
    float tile_max = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 8; i++) {
      tile_max = max(tile_max, key_ok[i] ? score[8 * (i / 4) + 4 * r + (i % 4)] : -INFINITY);
    }
    tile_max = max(
        max(tile_max, simd_shuffle_xor(tile_max, ushort(1))),
        max(simd_shuffle_xor(tile_max, ushort(8)), simd_shuffle_xor(tile_max, ushort(9))));
    const float new_max = max(running_max[r], tile_max);
    const bool any = new_max > -INFINITY;
    correction[r] = new_max > running_max[r] ? exp(running_max[r] - new_max) : 1.0f;
    rescale = rescale || correction[r] != 1.0f;
    float tile_sum = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 8; i++) {
      const short e = 8 * (i / 4) + 4 * r + (i % 4);
      p[e] = (any && key_ok[i]) ? exp(score[e] - new_max) : 0.0f;
      tile_sum += p[e];
    }
    tile_sum += simd_shuffle_xor(tile_sum, ushort(1)) + simd_shuffle_xor(tile_sum, ushort(8))
        + simd_shuffle_xor(tile_sum, ushort(9));
    running_sum[r] = running_sum[r] * correction[r] + tile_sum;
    running_max[r] = new_max;
  }
  if (rescale) {
    _Pragma("clang loop unroll(full)")
    for (short nf = 0; nf < 2; nf++) {
      _Pragma("clang loop unroll(full)")
      for (short r = 0; r < 2; r++) {
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 4; j++) {
          const short e = 8 * nf + 4 * r + j;
          o0[e] *= correction[r];
          o1[e] *= correction[r];
          o2[e] *= correction[r];
          o3[e] *= correction[r];
        }
      }
    }
  }

  // 3. P V over this half of the head dimension, 16 keys at a time.
  _Pragma("clang loop unroll(full)")
  for (short key_half = 0; key_half < 2; key_half++) {
    vec<T, 8> values[4][2];
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 4; j++) {
      values[j][0] = *(const device vec<T, 8>*)(v_row[4 * key_half + j]);
      values[j][1] = *(const device vec<T, 8>*)(v_row[4 * key_half + j] + 8);
    }
    _Pragma("clang loop unroll(full)")
    for (short r = 0; r < 2; r++) {
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; j++) {
        const float weight = p[8 * key_half + 4 * r + j];
        const half hi = static_cast<half>(weight);
        pa[4 * r + j] = hi;
        pa[8 + 4 * r + j] = static_cast<half>(weight - static_cast<float>(hi));
      }
    }
    _Pragma("clang loop unroll(full)")
    for (short g = 0; g < 4; g++) {
      _Pragma("clang loop unroll(full)")
      for (short nf = 0; nf < 2; nf++) {
        _Pragma("clang loop unroll(full)")
        for (short r = 0; r < 2; r++) {
          _Pragma("clang loop unroll(full)")
          for (short j = 0; j < 4; j++) {
            const short d = 4 * g + 2 * nf + r;
            const T value = values[j][d / 8][d % 8];
            vb[8 * nf + 4 * r + j] = value;
            vb[8 * (nf + 2) + 4 * r + j] = value;
          }
        }
      }
      if (g == 0) {
        pv_op.run(pa, vb, o0);
      } else if (g == 1) {
        pv_op.run(pa, vb, o1);
      } else if (g == 2) {
        pv_op.run(pa, vb, o2);
      } else {
        pv_op.run(pa, vb, o3);
      }
    }
  }
}

// A query that kept nothing writes zeros; the host never dispatches one, but
// the guard keeps a malformed index row from producing NaN. Output fragment
// `nf` of product g, element `4 r + j`, is head `fm + 8 r` and dimension
// `dim_base + 16 ((fn + j) % 8) + 4 g + 2 nf + (fn + j) / 8`.
_Pragma("clang loop unroll(full)")
for (short r = 0; r < 2; r++) {
  if (head_ok[r]) {
    const float normaliser = running_sum[r] > 0.0f ? 1.0f / running_sum[r] : 0.0f;
    device T* dst = out
        + (static_cast<int64_t>(query) * Heads + static_cast<int>(kv_head) * Group + fm + 8 * r)
            * HeadDim
        + dim_base;
    _Pragma("clang loop unroll(full)")
    for (short nf = 0; nf < 2; nf++) {
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 4; j++) {
        const short n = fn + j;
        const short e = 8 * nf + 4 * r + j;
        const short d = 16 * (n % 8) + 2 * nf + n / 8;
        dst[d] = static_cast<T>(o0[e] * normaliser);
        dst[d + 4] = static_cast<T>(o1[e] * normaliser);
        dst[d + 8] = static_cast<T>(o2[e] * normaliser);
        dst[d + 12] = static_cast<T>(o3[e] * normaliser);
      }
    }
  }
}
