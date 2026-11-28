
const int count_limit = limits[0];
const int blocks_n = limits[1];
const int seq_n = limits[2];
const int keep_q = limits[3];
const int decode_keys = limits[4];
const size_t i = thread_position_in_grid.x;
if (blocks_n <= 0 || seq_n <= 0 || i >= (size_t)count_limit) return;
const size_t blocks = (size_t)blocks_n;
const size_t block = i % blocks;
const size_t query = (i / blocks) % (size_t)seq_n;
const int limit = completes[query];
const uint raw = as_type<uint>(scores[i]);
const uint magnitude = raw & 0x7fffffffu;
const uint exponent = magnitude >> 23;
const int causal = ((int)block < limit) ? 1 : 0;
long key = 0x6000000000000000L | (long)block;
if (causal != 0) {
    if (decode_keys != 0) {
        // NaN: exponent all ones and a non-zero mantissa. +inf stays.
        const uint bits = (magnitude > 0x7f800000u) ? 0xff800000u : raw;
        const uint inverted = ax_qsa_inverted_finite_order(bits);
        key = ((long)inverted << 30) | (long)block;
    } else if (exponent == 0xffu) {
        key = 0x4000000000000000L | (long)block;
    } else {
        const uint inverted = ax_qsa_inverted_finite_order(raw);
        key = ((long)inverted << 30) | (long)block;
    }
}
keys[i] = key;
if (block == 0) {
    const size_t row = i / blocks;
    if (decode_keys != 0) {
        flags[row] = 0;
    } else {
        int finite = 0;
        int bad = 0;
        for (int b = 0; b < limit; ++b) {
            const uint causal_bits = as_type<uint>(scores[row * blocks + (size_t)b]);
            if (((causal_bits >> 23) & 0xffu) == 0xffu) {
                bad = 1;
            } else {
                finite += 1;
            }
        }
        flags[row] = (bad != 0 || finite < keep_q) ? 1 : 0;
    }
}
