
#include <metal_stdlib>
using namespace metal;

// Inverted order of one finite f32, matching `f32::partial_cmp` with signed
// zero collapsed to +0. `bits` is the IEEE pattern. Integer ops only: a float
// compare on this GPU flushes a subnormal into the zero class.
inline uint ax_qsa_inverted_finite_order(uint bits) {
    if ((bits & 0x7fffffffu) == 0u) {
        bits = 0u;
    }
    const uint mask = (uint)(((int)bits) >> 31);
    const uint order = bits ^ (mask | 0x80000000u);
    return ~order;
}
