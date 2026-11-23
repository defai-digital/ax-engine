// Copy token-major BF16 K and V segments directly into the padded SDPA tiles.
// A vector covers eight channels of one 256-channel KV head.
const uint width = uint(geometry[0]);
const uint sealed = uint(geometry[1]);
const int tokens = geometry[2];
const uint vectors_per_head = 32;
const uint packets_per_row = width * vectors_per_head;
const uint lane = thread_position_in_grid.x;
if (lane >= packets_per_row) {
    return;
}
const uint head = thread_position_in_grid.y;
const uint query = thread_position_in_grid.z;
const uint key = lane / vectors_per_head;
const uint packet = (query * 2 + head) * packets_per_row + lane;

// The picker supplies valid absolute indices or -1 padding. Padding copies
// token zero exactly as the stock gather does. Clamp for safe device access
// even if the caller violates the picker's range contract.
const int index = selection[query * width + key];
const uint token = uint(clamp(index, 0, tokens - 1));
const bool prefix = token < sealed;
const uint local_token = prefix ? token : token - sealed;
const uint source = (local_token * 2 + head) * vectors_per_head + lane % vectors_per_head;

const device uint4* pk = reinterpret_cast<const device uint4*>(key_prefix);
const device uint4* pv = reinterpret_cast<const device uint4*>(value_prefix);
const device uint4* tk = reinterpret_cast<const device uint4*>(key_tail);
const device uint4* tv = reinterpret_cast<const device uint4*>(value_tail);
device uint4* ko = reinterpret_cast<device uint4*>(key_out);
device uint4* vo = reinterpret_cast<device uint4*>(value_out);
ko[packet] = prefix ? pk[source] : tk[source];
vo[packet] = prefix ? pv[source] : tv[source];
