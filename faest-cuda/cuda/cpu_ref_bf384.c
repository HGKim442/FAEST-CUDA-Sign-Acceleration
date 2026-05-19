#include "cpu_ref_bf384.h"

// Pull in original faest types and static inline helpers.
// Must be compiled with the same flags as the original faest build
// so that HAVE_ATTR_VECTOR_SIZE is defined consistently.
#include "fields.h"
#include "universal_hashing.h"

void cpu_bf384_mul_128_ref(
    uint8_t       out[48],
    const uint8_t lhs[48],
    const uint8_t rhs[16]
) {
    bf384_store(out, bf384_mul_128(bf384_load(lhs), bf128_load(rhs)));
}

void cpu_leaf_hash_128_ref(
    uint8_t       out[48],
    const uint8_t uhash[48],
    const uint8_t x[64]      // x0 = x[0..15], x1 = x[16..63]
) {
    leaf_hash_128(out, uhash, x);
}

void cpu_bf576_mul_192_ref(
    uint8_t       out[72],
    const uint8_t lhs[72],
    const uint8_t rhs[24]
) {
    bf576_store(out, bf576_mul_192(bf576_load(lhs), bf192_load(rhs)));
}

void cpu_bf768_mul_256_ref(
    uint8_t       out[96],
    const uint8_t lhs[96],
    const uint8_t rhs[32]
) {
    bf768_store(out, bf768_mul_256(bf768_load(lhs), bf256_load(rhs)));
}

void cpu_leaf_hash_192_ref(
    uint8_t       out[72],
    const uint8_t uhash[72],
    const uint8_t x[96]
) {
    leaf_hash_192(out, uhash, x);
}

void cpu_leaf_hash_256_ref(
    uint8_t       out[96],
    const uint8_t uhash[96],
    const uint8_t x[128]
) {
    leaf_hash_256(out, uhash, x);
}
