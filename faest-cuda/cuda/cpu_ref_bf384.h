#pragma once

#include <stdint.h>

// CUDA C++ only. CPU reference wrappers for unit testing.
// Include from .cu files only.

#ifdef __cplusplus
extern "C" {
#endif

// Compute out = bf384_mul_128(lhs, rhs) using the original CPU implementation.
// lhs: 48 bytes (bf384), rhs: 16 bytes (bf128), out: 48 bytes (bf384)
void cpu_bf384_mul_128_ref(
    uint8_t       out[48],
    const uint8_t lhs[48],
    const uint8_t rhs[16]
);

// Compute out = leaf_hash_128(uhash, x) using the original CPU implementation.
// uhash: 48 bytes (bf384)
// x: 64 bytes — x0 = x[0..15] (bf128), x1 = x[16..63] (bf384)
// out: 48 bytes (bf384)
void cpu_leaf_hash_128_ref(
    uint8_t       out[48],
    const uint8_t uhash[48],
    const uint8_t x[64]
);

// Compute out = bf576_mul_192(lhs, rhs) using the original CPU implementation.
// lhs: 72 bytes (bf576, BF576_NUM_BYTES), rhs: 24 bytes (bf192, BF192_NUM_BYTES)
// out: 72 bytes (bf576)
// NOTE: do NOT use sizeof(bf576_t)=96 or sizeof(bf192_t)=32 — use NUM_BYTES only.
void cpu_bf576_mul_192_ref(
    uint8_t       out[72],
    const uint8_t lhs[72],
    const uint8_t rhs[24]
);

// Compute out = bf768_mul_256(lhs, rhs) using the original CPU implementation.
// lhs: 96 bytes (bf768, BF768_NUM_BYTES), rhs: 32 bytes (bf256, BF256_NUM_BYTES)
// out: 96 bytes (bf768)
void cpu_bf768_mul_256_ref(
    uint8_t       out[96],
    const uint8_t lhs[96],
    const uint8_t rhs[32]
);

// Compute out = leaf_hash_192(uhash, x) using the original CPU implementation.
// uhash: 72 bytes (bf576), x: 96 bytes (x0=x[0..23] bf192, x1=x[24..95] bf576)
// out: 72 bytes (bf576)
// NOTE: x0 offset = BF192_NUM_BYTES = 24, not sizeof(bf192_t) = 32
void cpu_leaf_hash_192_ref(
    uint8_t       out[72],
    const uint8_t uhash[72],
    const uint8_t x[96]
);

// Compute out = leaf_hash_256(uhash, x) using the original CPU implementation.
// uhash: 96 bytes (bf768), x: 128 bytes (x0=x[0..31] bf256, x1=x[32..127] bf768)
// out: 96 bytes (bf768)
void cpu_leaf_hash_256_ref(
    uint8_t       out[96],
    const uint8_t uhash[96],
    const uint8_t x[128]
);

#ifdef __cplusplus
}
#endif
