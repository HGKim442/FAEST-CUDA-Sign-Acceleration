#pragma once

#include <stdint.h>

// ---------------------------------------------------------------------------
// Types
// little-endian limb order: v[0] = least significant, v[5] = most significant
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t v[2];
} bf128_cuda_t;

typedef struct {
    uint64_t v[6];
} bf384_cuda_t;

// GF(2^384) irreducible polynomial: X^384 + X^12 + X^3 + X^2 + 1
// lower-degree terms only (X^384 term is implicit)
static constexpr uint64_t BF384_MODULUS_CUDA = UINT64_C(0x100D);

// ---------------------------------------------------------------------------
// Helper: extract bit as all-0 or all-1 uint64 mask
// ---------------------------------------------------------------------------

__device__ __forceinline__
uint64_t bf128_bit_to_uint64_mask_device(const bf128_cuda_t a, unsigned int bit) {
    const unsigned int limb_idx = bit / 64;
    const unsigned int bit_idx  = bit % 64;
    return -((a.v[limb_idx] >> bit_idx) & UINT64_C(1));
}

__device__ __forceinline__
uint64_t bf384_bit_to_uint64_mask_device(const bf384_cuda_t a, unsigned int bit) {
    const unsigned int limb_idx = bit / 64;
    const unsigned int bit_idx  = bit % 64;
    return -((a.v[limb_idx] >> bit_idx) & UINT64_C(1));
}

// ---------------------------------------------------------------------------
// GF(2^384) operations
// ---------------------------------------------------------------------------

// AND all 6 limbs with a 64-bit scalar mask
__device__ __forceinline__
bf384_cuda_t bf384_and_64_device(bf384_cuda_t a, uint64_t mask) {
    for (int i = 0; i < 6; ++i) {
        a.v[i] &= mask;
    }
    return a;
}

// GF(2^384) addition = bitwise XOR over all limbs
__device__ __forceinline__
bf384_cuda_t bf384_add_device(bf384_cuda_t a, const bf384_cuda_t b) {
    for (int i = 0; i < 6; ++i) {
        a.v[i] ^= b.v[i];
    }
    return a;
}

// Left shift by 1 bit across all 384 bits (little-endian limb order)
// carry propagates from v[i-1] bit 63 into v[i] bit 0
__device__ __forceinline__
bf384_cuda_t bf384_shift_left_1_device(bf384_cuda_t a) {
    for (int i = 5; i > 0; --i) {
        a.v[i] = (a.v[i] << 1) | (a.v[i - 1] >> 63);
    }
    a.v[0] <<= 1;
    return a;
}

// ---------------------------------------------------------------------------
// GF(2^384) x GF(2^128) multiplication
// shift-and-add with modular reduction
// mirrors CPU bf384_mul_128() exactly
// ---------------------------------------------------------------------------

__device__ __forceinline__
bf384_cuda_t bf384_mul_128_device(bf384_cuda_t lhs, bf128_cuda_t rhs) {
    // result = lhs * bit0(rhs)
    bf384_cuda_t result = bf384_and_64_device(lhs,
                              bf128_bit_to_uint64_mask_device(rhs, 0));

    for (unsigned int idx = 1; idx != 128; ++idx) {
        // extract MSB of lhs (bit 383)
        const uint64_t mask = bf384_bit_to_uint64_mask_device(lhs, 383);
        // lhs <<= 1
        lhs = bf384_shift_left_1_device(lhs);
        // modular reduction: if MSB was 1, XOR with irreducible polynomial
        lhs.v[0] ^= mask & BF384_MODULUS_CUDA;
        // accumulate: result ^= lhs * bit_idx(rhs)
        result = bf384_add_device(result,
                     bf384_and_64_device(lhs,
                         bf128_bit_to_uint64_mask_device(rhs, idx)));
    }
    return result;
}

// ===========================================================================
// Types for 192-bit / 256-bit variants
// little-endian limb order: v[0] = least significant
// IMPORTANT: sizeof(bf192_t)=32 on CPU (8B padding), but BF192_NUM_BYTES=24.
//            sizeof(bf576_t)=96 on CPU (24B padding), but BF576_NUM_BYTES=72.
//            CUDA types use only the valid data bytes (no padding).
// ===========================================================================

typedef struct {
    uint64_t v[3];
} bf192_cuda_t;   // 24 bytes = BF192_NUM_BYTES

typedef struct {
    uint64_t v[9];
} bf576_cuda_t;   // 72 bytes = BF576_NUM_BYTES

typedef struct {
    uint64_t v[4];
} bf256_cuda_t;   // 32 bytes = BF256_NUM_BYTES

typedef struct {
    uint64_t v[12];
} bf768_cuda_t;   // 96 bytes = BF768_NUM_BYTES

// GF(2^576): X^576 + X^13 + X^4 + X^3 + 1
static constexpr uint64_t BF576_MODULUS_CUDA = UINT64_C(0x2019);

// GF(2^768): X^768 + X^19 + X^17 + X^4 + 1
static constexpr uint64_t BF768_MODULUS_CUDA = UINT64_C(0xA0011);

// ---------------------------------------------------------------------------
// Helper: bit-to-mask for bf192 and bf576
// ---------------------------------------------------------------------------

__device__ __forceinline__
uint64_t bf192_bit_to_uint64_mask_device(const bf192_cuda_t a, unsigned int bit) {
    const unsigned int limb_idx = bit / 64;
    const unsigned int bit_idx  = bit % 64;
    return -((a.v[limb_idx] >> bit_idx) & UINT64_C(1));
}

__device__ __forceinline__
uint64_t bf576_bit_to_uint64_mask_device(const bf576_cuda_t a, unsigned int bit) {
    const unsigned int limb_idx = bit / 64;
    const unsigned int bit_idx  = bit % 64;
    return -((a.v[limb_idx] >> bit_idx) & UINT64_C(1));
}

__device__ __forceinline__
uint64_t bf256_bit_to_uint64_mask_device(const bf256_cuda_t a, unsigned int bit) {
    const unsigned int limb_idx = bit / 64;
    const unsigned int bit_idx  = bit % 64;
    return -((a.v[limb_idx] >> bit_idx) & UINT64_C(1));
}

__device__ __forceinline__
uint64_t bf768_bit_to_uint64_mask_device(const bf768_cuda_t a, unsigned int bit) {
    const unsigned int limb_idx = bit / 64;
    const unsigned int bit_idx  = bit % 64;
    return -((a.v[limb_idx] >> bit_idx) & UINT64_C(1));
}

// ---------------------------------------------------------------------------
// GF(2^576) operations
// ---------------------------------------------------------------------------

__device__ __forceinline__
bf576_cuda_t bf576_and_64_device(bf576_cuda_t a, uint64_t mask) {
    for (int i = 0; i < 9; ++i) a.v[i] &= mask;
    return a;
}

__device__ __forceinline__
bf576_cuda_t bf576_add_device(bf576_cuda_t a, const bf576_cuda_t b) {
    for (int i = 0; i < 9; ++i) a.v[i] ^= b.v[i];
    return a;
}

__device__ __forceinline__
bf576_cuda_t bf576_shift_left_1_device(bf576_cuda_t a) {
    for (int i = 8; i > 0; --i)
        a.v[i] = (a.v[i] << 1) | (a.v[i - 1] >> 63);
    a.v[0] <<= 1;
    return a;
}

// GF(2^576) x GF(2^192) multiplication
// mirrors CPU bf576_mul_192() exactly
__device__ __forceinline__
bf576_cuda_t bf576_mul_192_device(bf576_cuda_t lhs, bf192_cuda_t rhs) {
    bf576_cuda_t result = bf576_and_64_device(lhs,
                              bf192_bit_to_uint64_mask_device(rhs, 0));
    for (unsigned int idx = 1; idx != 192; ++idx) {
        const uint64_t mask = bf576_bit_to_uint64_mask_device(lhs, 575);
        lhs = bf576_shift_left_1_device(lhs);
        lhs.v[0] ^= mask & BF576_MODULUS_CUDA;
        result = bf576_add_device(result,
                     bf576_and_64_device(lhs,
                         bf192_bit_to_uint64_mask_device(rhs, idx)));
    }
    return result;
}

// ---------------------------------------------------------------------------
// GF(2^768) operations
// ---------------------------------------------------------------------------

__device__ __forceinline__
bf768_cuda_t bf768_and_64_device(bf768_cuda_t a, uint64_t mask) {
    for (int i = 0; i < 12; ++i) a.v[i] &= mask;
    return a;
}

__device__ __forceinline__
bf768_cuda_t bf768_add_device(bf768_cuda_t a, const bf768_cuda_t b) {
    for (int i = 0; i < 12; ++i) a.v[i] ^= b.v[i];
    return a;
}

__device__ __forceinline__
bf768_cuda_t bf768_shift_left_1_device(bf768_cuda_t a) {
    for (int i = 11; i > 0; --i)
        a.v[i] = (a.v[i] << 1) | (a.v[i - 1] >> 63);
    a.v[0] <<= 1;
    return a;
}

// GF(2^768) x GF(2^256) multiplication
// mirrors CPU bf768_mul_256() exactly
__device__ __forceinline__
bf768_cuda_t bf768_mul_256_device(bf768_cuda_t lhs, bf256_cuda_t rhs) {
    bf768_cuda_t result = bf768_and_64_device(lhs,
                              bf256_bit_to_uint64_mask_device(rhs, 0));
    for (unsigned int idx = 1; idx != 256; ++idx) {
        const uint64_t mask = bf768_bit_to_uint64_mask_device(lhs, 767);
        lhs = bf768_shift_left_1_device(lhs);
        lhs.v[0] ^= mask & BF768_MODULUS_CUDA;
        result = bf768_add_device(result,
                     bf768_and_64_device(lhs,
                         bf256_bit_to_uint64_mask_device(rhs, idx)));
    }
    return result;
}
