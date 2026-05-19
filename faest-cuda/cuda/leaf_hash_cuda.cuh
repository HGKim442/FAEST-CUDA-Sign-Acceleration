#pragma once

// Host-side C-callable interface for CUDA leaf_hash_128 batch execution.
// Safe to include from C or C++ translation units.
// Kernel implementation is in leaf_hash_cuda.cu.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Batch execution of leaf_hash_128 on GPU.
// h[i] = bf384_mul_128(uhash[i], x0[i]) XOR x1[i]
//
// All host pointers must be uint64_t-aligned and laid out as
// little-endian limbs (v[0] = least significant).
//   h_uhash : [N * 6]  uint64_t  (bf384 per element)
//   h_x0    : [N * 2]  uint64_t  (bf128 per element)
//   h_x1    : [N * 6]  uint64_t  (bf384 per element)
//   h_out   : [N * 6]  uint64_t  (bf384 per element, output)
//
// Returns 0 on success, -1 on error.
int leaf_hash_128_batch_cuda(
    const uint64_t* h_uhash,
    const uint64_t* h_x0,
    const uint64_t* h_x1,
    uint64_t*       h_out,
    int N
);

// Batch execution of leaf_hash_192 on GPU.
// h[i] = bf576_mul_192(uhash[i], x0[i]) XOR x1[i]
//
// All pointers are uint64_t-aligned, little-endian limb layout.
// Byte sizes are based on NUM_BYTES (no struct padding):
//   h_uhash : [N * 9]  uint64_t  = N * 72 bytes  (bf576 per element)
//   h_x0    : [N * 3]  uint64_t  = N * 24 bytes  (bf192 per element)
//   h_x1    : [N * 9]  uint64_t  = N * 72 bytes  (bf576 per element)
//   h_out   : [N * 9]  uint64_t  = N * 72 bytes  (bf576 per element, output)
//
// WARNING: do NOT pass sizeof(bf192_t)=32 or sizeof(bf576_t)=96 as strides.
//          Use BF192_NUM_BYTES=24 and BF576_NUM_BYTES=72 only.
//
// Returns 0 on success, -1 on error.
int leaf_hash_192_batch_cuda(
    const uint64_t* h_uhash,
    const uint64_t* h_x0,
    const uint64_t* h_x1,
    uint64_t*       h_out,
    int N
);

// Batch execution of leaf_hash_256 on GPU.
// h[i] = bf768_mul_256(uhash[i], x0[i]) XOR x1[i]
//
//   h_uhash : [N * 12] uint64_t  = N * 96 bytes  (bf768 per element)
//   h_x0    : [N * 4]  uint64_t  = N * 32 bytes  (bf256 per element)
//   h_x1    : [N * 12] uint64_t  = N * 96 bytes  (bf768 per element)
//   h_out   : [N * 12] uint64_t  = N * 96 bytes  (bf768 per element, output)
//
// Returns 0 on success, -1 on error.
int leaf_hash_256_batch_cuda(
    const uint64_t* h_uhash,
    const uint64_t* h_x0,
    const uint64_t* h_x1,
    uint64_t*       h_out,
    int N
);

#ifdef __cplusplus
}
#endif
