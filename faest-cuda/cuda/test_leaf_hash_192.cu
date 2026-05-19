#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include "leaf_hash_cuda.cuh"
#include "cpu_ref_bf384.h"

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        return 1;                                                       \
    }                                                                   \
} while (0)

// ---------------------------------------------------------------------------
// xorshift64 PRNG
// ---------------------------------------------------------------------------

static uint64_t xorshift64_state;
static void xorshift64_seed(uint64_t s) { xorshift64_state = s; }
static uint64_t xorshift64_next(void) {
    uint64_t x = xorshift64_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xorshift64_state = x;
}
static void rand_bytes(uint8_t* buf, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        uint64_t v = xorshift64_next();
        size_t chunk = (n - i) < 8 ? (n - i) : 8;
        memcpy(buf + i, &v, chunk);
    }
}

// ---------------------------------------------------------------------------
// run_one: compare CPU leaf_hash_192 vs GPU leaf_hash_192_batch_cuda
//
// cpu_uhash[72]: valid uhash bytes (BF576_NUM_BYTES=72)
// cpu_x[96]:     serialized x for CPU — x0[0..23] + x1[24..95]
//
// gpu_uhash_src[uhash_src_size]: source buffer for GPU packing
//   (may be 96B with 24B garbage if padding test, but we copy only 72B)
// gpu_x0_src[x0_src_size]: source for GPU x0 packing (copy only 24B)
// gpu_x1_src[x1_src_size]: source for GPU x1 packing (copy only 72B)
//
// CPU reference always uses cpu_uhash[72] and cpu_x[96] (valid only).
// GPU packing copies only valid NUM_BYTES from src buffers.
// ---------------------------------------------------------------------------

static int run_one(int idx,
                   const uint8_t cpu_uhash[72],
                   const uint8_t cpu_x[96],
                   const uint8_t* gpu_uhash_src,  // copy 72B from here
                   const uint8_t* gpu_x0_src,     // copy 24B from here
                   const uint8_t* gpu_x1_src,     // copy 72B from here
                   bool is_padding_test)
{
    // CPU reference: valid serialized input only
    uint8_t cpu_out[72];
    cpu_leaf_hash_192_ref(cpu_out, cpu_uhash, cpu_x);

    // GPU packing: copy only NUM_BYTES from potentially-padded sources
    uint64_t h_uhash[9], h_x0[3], h_x1[9], h_out[9];
    memset(h_uhash, 0, sizeof(h_uhash));
    memset(h_x0,    0, sizeof(h_x0));
    memset(h_x1,    0, sizeof(h_x1));
    memcpy(h_uhash, gpu_uhash_src, 72);  // BF576_NUM_BYTES=72
    memcpy(h_x0,    gpu_x0_src,    24);  // BF192_NUM_BYTES=24
    memcpy(h_x1,    gpu_x1_src,    72);  // BF576_NUM_BYTES=72

    int rc = leaf_hash_192_batch_cuda(h_uhash, h_x0, h_x1, h_out, 1);
    if (rc != 0) {
        fprintf(stderr, "FAIL [%d]: leaf_hash_192_batch_cuda returned %d\n", idx, rc);
        return 1;
    }

    if (memcmp(cpu_out, h_out, 72) != 0) {
        fprintf(stderr, "FAIL [%d]%s\n  uhash: ", idx,
                is_padding_test ? " (padding test)" : "");
        for (int i = 0; i < 72; i++) fprintf(stderr, "%02x", cpu_uhash[i]);
        fprintf(stderr, "\n  x:     ");
        for (int i = 0; i < 96; i++) fprintf(stderr, "%02x", cpu_x[i]);
        fprintf(stderr, "\n  cpu:   ");
        for (int i = 0; i < 72; i++) fprintf(stderr, "%02x", cpu_out[i]);
        fprintf(stderr, "\n  gpu:   ");
        for (int i = 0; i < 72; i++) fprintf(stderr, "%02x", ((uint8_t*)h_out)[i]);
        fprintf(stderr, "\n");
        return 1;
    }
    return 0;
}

// Convenience: no padding (gpu sources == cpu sources)
static int run_simple(int idx,
                      const uint8_t uhash[72],
                      const uint8_t x[96])
{
    return run_one(idx, uhash, x,
                   uhash,      // gpu_uhash_src
                   x,          // gpu_x0_src  (x[0..23])
                   x + 24,     // gpu_x1_src  (x[24..95])
                   false);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void) {
    printf("=== Layer 2 (Step 9-2): leaf_hash_192 CPU vs GPU unit test ===\n");
    xorshift64_seed(UINT64_C(0x192192192192192A));

    int failures = 0;
    int test_idx = 0;

    // --- Deterministic cases ---
    printf("[1] Deterministic cases...\n");

    uint8_t uhash[72], x[96];
    uint8_t zero72[72], zero96[96];
    memset(zero72, 0, 72);
    memset(zero96, 0, 96);

    // 0: all zero
    failures += run_simple(test_idx++, zero72, zero96);

    // 1: uhash=0, x0=1, x1=0 → h = mul(0, 1) XOR 0 = 0
    memset(x, 0, 96); x[0] = 1;
    failures += run_simple(test_idx++, zero72, x);

    // 2: uhash=1, x0=1, x1=0 → h = mul(1, 1) XOR 0 = 1
    memset(uhash, 0, 72); uhash[0] = 1;
    memset(x, 0, 96); x[0] = 1;
    failures += run_simple(test_idx++, uhash, x);

    // 3: uhash=1, x0=0, x1=rand → h = 0 XOR x1 = x1
    rand_bytes(x + 24, 72);  // x1 = random
    memset(x, 0, 24);        // x0 = 0
    failures += run_simple(test_idx++, uhash, x);

    // 4: uhash all-0xff, x0 all-0xff, x1 all-0xff
    memset(uhash, 0xff, 72); memset(x, 0xff, 96);
    failures += run_simple(test_idx++, uhash, x);

    // 5: uhash bit 575 set (MSB of bf576), x0=1, x1=0 → tests reduction
    memset(uhash, 0, 72); uhash[71] = 0x80;
    memset(x, 0, 96); x[0] = 1;
    failures += run_simple(test_idx++, uhash, x);

    // 6: x0 bit 191 set (MSB of bf192, last loop iteration in mul)
    rand_bytes(uhash, 72);
    memset(x, 0, 96); x[23] = 0x80;
    failures += run_simple(test_idx++, uhash, x);

    // 7: x0 last valid byte (byte 23) nonzero, x1=0
    memset(x, 0, 96); x[23] = 0xAB;
    failures += run_simple(test_idx++, uhash, x);

    // 8: x1 first byte (byte 24 of x) nonzero, x0=0
    memset(x, 0, 96); x[24] = 0x42;
    failures += run_simple(test_idx++, uhash, x);

    // 9: x1 last byte (byte 95 of x) nonzero, x0=0
    memset(x, 0, 96); x[95] = 0xCD;
    failures += run_simple(test_idx++, uhash, x);

    // 10: random uhash, x
    rand_bytes(uhash, 72); rand_bytes(x, 96);
    failures += run_simple(test_idx++, uhash, x);

    // --- Padding garbage tests ---
    // CPU reference uses valid 72B uhash and 96B x only.
    // GPU packing sources have garbage in the padding region.
    // Verifies that GPU packing copies only NUM_BYTES (not sizeof).

    // 11: uhash source has 24B garbage after valid 72B
    {
        uint8_t uhash_padded[96];  // 72B valid + 24B garbage
        rand_bytes(uhash_padded, 96);
        memcpy(uhash, uhash_padded, 72);  // CPU uses only valid 72B
        rand_bytes(x, 96);
        failures += run_one(test_idx++,
                            uhash, x,         // CPU: valid 72B uhash, valid 96B x
                            uhash_padded,     // GPU uhash src: 72B valid + 24B garbage
                            x,                // GPU x0 src: x[0..23]
                            x + 24,           // GPU x1 src: x[24..95]
                            true);
    }

    // 12: x0 source has 8B garbage after valid 24B (sizeof(bf192_t)=32 padding)
    {
        uint8_t x0_padded[32];  // 24B valid + 8B garbage
        rand_bytes(x0_padded, 32);
        memset(x, 0, 96);
        memcpy(x, x0_padded, 24);  // CPU x[0..23] = valid 24B
        rand_bytes(x + 24, 72);    // CPU x1 = random
        rand_bytes(uhash, 72);
        failures += run_one(test_idx++,
                            uhash, x,
                            uhash,      // GPU uhash src: no padding
                            x0_padded,  // GPU x0 src: 24B valid + 8B garbage
                            x + 24,     // GPU x1 src
                            true);
    }

    // 13: x1 source has 24B garbage after valid 72B (sizeof(bf576_t)=96 padding)
    {
        uint8_t x1_padded[96];  // 72B valid + 24B garbage
        rand_bytes(x1_padded, 96);
        memset(x, 0, 96);
        rand_bytes(x, 24);           // CPU x0
        memcpy(x + 24, x1_padded, 72);  // CPU x1 = valid 72B
        rand_bytes(uhash, 72);
        failures += run_one(test_idx++,
                            uhash, x,
                            uhash,      // GPU uhash src
                            x,          // GPU x0 src: x[0..23]
                            x1_padded,  // GPU x1 src: 72B valid + 24B garbage
                            true);
    }

    // 14: all three padded simultaneously
    {
        uint8_t uhash_padded[96], x0_padded[32], x1_padded[96];
        rand_bytes(uhash_padded, 96);
        rand_bytes(x0_padded, 32);
        rand_bytes(x1_padded, 96);
        memcpy(uhash, uhash_padded, 72);
        memset(x, 0, 96);
        memcpy(x, x0_padded, 24);
        memcpy(x + 24, x1_padded, 72);
        rand_bytes(uhash, 72); memcpy(uhash_padded, uhash, 72);
        failures += run_one(test_idx++,
                            uhash, x,
                            uhash_padded, x0_padded, x1_padded,
                            true);
    }

    printf("  Deterministic: %d case(s), %d failure(s)\n",
           test_idx, failures);

    // --- Random cases ---
    printf("[2] Random cases (1000)...\n");
    int rand_failures = 0;
    for (int i = 0; i < 1000; ++i) {
        rand_bytes(uhash, 72);
        rand_bytes(x, 96);
        rand_failures += run_simple(test_idx++, uhash, x);
    }
    failures += rand_failures;
    printf("  Random: 1000 case(s), %d failure(s)\n", rand_failures);

    if (failures == 0)
        printf("=== RESULT: PASS (0 failure(s)) ===\n");
    else
        printf("=== RESULT: FAIL (%d failure(s)) ===\n", failures);

    return failures > 0 ? 1 : 0;
}
