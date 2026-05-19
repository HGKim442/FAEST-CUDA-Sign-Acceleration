#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include "leaf_hash_cuda.cuh"
#include "cpu_ref_bf384.h"

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        return 1;                                                       \
    }                                                                   \
} while (0)

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

// run_one: CPU reference uses valid bytes only.
// GPU packing copies NUM_BYTES from src (may contain garbage beyond valid region).
static int run_one(int idx,
                   const uint8_t cpu_uhash[96],
                   const uint8_t cpu_x[128],
                   const uint8_t* gpu_uhash_src,  // copy 96B
                   const uint8_t* gpu_x0_src,     // copy 32B
                   const uint8_t* gpu_x1_src,     // copy 96B
                   bool is_padding_test)
{
    uint8_t cpu_out[96];
    cpu_leaf_hash_256_ref(cpu_out, cpu_uhash, cpu_x);

    uint64_t h_uhash[12], h_x0[4], h_x1[12], h_out[12];
    memset(h_uhash, 0, sizeof(h_uhash));
    memset(h_x0,    0, sizeof(h_x0));
    memset(h_x1,    0, sizeof(h_x1));
    memcpy(h_uhash, gpu_uhash_src, 96);  // BF768_NUM_BYTES=96
    memcpy(h_x0,    gpu_x0_src,    32);  // BF256_NUM_BYTES=32
    memcpy(h_x1,    gpu_x1_src,    96);  // BF768_NUM_BYTES=96

    int rc = leaf_hash_256_batch_cuda(h_uhash, h_x0, h_x1, h_out, 1);
    if (rc != 0) {
        fprintf(stderr, "FAIL [%d]: leaf_hash_256_batch_cuda returned %d\n", idx, rc);
        return 1;
    }

    if (memcmp(cpu_out, h_out, 96) != 0) {
        fprintf(stderr, "FAIL [%d]%s\n  uhash: ", idx,
                is_padding_test ? " (padding test)" : "");
        for (int i = 0; i < 96; i++) fprintf(stderr, "%02x", cpu_uhash[i]);
        fprintf(stderr, "\n  x:     ");
        for (int i = 0; i < 128; i++) fprintf(stderr, "%02x", cpu_x[i]);
        fprintf(stderr, "\n  cpu:   ");
        for (int i = 0; i < 96; i++) fprintf(stderr, "%02x", cpu_out[i]);
        fprintf(stderr, "\n  gpu:   ");
        for (int i = 0; i < 96; i++) fprintf(stderr, "%02x", ((uint8_t*)h_out)[i]);
        fprintf(stderr, "\n");
        return 1;
    }
    return 0;
}

static int run_simple(int idx,
                      const uint8_t uhash[96],
                      const uint8_t x[128])
{
    return run_one(idx, uhash, x,
                   uhash,      // gpu_uhash_src
                   x,          // gpu_x0_src (x[0..31])
                   x + 32,     // gpu_x1_src (x[32..127])
                   false);
}

int main(void) {
    printf("=== Layer 2 (Step 9-2): leaf_hash_256 CPU vs GPU unit test ===\n");
    xorshift64_seed(UINT64_C(0x256256256256256B));

    int failures = 0;
    int test_idx = 0;

    printf("[1] Deterministic cases...\n");

    uint8_t uhash[96], x[128];
    uint8_t zero96[96], zero128[128];
    memset(zero96, 0, 96); memset(zero128, 0, 128);

    // 0: all zero
    failures += run_simple(test_idx++, zero96, zero128);

    // 1: uhash=0, x0=1, x1=0
    memset(x, 0, 128); x[0] = 1;
    failures += run_simple(test_idx++, zero96, x);

    // 2: uhash=1, x0=1, x1=0
    memset(uhash, 0, 96); uhash[0] = 1;
    memset(x, 0, 128); x[0] = 1;
    failures += run_simple(test_idx++, uhash, x);

    // 3: uhash=1, x0=0, x1=rand
    rand_bytes(x + 32, 96); memset(x, 0, 32);
    failures += run_simple(test_idx++, uhash, x);

    // 4: all-0xff
    memset(uhash, 0xff, 96); memset(x, 0xff, 128);
    failures += run_simple(test_idx++, uhash, x);

    // 5: uhash bit 767 (MSB of bf768), x0=1, x1=0 → reduction path
    memset(uhash, 0, 96); uhash[95] = 0x80;
    memset(x, 0, 128); x[0] = 1;
    failures += run_simple(test_idx++, uhash, x);

    // 6: x0 bit 255 (MSB of bf256, last loop iteration)
    rand_bytes(uhash, 96);
    memset(x, 0, 128); x[31] = 0x80;
    failures += run_simple(test_idx++, uhash, x);

    // 7: x0 last valid byte (byte 31) nonzero
    memset(x, 0, 128); x[31] = 0xAB;
    failures += run_simple(test_idx++, uhash, x);

    // 8: x1 first byte (x[32]) nonzero
    memset(x, 0, 128); x[32] = 0x42;
    failures += run_simple(test_idx++, uhash, x);

    // 9: x1 last byte (x[127]) nonzero
    memset(x, 0, 128); x[127] = 0xCD;
    failures += run_simple(test_idx++, uhash, x);

    // 10: random
    rand_bytes(uhash, 96); rand_bytes(x, 128);
    failures += run_simple(test_idx++, uhash, x);

    // --- Padding garbage tests ---
    // 256 variants have no padding in CPU types (bf256_t=32B, bf768_t=96B).
    // Tests still verify that GPU packing copies exactly NUM_BYTES.

    // 11: extra garbage appended after valid uhash 96B
    {
        uint8_t uhash_extra[128];
        rand_bytes(uhash_extra, 128);
        memcpy(uhash, uhash_extra, 96);
        rand_bytes(x, 128);
        failures += run_one(test_idx++,
                            uhash, x,
                            uhash_extra,  // GPU uhash src: 96B valid + 32B extra
                            x, x + 32,
                            true);
    }

    // 12: extra garbage after x0 32B
    {
        uint8_t x0_extra[64];
        rand_bytes(x0_extra, 64);
        memset(x, 0, 128);
        memcpy(x, x0_extra, 32);
        rand_bytes(x + 32, 96);
        rand_bytes(uhash, 96);
        failures += run_one(test_idx++,
                            uhash, x,
                            uhash, x0_extra, x + 32,
                            true);
    }

    printf("  Deterministic: %d case(s), %d failure(s)\n",
           test_idx, failures);

    printf("[2] Random cases (1000)...\n");
    int rand_failures = 0;
    for (int i = 0; i < 1000; ++i) {
        rand_bytes(uhash, 96);
        rand_bytes(x, 128);
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
