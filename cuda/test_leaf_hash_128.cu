#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "cuda/leaf_hash_cuda.cuh"
#include "cuda/cpu_ref_bf384.h"

#define HOST_CHECK_ALLOC(ptr) do {                                      \
    if ((ptr) == NULL) {                                                \
        fprintf(stderr, "Host allocation failed: %s\n", #ptr);         \
        exit(1);                                                        \
    }                                                                   \
} while (0)

// ---------------------------------------------------------------------------
// xorshift64 PRNG
// ---------------------------------------------------------------------------
static uint64_t xorshift64_state = UINT64_C(0x123456789abcdef0);

static uint64_t xorshift64_next(void) {
    uint64_t x = xorshift64_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xorshift64_state = x;
    return x;
}

static void rand_bytes(uint8_t* buf, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        uint64_t r = xorshift64_next();
        size_t chunk = (n - i) < 8 ? (n - i) : 8;
        memcpy(buf + i, &r, chunk);
    }
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------
static void print_hex(const char* label, const uint8_t* buf, size_t n) {
    printf("  %s: ", label);
    for (size_t i = 0; i < n; ++i) printf("%02x", buf[i]);
    printf("\n");
}

// ---------------------------------------------------------------------------
// Test runner
// all_uhash: [N*48], all_x: [N*64]
// ---------------------------------------------------------------------------
static int run_tests(
    const uint8_t* all_uhash,
    const uint8_t* all_x,
    int N,
    const char* label
) {
    // CPU reference
    uint8_t* cpu_out = (uint8_t*)malloc(N * 48);
    HOST_CHECK_ALLOC(cpu_out);
    for (int i = 0; i < N; ++i) {
        cpu_leaf_hash_128_ref(cpu_out + i * 48,
                              all_uhash + i * 48,
                              all_x + i * 64);
    }

    // Pack GPU inputs from the same x[64] source
    uint64_t* h_uhash = (uint64_t*)malloc(N * 6 * sizeof(uint64_t));
    HOST_CHECK_ALLOC(h_uhash);
    uint64_t* h_x0 = (uint64_t*)malloc(N * 2 * sizeof(uint64_t));
    HOST_CHECK_ALLOC(h_x0);
    uint64_t* h_x1 = (uint64_t*)malloc(N * 6 * sizeof(uint64_t));
    HOST_CHECK_ALLOC(h_x1);
    uint64_t* h_out = (uint64_t*)malloc(N * 6 * sizeof(uint64_t));
    HOST_CHECK_ALLOC(h_out);

    for (int i = 0; i < N; ++i) {
        memcpy(h_uhash + i * 6, all_uhash + i * 48,      48);
        memcpy(h_x0   + i * 2, all_x    + i * 64,        16);
        memcpy(h_x1   + i * 6, all_x    + i * 64 + 16,   48);
    }

    // GPU wrapper
    int ret = leaf_hash_128_batch_cuda(h_uhash, h_x0, h_x1, h_out, N);
    if (ret != 0) {
        printf("  [ERROR] leaf_hash_128_batch_cuda returned %d\n", ret);
        free(cpu_out); free(h_uhash); free(h_x0); free(h_x1); free(h_out);
        return N;
    }

    // byte-by-byte compare
    int failures = 0;
    for (int i = 0; i < N; ++i) {
        uint8_t gpu_bytes[48];
        memcpy(gpu_bytes, h_out + i * 6, 48);
        if (memcmp(cpu_out + i * 48, gpu_bytes, 48) != 0) {
            printf("  [FAIL] %s case %d\n", label, i);
            print_hex("uhash", all_uhash + i * 48,    48);
            print_hex("x0",   all_x + i * 64,         16);
            print_hex("x1",   all_x + i * 64 + 16,    48);
            print_hex("cpu",  cpu_out + i * 48,        48);
            print_hex("gpu",  gpu_bytes,                48);
            ++failures;
            if (failures >= 5) {
                printf("  ... (stopping at 5 failures)\n");
                break;
            }
        }
    }

    free(cpu_out); free(h_uhash); free(h_x0); free(h_x1); free(h_out);
    return failures;
}

// ---------------------------------------------------------------------------
// Deterministic test cases (DET_N=16)
// ---------------------------------------------------------------------------
#define DET_N 16

static void build_deterministic(
    uint8_t uhash_arr[][48],
    uint8_t x_arr[][64]
) {
    memset(uhash_arr, 0, DET_N * 48);
    memset(x_arr,     0, DET_N * 64);
    int i = 0;

    // case 0: uhash=0, x=random → out = x1
    rand_bytes(x_arr[i], 64); i++;
    // case 1: x0=0, uhash=random, x1=random → out = x1
    rand_bytes(uhash_arr[i], 48); rand_bytes(x_arr[i] + 16, 48); i++;
    // case 2: x1=0, uhash=random, x0=random → out = mul
    rand_bytes(uhash_arr[i], 48); rand_bytes(x_arr[i], 16); i++;
    // case 3: x0=1, x1=0 → out = uhash
    rand_bytes(uhash_arr[i], 48); x_arr[i][0] = 0x01; i++;
    // case 4: uhash bit383, x0 bit1, x1=0 → reduction into result
    uhash_arr[i][47] = 0x80; x_arr[i][0] = 0x02; i++;
    // case 5: x1=all-ff → final XOR test
    rand_bytes(uhash_arr[i], 48); rand_bytes(x_arr[i], 16);
    memset(x_arr[i] + 16, 0xff, 48); i++;
    // case 6: all zero
    i++;
    // case 7: uhash=all-ff, x=all-ff
    memset(uhash_arr[i], 0xff, 48); memset(x_arr[i], 0xff, 64); i++;
    // case 8~9: random pairs
    rand_bytes(uhash_arr[i], 48); rand_bytes(x_arr[i], 64); i++;
    rand_bytes(uhash_arr[i], 48); rand_bytes(x_arr[i], 64); i++;
    // case 10: x0 bit 63 only (bf128 limb 0 MSB), x1 random
    rand_bytes(uhash_arr[i], 48); x_arr[i][7] = 0x80;
    rand_bytes(x_arr[i] + 16, 48); i++;
    // case 11: x0 bit 64 only (bf128 limb 1 LSB), x1 random
    rand_bytes(uhash_arr[i], 48); x_arr[i][8] = 0x01;
    rand_bytes(x_arr[i] + 16, 48); i++;
    // case 12: x0 bit 127 only (bf128 MSB), x1 random
    rand_bytes(uhash_arr[i], 48); x_arr[i][15] = 0x80;
    rand_bytes(x_arr[i] + 16, 48); i++;
    // case 13: x1 bit 0 only
    rand_bytes(uhash_arr[i], 48); rand_bytes(x_arr[i], 16);
    x_arr[i][16] = 0x01; i++;
    // case 14: x1 bit 383 only (bf384 MSB)
    rand_bytes(uhash_arr[i], 48); rand_bytes(x_arr[i], 16);
    x_arr[i][63] = 0x80; i++;
    // case 15: uhash all-ff, x0=1, x1=random
    memset(uhash_arr[i], 0xff, 48); x_arr[i][0] = 0x01;
    rand_bytes(x_arr[i] + 16, 48); i++;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(void) {
    printf("=== Layer 2: leaf_hash_128 CPU vs GPU unit test ===\n");
    printf("PRNG seed: 0x123456789abcdef0\n\n");

    int total_fail = 0;

    printf("[1] Deterministic cases (%d)...\n", DET_N);
    uint8_t det_uhash[DET_N][48];
    uint8_t det_x[DET_N][64];
    build_deterministic(det_uhash, det_x);
    int det_fail = run_tests((uint8_t*)det_uhash, (uint8_t*)det_x, DET_N, "det");
    if (det_fail == 0) printf("  PASS (%d cases)\n", DET_N);
    total_fail += det_fail;

    printf("[2] Random cases (1000)...\n");
    int RAND_N = 1000;
    uint8_t* rnd_uhash = (uint8_t*)malloc(RAND_N * 48);
    HOST_CHECK_ALLOC(rnd_uhash);
    uint8_t* rnd_x = (uint8_t*)malloc(RAND_N * 64);
    HOST_CHECK_ALLOC(rnd_x);
    rand_bytes(rnd_uhash, RAND_N * 48);
    rand_bytes(rnd_x,     RAND_N * 64);
    int rnd_fail = run_tests(rnd_uhash, rnd_x, RAND_N, "rand");
    if (rnd_fail == 0) printf("  PASS (1000 cases)\n");
    total_fail += rnd_fail;
    free(rnd_uhash); free(rnd_x);

    printf("\n=== RESULT: %s (%d failure(s)) ===\n",
           total_fail == 0 ? "PASS" : "FAIL", total_fail);
    return total_fail == 0 ? 0 : 1;
}
