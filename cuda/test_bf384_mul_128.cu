#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "cuda/fields_cuda.cuh"
#include "cuda/cpu_ref_bf384.h"

// ---------------------------------------------------------------------------
// CUDA error check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err__ = (call);                                         \
    if (err__ != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err__));         \
        exit(1);                                                        \
    }                                                                   \
} while (0)

// ---------------------------------------------------------------------------
// Host allocation check macro
// ---------------------------------------------------------------------------
#define HOST_CHECK_ALLOC(ptr) do {                                      \
    if ((ptr) == NULL) {                                                \
        fprintf(stderr, "Host allocation failed: %s\n", #ptr);         \
        exit(1);                                                        \
    }                                                                   \
} while (0)

// ---------------------------------------------------------------------------
// Simple xorshift64 PRNG for reproducibility
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
// GPU test kernel: 1 thread = 1 bf384_mul_128_device call
// ---------------------------------------------------------------------------
__global__ void test_kernel(
    const bf384_cuda_t* lhs,
    const bf128_cuda_t* rhs,
    bf384_cuda_t*       out,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = bf384_mul_128_device(lhs[tid], rhs[tid]);
    }
}

// ---------------------------------------------------------------------------
// Convert raw bytes to cuda types (little-endian)
// ---------------------------------------------------------------------------
static bf384_cuda_t bytes_to_bf384(const uint8_t src[48]) {
    bf384_cuda_t r;
    memcpy(r.v, src, 48);
    return r;
}

static bf128_cuda_t bytes_to_bf128(const uint8_t src[16]) {
    bf128_cuda_t r;
    memcpy(r.v, src, 16);
    return r;
}

static void bf384_to_bytes(uint8_t dst[48], const bf384_cuda_t* src) {
    memcpy(dst, src->v, 48);
}

// ---------------------------------------------------------------------------
// Print helpers for failure output
// ---------------------------------------------------------------------------
static void print_hex(const char* label, const uint8_t* buf, size_t n) {
    printf("  %s: ", label);
    for (size_t i = 0; i < n; ++i) printf("%02x", buf[i]);
    printf("\n");
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
#define BLOCK_SIZE 256

static int run_tests(
    const uint8_t* all_lhs,
    const uint8_t* all_rhs,
    int N,
    const char* label
) {
    // CPU reference
    uint8_t* cpu_out = (uint8_t*)malloc(N * 48);
    HOST_CHECK_ALLOC(cpu_out);
    for (int i = 0; i < N; ++i) {
        cpu_bf384_mul_128_ref(cpu_out + i * 48,
                              all_lhs + i * 48,
                              all_rhs + i * 16);
    }

    // Pack host arrays
    bf384_cuda_t* h_lhs = (bf384_cuda_t*)malloc(N * sizeof(bf384_cuda_t));
    HOST_CHECK_ALLOC(h_lhs);
    bf128_cuda_t* h_rhs = (bf128_cuda_t*)malloc(N * sizeof(bf128_cuda_t));
    HOST_CHECK_ALLOC(h_rhs);
    bf384_cuda_t* h_out = (bf384_cuda_t*)malloc(N * sizeof(bf384_cuda_t));
    HOST_CHECK_ALLOC(h_out);

    for (int i = 0; i < N; ++i) {
        h_lhs[i] = bytes_to_bf384(all_lhs + i * 48);
        h_rhs[i] = bytes_to_bf128(all_rhs + i * 16);
    }

    // GPU memory
    bf384_cuda_t *d_lhs, *d_out;
    bf128_cuda_t *d_rhs;
    CUDA_CHECK(cudaMalloc((void**)&d_lhs, N * sizeof(bf384_cuda_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_rhs, N * sizeof(bf128_cuda_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(bf384_cuda_t)));

    CUDA_CHECK(cudaMemcpy(d_lhs, h_lhs, N * sizeof(bf384_cuda_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, h_rhs, N * sizeof(bf128_cuda_t), cudaMemcpyHostToDevice));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    test_kernel<<<grid, BLOCK_SIZE>>>(d_lhs, d_rhs, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(bf384_cuda_t), cudaMemcpyDeviceToHost));

    // byte-by-byte compare
    int failures = 0;
    for (int i = 0; i < N; ++i) {
        uint8_t gpu_bytes[48];
        bf384_to_bytes(gpu_bytes, &h_out[i]);
        if (memcmp(cpu_out + i * 48, gpu_bytes, 48) != 0) {
            printf("  [FAIL] %s case %d\n", label, i);
            print_hex("lhs", all_lhs + i * 48, 48);
            print_hex("rhs", all_rhs + i * 16, 16);
            print_hex("cpu", cpu_out + i * 48, 48);
            print_hex("gpu", gpu_bytes, 48);
            ++failures;
            if (failures >= 5) {
                printf("  ... (stopping at 5 failures)\n");
                break;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_lhs));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_out));
    free(h_lhs); free(h_rhs); free(h_out); free(cpu_out);
    return failures;
}

// ---------------------------------------------------------------------------
// Build deterministic test cases
// ---------------------------------------------------------------------------
#define DET_N 22

static void build_deterministic(uint8_t lhs_arr[][48], uint8_t rhs_arr[][16]) {
    memset(lhs_arr, 0, DET_N * 48);
    memset(rhs_arr, 0, DET_N * 16);

    int i = 0;

    // case 0: lhs=0, rhs=random
    rand_bytes(rhs_arr[i], 16); i++;
    // case 1: lhs=random, rhs=0
    rand_bytes(lhs_arr[i], 48); i++;
    // case 2: rhs=1 (result should equal lhs)
    rand_bytes(lhs_arr[i], 48); rhs_arr[i][0] = 0x01; i++;
    // case 3: lhs bit 383 set, rhs random (forces reduction)
    lhs_arr[i][47] = 0x80; rand_bytes(rhs_arr[i], 16); i++;
    // case 4: lhs all-ff, rhs all-ff
    memset(lhs_arr[i], 0xff, 48); memset(rhs_arr[i], 0xff, 16); i++;
    // case 5: lhs bit 0 only
    lhs_arr[i][0] = 0x01; rand_bytes(rhs_arr[i], 16); i++;
    // case 6: lhs bit 63 only (limb 0 MSB)
    lhs_arr[i][7] = 0x80; rand_bytes(rhs_arr[i], 16); i++;
    // case 7: lhs bit 64 only (limb 1 LSB)
    lhs_arr[i][8] = 0x01; rand_bytes(rhs_arr[i], 16); i++;
    // case 8: lhs bit 383 only, rhs bit 1 only (forces reduction into result)
    lhs_arr[i][47] = 0x80; rhs_arr[i][0] = 0x02; i++;
    // case 9: rhs bit 1 only
    rand_bytes(lhs_arr[i], 48); rhs_arr[i][0] = 0x02; i++;
    // case 10: rhs bit 63 only (limb 0 MSB)
    rand_bytes(lhs_arr[i], 48); rhs_arr[i][7] = 0x80; i++;
    // case 11: rhs bit 64 only (limb 1 LSB)
    rand_bytes(lhs_arr[i], 48); rhs_arr[i][8] = 0x01; i++;
    // case 12: rhs bit 127 only (last loop iteration)
    rand_bytes(lhs_arr[i], 48); rhs_arr[i][15] = 0x80; i++;
    // case 13: rhs all-ff, lhs random
    rand_bytes(lhs_arr[i], 48); memset(rhs_arr[i], 0xff, 16); i++;
    // case 14: lhs bit 64, rhs bit 63 (cross-limb boundary)
    lhs_arr[i][8] = 0x01; rhs_arr[i][7] = 0x80; i++;
    // case 15: lhs all-ff, rhs=1
    memset(lhs_arr[i], 0xff, 48); rhs_arr[i][0] = 0x01; i++;
    // case 16~21: random pairs
    for (; i < DET_N; ++i) {
        rand_bytes(lhs_arr[i], 48);
        rand_bytes(rhs_arr[i], 16);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(void) {
    printf("=== Layer 1: bf384_mul_128 CPU vs GPU unit test ===\n");
    printf("PRNG seed: 0x123456789abcdef0\n\n");

    int total_fail = 0;

    // Deterministic tests
    printf("[1] Deterministic cases (%d)...\n", DET_N);
    uint8_t det_lhs[DET_N][48];
    uint8_t det_rhs[DET_N][16];
    build_deterministic(det_lhs, det_rhs);
    int det_fail = run_tests((uint8_t*)det_lhs, (uint8_t*)det_rhs, DET_N, "det");
    if (det_fail == 0) printf("  PASS (%d cases)\n", DET_N);
    total_fail += det_fail;

    // Random tests
    printf("[2] Random cases (1000)...\n");
    int RAND_N = 1000;
    uint8_t* rnd_lhs = (uint8_t*)malloc(RAND_N * 48);
    HOST_CHECK_ALLOC(rnd_lhs);
    uint8_t* rnd_rhs = (uint8_t*)malloc(RAND_N * 16);
    HOST_CHECK_ALLOC(rnd_rhs);
    rand_bytes(rnd_lhs, RAND_N * 48);
    rand_bytes(rnd_rhs, RAND_N * 16);
    int rnd_fail = run_tests(rnd_lhs, rnd_rhs, RAND_N, "rand");
    if (rnd_fail == 0) printf("  PASS (1000 cases)\n");
    total_fail += rnd_fail;
    free(rnd_lhs); free(rnd_rhs);

    printf("\n=== RESULT: %s (%d failure(s)) ===\n",
           total_fail == 0 ? "PASS" : "FAIL", total_fail);
    return total_fail == 0 ? 0 : 1;
}
