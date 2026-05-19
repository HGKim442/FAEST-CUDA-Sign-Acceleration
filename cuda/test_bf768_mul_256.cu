#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include "fields_cuda.cuh"
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

__global__
void test_bf768_mul_256_kernel(
    const bf768_cuda_t* lhs,
    const bf256_cuda_t* rhs,
    bf768_cuda_t*       out,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        out[tid] = bf768_mul_256_device(lhs[tid], rhs[tid]);
}

static bf768_cuda_t pack_bf768(const uint8_t src[96]) {
    bf768_cuda_t r; memcpy(r.v, src, 96); return r;
}
static bf256_cuda_t pack_bf256(const uint8_t src[32]) {
    bf256_cuda_t r; memcpy(r.v, src, 32); return r;
}
static void unpack_bf768(const bf768_cuda_t* src, uint8_t dst[96]) {
    memcpy(dst, src->v, 96);
}

static int run_one(int idx,
                   const uint8_t lhs[96],
                   const uint8_t rhs[32])
{
    uint8_t cpu_out[96];
    cpu_bf768_mul_256_ref(cpu_out, lhs, rhs);

    bf768_cuda_t h_lhs = pack_bf768(lhs);
    bf256_cuda_t h_rhs = pack_bf256(rhs);

    bf768_cuda_t* d_lhs; bf256_cuda_t* d_rhs; bf768_cuda_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_lhs, sizeof(bf768_cuda_t)));
    CUDA_CHECK(cudaMalloc(&d_rhs, sizeof(bf256_cuda_t)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(bf768_cuda_t)));
    CUDA_CHECK(cudaMemcpy(d_lhs, &h_lhs, sizeof(bf768_cuda_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, &h_rhs, sizeof(bf256_cuda_t), cudaMemcpyHostToDevice));

    test_bf768_mul_256_kernel<<<1, 1>>>(d_lhs, d_rhs, d_out, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bf768_cuda_t h_out;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(bf768_cuda_t), cudaMemcpyDeviceToHost));
    cudaFree(d_lhs); cudaFree(d_rhs); cudaFree(d_out);

    uint8_t gpu_out[96];
    unpack_bf768(&h_out, gpu_out);

    if (memcmp(cpu_out, gpu_out, 96) != 0) {
        fprintf(stderr, "FAIL [%d]\n  lhs: ", idx);
        for (int i = 0; i < 96; i++) fprintf(stderr, "%02x", lhs[i]);
        fprintf(stderr, "\n  rhs: ");
        for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", rhs[i]);
        fprintf(stderr, "\n  cpu: ");
        for (int i = 0; i < 96; i++) fprintf(stderr, "%02x", cpu_out[i]);
        fprintf(stderr, "\n  gpu: ");
        for (int i = 0; i < 96; i++) fprintf(stderr, "%02x", gpu_out[i]);
        fprintf(stderr, "\n");
        return 1;
    }
    return 0;
}

int main(void) {
    printf("=== Layer 1 (Step 9-1): bf768_mul_256 CPU vs GPU unit test ===\n");
    xorshift64_seed(UINT64_C(0xfedcba9876543210));

    int failures = 0;
    int test_idx = 0;

    printf("[1] Deterministic cases...\n");

    uint8_t lhs[96], rhs[32];
    uint8_t zero96[96], zero32[32];
    memset(zero96, 0, 96); memset(zero32, 0, 32);

    // 0: both zero
    failures += run_one(test_idx++, zero96, zero32);
    // 1: lhs=0, rhs=1
    memset(rhs, 0, 32); rhs[0] = 1;
    failures += run_one(test_idx++, zero96, rhs);
    // 2: lhs=1, rhs=1
    memset(lhs, 0, 96); lhs[0] = 1;
    failures += run_one(test_idx++, lhs, rhs);
    // 3: all-0xff
    memset(lhs, 0xff, 96); memset(rhs, 0xff, 32);
    failures += run_one(test_idx++, lhs, rhs);
    // 4: lhs bit 767 (MSB), rhs=1 → reduction path
    memset(lhs, 0, 96); lhs[95] = 0x80;
    memset(rhs, 0, 32); rhs[0] = 1;
    failures += run_one(test_idx++, lhs, rhs);
    // 5: lhs bit 767, rhs bit 1
    memset(rhs, 0, 32); rhs[0] = 2;
    failures += run_one(test_idx++, lhs, rhs);
    // 6: lhs bit 0, rhs all-0xff
    memset(lhs, 0, 96); lhs[0] = 1;
    memset(rhs, 0xff, 32);
    failures += run_one(test_idx++, lhs, rhs);
    // 7: lhs bit 63 (limb 0 MSB)
    memset(lhs, 0, 96); lhs[7] = 0x80;
    memset(rhs, 0, 32); rhs[0] = 1;
    failures += run_one(test_idx++, lhs, rhs);
    // 8: lhs bit 64 (limb 1 LSB)
    memset(lhs, 0, 96); lhs[8] = 0x01;
    failures += run_one(test_idx++, lhs, rhs);
    // 9: rhs bit 255 (last loop iteration)
    rand_bytes(lhs, 96);
    memset(rhs, 0, 32); rhs[31] = 0x80;
    failures += run_one(test_idx++, lhs, rhs);
    // 10: rhs bit 191 (limb 2 MSB of rhs)
    memset(rhs, 0, 32); rhs[23] = 0x80;
    failures += run_one(test_idx++, lhs, rhs);
    // 11: rhs bit 64 (limb 1 LSB)
    memset(rhs, 0, 32); rhs[8] = 0x01;
    failures += run_one(test_idx++, lhs, rhs);
    // 12: last valid byte of lhs (byte 95) nonzero, rhs=1
    memset(lhs, 0, 96); lhs[95] = 0xCD;
    memset(rhs, 0, 32); rhs[0] = 1;
    failures += run_one(test_idx++, lhs, rhs);
    // 13: last valid byte of rhs (byte 31) nonzero
    rand_bytes(lhs, 96);
    memset(rhs, 0, 32); rhs[31] = 0xAB;
    failures += run_one(test_idx++, lhs, rhs);

    printf("  Deterministic: %d case(s), %d failure(s)\n",
           test_idx, failures);

    printf("[2] Random cases (1000)...\n");
    int rand_failures = 0;
    for (int i = 0; i < 1000; ++i) {
        rand_bytes(lhs, 96);
        rand_bytes(rhs, 32);
        rand_failures += run_one(test_idx++, lhs, rhs);
    }
    failures += rand_failures;
    printf("  Random: 1000 case(s), %d failure(s)\n", rand_failures);

    if (failures == 0)
        printf("=== RESULT: PASS (0 failure(s)) ===\n");
    else
        printf("=== RESULT: FAIL (%d failure(s)) ===\n", failures);

    return failures > 0 ? 1 : 0;
}
