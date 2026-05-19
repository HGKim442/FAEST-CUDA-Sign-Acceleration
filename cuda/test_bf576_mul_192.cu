#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include "fields_cuda.cuh"
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
// xorshift64 PRNG (fixed seed for reproducibility)
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
// GPU kernel: one thread computes one bf576_mul_192
// lhs[N*9], rhs[N*3], out[N*9]  (all uint64_t arrays, limb layout)
// ---------------------------------------------------------------------------

__global__
void test_bf576_mul_192_kernel(
    const bf576_cuda_t* lhs,
    const bf192_cuda_t* rhs,
    bf576_cuda_t*       out,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        out[tid] = bf576_mul_192_device(lhs[tid], rhs[tid]);
}

// ---------------------------------------------------------------------------
// Pack/unpack helpers (NUM_BYTES only — no padding)
// ---------------------------------------------------------------------------

// Pack CPU byte array (72B, BF576_NUM_BYTES) into bf576_cuda_t
static bf576_cuda_t pack_bf576(const uint8_t src[72]) {
    bf576_cuda_t r;
    memcpy(r.v, src, 72);
    return r;
}

// Pack CPU byte array (24B, BF192_NUM_BYTES) into bf192_cuda_t
static bf192_cuda_t pack_bf192(const uint8_t src[24]) {
    bf192_cuda_t r;
    memcpy(r.v, src, 24);
    return r;
}

// Unpack bf576_cuda_t to byte array (72B)
static void unpack_bf576(const bf576_cuda_t* src, uint8_t dst[72]) {
    memcpy(dst, src->v, 72);
}

// ---------------------------------------------------------------------------
// Run a single test case
// ---------------------------------------------------------------------------

static int run_one(int idx,
                   const uint8_t lhs[72],
                   const uint8_t rhs[24],
                   bool inject_padding)
{
    // CPU reference
    uint8_t cpu_out[72];
    cpu_bf576_mul_192_ref(cpu_out, lhs, rhs);

    // Pack for GPU
    bf576_cuda_t h_lhs = pack_bf576(lhs);
    bf192_cuda_t h_rhs = pack_bf192(rhs);

    // If padding injection test: write garbage into bytes 24..31 of a
    // hypothetical 32-byte buffer — but our bf192_cuda_t only has 24B,
    // so the garbage never enters the kernel. Verify CPU ref is unaffected.
    // (The real padding risk is in packing: we use BF192_NUM_BYTES=24 above.)

    bf576_cuda_t* d_lhs; bf192_cuda_t* d_rhs; bf576_cuda_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_lhs, sizeof(bf576_cuda_t)));
    CUDA_CHECK(cudaMalloc(&d_rhs, sizeof(bf192_cuda_t)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(bf576_cuda_t)));
    CUDA_CHECK(cudaMemcpy(d_lhs, &h_lhs, sizeof(bf576_cuda_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, &h_rhs, sizeof(bf192_cuda_t), cudaMemcpyHostToDevice));

    test_bf576_mul_192_kernel<<<1, 1>>>(d_lhs, d_rhs, d_out, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bf576_cuda_t h_out;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(bf576_cuda_t), cudaMemcpyDeviceToHost));
    cudaFree(d_lhs); cudaFree(d_rhs); cudaFree(d_out);

    uint8_t gpu_out[72];
    unpack_bf576(&h_out, gpu_out);

    if (memcmp(cpu_out, gpu_out, 72) != 0) {
        fprintf(stderr, "FAIL [%d]%s\n  lhs: ", idx, inject_padding ? " (padding)" : "");
        for (int i = 0; i < 72; i++) fprintf(stderr, "%02x", lhs[i]);
        fprintf(stderr, "\n  rhs: ");
        for (int i = 0; i < 24; i++) fprintf(stderr, "%02x", rhs[i]);
        fprintf(stderr, "\n  cpu: ");
        for (int i = 0; i < 72; i++) fprintf(stderr, "%02x", cpu_out[i]);
        fprintf(stderr, "\n  gpu: ");
        for (int i = 0; i < 72; i++) fprintf(stderr, "%02x", gpu_out[i]);
        fprintf(stderr, "\n");
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void) {
    printf("=== Layer 1 (Step 9-1): bf576_mul_192 CPU vs GPU unit test ===\n");
    xorshift64_seed(UINT64_C(0x123456789abcdef0));

    int failures = 0;
    int test_idx = 0;

    // --- Deterministic cases ---
    printf("[1] Deterministic cases...\n");

    uint8_t lhs[72], rhs[24], zero72[72], zero24[24];
    memset(zero72, 0, 72); memset(zero24, 0, 24);

    // 0: lhs=0, rhs=0
    failures += run_one(test_idx++, zero72, zero24, false);
    // 1: lhs=0, rhs=1
    memset(rhs, 0, 24); rhs[0] = 1;
    failures += run_one(test_idx++, zero72, rhs, false);
    // 2: lhs=1, rhs=1 → out=1
    memset(lhs, 0, 72); lhs[0] = 1;
    memset(rhs, 0, 24); rhs[0] = 1;
    failures += run_one(test_idx++, lhs, rhs, false);
    // 3: lhs all-0xff, rhs all-0xff
    memset(lhs, 0xff, 72); memset(rhs, 0xff, 24);
    failures += run_one(test_idx++, lhs, rhs, false);
    // 4: lhs bit 575 (MSB) set, rhs=1 → tests reduction path
    memset(lhs, 0, 72); lhs[71] = 0x80;
    memset(rhs, 0, 24); rhs[0] = 1;
    failures += run_one(test_idx++, lhs, rhs, false);
    // 5: lhs bit 575, rhs bit 1 → reduction + accumulate
    memset(rhs, 0, 24); rhs[0] = 2;
    failures += run_one(test_idx++, lhs, rhs, false);
    // 6: lhs bit 0 (LSB), rhs all-0xff
    memset(lhs, 0, 72); lhs[0] = 1;
    memset(rhs, 0xff, 24);
    failures += run_one(test_idx++, lhs, rhs, false);
    // 7: lhs bit 63 (limb 0 MSB), rhs=1 — limb boundary
    memset(lhs, 0, 72); lhs[7] = 0x80;
    memset(rhs, 0, 24); rhs[0] = 1;
    failures += run_one(test_idx++, lhs, rhs, false);
    // 8: lhs bit 64 (limb 1 LSB), rhs=1 — limb boundary
    memset(lhs, 0, 72); lhs[8] = 0x01;
    failures += run_one(test_idx++, lhs, rhs, false);
    // 9: rhs bit 191 (last loop iteration)
    memset(lhs, 0, 72); rand_bytes(lhs, 72);
    memset(rhs, 0, 24); rhs[23] = 0x80;
    failures += run_one(test_idx++, lhs, rhs, false);
    // 10: rhs bit 23 (limb boundary in rhs)
    memset(rhs, 0, 24); rhs[2] = 0x80;
    failures += run_one(test_idx++, lhs, rhs, false);
    // 11: rhs bit 64 (limb 1 LSB of rhs)
    memset(rhs, 0, 24); rhs[8] = 0x01;
    failures += run_one(test_idx++, lhs, rhs, false);

    // 12-15: bf192 rhs padding injection test
    // Real bf192 data is 24 bytes; CPU load reads exactly 24B (BF192_NUM_BYTES).
    // We simulate: valid rhs in [0..23], garbage in [24..31] if sizeof were used.
    // Since our pack_bf192 copies only 24B, the garbage never enters GPU.
    // CPU ref also reads only 24B via bf192_load. Both should give same result.
    {
        uint8_t rhs_padded[32];
        rand_bytes(rhs_padded, 32); // fill all 32B including "padding"
        // Only [0..23] is valid bf192 data. run_one takes rhs[24].
        // This confirms packing uses BF192_NUM_BYTES=24 correctly.
        rand_bytes(lhs, 72);
        failures += run_one(test_idx++, lhs, rhs_padded, true);
        rand_bytes(rhs_padded, 32);
        rand_bytes(lhs, 72);
        failures += run_one(test_idx++, lhs, rhs_padded, true);
        // last valid byte of rhs (byte 23) is nonzero
        memset(rhs_padded, 0, 32); rhs_padded[23] = 0xAB;
        rand_bytes(lhs, 72);
        failures += run_one(test_idx++, lhs, rhs_padded, true);
        // last valid byte of lhs (byte 71) is nonzero — bf576 no padding
        memset(lhs, 0, 72); lhs[71] = 0xCD;
        memset(rhs_padded, 0, 32); rhs_padded[0] = 1;
        failures += run_one(test_idx++, lhs, rhs_padded, true);
    }

    printf("  Deterministic: %d case(s), %d failure(s)\n",
           test_idx, failures);

    // --- Random cases ---
    printf("[2] Random cases (1000)...\n");
    int rand_failures = 0;
    for (int i = 0; i < 1000; ++i) {
        rand_bytes(lhs, 72);
        rand_bytes(rhs, 24);
        rand_failures += run_one(test_idx++, lhs, rhs, false);
    }
    failures += rand_failures;
    printf("  Random: 1000 case(s), %d failure(s)\n", rand_failures);

    if (failures == 0)
        printf("=== RESULT: PASS (0 failure(s)) ===\n");
    else
        printf("=== RESULT: FAIL (%d failure(s)) ===\n", failures);

    return failures > 0 ? 1 : 0;
}
