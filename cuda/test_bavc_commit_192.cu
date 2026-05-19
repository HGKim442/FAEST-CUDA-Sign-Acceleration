#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "instances.h"
#include "bavc.h"

#define HOST_CHECK_ALLOC(ptr) do {                                      \
    if ((ptr) == NULL) {                                                \
        fprintf(stderr, "Host allocation failed: %s\n", #ptr);         \
        exit(1);                                                        \
    }                                                                   \
} while (0)

// ---------------------------------------------------------------------------
// xorshift64 PRNG
// ---------------------------------------------------------------------------
static uint64_t xorshift64_state = UINT64_C(0x192192192192192A);

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
// Compare two byte arrays, report first mismatch
// ---------------------------------------------------------------------------
static int compare_bytes(const char* name, const uint8_t* cpu,
                          const uint8_t* gpu, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        if (cpu[i] != gpu[i]) {
            printf("  [FAIL] %s: first mismatch at offset %zu"
                   " — cpu=0x%02x gpu=0x%02x\n",
                   name, i, cpu[i], gpu[i]);
            return 1;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(void) {
    printf("=== Layer 3 (Step 9-3A): bavc_commit_faest_192 CPU vs GPU test ===\n");
    printf("PRNG seed: 0x192192192192192A\n\n");

    const faest_paramset_t* params = faest_get_paramset(FAEST_192S);
    if (!params) {
        fprintf(stderr, "faest_get_paramset(FAEST_192S) returned NULL\n");
        return 1;
    }

    const unsigned int lambda_bytes = params->lambda / 8;
    const unsigned int L            = params->L;
    const unsigned int com_size     = lambda_bytes * 3;

    const size_t h_len   = (size_t)lambda_bytes * 2;
    const size_t com_len = (size_t)L * com_size;
    const size_t sd_len  = (size_t)L * lambda_bytes;

    int total_fail = 0;

    for (int trial = 0; trial < 3; ++trial) {
        printf("[trial %d]\n", trial);

        uint8_t root_key[MAX_LAMBDA_BYTES];
        uint8_t iv[IV_SIZE];
        rand_bytes(root_key, lambda_bytes);
        rand_bytes(iv, IV_SIZE);

        /* CPU path */
        bavc_t cpu_bavc = {0};
        bavc_commit(&cpu_bavc, root_key, iv, params);

        /* GPU path */
        bavc_t gpu_bavc = {0};
        bool ok = bavc_commit_faest_192_cuda_test(&gpu_bavc, root_key, iv, params);
        if (!ok) {
            printf("  [FAIL] bavc_commit_faest_192_cuda_test returned false\n");
            bavc_clear(&cpu_bavc);
            bavc_clear(&gpu_bavc);
            ++total_fail;
            continue;
        }

        /* Compare h, com, sd */
        int fail = 0;
        fail += compare_bytes("h",   cpu_bavc.h,   gpu_bavc.h,   h_len);
        fail += compare_bytes("com", cpu_bavc.com, gpu_bavc.com, com_len);
        fail += compare_bytes("sd",  cpu_bavc.sd,  gpu_bavc.sd,  sd_len);

        if (fail == 0) {
            printf("  PASS (h=%zuB com=%zuB sd=%zuB)\n",
                   h_len, com_len, sd_len);
        }
        total_fail += fail;

        bavc_clear(&cpu_bavc);
        bavc_clear(&gpu_bavc);
    }

    printf("\n=== RESULT: %s (%d failure(s)) ===\n",
           total_fail == 0 ? "PASS" : "FAIL", total_fail);
    return total_fail == 0 ? 0 : 1;
}
