#include "cuda/leaf_hash_cuda.cuh"
#include "cuda/fields_cuda.cuh"

#include <stdio.h>
#include <cuda_runtime.h>

#ifdef CUDA_TIMING
#include <time.h>
static inline long long lh_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}
#endif

// ---------------------------------------------------------------------------
// Internal error check: sets status=-1 and jumps to cleanup on failure
// ---------------------------------------------------------------------------
#define CUDA_GOTO_IF_ERROR(call) do {                                   \
    cudaError_t err__ = (call);                                         \
    if (err__ != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err__));         \
        status = -1; goto cleanup;                                      \
    }                                                                   \
} while (0)

// ---------------------------------------------------------------------------
// Kernel: 1 thread = 1 leaf_hash_128 call
// h[tid] = bf384_mul_128(uhash[tid], x0[tid]) XOR x1[tid]
// ---------------------------------------------------------------------------
#define BLOCK_SIZE 128

__global__ void leaf_hash_128_batch_kernel(
    const bf384_cuda_t* uhash,
    const bf128_cuda_t* x0,
    const bf384_cuda_t* x1,
    bf384_cuda_t*       out,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        bf384_cuda_t mul = bf384_mul_128_device(uhash[tid], x0[tid]);
        out[tid] = bf384_add_device(mul, x1[tid]);
    }
}

// ---------------------------------------------------------------------------
// C-callable wrapper
// ---------------------------------------------------------------------------
extern "C"
int leaf_hash_128_batch_cuda(
    const uint64_t* h_uhash,
    const uint64_t* h_x0,
    const uint64_t* h_x1,
    uint64_t*       h_out,
    int N
) {
    if (N < 0) return -1;
    if (N == 0) return 0;
    if (!h_uhash || !h_x0 || !h_x1 || !h_out) return -1;

    const bf384_cuda_t* uhash = (const bf384_cuda_t*)h_uhash;
    const bf128_cuda_t* x0    = (const bf128_cuda_t*)h_x0;
    const bf384_cuda_t* x1    = (const bf384_cuda_t*)h_x1;

    int status = 0;
    bf384_cuda_t *d_uhash = NULL, *d_x1 = NULL, *d_out = NULL;
    bf128_cuda_t *d_x0 = NULL;

#ifdef CUDA_TIMING
    /* Declare all timing variables at top to avoid goto-bypass errors */
    long long t_wrapper_start = 0, t_malloc_end = 0,
              t_free_start = 0, t_wrapper_end = 0;
    float h2d_ms = 0, kern_ms = 0, d2h_ms = 0;
    cudaEvent_t e_h2d_start, e_h2d_end, e_kern_start, e_kern_end,
                e_d2h_start, e_d2h_end;
    cudaEventCreate(&e_h2d_start); cudaEventCreate(&e_h2d_end);
    cudaEventCreate(&e_kern_start); cudaEventCreate(&e_kern_end);
    cudaEventCreate(&e_d2h_start); cudaEventCreate(&e_d2h_end);
    t_wrapper_start = lh_now_us();
#endif

    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_uhash, N * sizeof(bf384_cuda_t)));
    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_x0,    N * sizeof(bf128_cuda_t)));
    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_x1,    N * sizeof(bf384_cuda_t)));
    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_out,   N * sizeof(bf384_cuda_t)));

#ifdef CUDA_TIMING
    t_malloc_end = lh_now_us();
    cudaEventRecord(e_h2d_start);
#endif

    CUDA_GOTO_IF_ERROR(cudaMemcpy(d_uhash, uhash, N * sizeof(bf384_cuda_t), cudaMemcpyHostToDevice));
    CUDA_GOTO_IF_ERROR(cudaMemcpy(d_x0,    x0,    N * sizeof(bf128_cuda_t), cudaMemcpyHostToDevice));
    CUDA_GOTO_IF_ERROR(cudaMemcpy(d_x1,    x1,    N * sizeof(bf384_cuda_t), cudaMemcpyHostToDevice));

    {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
#ifdef CUDA_TIMING
        cudaEventRecord(e_h2d_end);
        cudaEventRecord(e_kern_start);
#endif
        leaf_hash_128_batch_kernel<<<grid, BLOCK_SIZE>>>(d_uhash, d_x0, d_x1, d_out, N);
        CUDA_GOTO_IF_ERROR(cudaGetLastError());
        CUDA_GOTO_IF_ERROR(cudaDeviceSynchronize());
#ifdef CUDA_TIMING
        cudaEventRecord(e_kern_end);
        cudaEventRecord(e_d2h_start);
#endif
    }

    CUDA_GOTO_IF_ERROR(cudaMemcpy((bf384_cuda_t*)h_out, d_out,
                                   N * sizeof(bf384_cuda_t), cudaMemcpyDeviceToHost));
#ifdef CUDA_TIMING
    cudaEventRecord(e_d2h_end);
    cudaEventSynchronize(e_d2h_end);
#endif

cleanup:
#ifdef CUDA_TIMING
    t_free_start = lh_now_us();
#endif
    if (d_uhash) cudaFree(d_uhash);
    if (d_x0)    cudaFree(d_x0);
    if (d_x1)    cudaFree(d_x1);
    if (d_out)   cudaFree(d_out);
#ifdef CUDA_TIMING
    t_wrapper_end = lh_now_us();
    if (status == 0) {
        cudaEventElapsedTime(&h2d_ms,  e_h2d_start,  e_h2d_end);
        cudaEventElapsedTime(&kern_ms, e_kern_start, e_kern_end);
        cudaEventElapsedTime(&d2h_ms,  e_d2h_start,  e_d2h_end);
    }
    fprintf(stderr,
        "[CUDA_TIMING_WRAPPER] N=%d"
        " malloc_wall_us=%lld"
        " h2d_us=%.1f kern_us=%.1f d2h_us=%.1f"
        " free_wall_us=%lld wrapper_wall_us=%lld\n",
        N,
        t_malloc_end    - t_wrapper_start,
        h2d_ms * 1000.0f, kern_ms * 1000.0f, d2h_ms * 1000.0f,
        t_wrapper_end   - t_free_start,
        t_wrapper_end   - t_wrapper_start);
    cudaEventDestroy(e_h2d_start); cudaEventDestroy(e_h2d_end);
    cudaEventDestroy(e_kern_start); cudaEventDestroy(e_kern_end);
    cudaEventDestroy(e_d2h_start); cudaEventDestroy(e_d2h_end);
#endif
    return status;
}

// ---------------------------------------------------------------------------
// Kernel: 1 thread = 1 leaf_hash_192 call
// h[tid] = bf576_mul_192(uhash[tid], x0[tid]) XOR x1[tid]
// ---------------------------------------------------------------------------

__global__ void leaf_hash_192_batch_kernel(
    const bf576_cuda_t* uhash,
    const bf192_cuda_t* x0,
    const bf576_cuda_t* x1,
    bf576_cuda_t*       out,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        bf576_cuda_t mul = bf576_mul_192_device(uhash[tid], x0[tid]);
        out[tid] = bf576_add_device(mul, x1[tid]);
    }
}

// ---------------------------------------------------------------------------
// C-callable wrapper for leaf_hash_192
// ---------------------------------------------------------------------------
extern "C"
int leaf_hash_192_batch_cuda(
    const uint64_t* h_uhash,
    const uint64_t* h_x0,
    const uint64_t* h_x1,
    uint64_t*       h_out,
    int N
) {
    if (N < 0) return -1;
    if (N == 0) return 0;
    if (!h_uhash || !h_x0 || !h_x1 || !h_out) return -1;

    const bf576_cuda_t* uhash = (const bf576_cuda_t*)h_uhash;
    const bf192_cuda_t* x0    = (const bf192_cuda_t*)h_x0;
    const bf576_cuda_t* x1    = (const bf576_cuda_t*)h_x1;

    int status = 0;
    bf576_cuda_t *d_uhash = NULL, *d_x1 = NULL, *d_out = NULL;
    bf192_cuda_t *d_x0 = NULL;

    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_uhash, N * sizeof(bf576_cuda_t)));
    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_x0,    N * sizeof(bf192_cuda_t)));
    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_x1,    N * sizeof(bf576_cuda_t)));
    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_out,   N * sizeof(bf576_cuda_t)));

    CUDA_GOTO_IF_ERROR(cudaMemcpy(d_uhash, uhash, N * sizeof(bf576_cuda_t), cudaMemcpyHostToDevice));
    CUDA_GOTO_IF_ERROR(cudaMemcpy(d_x0,    x0,    N * sizeof(bf192_cuda_t), cudaMemcpyHostToDevice));
    CUDA_GOTO_IF_ERROR(cudaMemcpy(d_x1,    x1,    N * sizeof(bf576_cuda_t), cudaMemcpyHostToDevice));

    {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        leaf_hash_192_batch_kernel<<<grid, BLOCK_SIZE>>>(d_uhash, d_x0, d_x1, d_out, N);
        CUDA_GOTO_IF_ERROR(cudaGetLastError());
        CUDA_GOTO_IF_ERROR(cudaDeviceSynchronize());
    }

    CUDA_GOTO_IF_ERROR(cudaMemcpy((bf576_cuda_t*)h_out, d_out,
                                   N * sizeof(bf576_cuda_t), cudaMemcpyDeviceToHost));

cleanup:
    if (d_uhash) cudaFree(d_uhash);
    if (d_x0)    cudaFree(d_x0);
    if (d_x1)    cudaFree(d_x1);
    if (d_out)   cudaFree(d_out);
    return status;
}

// ---------------------------------------------------------------------------
// Kernel: 1 thread = 1 leaf_hash_256 call
// h[tid] = bf768_mul_256(uhash[tid], x0[tid]) XOR x1[tid]
// ---------------------------------------------------------------------------

__global__ void leaf_hash_256_batch_kernel(
    const bf768_cuda_t* uhash,
    const bf256_cuda_t* x0,
    const bf768_cuda_t* x1,
    bf768_cuda_t*       out,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        bf768_cuda_t mul = bf768_mul_256_device(uhash[tid], x0[tid]);
        out[tid] = bf768_add_device(mul, x1[tid]);
    }
}

// ---------------------------------------------------------------------------
// C-callable wrapper for leaf_hash_256
// ---------------------------------------------------------------------------
extern "C"
int leaf_hash_256_batch_cuda(
    const uint64_t* h_uhash,
    const uint64_t* h_x0,
    const uint64_t* h_x1,
    uint64_t*       h_out,
    int N
) {
    if (N < 0) return -1;
    if (N == 0) return 0;
    if (!h_uhash || !h_x0 || !h_x1 || !h_out) return -1;

    const bf768_cuda_t* uhash = (const bf768_cuda_t*)h_uhash;
    const bf256_cuda_t* x0    = (const bf256_cuda_t*)h_x0;
    const bf768_cuda_t* x1    = (const bf768_cuda_t*)h_x1;

    int status = 0;
    bf768_cuda_t *d_uhash = NULL, *d_x1 = NULL, *d_out = NULL;
    bf256_cuda_t *d_x0 = NULL;

    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_uhash, N * sizeof(bf768_cuda_t)));
    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_x0,    N * sizeof(bf256_cuda_t)));
    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_x1,    N * sizeof(bf768_cuda_t)));
    CUDA_GOTO_IF_ERROR(cudaMalloc((void**)&d_out,   N * sizeof(bf768_cuda_t)));

    CUDA_GOTO_IF_ERROR(cudaMemcpy(d_uhash, uhash, N * sizeof(bf768_cuda_t), cudaMemcpyHostToDevice));
    CUDA_GOTO_IF_ERROR(cudaMemcpy(d_x0,    x0,    N * sizeof(bf256_cuda_t), cudaMemcpyHostToDevice));
    CUDA_GOTO_IF_ERROR(cudaMemcpy(d_x1,    x1,    N * sizeof(bf768_cuda_t), cudaMemcpyHostToDevice));

    {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        leaf_hash_256_batch_kernel<<<grid, BLOCK_SIZE>>>(d_uhash, d_x0, d_x1, d_out, N);
        CUDA_GOTO_IF_ERROR(cudaGetLastError());
        CUDA_GOTO_IF_ERROR(cudaDeviceSynchronize());
    }

    CUDA_GOTO_IF_ERROR(cudaMemcpy((bf768_cuda_t*)h_out, d_out,
                                   N * sizeof(bf768_cuda_t), cudaMemcpyDeviceToHost));

cleanup:
    if (d_uhash) cudaFree(d_uhash);
    if (d_x0)    cudaFree(d_x0);
    if (d_x1)    cudaFree(d_x1);
    if (d_out)   cudaFree(d_out);
    return status;
}
