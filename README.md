# faest-cuda

CUDA-based GPU acceleration of the [FAEST](https://faest.info) post-quantum
digital signature scheme.

> **Paper:** Accelerating FAEST Signing on GPU via CUDA Batch Parallelization  
> **Authors:** Ha-Gyeong Kim, Seung-Won Lee, Min-Ho Song, Si-Woo Eum, and Hwa-Jeong Seo  
> **Affiliation:** Hansung University, Seoul, Korea

---

## What this is

FAEST is a post-quantum digital signature scheme based on the
VOLE-in-the-Head (VOLEitH) paradigm. The Sign operation performs a large
number of independent `leaf_hash` calls — GF(2^{3λ}) field multiplications —
which are the dominant computational bottleneck.

This implementation accelerates the Sign path by offloading batched
`leaf_hash` computations in `bavc_commit_faest()` to NVIDIA CUDA.
It does **not** accelerate the Verify path.

End-to-end Sign speedups on an RTX 3060 Laptop GPU (Ampere, CC 8.6):

| Variant    | CPU baseline | CUDA (BS=128) | Speedup |
|------------|-------------|---------------|---------|
| faest_128s | 24.6 ms     | 17.7 ms       | **1.39×** |
| faest_192s | 227.1 ms    | 107.2 ms      | **2.12×** |
| faest_256s | 257.1 ms    | 113.4 ms      | **2.27×** |

CPU baseline: faest-ref v2.0.4 release+LTO build.  
CUDA: BLOCK_SIZE=128, separate warm-up run, steady-state median (n=29, rows 2–30).

---

## Scope

**Implemented:**
- Sign-path `leaf_hash` offload for AES parameter sets with λ = 128, 192, 256
- Official performance evaluation for `faest_128s`, `faest_192s`, `faest_256s`

**Not covered in this evaluation:**
- Verify-path offload
- EM variants (no `leaf_hash` bottleneck in EM path)
- Official performance evaluation for f-variants

---

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 3060 Laptop (Ampere, CC 8.6, 6 GB GDDR6) |
| CUDA | 13.2 (V13.2.51) |
| Driver | 595.58.03 |
| GCC | 13.3.0 |
| OS | Ubuntu 24.04.3 LTS |
| Meson | 1.3.2 / Ninja 1.11.1 |

---

## Based on

This repository is based on
[faest-sign/faest-ref v2.0.4](https://github.com/faest-sign/faest-ref)
(upstream commit `5113c66`), licensed under the MIT License.
The original `LICENSE` file is preserved.

Modified tracked files: `.gitignore`, `bavc.c`, `bavc.h`, `meson.build`,
`meson_options.txt`, `tests/meson.build`.  
All new CUDA source files are in the `cuda/` directory (15 files).

---

## Build

### Requirements

- CUDA Toolkit 13.x (`nvcc`)
- GCC 13.x
- Meson ≥ 1.0, Ninja
- Boost (`program_options`, `unit_test_framework`)
- OpenSSL

### CUDA correctness build

```bash
meson setup builddir-cuda \
  -Dbuildtype=release \
  -Duse_cuda=true \
  -Dfaest_cuda_arch=sm_86 \
  -Dfaest_cuda_path=/usr/local/cuda
meson compile -C builddir-cuda
```

Replace `sm_86` with your GPU's compute capability
(e.g. `sm_89` for RTX 4080, `sm_86` for RTX 3090).

### CUDA benchmark build

```bash
meson setup builddir-cuda-bench \
  -Dbuildtype=release \
  -Duse_cuda=true \
  -Dfaest_cuda_arch=sm_86 \
  -Dfaest_cuda_path=/usr/local/cuda \
  -Dbenchmarks=enabled
meson compile -C builddir-cuda-bench \
  faest_128s_cuda_bench \
  faest_192s_cuda_bench \
  faest_256s_cuda_bench
```

---

## Test

### Meson CUDA tests (correctness)

```bash
meson test -C builddir-cuda \
  cuda_faest_128s cuda_faest_128f \
  cuda_faest_192s cuda_faest_256s \
  cuda_faest_em_128s
```

### Unit tests (Layer 1–3A)

```bash
cd cuda
make test-all
```

---

## Benchmark

```bash
cd builddir-cuda-bench

# Run warm-up separately before measuring
./faest_128s_cuda_bench -i 1
./faest_192s_cuda_bench -i 1
./faest_256s_cuda_bench -i 1

# Official measurement: -i 30, use rows 2–30 (n=29) as steady-state.
# The first recorded iteration may include CUDA initialization overhead
# even after a separate warm-up run.
./faest_128s_cuda_bench -i 30
./faest_192s_cuda_bench -i 30
./faest_256s_cuda_bench -i 30
```

CSV output format: `keygen_us, sign_us, verify_us`

---

## Implementation Notes

**Two-pass GPU offload structure:**

```
Pass 1 (CPU): PRG → collect uhash/x0/x1 buffers
Pass 2 (GPU): leaf_hash_*_batch_cuda() — N independent calls in parallel
Pass 3 (CPU): copy results to com[] + H1_update (order preserved)
```

**Key design decisions:**
- `sizeof(bf192_t) = 32` but `BF192_NUM_BYTES = 24` —
  all packing uses serialized byte length (NUM_BYTES), not struct sizeof
- BLOCK_SIZE=128 adopted for all variants after comparing BS=64/128/256/512
- GPU path unified for λ=128/192/256 in a single `bavc_commit_faest()` branch

---

## License

This project inherits the MIT License from the original faest-ref implementation.  
Copyright (c) 2023 Sebastian Ramacher, AIT Austrian Institute of Technology.  
See [LICENSE](LICENSE) for details.
