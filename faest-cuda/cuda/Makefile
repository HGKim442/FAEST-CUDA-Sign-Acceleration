ROOT      ?= $(HOME)/faest-ref-cuda
CUDA_DIR  := $(ROOT)/cuda
BUILD_DIR := $(ROOT)/builddir

# sm_86: RTX 3060 Laptop / RTX 3090
# sm_89: RTX 4080
ARCH ?= sm_86

CC   := gcc
NVCC := nvcc

CFLAGS := -std=c11 -O3 \
          -I$(ROOT) -I$(BUILD_DIR) \
          -I$(ROOT)/sha3 -I$(ROOT)/sha3/opt64 \
          -DHAVE_CONFIG_H -D_GNU_SOURCE \
          -DFAEST_STATIC -DFAEST_EXPORT= -DFAEST_TESTS \
          -Wno-psabi

NVCCFLAGS := -O2 -arch=$(ARCH) -ccbin gcc \
             -I$(ROOT) -I$(BUILD_DIR)

# Additional defines for CUDA-enabled test targets
CUDA_TEST_DEFS := -DUSE_CUDA_GPU -DFAEST_TESTS

.PHONY: all clean test test-bf384 test-bf576 test-bf768 test-leaf test-leaf192 test-leaf256 test-bavc test-bavc192 test-bavc256 test-all

all: $(CUDA_DIR)/test_bf384_mul_128 \
     $(CUDA_DIR)/test_leaf_hash_128 \
     $(CUDA_DIR)/test_bavc_commit_128 \
     $(CUDA_DIR)/test_bf576_mul_192 \
     $(CUDA_DIR)/test_bf768_mul_256 \
     $(CUDA_DIR)/test_leaf_hash_192 \
     $(CUDA_DIR)/test_leaf_hash_256 \
     $(CUDA_DIR)/test_bavc_commit_192 \
     $(CUDA_DIR)/test_bavc_commit_256

# Convert thin archive to fat archive for nvlink
$(CUDA_DIR)/libfaest_cpu.a: $(BUILD_DIR)/libfaest_no_random.a
	printf 'CREATE $@\nADDLIB $<\nSAVE\nEND\n' | ar -M

$(CUDA_DIR)/cpu_ref_bf384.o: $(CUDA_DIR)/cpu_ref_bf384.c \
                              $(CUDA_DIR)/cpu_ref_bf384.h
	$(CC) $(CFLAGS) -c $< -o $@

$(CUDA_DIR)/leaf_hash_cuda.o: $(CUDA_DIR)/leaf_hash_cuda.cu \
                               $(CUDA_DIR)/leaf_hash_cuda.cuh \
                               $(CUDA_DIR)/fields_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# bavc.c compiled with USE_CUDA_GPU to expose the test helper
$(CUDA_DIR)/bavc_test.o: $(ROOT)/bavc.c $(ROOT)/bavc.h
	$(CC) $(CFLAGS) $(CUDA_TEST_DEFS) -c $< -o $@

$(CUDA_DIR)/test_bf384_mul_128: $(CUDA_DIR)/test_bf384_mul_128.cu \
                                 $(CUDA_DIR)/cpu_ref_bf384.o \
                                 $(CUDA_DIR)/libfaest_cpu.a \
                                 $(CUDA_DIR)/fields_cuda.cuh \
                                 $(CUDA_DIR)/cpu_ref_bf384.h
	$(NVCC) $(NVCCFLAGS) $< \
	    $(CUDA_DIR)/cpu_ref_bf384.o \
	    $(CUDA_DIR)/libfaest_cpu.a \
	    -o $@

$(CUDA_DIR)/test_leaf_hash_128: $(CUDA_DIR)/test_leaf_hash_128.cu \
                                 $(CUDA_DIR)/leaf_hash_cuda.o \
                                 $(CUDA_DIR)/cpu_ref_bf384.o \
                                 $(CUDA_DIR)/libfaest_cpu.a \
                                 $(CUDA_DIR)/leaf_hash_cuda.cuh \
                                 $(CUDA_DIR)/cpu_ref_bf384.h
	$(NVCC) $(NVCCFLAGS) $< \
	    $(CUDA_DIR)/leaf_hash_cuda.o \
	    $(CUDA_DIR)/cpu_ref_bf384.o \
	    $(CUDA_DIR)/libfaest_cpu.a \
	    -o $@

$(CUDA_DIR)/test_bavc_commit_128: $(CUDA_DIR)/test_bavc_commit_128.cu \
                                   $(CUDA_DIR)/bavc_test.o \
                                   $(CUDA_DIR)/leaf_hash_cuda.o \
                                   $(CUDA_DIR)/cpu_ref_bf384.o \
                                   $(CUDA_DIR)/libfaest_cpu.a \
                                   $(ROOT)/bavc.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_TEST_DEFS) $< \
	    $(CUDA_DIR)/bavc_test.o \
	    $(CUDA_DIR)/leaf_hash_cuda.o \
	    $(CUDA_DIR)/cpu_ref_bf384.o \
	    $(CUDA_DIR)/libfaest_cpu.a \
	    -lssl -lcrypto \
	    -o $@

$(CUDA_DIR)/test_leaf_hash_192: $(CUDA_DIR)/test_leaf_hash_192.cu \
                                 $(CUDA_DIR)/leaf_hash_cuda.o \
                                 $(CUDA_DIR)/cpu_ref_bf384.o \
                                 $(CUDA_DIR)/libfaest_cpu.a \
                                 $(CUDA_DIR)/leaf_hash_cuda.cuh \
                                 $(CUDA_DIR)/cpu_ref_bf384.h
	$(NVCC) $(NVCCFLAGS) $< \
	    $(CUDA_DIR)/leaf_hash_cuda.o \
	    $(CUDA_DIR)/cpu_ref_bf384.o \
	    $(CUDA_DIR)/libfaest_cpu.a \
	    -o $@

$(CUDA_DIR)/test_leaf_hash_256: $(CUDA_DIR)/test_leaf_hash_256.cu \
                                 $(CUDA_DIR)/leaf_hash_cuda.o \
                                 $(CUDA_DIR)/cpu_ref_bf384.o \
                                 $(CUDA_DIR)/libfaest_cpu.a \
                                 $(CUDA_DIR)/leaf_hash_cuda.cuh \
                                 $(CUDA_DIR)/cpu_ref_bf384.h
	$(NVCC) $(NVCCFLAGS) $< \
	    $(CUDA_DIR)/leaf_hash_cuda.o \
	    $(CUDA_DIR)/cpu_ref_bf384.o \
	    $(CUDA_DIR)/libfaest_cpu.a \
	    -o $@

$(CUDA_DIR)/test_bavc_commit_192: $(CUDA_DIR)/test_bavc_commit_192.cu \
                                   $(CUDA_DIR)/bavc_test.o \
                                   $(CUDA_DIR)/leaf_hash_cuda.o \
                                   $(CUDA_DIR)/cpu_ref_bf384.o \
                                   $(CUDA_DIR)/libfaest_cpu.a \
                                   $(ROOT)/bavc.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_TEST_DEFS) $< \
	    $(CUDA_DIR)/bavc_test.o \
	    $(CUDA_DIR)/leaf_hash_cuda.o \
	    $(CUDA_DIR)/cpu_ref_bf384.o \
	    $(CUDA_DIR)/libfaest_cpu.a \
	    -lssl -lcrypto \
	    -o $@

$(CUDA_DIR)/test_bavc_commit_256: $(CUDA_DIR)/test_bavc_commit_256.cu \
                                   $(CUDA_DIR)/bavc_test.o \
                                   $(CUDA_DIR)/leaf_hash_cuda.o \
                                   $(CUDA_DIR)/cpu_ref_bf384.o \
                                   $(CUDA_DIR)/libfaest_cpu.a \
                                   $(ROOT)/bavc.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_TEST_DEFS) $< \
	    $(CUDA_DIR)/bavc_test.o \
	    $(CUDA_DIR)/leaf_hash_cuda.o \
	    $(CUDA_DIR)/cpu_ref_bf384.o \
	    $(CUDA_DIR)/libfaest_cpu.a \
	    -lssl -lcrypto \
	    -o $@

$(CUDA_DIR)/test_bf576_mul_192: $(CUDA_DIR)/test_bf576_mul_192.cu \
                                 $(CUDA_DIR)/cpu_ref_bf384.o \
                                 $(CUDA_DIR)/libfaest_cpu.a \
                                 $(CUDA_DIR)/fields_cuda.cuh \
                                 $(CUDA_DIR)/cpu_ref_bf384.h
	$(NVCC) $(NVCCFLAGS) $< \
	    $(CUDA_DIR)/cpu_ref_bf384.o \
	    $(CUDA_DIR)/libfaest_cpu.a \
	    -o $@

$(CUDA_DIR)/test_bf768_mul_256: $(CUDA_DIR)/test_bf768_mul_256.cu \
                                 $(CUDA_DIR)/cpu_ref_bf384.o \
                                 $(CUDA_DIR)/libfaest_cpu.a \
                                 $(CUDA_DIR)/fields_cuda.cuh \
                                 $(CUDA_DIR)/cpu_ref_bf384.h
	$(NVCC) $(NVCCFLAGS) $< \
	    $(CUDA_DIR)/cpu_ref_bf384.o \
	    $(CUDA_DIR)/libfaest_cpu.a \
	    -o $@

test-bf384: $(CUDA_DIR)/test_bf384_mul_128
	$(CUDA_DIR)/test_bf384_mul_128

test_bf576_mul_192: $(CUDA_DIR)/test_bf576_mul_192

test_bf768_mul_256: $(CUDA_DIR)/test_bf768_mul_256

test_leaf_hash_192: $(CUDA_DIR)/test_leaf_hash_192

test_leaf_hash_256: $(CUDA_DIR)/test_leaf_hash_256

test-bf576: $(CUDA_DIR)/test_bf576_mul_192
	$(CUDA_DIR)/test_bf576_mul_192

test-bf768: $(CUDA_DIR)/test_bf768_mul_256
	$(CUDA_DIR)/test_bf768_mul_256

test-leaf: $(CUDA_DIR)/test_leaf_hash_128
	$(CUDA_DIR)/test_leaf_hash_128

test-leaf192: $(CUDA_DIR)/test_leaf_hash_192
	$(CUDA_DIR)/test_leaf_hash_192

test-leaf256: $(CUDA_DIR)/test_leaf_hash_256
	$(CUDA_DIR)/test_leaf_hash_256

test-bavc: $(CUDA_DIR)/test_bavc_commit_128
	$(CUDA_DIR)/test_bavc_commit_128

test_bavc_commit_192: $(CUDA_DIR)/test_bavc_commit_192

test_bavc_commit_256: $(CUDA_DIR)/test_bavc_commit_256

test-bavc192: $(CUDA_DIR)/test_bavc_commit_192
	$(CUDA_DIR)/test_bavc_commit_192

test-bavc256: $(CUDA_DIR)/test_bavc_commit_256
	$(CUDA_DIR)/test_bavc_commit_256

test-all: all
	$(CUDA_DIR)/test_bf384_mul_128
	$(CUDA_DIR)/test_bf576_mul_192
	$(CUDA_DIR)/test_bf768_mul_256
	$(CUDA_DIR)/test_leaf_hash_128
	$(CUDA_DIR)/test_leaf_hash_192
	$(CUDA_DIR)/test_leaf_hash_256
	$(CUDA_DIR)/test_bavc_commit_128
	$(CUDA_DIR)/test_bavc_commit_192
	$(CUDA_DIR)/test_bavc_commit_256

test: test-all

clean:
	rm -f $(CUDA_DIR)/cpu_ref_bf384.o \
	      $(CUDA_DIR)/leaf_hash_cuda.o \
	      $(CUDA_DIR)/bavc_test.o \
	      $(CUDA_DIR)/libfaest_cpu.a \
	      $(CUDA_DIR)/test_bf384_mul_128 \
	      $(CUDA_DIR)/test_leaf_hash_128 \
	      $(CUDA_DIR)/test_bavc_commit_128 \
	      $(CUDA_DIR)/test_bf576_mul_192 \
	      $(CUDA_DIR)/test_bf768_mul_256 \
	      $(CUDA_DIR)/test_leaf_hash_192 \
	      $(CUDA_DIR)/test_leaf_hash_256 \
	      $(CUDA_DIR)/test_bavc_commit_192 \
	      $(CUDA_DIR)/test_bavc_commit_256
