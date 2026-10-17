/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <cstdio>
#include <vector>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

// Run with P2P disabled and GPUNetIO GDAKI enabled to exercise transport
// coalescing. The same payload oracle is also valid on other transports.
static constexpr size_t kGuard = 32;
__host__ __device__ static unsigned char payload(size_t offset, int pe, int epoch) {
    return static_cast<unsigned char>((offset * 13 + pe * 67 + epoch * 17) % 251);
}
__host__ __device__ static size_t slot_offset(int lane, size_t bytes, int mode, bool source) {
    // Modes: contiguous, source stride, destination stride, both strides,
    // unequal lengths, partial warp, mixed peers, destination address gap.
    bool strided = mode == 3 || (source ? mode == 1 : mode == 2);
    size_t offset = kGuard + static_cast<size_t>(lane) * bytes * (strided ? 2 : 1);
    if (mode == 7 && !source && lane >= 16) offset += 5;
    return offset;
}
__global__ void initialize(unsigned char *source, size_t size, int pe, int epoch) {
    for (size_t i = threadIdx.x; i < size; i += blockDim.x) source[i] = payload(i, pe, epoch);
}
__global__ void issue(unsigned char *dest, unsigned char *source, size_t region, size_t bytes,
                      int mode, int pe, int npes, int repeats) {
    int lane = threadIdx.x;
    size_t count = mode == 4 && bytes >= 4 ? bytes - lane % 3 : bytes;
    int target = (pe + (mode == 6 && lane % 2 ? 2 : 1)) % npes;
    for (int i = 0; i < repeats; ++i) {
        if (mode != 5 || lane < 17) {
            nvshmem_putmem_nbi(dest + static_cast<size_t>(pe) * region + slot_offset(lane, bytes, mode, false),
                              source + slot_offset(lane, bytes, mode, true), count, target);
        }
    }
    __syncwarp();
    if (lane == 0) nvshmem_quiet();
}
int main(int argc, char **argv) {
    init_wrapper(&argc, &argv);
    int pe = nvshmem_my_pe(), npes = nvshmem_n_pes();
    if (npes < 2) {
        if (pe == 0) std::printf("SKIP: requires at least two PEs\n");
        finalize_wrapper();
        return 0;
    }
    constexpr size_t capacity = 2 * kGuard + 32 * 65536 * 2;
    auto *source = static_cast<unsigned char *>(nvshmem_malloc(capacity));
    auto *dest = static_cast<unsigned char *>(nvshmem_malloc(capacity * npes));
    if (!source || !dest) nvshmem_global_exit(2);
    int errors = 0, epoch = 0;
    for (size_t bytes : {size_t(1), size_t(4), size_t(16), size_t(256), size_t(4096), size_t(65536)}) {
        const size_t region = 2 * kGuard + 32 * bytes * 2;
        const size_t total = region * npes;
        std::vector<unsigned char> observed(total), expected(total);
        for (int mode = 0; mode < 8; ++mode, ++epoch) {
            initialize<<<1, 256>>>(source, region, pe, epoch);
            CUDA_CHECK(cudaMemset(dest, 0xa5, total));
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();
            // Force multiple SQ wraps in the combined path, and mix completed
            // combined/fallback calls across persistent QP epochs.
            int repeats = bytes == 16 && mode == 0 ? 4097 : 3;
            issue<<<1, 32>>>(dest, source, region, bytes, mode, pe, npes, repeats);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();
            CUDA_CHECK(cudaMemcpy(observed.data(), dest, total, cudaMemcpyDeviceToHost));
            std::fill(expected.begin(), expected.end(), 0xa5);
            for (int sender = 0; sender < npes; ++sender) {
                for (int lane = 0; lane < 32; ++lane) {
                    int target = (sender + (mode == 6 && lane % 2 ? 2 : 1)) % npes;
                    if (target != pe || (mode == 5 && lane >= 17)) continue;
                    size_t count = mode == 4 && bytes >= 4 ? bytes - lane % 3 : bytes;
                    size_t src = slot_offset(lane, bytes, mode, true);
                    size_t dst = static_cast<size_t>(sender) * region + slot_offset(lane, bytes, mode, false);
                    for (size_t i = 0; i < count; ++i) expected[dst + i] = payload(src + i, sender, epoch);
                }
            }
            if (observed != expected) {
                std::fprintf(stderr, "PE %d: bytes=%zu mode=%d failed payload/guard check\n", pe, bytes, mode);
                ++errors;
            }
            nvshmem_barrier_all();
        }
    }
    // A contiguous warp crosses two separately registered source ranges.
    // The guard offset also makes one lane's PUT cross the boundary itself.
    unsigned char *registered = nullptr;
    CUDA_CHECK(cudaMalloc(&registered, capacity));
    constexpr size_t split = 16 * 65536;
    if (nvshmemx_buffer_register(registered, split) != NVSHMEMX_SUCCESS ||
        nvshmemx_buffer_register(registered + split, capacity - split) != NVSHMEMX_SUCCESS) {
        nvshmem_global_exit(3);
    }
    initialize<<<1, 256>>>(registered, capacity, pe, epoch);
    CUDA_CHECK(cudaMemset(dest, 0xa5, capacity * npes));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    issue<<<1, 32>>>(dest, registered, capacity, 65536, 0, pe, npes, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    std::vector<unsigned char> observed(capacity * npes), expected(capacity * npes, 0xa5);
    CUDA_CHECK(cudaMemcpy(observed.data(), dest, observed.size(), cudaMemcpyDeviceToHost));
    const int previous = (pe + npes - 1) % npes;
    for (size_t i = kGuard; i < kGuard + 32 * 65536; ++i) {
        expected[static_cast<size_t>(previous) * capacity + i] = payload(i, previous, epoch);
    }
    if (observed != expected) {
        std::fprintf(stderr, "PE %d: registered source boundary failed payload/guard check\n", pe);
        ++errors;
    }
    nvshmem_barrier_all();
    if (nvshmemx_buffer_unregister(registered) != NVSHMEMX_SUCCESS ||
        nvshmemx_buffer_unregister(registered + split) != NVSHMEMX_SUCCESS) {
        nvshmem_global_exit(4);
    }
    CUDA_CHECK(cudaFree(registered));
    if (errors) nvshmem_global_exit(1);
    if (pe == 0) std::printf("PASS: 49 contiguous/fallback cases, guards, registration boundaries and SQ reuse\n");
    nvshmem_free(dest);
    nvshmem_free(source);
    finalize_wrapper();
    return 0;
}
