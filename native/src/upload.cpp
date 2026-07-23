// upload.cpp — pinned-host async transfer + named-stream cache.
//
// Streams: a thread-local map from string-name to cudaStream_t; lazily
// created on first use. Same lifecycle as the Python streams module —
// the two never share state but operate identically.
//
// Pinned host memory: callers are expected to own their host buffers;
// for now we just call cudaMemcpyAsync. A pinned-pool helper (bonafide
// ::register_host_buffer) can land in a follow-up.
#include "bonafide/api.hpp"
#include "bonafide/upload.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

#define CUDA_CHECK(stmt) do {                                            \
    cudaError_t err__ = (stmt);                                          \
    if (err__ != cudaSuccess) {                                          \
        std::fprintf(stderr, "[bonafide_native] CUDA error %s at %s:%d: %s\n", \
                     #stmt, __FILE__, __LINE__, cudaGetErrorString(err__));    \
    }                                                                     \
} while (0)

namespace bonafide {

namespace {
std::unordered_map<std::string, cudaStream_t> g_streams;
std::mutex g_streams_mu;
}  // namespace

void *get_or_create_stream(const char *name) {
    std::lock_guard<std::mutex> lock(g_streams_mu);
    auto it = g_streams.find(name);
    if (it != g_streams.end()) return it->second;
    cudaStream_t s = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    g_streams.emplace(name, s);
    return s;
}

std::size_t upload_async(const void *host_src, void *device_dst,
                         std::size_t nbytes, const char *stream_name) {
    if (nbytes == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(get_or_create_stream(stream_name));
    CUDA_CHECK(cudaMemcpyAsync(device_dst, host_src, nbytes,
                               cudaMemcpyHostToDevice, s));
    return nbytes;
}

void upload_synchronize(const char *stream_name) {
    cudaStream_t s = static_cast<cudaStream_t>(get_or_create_stream(stream_name));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

}  // namespace bonafide
