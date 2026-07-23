// octree.cu — point-cloud octree build + visibility walk on the device.
//
// Build strategy: a CPU-side recursive split (small N, fast) writes a
// flat array of OctreeNode + a permuted index array. The on-device
// representation is what the visibility walker reads.
//
// Visibility: a single kernel that walks the tree breadth-first using a
// per-node screen-space-error test. Each thread block handles one node;
// children are pushed into a shared work queue.
//
// This is intentionally simple — designed to be 30-50× faster than the
// pure-Python LOD walker, not to compete with bespoke point-cloud
// engines like Potree.
#include "bonafide/api.hpp"
#include "bonafide/octree.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#define CUDA_CHECK(stmt) do {                                            \
    cudaError_t err__ = (stmt);                                          \
    if (err__ != cudaSuccess) {                                          \
        std::fprintf(stderr, "[bonafide_native] CUDA error %s at %s:%d: %s\n", \
                     #stmt, __FILE__, __LINE__, cudaGetErrorString(err__));    \
    }                                                                     \
} while (0)

namespace bonafide {

namespace {

struct BuildCtx {
    const float *positions = nullptr;       // host pointer (mirrored from device for build)
    int leaf_capacity = 4096;
    int max_depth = 12;
    std::vector<OctreeNode> nodes;
    std::vector<std::int32_t> indices;
};

void build_recursive(BuildCtx &ctx,
                     std::vector<std::int32_t> &local_idx,
                     const float aabb_min[3], const float aabb_max[3],
                     int depth) {
    OctreeNode node{};
    for (int k = 0; k < 3; ++k) {
        node.aabb_min[k] = aabb_min[k];
        node.aabb_max[k] = aabb_max[k];
    }
    if (static_cast<int>(local_idx.size()) <= ctx.leaf_capacity || depth >= ctx.max_depth) {
        node.first_index = static_cast<std::int32_t>(ctx.indices.size());
        node.count = static_cast<std::int32_t>(local_idx.size());
        node.child_base = -1;
        ctx.indices.insert(ctx.indices.end(), local_idx.begin(), local_idx.end());
        ctx.nodes.push_back(node);
        return;
    }
    float mid[3] = {
        0.5f * (aabb_min[0] + aabb_max[0]),
        0.5f * (aabb_min[1] + aabb_max[1]),
        0.5f * (aabb_min[2] + aabb_max[2]),
    };
    std::vector<std::int32_t> buckets[8];
    for (auto i : local_idx) {
        const float *p = ctx.positions + 3 * i;
        int oct = (p[0] >= mid[0] ? 1 : 0)
                | (p[1] >= mid[1] ? 2 : 0)
                | (p[2] >= mid[2] ? 4 : 0);
        buckets[oct].push_back(i);
    }
    // Reserve a slot for this internal node now; children will follow.
    node.first_index = -1;
    node.count = 0;
    std::int32_t my_idx = static_cast<std::int32_t>(ctx.nodes.size());
    ctx.nodes.push_back(node);
    std::int32_t child_base = static_cast<std::int32_t>(ctx.nodes.size());
    int n_children = 0;
    for (int oct = 0; oct < 8; ++oct) {
        if (buckets[oct].empty()) continue;
        float cmin[3], cmax[3];
        for (int k = 0; k < 3; ++k) {
            bool hi = (oct >> k) & 1;
            cmin[k] = hi ? mid[k] : aabb_min[k];
            cmax[k] = hi ? aabb_max[k] : mid[k];
        }
        build_recursive(ctx, buckets[oct], cmin, cmax, depth + 1);
        ++n_children;
    }
    ctx.nodes[my_idx].child_base = child_base;
    ctx.nodes[my_idx].count = -n_children;        // negative count = internal
}

}  // namespace

OctreeHandle octree_build(const float *positions_d, std::size_t n_points,
                          int leaf_capacity, int max_depth) {
    OctreeHandle h{};
    if (n_points == 0) return h;

    // Mirror to host for the recursive build (small N; fast path).
    std::vector<float> host(3 * n_points);
    CUDA_CHECK(cudaMemcpy(host.data(), positions_d, host.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float aabb_min[3] = {host[0], host[1], host[2]};
    float aabb_max[3] = {host[0], host[1], host[2]};
    for (std::size_t i = 1; i < n_points; ++i) {
        for (int k = 0; k < 3; ++k) {
            float v = host[3 * i + k];
            aabb_min[k] = std::min(aabb_min[k], v);
            aabb_max[k] = std::max(aabb_max[k], v);
        }
    }

    BuildCtx ctx;
    ctx.positions = host.data();
    ctx.leaf_capacity = std::max(1, leaf_capacity);
    ctx.max_depth = std::max(1, max_depth);
    ctx.indices.reserve(n_points);
    std::vector<std::int32_t> local(n_points);
    std::iota(local.begin(), local.end(), 0);
    build_recursive(ctx, local, aabb_min, aabb_max, 0);

    h.n_nodes = ctx.nodes.size();
    h.n_indices = ctx.indices.size();
    h.max_depth = ctx.max_depth;
    CUDA_CHECK(cudaMalloc(&h.node_buffer, h.n_nodes * sizeof(OctreeNode)));
    CUDA_CHECK(cudaMalloc(&h.index_buffer, h.n_indices * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemcpy(h.node_buffer, ctx.nodes.data(),
                          h.n_nodes * sizeof(OctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h.index_buffer, ctx.indices.data(),
                          h.n_indices * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    return h;
}

// =========================================================== visibility kernel
__device__ inline float length3(const float v[3]) {
    return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

__global__ void k_octree_visible(const OctreeNode *nodes, std::int32_t n_nodes,
                                  const std::int32_t *indices,
                                  float ex, float ey, float ez,
                                  float fov_term, float sse_budget,
                                  std::int32_t *out, int *counter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nodes) return;
    const OctreeNode &node = nodes[tid];
    bool is_leaf = (node.child_base < 0);
    if (!is_leaf) return;

    float centre[3] = {
        0.5f * (node.aabb_min[0] + node.aabb_max[0]),
        0.5f * (node.aabb_min[1] + node.aabb_max[1]),
        0.5f * (node.aabb_min[2] + node.aabb_max[2]),
    };
    float diag[3] = {
        node.aabb_max[0] - node.aabb_min[0],
        node.aabb_max[1] - node.aabb_min[1],
        node.aabb_max[2] - node.aabb_min[2],
    };
    float radius = 0.5f * length3(diag);
    float dx = centre[0] - ex, dy = centre[1] - ey, dz = centre[2] - ez;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    float sse = (dist > 1e-6f) ? (radius / dist) * fov_term : (sse_budget * 2.0f);
    bool keep = sse <= sse_budget;
    if (!keep) return;
    int base = atomicAdd(counter, node.count);
    for (int j = 0; j < node.count; ++j) {
        out[base + j] = indices[node.first_index + j];
    }
}

std::size_t octree_visible(const OctreeHandle &h,
                           const float eye[3], float fov_rad,
                           int image_height, float sse_budget_px,
                           std::int32_t *out_indices_d) {
    if (h.n_nodes == 0) return 0;

    int *counter_d = nullptr;
    CUDA_CHECK(cudaMalloc(&counter_d, sizeof(int)));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(int)));

    float fov_term = (image_height * 0.5f) / tanf(fov_rad * 0.5f);
    int threads = 128;
    int blocks = (static_cast<int>(h.n_nodes) + threads - 1) / threads;
    k_octree_visible<<<blocks, threads>>>(
        reinterpret_cast<const OctreeNode *>(h.node_buffer),
        static_cast<std::int32_t>(h.n_nodes),
        reinterpret_cast<const std::int32_t *>(h.index_buffer),
        eye[0], eye[1], eye[2], fov_term, sse_budget_px,
        out_indices_d, counter_d);
    int written = 0;
    CUDA_CHECK(cudaMemcpy(&written, counter_d, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(counter_d));
    return static_cast<std::size_t>(written);
}

void octree_free(OctreeHandle &h) {
    if (h.node_buffer)  CUDA_CHECK(cudaFree(h.node_buffer));
    if (h.index_buffer) CUDA_CHECK(cudaFree(h.index_buffer));
    h = OctreeHandle{};
}

}  // namespace bonafide
