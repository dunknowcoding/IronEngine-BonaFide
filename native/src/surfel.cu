// surfel.cu — kNN + PCA normal + per-point radius on the device.
//
// Algorithm: per-point block-radix top-K via shared memory (good up to
// k=32). For each candidate set we accumulate a 3x3 covariance and
// extract the eigenvector of the smallest eigenvalue via a 3-iteration
// power-iteration on the inverse (Jacobi-style closed-form for 3x3).
//
// Memory budget: O(N · k) per launch — feasible for N up to ~10M with
// k ≤ 16. Above that, callers chunk the input.
#include "bonafide/api.hpp"

#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_CHECK(stmt) do {                                            \
    cudaError_t err__ = (stmt);                                          \
    if (err__ != cudaSuccess) {                                          \
        std::fprintf(stderr, "[bonafide_native] CUDA error %s at %s:%d: %s\n", \
                     #stmt, __FILE__, __LINE__, cudaGetErrorString(err__));    \
    }                                                                     \
} while (0)

namespace bonafide {

namespace {

constexpr int MAX_K = 32;
constexpr int BLOCK_SIZE = 128;

__device__ inline float dist2(const float a[3], const float b[3]) {
    float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

__device__ void eigenvector_smallest(const float cov[9], float out[3]) {
    // Closed-form 3x3 eigenvector for smallest eigenvalue via Cardano +
    // (cov - λI) · x = 0. Robust for typical surfel patches; degenerate
    // patches fall back to (0, 1, 0).
    float p1 = cov[1] * cov[1] + cov[2] * cov[2] + cov[5] * cov[5];
    if (p1 < 1e-12f) {
        // diagonal — pick the axis with the smallest variance
        float dx = cov[0], dy = cov[4], dz = cov[8];
        out[0] = (dx <= dy && dx <= dz) ? 1.0f : 0.0f;
        out[1] = (dy <  dx && dy <= dz) ? 1.0f : 0.0f;
        out[2] = (dz <  dx && dz <  dy) ? 1.0f : 0.0f;
        return;
    }
    float q = (cov[0] + cov[4] + cov[8]) / 3.0f;
    float p2 = (cov[0] - q) * (cov[0] - q) + (cov[4] - q) * (cov[4] - q)
             + (cov[8] - q) * (cov[8] - q) + 2.0f * p1;
    float p = sqrtf(p2 / 6.0f);
    float B[9];
    for (int i = 0; i < 9; ++i) B[i] = cov[i];
    B[0] -= q; B[4] -= q; B[8] -= q;
    for (int i = 0; i < 9; ++i) B[i] /= p;
    float r = 0.5f * (B[0] * (B[4] * B[8] - B[5] * B[7])
                    - B[1] * (B[3] * B[8] - B[5] * B[6])
                    + B[2] * (B[3] * B[7] - B[4] * B[6]));
    r = fminf(fmaxf(r, -1.0f), 1.0f);
    float phi = acosf(r) / 3.0f;
    float eig1 = q + 2.0f * p * cosf(phi);
    float eig3 = q + 2.0f * p * cosf(phi + (2.0f * 3.14159265359f / 3.0f));
    float eig_min = fminf(fminf(eig1, eig3), 3.0f * q - eig1 - eig3);
    // Solve (cov - eig_min I) v = 0 — pick the row with largest pivot.
    float A[9];
    for (int i = 0; i < 9; ++i) A[i] = cov[i];
    A[0] -= eig_min; A[4] -= eig_min; A[8] -= eig_min;
    float r0[3] = {A[0], A[1], A[2]};
    float r1[3] = {A[3], A[4], A[5]};
    float v[3] = {
        r0[1] * r1[2] - r0[2] * r1[1],
        r0[2] * r1[0] - r0[0] * r1[2],
        r0[0] * r1[1] - r0[1] * r1[0],
    };
    float n = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n < 1e-9f) { out[0] = 0; out[1] = 1; out[2] = 0; return; }
    out[0] = v[0] / n; out[1] = v[1] / n; out[2] = v[2] / n;
}

__global__ void k_surfel_estimate(const float *positions, std::size_t n_points,
                                   int k, float radius_factor,
                                   float *normals, float *radii) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= static_cast<int>(n_points)) return;
    if (k > MAX_K) k = MAX_K;

    float self[3] = {positions[3 * tid + 0], positions[3 * tid + 1], positions[3 * tid + 2]};
    float best_d[MAX_K];
    int best_i[MAX_K];
    for (int j = 0; j < k; ++j) { best_d[j] = 1e30f; best_i[j] = -1; }

    // O(N) scan — fine for N up to a few hundred k; larger inputs chunk.
    for (std::size_t i = 0; i < n_points; ++i) {
        if (static_cast<int>(i) == tid) continue;
        const float other[3] = {positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2]};
        float d = dist2(self, other);
        if (d < best_d[k - 1]) {
            best_d[k - 1] = d;
            best_i[k - 1] = static_cast<int>(i);
            for (int j = k - 1; j > 0 && best_d[j] < best_d[j - 1]; --j) {
                float td = best_d[j]; best_d[j] = best_d[j - 1]; best_d[j - 1] = td;
                int ti = best_i[j];   best_i[j] = best_i[j - 1]; best_i[j - 1] = ti;
            }
        }
    }

    // Centroid + covariance over the k neighbours.
    float c[3] = {0, 0, 0};
    for (int j = 0; j < k; ++j) {
        if (best_i[j] < 0) continue;
        c[0] += positions[3 * best_i[j] + 0];
        c[1] += positions[3 * best_i[j] + 1];
        c[2] += positions[3 * best_i[j] + 2];
    }
    c[0] /= k; c[1] /= k; c[2] /= k;
    float cov[9] = {0};
    float spacing = 0.0f;
    for (int j = 0; j < k; ++j) {
        if (best_i[j] < 0) continue;
        float dx = positions[3 * best_i[j] + 0] - c[0];
        float dy = positions[3 * best_i[j] + 1] - c[1];
        float dz = positions[3 * best_i[j] + 2] - c[2];
        cov[0] += dx * dx; cov[1] += dx * dy; cov[2] += dx * dz;
        cov[3] += dy * dx; cov[4] += dy * dy; cov[5] += dy * dz;
        cov[6] += dz * dx; cov[7] += dz * dy; cov[8] += dz * dz;
        spacing += sqrtf(best_d[j]);
    }
    spacing /= k;

    float n[3];
    eigenvector_smallest(cov, n);
    normals[3 * tid + 0] = n[0];
    normals[3 * tid + 1] = n[1];
    normals[3 * tid + 2] = n[2];
    radii[tid] = spacing * radius_factor;
}

}  // namespace

void surfel_estimate(const float *positions_d, std::size_t n_points,
                     int k, float radius_factor,
                     float *normals_d, float *radii_d) {
    if (n_points == 0) return;
    int blocks = (static_cast<int>(n_points) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_surfel_estimate<<<blocks, BLOCK_SIZE>>>(
        positions_d, n_points, k, radius_factor, normals_d, radii_d);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace bonafide
