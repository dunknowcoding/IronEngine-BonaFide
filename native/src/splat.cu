// splat.cu — disk splatting CUDA kernel.
//
// Each thread processes one point: project to screen, then atomically
// stamp a screen-space disk into the rgb / depth buffers using
// atomicCAS-based depth tests so closer fragments win.
//
// This is the "compete with gsplat baseline" path. For full 3DGS
// quality (anisotropic Gaussians + spherical harmonics) the wrapper
// still defers to the gsplat library; this kernel is the always-on
// fast path used when the user just wants disks.
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

constexpr int BLOCK_SIZE = 256;

__device__ inline float fatomicMin(float *addr, float value) {
    int *iaddr = reinterpret_cast<int *>(addr);
    int old = __float_as_int(*addr), assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) break;
        old = atomicCAS(iaddr, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void k_splat(const float *positions, const float *colors,
                         std::size_t n_points, const float *vp,
                         int width, int height, int radius_px,
                         float *rgb, float *depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= static_cast<int>(n_points)) return;
    float x = positions[3 * tid + 0];
    float y = positions[3 * tid + 1];
    float z = positions[3 * tid + 2];
    float clip[4] = {
        vp[0]  * x + vp[1]  * y + vp[2]  * z + vp[3],
        vp[4]  * x + vp[5]  * y + vp[6]  * z + vp[7],
        vp[8]  * x + vp[9]  * y + vp[10] * z + vp[11],
        vp[12] * x + vp[13] * y + vp[14] * z + vp[15],
    };
    if (clip[3] <= 0.0f) return;
    float ndc_x = clip[0] / clip[3];
    float ndc_y = clip[1] / clip[3];
    float ndc_z = clip[2] / clip[3];
    int px = static_cast<int>((ndc_x * 0.5f + 0.5f) * width + 0.5f);
    int py = static_cast<int>((1.0f - (ndc_y * 0.5f + 0.5f)) * height + 0.5f);
    if (px < 0 || px >= width || py < 0 || py >= height) return;

    float r = colors[3 * tid + 0];
    float g = colors[3 * tid + 1];
    float b = colors[3 * tid + 2];
    int r2 = radius_px * radius_px;

    for (int dy = -radius_px; dy <= radius_px; ++dy) {
        int yy = py + dy;
        if (yy < 0 || yy >= height) continue;
        for (int dx = -radius_px; dx <= radius_px; ++dx) {
            if (dx * dx + dy * dy > r2 && radius_px > 0) continue;
            int xx = px + dx;
            if (xx < 0 || xx >= width) continue;
            int idx = yy * width + xx;
            float old = fatomicMin(&depth[idx], ndc_z);
            // If our z is the new minimum we won — write the colour.
            if (ndc_z < old) {
                rgb[3 * idx + 0] = r;
                rgb[3 * idx + 1] = g;
                rgb[3 * idx + 2] = b;
            }
        }
    }
}

}  // namespace

void splat_render(const float *positions_d, const float *colors_d,
                  std::size_t n_points, const float *view_proj_d,
                  int width, int height, float point_size_px,
                  float *rgb_d, float *depth_d) {
    if (n_points == 0) return;
    int radius_px = static_cast<int>(0.5f * point_size_px + 0.5f);
    int blocks = (static_cast<int>(n_points) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_splat<<<blocks, BLOCK_SIZE>>>(positions_d, colors_d, n_points, view_proj_d,
                                     width, height, radius_px, rgb_d, depth_d);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace bonafide
