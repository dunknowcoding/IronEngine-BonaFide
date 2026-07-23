// bonafide/api.hpp — host-callable C++ surface used by the nanobind layer.
//
// Each function is a thin shim over a CUDA kernel launch; the bindings
// in src/bindings.cpp marshal numpy / torch tensors to raw device pointers
// and dispatch here. Tensors flow as (ptr, dims, strides) tuples — see
// nanobind's `nb::ndarray<>` docs.
#pragma once

#include <cstddef>
#include <cstdint>

namespace bonafide {

// =================================================================== octree
// Build a balanced octree over `positions` (N, 3 float32) on the device.
// Returns an opaque handle the caller passes back to subsequent walkers.
struct OctreeHandle {
    void *node_buffer = nullptr;
    void *index_buffer = nullptr;
    std::size_t n_nodes = 0;
    std::size_t n_indices = 0;
    int max_depth = 0;
};

OctreeHandle octree_build(const float *positions_d, std::size_t n_points,
                          int leaf_capacity, int max_depth);

// Walk the octree against a camera frustum + screen-space-error budget.
// Writes visible point indices into `out_indices_d` (pre-allocated, size
// hint = n_points). Returns the number of indices actually written.
std::size_t octree_visible(const OctreeHandle &h,
                           const float eye[3], float fov_rad,
                           int image_height, float sse_budget_px,
                           std::int32_t *out_indices_d);

void octree_free(OctreeHandle &h);

// =================================================================== surfel
// kNN + PCA normal + per-point radius.
// Inputs:
//   positions_d     (N, 3) float32 device pointer
//   k               number of neighbours
//   radius_factor   multiply mean k-NN spacing
// Outputs (pre-allocated):
//   normals_d       (N, 3) float32
//   radii_d         (N,)   float32
void surfel_estimate(const float *positions_d, std::size_t n_points,
                     int k, float radius_factor,
                     float *normals_d, float *radii_d);

// =================================================================== splat
// Disk-splat raster — competes with gsplat baseline for sparse clouds.
// `view_proj_d` is a 4x4 row-major float32 matrix on device (16 floats).
void splat_render(const float *positions_d, const float *colors_d,
                  std::size_t n_points,
                  const float *view_proj_d,
                  int width, int height, float point_size_px,
                  float *rgb_d, float *depth_d);

// =================================================================== upload
// Async pinned-host → device transfer on the named stream. Returns the
// number of bytes transferred (== nbytes on success).
std::size_t upload_async(const void *host_src, void *device_dst,
                         std::size_t nbytes, const char *stream_name);

void upload_synchronize(const char *stream_name);

}  // namespace bonafide
