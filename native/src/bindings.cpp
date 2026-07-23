// bindings.cpp -- nanobind module surface.
//
// Maps numpy / torch tensors to raw device pointers and invokes the
// host-callable shims in `include/bonafide/api.hpp`. The Python layer
// in `ironengine_bonafide/backends/cuda/native_bridge.py` is the only
// importer of this module.
//
// Targets nanobind >= 2.0. Notes vs. the 1.x API:
//   * the `_a` argument-literal lives in `nanobind::literals` and must
//     be `using`-imported explicitly.
//   * there is no `nb::any` scalar placeholder -- omit the dtype to
//     accept any scalar type.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cstdint>

#include "bonafide/api.hpp"

namespace nb = nanobind;
using namespace nb::literals;   // brings in the `_a` argument literal
using namespace bonafide;

// Convenience aliases. `nb::c_contig` guarantees a dense row-major
// buffer so `.data()` is a plain pointer we can hand to CUDA.
using CudaFloat = nb::ndarray<nb::pytorch, float, nb::device::cuda, nb::c_contig>;
using CudaI32   = nb::ndarray<nb::pytorch, std::int32_t, nb::device::cuda, nb::c_contig>;
// Untyped views for the async-upload path (any dtype; we only need bytes).
using HostAny   = nb::ndarray<nb::device::cpu, nb::c_contig, nb::ro>;
using CudaAny   = nb::ndarray<nb::device::cuda, nb::c_contig>;

namespace {

OctreeHandle py_octree_build(CudaFloat positions, int leaf_capacity, int max_depth) {
    if (positions.ndim() != 2 || positions.shape(1) != 3) {
        throw nb::type_error("positions must be (N, 3) float32 cuda");
    }
    return octree_build(positions.data(),
                        static_cast<std::size_t>(positions.shape(0)),
                        leaf_capacity, max_depth);
}

std::size_t py_octree_visible(const OctreeHandle &h, std::array<float, 3> eye,
                              float fov_rad, int image_height,
                              float sse_budget_px, CudaI32 out) {
    return octree_visible(h, eye.data(), fov_rad, image_height, sse_budget_px,
                          out.data());
}

void py_octree_free(OctreeHandle &h) { octree_free(h); }

void py_surfel_estimate(CudaFloat positions, int k, float radius_factor,
                        CudaFloat normals, CudaFloat radii) {
    if (positions.ndim() != 2 || positions.shape(1) != 3) {
        throw nb::type_error("positions must be (N, 3) float32 cuda");
    }
    surfel_estimate(positions.data(),
                    static_cast<std::size_t>(positions.shape(0)),
                    k, radius_factor, normals.data(), radii.data());
}

void py_splat_render(CudaFloat positions, CudaFloat colors,
                     CudaFloat view_proj, int width, int height,
                     float point_size_px, CudaFloat rgb, CudaFloat depth) {
    if (positions.shape(0) != colors.shape(0)) {
        throw nb::type_error("positions / colors size mismatch");
    }
    splat_render(positions.data(), colors.data(),
                 static_cast<std::size_t>(positions.shape(0)),
                 view_proj.data(), width, height, point_size_px,
                 rgb.data(), depth.data());
}

std::size_t py_upload_async(HostAny host, CudaAny dev, const std::string &stream) {
    return upload_async(host.data(), dev.data(), host.nbytes(), stream.c_str());
}

void py_upload_sync(const std::string &stream) { upload_synchronize(stream.c_str()); }

}  // namespace

NB_MODULE(bonafide_native, m) {
    m.doc() = "IronEngine-BonaFide native CUDA acceleration layer.";

    nb::class_<OctreeHandle>(m, "OctreeHandle")
        .def_ro("n_nodes", &OctreeHandle::n_nodes)
        .def_ro("n_indices", &OctreeHandle::n_indices)
        .def_ro("max_depth", &OctreeHandle::max_depth);

    m.def("octree_build", &py_octree_build,
          "positions"_a, "leaf_capacity"_a = 4096, "max_depth"_a = 12);
    m.def("octree_visible", &py_octree_visible,
          "handle"_a, "eye"_a, "fov_rad"_a, "image_height"_a,
          "sse_budget_px"_a, "out_indices"_a);
    m.def("octree_free", &py_octree_free, "handle"_a);

    m.def("surfel_estimate", &py_surfel_estimate,
          "positions"_a, "k"_a, "radius_factor"_a,
          "out_normals"_a, "out_radii"_a);

    m.def("splat_render", &py_splat_render,
          "positions"_a, "colors"_a, "view_proj"_a,
          "width"_a, "height"_a, "point_size_px"_a,
          "out_rgb"_a, "out_depth"_a);

    m.def("upload_async", &py_upload_async,
          "host"_a, "device"_a, "stream"_a = "transfer");
    m.def("upload_sync", &py_upload_sync, "stream"_a = "transfer");
}
